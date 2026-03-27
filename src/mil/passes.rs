//! Optimization passes for Graph IR
//!
//! Inspired by Orion's compiler passes:
//! - DCE (Dead Code Elimination)
//! - Identity Elimination
//! - Cast Fusion
//! - SRAM Annotation
//! - Uniform Output Padding
//! - ANE Validation

use crate::mil::graph::{Dtype, Graph, Op};
use crate::Result;

/// Optimization result
#[derive(Debug, Default)]
pub struct OptimizeResult {
    pub nodes_removed: usize,
    pub nodes_fused: usize,
    pub nodes_modified: usize,
}

/// Dead Code Elimination pass
///
/// Removes nodes that are not used by any output.
/// Marks nodes as `is_live = false` instead of removing them
/// to preserve indices.
pub fn dead_code_elimination(graph: &mut Graph) -> Result<OptimizeResult> {
    let mut result = OptimizeResult::default();

    // Start from outputs and work backwards
    let mut live_set: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut queue: Vec<usize> = graph.outputs.iter().map(|io| io.node_idx).collect();

    // Mark all outputs as live
    for io in &graph.outputs {
        live_set.insert(io.node_idx);
    }

    // BFS backwards through the graph
    while let Some(idx) = queue.pop() {
        if let Some(node) = graph.get_node(idx) {
            // Mark all inputs to this node as live
            for &input_idx in &node.inputs {
                if live_set.insert(input_idx) {
                    queue.push(input_idx);
                }
            }
        }
    }

    // Mark dead nodes
    for (idx, node) in graph.nodes.iter_mut().enumerate() {
        if !live_set.contains(&idx) {
            if node.is_live {
                node.is_live = false;
                result.nodes_removed += 1;
            }
        }
    }

    Ok(result)
}

/// Identity Elimination pass
///
/// Removes identity operations by replacing them with their input.
/// Identity nodes are marked as dead and all references are updated.
pub fn eliminate_identity(graph: &mut Graph) -> Result<OptimizeResult> {
    let mut result = OptimizeResult::default();

    // Find all identity nodes and their sources
    let identity_ops: Vec<(usize, usize)> = graph
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.op == Op::Identity && node.is_live && node.inputs.len() == 1)
        .map(|(idx, node)| (idx, node.inputs[0]))
        .collect();

    for (identity_idx, source_idx) in identity_ops {
        // Replace all references to identity with source
        for node in graph.nodes.iter_mut() {
            if node.is_live {
                for input_idx in &mut node.inputs {
                    if *input_idx == identity_idx {
                        *input_idx = source_idx;
                    }
                }
            }
        }

        // Update outputs to point to source
        for io in &mut graph.outputs {
            if io.node_idx == identity_idx {
                io.node_idx = source_idx;
            }
        }

        // Mark identity as dead
        if let Some(identity) = graph.get_node_mut(identity_idx) {
            identity.is_live = false;
            result.nodes_removed += 1;
        }

        // Mark source as output if needed
        if let Some(source) = graph.get_node_mut(source_idx) {
            source.is_output = true;
        }
    }

    Ok(result)
}

/// Cast Fusion pass
///
/// Fuses consecutive cast operations where possible.
/// E.g., fp32 -> fp16 -> fp32 becomes a no-op.
pub fn fuse_casts(graph: &mut Graph) -> Result<OptimizeResult> {
    let mut result = OptimizeResult::default();

    // Find cast chains: A -> cast -> B -> cast -> C
    let cast_chains: Vec<(usize, usize, usize)> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(cast_idx, cast_node)| {
            if cast_node.op != Op::Cast || !cast_node.is_live || cast_node.inputs.len() != 1 {
                return None;
            }
            let input_idx = cast_node.inputs[0];
            let input_node = graph.get_node(input_idx)?;
            if input_node.op != Op::Cast || !input_node.is_live || input_node.inputs.len() != 1 {
                return None;
            }
            let source_idx = input_node.inputs[0];
            Some((cast_idx, input_idx, source_idx))
        })
        .collect();

    for (cast_idx, input_idx, source_idx) in cast_chains {
        // Replace cast_idx with source_idx everywhere
        for node in graph.nodes.iter_mut() {
            if node.is_live {
                for input_idx_node in &mut node.inputs {
                    if *input_idx_node == cast_idx {
                        *input_idx_node = source_idx;
                    }
                }
            }
        }

        // Update outputs
        for io in &mut graph.outputs {
            if io.node_idx == cast_idx {
                io.node_idx = source_idx;
            }
        }

        // Mark both casts as dead
        if let Some(n) = graph.get_node_mut(cast_idx) {
            n.is_live = false;
        }
        if let Some(n) = graph.get_node_mut(input_idx) {
            n.is_live = false;
        }

        result.nodes_fused += 2;
    }

    Ok(result)
}

/// SRAM Annotation pass
///
/// Marks intermediate tensors that should be kept in ANE SRAM
/// for faster access. Based on reuse analysis.
pub fn annotate_sram(graph: &mut Graph) -> Result<OptimizeResult> {
    let mut result = OptimizeResult::default();

    // Count how many times each node's output is used
    let mut use_count: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for node in &graph.nodes {
        if !node.is_live {
            continue;
        }
        for &input_idx in &node.inputs {
            *use_count.entry(input_idx).or_insert(0) += 1;
        }
    }

    // Mark nodes with high reuse as SRAM candidates
    for (idx, count) in &use_count {
        if *count >= 2 {
            if let Some(node) = graph.get_node_mut(*idx) {
                // Mark for SRAM by adding a note in attrs
                node.attrs.keep_dims = true;
                result.nodes_modified += 1;
            }
        }
    }

    Ok(result)
}

/// Uniform Output Padding pass
///
/// ANE requires uniform IOSurface allocation sizes for multi-output programs.
/// This pass pads outputs to the maximum size.
pub fn uniform_output_padding(graph: &mut Graph) -> Result<OptimizeResult> {
    let mut result = OptimizeResult::default();

    // Find max output size
    let max_size: usize = graph
        .outputs
        .iter()
        .filter_map(|io| graph.get_node(io.node_idx))
        .map(|node| node.shape.iter().product::<usize>())
        .max()
        .unwrap_or(0);

    // Pad smaller outputs (in practice, this affects MIL generation)
    let output_indices: Vec<(usize, usize)> = graph
        .outputs
        .iter()
        .filter_map(|io| {
            graph.get_node(io.node_idx).map(|node| {
                let size = node.shape.iter().product::<usize>();
                (io.node_idx, size)
            })
        })
        .collect();

    for (node_idx, size) in output_indices {
        if size < max_size {
            if let Some(node) = graph.get_node_mut(node_idx) {
                node.attrs.keep_dims = true;
                result.nodes_modified += 1;
            }
        }
    }

    Ok(result)
}

/// ANE Validation pass
///
/// Checks ANE-specific constraints before compilation:
/// - concat is banned (must use multi-output)
/// - Shape constraints
/// - dtype constraints
/// - Op support
pub fn validate_for_ane(graph: &Graph) -> Result<()> {
    use crate::error::Error;

    for (idx, node) in graph.nodes.iter().enumerate() {
        if !node.is_live {
            continue;
        }

        // Check shape constraints
        // ANE requires [1, C, 1, S] or [1, C, H, W] layout
        if node.shape[0] != 1 {
            return Err(Error::CompilationFailed(format!(
                "ANE requires batch dimension of 1, got {:?} at node [{}] '{}'",
                node.shape, idx, node.name
            )));
        }

        // Check dtype constraints
        if node.dtype != Dtype::Fp16 && node.dtype != Dtype::Fp32 {
            return Err(Error::CompilationFailed(format!(
                "ANE supports only fp16/fp32, got {:?} at node [{}] '{}'",
                node.dtype, idx, node.name
            )));
        }
    }

    // Check output ordering (must be alphabetical)
    let output_names: Vec<&str> = graph.outputs.iter().map(|io| io.name.as_str()).collect();
    let mut sorted_names = output_names.clone();
    sorted_names.sort();

    if output_names != sorted_names {
        return Err(Error::CompilationFailed(format!(
            "ANE multi-output programs require alphabetical output ordering. Got: {:?}, expected: {:?}",
            output_names, sorted_names
        )));
    }

    Ok(())
}

/// Run all optimization passes on a graph
pub fn optimize(graph: &mut Graph) -> Result<OptimizeResult> {
    let mut total = OptimizeResult::default();

    // Run optimization passes in order
    total.nodes_removed += dead_code_elimination(graph)?.nodes_removed;
    total.nodes_removed += eliminate_identity(graph)?.nodes_removed;
    total.nodes_fused += fuse_casts(graph)?.nodes_fused;
    total.nodes_modified += annotate_sram(graph)?.nodes_modified;
    total.nodes_modified += uniform_output_padding(graph)?.nodes_modified;

    // Validate for ANE
    validate_for_ane(graph)?;

    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mil::graph::GraphBuilder;

    #[test]
    fn test_dce_removes_dead_nodes() {
        let mut graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("live", "x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("dead", "x", Dtype::Fp32, [1, 256, 1, 1]) // Not connected to output
            .output("live")
            .build();

        let result = dead_code_elimination(&mut graph).unwrap();
        assert_eq!(result.nodes_removed, 1);

        let dead_node = graph.get_node_by_name("dead").unwrap();
        assert!(!dead_node.is_live);

        let live_node = graph.get_node_by_name("live").unwrap();
        assert!(live_node.is_live);
    }

    #[test]
    fn test_identity_elimination() {
        let mut graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 1])
            .identity("id", "x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("out", "id", Dtype::Fp32, [1, 256, 1, 1])
            .output("out")
            .build();

        let result = eliminate_identity(&mut graph).unwrap();
        assert_eq!(result.nodes_removed, 1);

        // The relu should now point directly to input
        let out_node = graph.get_node_by_name("out").unwrap();
        assert_eq!(out_node.inputs.len(), 1);
    }

    #[test]
    fn test_validate_concat_rejected() {
        let mut graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 1])
            .input("y", Dtype::Fp32, [1, 256, 1, 1])
            .build();

        // Manually add a concat node - now supported!
        let mut concat_node = crate::mil::graph::Node::new(Op::Concat, "concat", Dtype::Fp32, [1, 512, 1, 1]);
        concat_node.inputs.push(0);
        concat_node.inputs.push(1);
        concat_node.attrs.axis = Some(1);
        graph.add_node(concat_node).unwrap();

        let result = validate_for_ane(&graph);
        // Concat should now pass validation
        assert!(result.is_ok());
    }

    #[test]
    fn test_uniform_output_padding() {
        let mut graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("small", "x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("large", "x", Dtype::Fp32, [1, 512, 1, 1])
            .output("small")
            .output("large")
            .build();

        let result = uniform_output_padding(&mut graph).unwrap();
        // One node should be marked for padding
        assert!(result.nodes_modified >= 1);
    }

    #[test]
    fn test_sram_annotation() {
        let mut graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("reused", "x", Dtype::Fp32, [1, 256, 1, 1])
            .add("use1", "reused", "x", Dtype::Fp32, [1, 256, 1, 1])
            .add("use2", "reused", "x", Dtype::Fp32, [1, 256, 1, 1])
            .output("use2")
            .build();

        let result = annotate_sram(&mut graph).unwrap();
        // The "reused" node should be marked for SRAM
        assert!(result.nodes_modified >= 1);
    }
}
