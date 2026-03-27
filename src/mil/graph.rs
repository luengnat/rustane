//! Graph IR for ANE MIL compilation
//!
//! Single-level graph intermediate representation with optimization passes.
//! Inspired by Orion's compiler/graph.h
//!
//! # Architecture
//!
//! ```text
//! Graph
//! ├── inputs: [GraphIO]
//! ├── outputs: [GraphIO]
//! └── nodes: [Node]
//!     ├── op: Op
//!     ├── inputs: [node_index]
//!     ├── attrs: Attrs
//!     └── shape: [usize; 4]
//! ```

use crate::error::{Error, Result};
use std::collections::{HashMap, VecDeque};

/// Maximum limits (matching Orion)
pub const MAX_INPUTS: usize = 8;
pub const MAX_OUTPUTS: usize = 8;
pub const MAX_NODES: usize = 4096;
pub const MAX_NAME: usize = 64;
pub const MAX_GRAPH_IO: usize = 16;

/// Operations - map 1:1 to MIL ops
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Op {
    Input = 0,        // Graph input placeholder
    Const,            // Constant / weight reference
    Conv1x1,          // 1x1 convolution (linear layer)
    Add,
    Sub,
    Mul,
    MatMul,
    Reshape,
    Transpose,
    Cast,
    ReLU,
    Tanh,
    Sigmoid,
    Softmax,
    Exp,
    Pow,
    ReduceSum,
    ReduceMean,
    ReduceMax,
    Neg,
    Sqrt,
    Rsqrt,
    RMSNorm,        // RMS normalization
    Concat,         // Concatenation along specified axis
    Split,
    Pad,
    Slice,
    Identity,
    Gather,         // Gather elements from input tensor at specified indices
    RoPE,           // Rotary position embeddings
    Count,
}

impl Op {
    /// Get the MIL op name for this operation
    pub fn mil_name(self) -> &'static str {
        match self {
            Op::Input => "input",
            Op::Const => "const",
            Op::Conv1x1 => "nn.convolution",
            Op::Add => "mb.add",
            Op::Sub => "mb.sub",
            Op::Mul => "mb.mul",
            Op::MatMul => "mb.matmul",
            Op::Reshape => "mb.reshape",
            Op::Transpose => "mb.transpose",
            Op::Cast => "mb.cast",
            Op::ReLU => "mb.relu",
            Op::Tanh => "mb.tanh",
            Op::Sigmoid => "mb.sigmoid",
            Op::Softmax => "mb.softmax",
            Op::Exp => "mb.exp",
            Op::Pow => "mb.pow",
            Op::ReduceSum => "mb.reduce_sum",
            Op::ReduceMean => "mb.reduce_mean",
            Op::ReduceMax => "mb.reduce_max",
            Op::Neg => "mb.negative",
            Op::Sqrt => "mb.sqrt",
            Op::Rsqrt => "mb.rsqrt",
            Op::RMSNorm => "mb.rms_norm",
            Op::Concat => "mb.concat",
            Op::Split => "mb.split",
            Op::Pad => "mb.pad",
            Op::Slice => "mb.slice_by_index",
            Op::Gather => "mb.gather",
            Op::RoPE => "mb.real",  // RoPE is implemented via complex multiplication
            Op::Identity => "identity",
            Op::Count => "invalid",
        }
    }

    /// Get all operations as a vector
    pub fn all_ops() -> Vec<Op> {
        (0..Op::Count as u8)
            .filter_map(|i| match i {
                0..=30 => Some(unsafe { std::mem::transmute(i) }),
                _ => None,
            })
            .collect()
    }
}

/// Data types
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dtype {
    Fp16 = 0,
    Fp32,
    Int32,
    Bool,
    String, // For cast dtype argument
}

impl Dtype {
    /// Get the MIL dtype name
    pub fn mil_name(self) -> &'static str {
        match self {
            Dtype::Fp16 => "fp16",
            Dtype::Fp32 => "fp32",
            Dtype::Int32 => "int32",
            Dtype::Bool => "bool",
            Dtype::String => "string",
        }
    }

    /// Get size in bytes
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::Fp16 => 2,
            Dtype::Fp32 => 4,
            Dtype::Int32 => 4,
            Dtype::Bool => 1,
            Dtype::String => 0, // Variable
        }
    }
}

/// Node attributes (union of all op-specific attributes)
#[derive(Debug, Clone)]
pub struct Attrs {
    /// For transpose: perm[4]
    pub perm: [i32; 4],
    /// For reduce ops / softmax
    pub axis: Option<i32>,
    pub keep_dims: bool,
    /// For matmul
    pub transpose_x: bool,
    pub transpose_y: bool,
    /// For const: weight blob path + scalar value
    pub blob_path: Option<String>,
    pub blob_offset: Option<u64>,
    pub scalar_val: Option<f32>,
    /// For cast: target dtype
    pub cast_dtype: Option<Dtype>,
    /// For conv: bias input index (-1 = no bias)
    pub bias_input: Option<i32>,
    /// For conv: groups, strides
    pub groups: i32,
    pub stride_h: i32,
    pub stride_w: i32,
    /// For slice_by_index
    pub slice_begin: [i32; 4],
    pub slice_end: [i32; 4],
    /// For RMSNorm: epsilon value
    pub eps: f32,
}

impl Default for Attrs {
    fn default() -> Self {
        Attrs {
            perm: [0, 1, 2, 3],
            axis: None,
            keep_dims: false,
            transpose_x: false,
            transpose_y: false,
            blob_path: None,
            blob_offset: None,
            scalar_val: None,
            cast_dtype: None,
            bias_input: None,
            groups: 1,
            stride_h: 1,
            stride_w: 1,
            slice_begin: [0; 4],
            slice_end: [-1; 4], // -1 means "to end"
            eps: 1e-5,
        }
    }
}

/// Named I/O for the graph
#[derive(Debug, Clone)]
pub struct GraphIO {
    pub name: String,
    pub node_idx: usize,
}

/// A single node in the graph
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Op,
    pub name: String,
    pub dtype: Dtype,
    pub shape: [usize; 4], // ANE layout [1, C, 1, S] or [1, C, H, W]
    pub inputs: Vec<usize>, // Indices into graph.nodes[]
    pub attrs: Attrs,
    pub is_output: bool,    // Marked as a graph output
    pub is_live: bool,      // Used by DCE pass
}

impl Node {
    /// Create a new node
    pub fn new(op: Op, name: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        Node {
            op,
            name: name.to_string(),
            dtype,
            shape,
            inputs: Vec::new(),
            attrs: Attrs::default(),
            is_output: false,
            is_live: true,
        }
    }

    /// Get the number of elements in the shape
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// The graph
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<GraphIO>,
    pub outputs: Vec<GraphIO>,
    /// Name -> node index mapping for fast lookup
    name_to_idx: HashMap<String, usize>,
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Graph {
            nodes: Vec::with_capacity(64),
            inputs: Vec::new(),
            outputs: Vec::new(),
            name_to_idx: HashMap::new(),
        }
    }

    /// Get the number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Add a node to the graph, returning its index
    pub fn add_node(&mut self, node: Node) -> Result<usize> {
        if self.nodes.len() >= MAX_NODES {
            return Err(Error::GraphError(format!(
                "Graph exceeds maximum node limit ({})",
                MAX_NODES
            )));
        }

        let idx = self.nodes.len();
        self.name_to_idx.insert(node.name.clone(), idx);
        self.nodes.push(node);
        Ok(idx)
    }

    /// Add an input to the graph
    pub fn add_input(&mut self, name: &str, node_idx: usize) -> Result<()> {
        if self.inputs.len() >= MAX_GRAPH_IO {
            return Err(Error::GraphError(format!(
                "Graph exceeds maximum input limit ({})",
                MAX_GRAPH_IO
            )));
        }
        self.inputs.push(GraphIO {
            name: name.to_string(),
            node_idx,
        });
        Ok(())
    }

    /// Add an output to the graph
    pub fn add_output(&mut self, name: &str, node_idx: usize) -> Result<()> {
        if self.outputs.len() >= MAX_GRAPH_IO {
            return Err(Error::GraphError(format!(
                "Graph exceeds maximum output limit ({})",
                MAX_GRAPH_IO
            )));
        }
        self.outputs.push(GraphIO {
            name: name.to_string(),
            node_idx,
        });
        // Mark the node as an output
        if let Some(node) = self.nodes.get_mut(node_idx) {
            node.is_output = true;
        }
        Ok(())
    }

    /// Get a node by name
    pub fn get_node_by_name(&self, name: &str) -> Option<&Node> {
        self.name_to_idx
            .get(name)
            .and_then(|&idx| self.nodes.get(idx))
    }

    /// Get a node index by name
    pub fn get_node_idx(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    /// Get a node by index
    pub fn get_node(&self, idx: usize) -> Option<&Node> {
        self.nodes.get(idx)
    }

    /// Get a mutable node by index
    pub fn get_node_mut(&mut self, idx: usize) -> Option<&mut Node> {
        self.nodes.get_mut(idx)
    }

    /// Perform topological sort, returning node indices in execution order
    pub fn topological_sort(&self) -> Result<Vec<usize>> {
        let mut in_degree = vec![0; self.nodes.len()];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];

        // Build adjacency list and compute in-degrees
        for (idx, node) in self.nodes.iter().enumerate() {
            for &input_idx in &node.inputs {
                if input_idx < self.nodes.len() {
                    adj[input_idx].push(idx);
                    in_degree[idx] += 1;
                }
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(idx, _)| idx)
            .collect();

        let mut result = Vec::with_capacity(self.nodes.len());

        while let Some(idx) = queue.pop_front() {
            result.push(idx);
            for &neighbor in &adj[idx] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(Error::GraphError(
                "Graph contains a cycle - topological sort failed".to_string(),
            ));
        }

        Ok(result)
    }

    /// Debug print the graph
    pub fn dump(&self) {
        println!("Graph ({} nodes):", self.nodes.len());
        println!("  Inputs:");
        for io in &self.inputs {
            let node = &self.nodes[io.node_idx];
            println!(
                "    {} -> {} ({:?})",
                io.name, node.name, node.shape
            );
        }
        println!("  Outputs:");
        for io in &self.outputs {
            let node = &self.nodes[io.node_idx];
            println!(
                "    {} <- {} ({:?})",
                io.name, node.name, node.shape
            );
        }
        println!("  Nodes:");
        for (idx, node) in self.nodes.iter().enumerate() {
            let inputs_str: Vec<String> = node
                .inputs
                .iter()
                .map(|&i| format!("{}", i))
                .collect();
            println!(
                "    [{}] {} = {}({}) -> {:?}",
                idx, node.name, node.op.mil_name(),
                inputs_str.join(", "),
                node.shape
            );
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph builder - fluent API for constructing graphs
pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        GraphBuilder {
            graph: Graph::new(),
        }
    }

    /// Add an input placeholder
    pub fn input(mut self, name: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let node = Node::new(Op::Input, name, dtype, shape);
        let idx = self.graph.add_node(node).unwrap();
        self.graph.add_input(name, idx).unwrap();
        self
    }

    /// Add a constant/weight reference
    pub fn constant(
        mut self,
        name: &str,
        dtype: Dtype,
        shape: [usize; 4],
        blob_path: &str,
        offset: u64,
    ) -> Self {
        let mut node = Node::new(Op::Const, name, dtype, shape);
        node.attrs.blob_path = Some(blob_path.to_string());
        node.attrs.blob_offset = Some(offset);
        self.graph.add_node(node).unwrap();
        self
    }

    /// Get a node index by name for use as input
    fn get_node_idx(&self, name: &str) -> Option<usize> {
        self.graph.get_node_idx(name)
    }

    /// Add a matmul operation
    pub fn matmul(
        mut self,
        name: &str,
        x: &str,
        y: &str,
        dtype: Dtype,
        shape: [usize; 4],
        transpose_y: bool,
    ) -> Self {
        let mut node = Node::new(Op::MatMul, name, dtype, shape);
        if let Some(x_idx) = self.get_node_idx(x) {
            node.inputs.push(x_idx);
        }
        if let Some(y_idx) = self.get_node_idx(y) {
            node.inputs.push(y_idx);
        }
        node.attrs.transpose_y = transpose_y;
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a conv1x1 operation (linear layer)
    pub fn conv1x1(
        mut self,
        name: &str,
        input: &str,
        weight: &str,
        dtype: Dtype,
        shape: [usize; 4],
        groups: i32,
    ) -> Self {
        let mut node = Node::new(Op::Conv1x1, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        if let Some(idx) = self.get_node_idx(weight) {
            node.inputs.push(idx);
        }
        node.attrs.groups = groups;
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a ReLU activation
    pub fn relu(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::ReLU, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a Tanh activation
    pub fn tanh(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Tanh, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a Sigmoid activation
    pub fn sigmoid(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Sigmoid, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add an element-wise sub
    pub fn sub(mut self, name: &str, a: &str, b: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Sub, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(a) {
            node.inputs.push(idx);
        }
        if let Some(idx) = self.get_node_idx(b) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add an exponential operation
    pub fn exp(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Exp, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a power operation (x^y with scalar y)
    pub fn pow_scalar(mut self, name: &str, input: &str, exponent: f32, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Pow, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        node.attrs.scalar_val = Some(exponent);
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a negation operation
    pub fn neg(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Neg, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a softmax operation
    pub fn softmax(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4], axis: i32) -> Self {
        let mut node = Node::new(Op::Softmax, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        node.attrs.axis = Some(axis);
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a cast operation
    pub fn cast(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4], target_dtype: Dtype) -> Self {
        let mut node = Node::new(Op::Cast, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        node.attrs.cast_dtype = Some(target_dtype);
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a transpose operation
    pub fn transpose(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4], perm: [i32; 4]) -> Self {
        let mut node = Node::new(Op::Transpose, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        node.attrs.perm = perm;
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a reshape operation
    pub fn reshape(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Reshape, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add an element-wise add
    pub fn add(mut self, name: &str, a: &str, b: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Add, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(a) {
            node.inputs.push(idx);
        }
        if let Some(idx) = self.get_node_idx(b) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a sqrt operation
    pub fn sqrt(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Sqrt, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a rsqrt operation
    pub fn rsqrt(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Rsqrt, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a mul operation
    pub fn mul(mut self, name: &str, a: &str, b: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Mul, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(a) {
            node.inputs.push(idx);
        }
        if let Some(idx) = self.get_node_idx(b) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a reduce_mean operation
    pub fn reduce_mean(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4], axis: i32, keep_dims: bool) -> Self {
        let mut node = Node::new(Op::ReduceMean, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        node.attrs.axis = Some(axis);
        node.attrs.keep_dims = keep_dims;
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add an identity operation (no-op, used for debugging/fusion)
    pub fn identity(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::Identity, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add RMSNorm operation
    pub fn rms_norm(mut self, name: &str, input: &str, dtype: Dtype, shape: [usize; 4], eps: f32) -> Self {
        let mut node = Node::new(Op::RMSNorm, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        node.attrs.eps = eps;
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add concat operation - concatenate multiple inputs along specified axis
    pub fn concat(mut self, name: &str, inputs: &[&str], dtype: Dtype, shape: [usize; 4], axis: i32) -> Self {
        let mut node = Node::new(Op::Concat, name, dtype, shape);
        for &input in inputs {
            if let Some(idx) = self.get_node_idx(input) {
                node.inputs.push(idx);
            }
        }
        node.attrs.axis = Some(axis);
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a slice operation - slice a tensor by index ranges
    pub fn slice(
        mut self,
        name: &str,
        input: &str,
        dtype: Dtype,
        shape: [usize; 4],
        begin: [i32; 4],
        end: [i32; 4],
    ) -> Self {
        let mut node = Node::new(Op::Slice, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        node.attrs.slice_begin = begin;
        node.attrs.slice_end = end;
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a gather operation - gather elements from input tensor at specified indices
    pub fn gather(mut self, name: &str, input: &str, indices: &str, dtype: Dtype, shape: [usize; 4], axis: i32) -> Self {
        let mut node = Node::new(Op::Gather, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        if let Some(idx) = self.get_node_idx(indices) {
            node.inputs.push(idx);
        }
        node.attrs.axis = Some(axis);
        self.graph.add_node(node).unwrap();
        self
    }

    /// Add a RoPE (Rotary Position Embeddings) operation
    ///
    /// RoPE applies rotary position embeddings to the input tensor.
    /// Formula: out_even = in_even * cos - in_odd * sin
    ///          out_odd = in_even * sin + in_odd * cos
    ///
    /// # Arguments
    /// * `name` - Output node name
    /// * `input` - Input tensor (query/key projections)
    /// * `cos_table` - Cosine rotation table
    /// * `sin_table` - Sine rotation table
    /// * `dtype` - Data type
    /// * `shape` - Output shape (same as input)
    pub fn rope(mut self, name: &str, input: &str, cos_table: &str, sin_table: &str, dtype: Dtype, shape: [usize; 4]) -> Self {
        let mut node = Node::new(Op::RoPE, name, dtype, shape);
        if let Some(idx) = self.get_node_idx(input) {
            node.inputs.push(idx);
        }
        if let Some(idx) = self.get_node_idx(cos_table) {
            node.inputs.push(idx);
        }
        if let Some(idx) = self.get_node_idx(sin_table) {
            node.inputs.push(idx);
        }
        self.graph.add_node(node).unwrap();
        self
    }

    /// Mark a node as a graph output
    pub fn output(mut self, name: &str) -> Self {
        if let Some(idx) = self.graph.get_node_idx(name) {
            self.graph.add_output(name, idx).unwrap();
        }
        self
    }

    /// Build the graph
    pub fn build(self) -> Graph {
        self.graph
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::new();
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_graph_builder_basic() {
        let graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("out", "x", Dtype::Fp32, [1, 256, 1, 1])
            .output("out")
            .build();

        assert_eq!(graph.len(), 2);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("relu1", "x", Dtype::Fp32, [1, 256, 1, 1])
            .relu("relu2", "relu1", Dtype::Fp32, [1, 256, 1, 1])
            .output("relu2")
            .build();

        let sorted = graph.topological_sort().unwrap();
        assert_eq!(sorted.len(), 3);
        // First node should be the input
        assert_eq!(graph.nodes[sorted[0]].op, Op::Input);
    }

    #[test]
    fn test_matmul_graph() {
        let graph = GraphBuilder::new()
            .input("x", Dtype::Fp32, [1, 256, 1, 512])
            .constant("w", Dtype::Fp32, [1, 512, 1, 256], "weights.bin", 0)
            .matmul("out", "x", "w", Dtype::Fp32, [1, 512, 1, 256], false)
            .output("out")
            .build();

        assert_eq!(graph.len(), 3);
        let matmul_node = graph.get_node_by_name("out").unwrap();
        assert_eq!(matmul_node.op, Op::MatMul);
        assert!(!matmul_node.attrs.transpose_y);
    }

    #[test]
    fn test_op_mil_names() {
        assert_eq!(Op::MatMul.mil_name(), "mb.matmul");
        assert_eq!(Op::ReLU.mil_name(), "mb.relu");
        assert_eq!(Op::Conv1x1.mil_name(), "nn.convolution");
    }

    #[test]
    fn test_dtype_names() {
        assert_eq!(Dtype::Fp16.mil_name(), "fp16");
        assert_eq!(Dtype::Fp32.mil_name(), "fp32");
        assert_eq!(Dtype::Fp16.size_bytes(), 2);
        assert_eq!(Dtype::Fp32.size_bytes(), 4);
    }
}
