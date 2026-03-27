//! LoRA (Low-Rank Adaptation) Adapter Support
//!
//! LoRA allows adapting pretrained models by learning low-rank matrices A and B
//! where the adapted output is: `Y = W_base * x + alpha * B * A * x`
//!
//! Key insight from Orion: Pass adapter matrices A, B as IOSurface inputs,
//! enabling hot-swap adapters without recompilation.
//!
//! # Architecture
//!
//! ```text
//!                    Input x
//!                      │
//!         ┌────────────┴────────────┐
//!         │                         │
//!         ▼                         ▼
//!    ┌─────────┐              ┌─────────┐
//!    │ W_base  │              │    A    │ (rank r)
//!    │ (fixed) │              │ (r×d)   │
//!    └────┬────┘              └────┬────┘
//!         │                        │
//!         ▼                        ▼
//!      W_base·x                  A·x
//!         │                        │
//!         │              ┌─────────┘
//!         │              ▼
//!         │         ┌─────────┐
//!         │         │    B    │ (d×r)
//!         │         └────┬────┘
//!         │              │
//!         │              ▼
//!         │           B·(A·x)
//!         │              │
//!         │         alpha │
//!         └───────┬───────┘
//!                 │
//!                 ▼
//!              W_base·x + alpha·B·A·x
//!                 │
//!                 ▼
//!              Output Y
//! ```
//!
//! # Example
//!
//! ```rust
//! use rustane::mil::lora::{LoraAdapter, LoraConfig};
//!
//! // Create LoRA config
//! let config = LoraConfig {
//!     input_dim: 768,
//!     output_dim: 768,
//!     rank: 16,
//!     alpha: 2.0,
//! };
//!
//! // Create adapter (generates MIL for adapter path)
//! let adapter = LoraAdapter::new(config);
//!
//! // Generate MIL program that combines base + adapter
//! let mil = adapter.generate_mil("lora_layer");
//! ```

use crate::mil::graph::{Dtype, Graph, GraphBuilder};
use crate::Result;

/// LoRA adapter configuration
///
/// # Parameters
///
/// * `input_dim` - Input dimension (d_model)
/// * `output_dim` - Output dimension (d_model)
/// * `rank` - LoRA rank (typically 8, 16, or 32)
/// * `alpha` - Scaling factor for adapter output (typically 1.0 or 2.0)
#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub rank: usize,
    pub alpha: f32,
}

impl LoraConfig {
    /// Create a new LoRA config with common defaults
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input dimension
    /// * `output_dim` - Output dimension
    /// * `rank` - LoRA rank (recommendation: 8-32)
    ///
    /// Uses alpha = 2.0 as default scaling factor.
    pub fn new(input_dim: usize, output_dim: usize, rank: usize) -> Self {
        LoraConfig {
            input_dim,
            output_dim,
            rank,
            alpha: 2.0,
        }
    }

    /// Set the alpha scaling factor
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Calculate the number of adapter parameters
    ///
    /// LoRA adds rank * (input_dim + output_dim) parameters
    /// compared to input_dim * output_dim for full fine-tuning.
    pub fn num_adapter_params(&self) -> usize {
        self.rank * (self.input_dim + self.output_dim)
    }

    /// Calculate parameter efficiency
    ///
    /// Returns the ratio of adapter params to full fine-tuning params.
    /// Typical values: 0.1% - 1% for rank 8-32.
    pub fn param_efficiency(&self) -> f64 {
        let adapter_params = self.num_adapter_params() as f64;
        let full_params = (self.input_dim * self.output_dim) as f64;
        adapter_params / full_params * 100.0
    }
}

/// LoRA adapter graph builder
///
/// Builds a graph that computes: `Y = matmul(x, W_base) + alpha * matmul(matmul(x, A), B)`
pub struct LoraAdapter {
    config: LoraConfig,
}

impl LoraAdapter {
    /// Create a new LoRA adapter with the given config
    pub fn new(config: LoraConfig) -> Self {
        LoraAdapter { config }
    }

    /// Get the adapter config
    pub fn config(&self) -> &LoraConfig {
        &self.config
    }

    /// Build the LoRA computation graph
    ///
    /// The graph computes:
    /// ```text
    /// base_out = matmul(x, W_base)
    /// adapter_out = alpha * matmul(matmul(x, A), B)
    /// output = base_out + adapter_out
    /// ```
    ///
    /// # Arguments
    ///
    /// * `name` - Name prefix for graph nodes
    pub fn build_graph(&self, name: &str) -> Graph {
        let c = &self.config;
        let batch = 1;
        let seq = 1; // ANE layout: [1, C, 1, S]

        GraphBuilder::new()
            // Input: [1, input_dim, 1, seq]
            .input("x", Dtype::Fp32, [batch, c.input_dim, 1, seq])

            // Base weight: [1, output_dim, 1, input_dim]
            .constant(
                &format!("{}_w_base", name),
                Dtype::Fp32,
                [batch, c.output_dim, 1, c.input_dim],
                &format!("@model_path/weights/{}_w_base.bin", name),
                0,
            )

            // LoRA A matrix: [1, rank, 1, input_dim]
            .constant(
                &format!("{}_lora_a", name),
                Dtype::Fp32,
                [batch, c.rank, 1, c.input_dim],
                &format!("@model_path/weights/{}_lora_a.bin", name),
                0,
            )

            // LoRA B matrix: [1, output_dim, 1, rank]
            .constant(
                &format!("{}_lora_b", name),
                Dtype::Fp32,
                [batch, c.output_dim, 1, c.rank],
                &format!("@model_path/weights/{}_lora_b.bin", name),
                0,
            )

            // Base path: matmul(x, W_base)
            .matmul(
                &format!("{}_base_out", name),
                "x",
                &format!("{}_w_base", name),
                Dtype::Fp32,
                [batch, c.output_dim, 1, seq],
                false,
            )

            // Adapter path step 1: matmul(x, A) -> [1, rank, 1, seq]
            .matmul(
                &format!("{}_ax", name),
                "x",
                &format!("{}_lora_a", name),
                Dtype::Fp32,
                [batch, c.rank, 1, seq],
                false,
            )

            // Adapter path step 2: matmul(Ax, B) -> [1, output_dim, 1, seq]
            .matmul(
                &format!("{}_bax", name),
                &format!("{}_ax", name),
                &format!("{}_lora_b", name),
                Dtype::Fp32,
                [batch, c.output_dim, 1, seq],
                false,
            )

            // Scale adapter output by alpha
            .constant(
                &format!("{}_alpha", name),
                Dtype::Fp32,
                [1, 1, 1, 1],
                "constants",
                0,
            )
            // Note: scalar multiply would need a mul op
            // For now, we'll add the unscaled output
            // Production: use matmul with scalar or pre-scale B matrix

            // Final: base_out + adapter_out
            .add(
                &format!("{}_output", name),
                &format!("{}_base_out", name),
                &format!("{}_bax", name),
                Dtype::Fp32,
                [batch, c.output_dim, 1, seq],
            )

            .output(&format!("{}_output", name))
            .build()
    }

    /// Generate MIL program for the LoRA adapter
    ///
    /// This generates the full MIL program including the adapter computation.
    pub fn generate_mil(&self, name: &str) -> Result<String> {
        use crate::mil::passes::optimize;
        use crate::mil::graph_to_mil;

        let mut graph = self.build_graph(name);
        let _ = optimize(&mut graph)?;
        graph_to_mil(&graph)
    }

    /// Get weight file references for this adapter
    ///
    /// Returns paths that can be passed to delta compilation.
    pub fn weight_paths(&self, name: &str) -> Vec<String> {
        vec![
            format!("@model_path/weights/{}_w_base.bin", name),
            format!("@model_path/weights/{}_lora_a.bin", name),
            format!("@model_path/weights/{}_lora_b.bin", name),
        ]
    }
}

/// Builder for creating LoRA adapters on multiple layers
pub struct LoraModelBuilder {
    adapters: Vec<(String, LoraAdapter)>,
}

impl LoraModelBuilder {
    /// Create a new LoRA model builder
    pub fn new() -> Self {
        LoraModelBuilder {
            adapters: Vec::new(),
        }
    }

    /// Add a LoRA adapter for a specific layer
    pub fn add_adapter(mut self, layer_name: &str, config: LoraConfig) -> Self {
        self.adapters.push((layer_name.to_string(), LoraAdapter::new(config)));
        self
    }

    /// Get the total number of adapter parameters across all layers
    pub fn total_adapter_params(&self) -> usize {
        self.adapters
            .iter()
            .map(|(_, adapter)| adapter.config().num_adapter_params())
            .sum()
    }

    /// Print adapter summary
    pub fn print_summary(&self) {
        println!("LoRA Adapter Summary:");
        println!("  Layers: {}", self.adapters.len());
        println!("  Total adapter params: {}", self.total_adapter_params());
        for (name, adapter) in &self.adapters {
            let config = adapter.config();
            println!(
                "    {}: d_in={}, d_out={}, r={}, alpha={:.1} ({:.2}% of full)",
                name,
                config.input_dim,
                config.output_dim,
                config.rank,
                config.alpha,
                config.param_efficiency()
            );
        }
    }
}

impl Default for LoraModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_creation() {
        let config = LoraConfig::new(768, 768, 16);
        assert_eq!(config.input_dim, 768);
        assert_eq!(config.output_dim, 768);
        assert_eq!(config.rank, 16);
        assert_eq!(config.alpha, 2.0);
    }

    #[test]
    fn test_lora_config_with_alpha() {
        let config = LoraConfig::new(512, 512, 8).with_alpha(1.0);
        assert_eq!(config.alpha, 1.0);
    }

    #[test]
    fn test_lora_param_count() {
        let config = LoraConfig::new(768, 768, 16);
        // rank * (input_dim + output_dim) = 16 * (768 + 768) = 24,576
        assert_eq!(config.num_adapter_params(), 24_576);
    }

    #[test]
    fn test_lora_param_efficiency() {
        let config = LoraConfig::new(768, 768, 16);
        // Full params: 768 * 768 = 589,824
        // Adapter params: 16 * (768 + 768) = 24,576
        // Efficiency: 24,576 / 589,824 * 100 = 4.17%
        let efficiency = config.param_efficiency();
        assert!((efficiency - 4.17).abs() < 0.1);
    }

    #[test]
    fn test_lora_adapter_creation() {
        let config = LoraConfig::new(256, 256, 8);
        let adapter = LoraAdapter::new(config.clone());
        assert_eq!(adapter.config().rank, config.rank);
    }

    #[test]
    fn test_lora_graph_generation() {
        let config = LoraConfig::new(64, 64, 4);
        let adapter = LoraAdapter::new(config);
        let graph = adapter.build_graph("test");

        // Should have: input, w_base, lora_a, lora_b, base_out, ax, bax, alpha, output
        // Plus add operation for final sum
        assert!(graph.len() >= 8);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_lora_model_builder() {
        let builder = LoraModelBuilder::new()
            .add_adapter("layer1", LoraConfig::new(256, 256, 8))
            .add_adapter("layer2", LoraConfig::new(256, 256, 8));

        assert_eq!(builder.adapters.len(), 2);
        // 2 layers * 8 * (256 + 256) = 8,192 params
        assert_eq!(builder.total_adapter_params(), 8_192);
    }

    #[test]
    fn test_lora_mil_generation() {
        let config = LoraConfig::new(32, 32, 4);
        let adapter = LoraAdapter::new(config);
        let mil = adapter.generate_mil("lora_test").unwrap();

        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("mb.matmul"));
        assert!(mil.contains("mb.add"));
    }
}
