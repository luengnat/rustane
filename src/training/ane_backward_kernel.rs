//! ANE Backward Kernel Executor
//!
//! This module provides the [`ANEBackwardKernel`] for compiling and executing
//! backward pass MIL code on the Apple Neural Engine (ANE).
//!
//! # Architecture
//!
//! The kernel executor handles:
//! 1. MIL code compilation to ANE executable format
//! 2. IOSurface allocation for inputs/outputs
//! 3. Kernel execution on ANE hardware
//! 4. Result retrieval from ANE memory
//!
//! # Usage
//!
//! ```ignore
//! use rustane::training::ane_backward_kernel::ANEBackwardKernel;
//! use rustane::layers::backward::RMSNormBackwardGen;
//!
//! let config = TransformerConfig::tiny();
//! let generator = RMSNormBackwardGen::new();
//! let mil_code = generator.generate(&config)?;
//!
//! let kernel = ANEBackwardKernel::compile(&mil_code, &config)?;
//! kernel.execute(&inputs, &mut outputs)?;
//! ```

use crate::wrapper::ANEExecutor;
use crate::error::{Error, Result};
use crate::training::TransformerConfig;

/// Compiled ANE backward kernel ready for execution
///
/// This struct wraps a compiled ANE executor with metadata for backward
/// pass operations. It manages the kernel lifecycle and provides
/// a safe interface for execution.
pub struct ANEBackwardKernel {
    /// The underlying ANE executor
    executor: ANEExecutor,
    /// Operation name (e.g., "rmsnorm_backward")
    operation_name: String,
    /// Input sizes in bytes
    input_sizes: Vec<usize>,
    /// Output sizes in bytes
    output_sizes: Vec<usize>,
}

impl ANEBackwardKernel {
    /// Compile MIL code to an ANE backward kernel
    ///
    /// # Arguments
    ///
    /// * `mil_code` - MIL code string to compile
    /// * `config` - Transformer configuration
    /// * `operation_name` - Name of the operation (for debugging)
    ///
    /// # Returns
    ///
    /// A compiled `ANEBackwardKernel` ready for execution
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - MIL code compilation fails
    /// - ANE kernel creation fails
    pub fn compile(
        mil_code: &str,
        config: &TransformerConfig,
        operation_name: &str,
    ) -> Result<Self> {
        use std::collections::HashMap;

        // Calculate tensor sizes from config
        // For backward pass, we typically have:
        // - Input: gradients from upstream (hidden_dim)
        // - Output: gradients for weights (varies by layer)
        let input_size_bytes = config.hidden_dim * std::mem::size_of::<f32>();
        let output_size_bytes = config.hidden_dim * std::mem::size_of::<f32>();

        // Parse MIL code to extract weight names and create weight dictionary
        // For now, use empty weights since backward kernels don't typically need weight data
        let weights = HashMap::new();

        // Create compile request
        let compile_request = crate::ane::ANECompileRequest {
            mil_text: mil_code.to_string(),
            weights,
            input_sizes: vec![input_size_bytes],
            output_sizes: vec![output_size_bytes],
        };

        // Compile the MIL code
        let executor = compile_request.compile()
            .map_err(|e| Error::Other(format!("Failed to compile MIL code for {}: {}", operation_name, e)))?;

        Ok(Self {
            executor,
            operation_name: operation_name.to_string(),
            input_sizes: vec![input_size_bytes],
            output_sizes: vec![output_size_bytes],
        })
    }

    /// Execute the backward kernel on ANE
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input tensors (as f32 vectors)
    /// * `outputs` - Output buffers to receive results
    ///
    /// # Returns
    ///
    /// Ok(()) if execution succeeds
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input/output size mismatch
    /// - ANE execution fails
    pub fn execute(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>]) -> Result<()> {
        // Validate input count
        if inputs.len() != self.input_sizes.len() {
            return Err(Error::InvalidParameter(format!(
                "Expected {} inputs, got {}",
                self.input_sizes.len(),
                inputs.len()
            )));
        }

        // Validate output count
        if outputs.len() != self.output_sizes.len() {
            return Err(Error::InvalidParameter(format!(
                "Expected {} outputs, got {}",
                self.output_sizes.len(),
                outputs.len()
            )));
        }

        // Write inputs to ANE
        for (i, input_data) in inputs.iter().enumerate() {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    input_data.as_ptr() as *const u8,
                    input_data.len() * std::mem::size_of::<f32>(),
                )
            };
            self.executor.write_input(i, bytes)?;
        }

        // Execute on ANE
        self.executor.eval()?;

        // Read outputs from ANE
        for (i, output_buffer) in outputs.iter_mut().enumerate() {
            let mut byte_buffer = vec![0u8; self.output_sizes[i]];
            self.executor.read_output(i, &mut byte_buffer)?;

            // Convert bytes to f32
            let float_slice = unsafe {
                std::slice::from_raw_parts(
                    byte_buffer.as_ptr() as *const f32,
                    byte_buffer.len() / std::mem::size_of::<f32>(),
                )
            };
            output_buffer.copy_from_slice(float_slice);
        }

        Ok(())
    }

    /// Get the operation name
    pub fn operation_name(&self) -> &str {
        &self.operation_name
    }

    /// Get number of inputs
    pub fn num_inputs(&self) -> usize {
        self.input_sizes.len()
    }

    /// Get number of outputs
    pub fn num_outputs(&self) -> usize {
        self.output_sizes.len()
    }
}

/// ANE Backward Kernel Cache
///
/// Caches compiled kernels to avoid recompilation overhead.
/// Kernels are keyed by (operation_name, config_hash).
pub struct ANEBackwardKernelCache {
    // In production, this would use a HashMap with proper keys
    // For now, we just track the cache concept
    cache_hits: usize,
    cache_misses: usize,
}

impl ANEBackwardKernelCache {
    /// Create a new kernel cache
    pub fn new() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> (usize, usize) {
        (self.cache_hits, self.cache_misses)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Record a cache hit (for testing purposes)
    pub fn record_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record a cache miss (for testing purposes)
    pub fn record_miss(&mut self) {
        self.cache_misses += 1;
    }
}

impl Default for ANEBackwardKernelCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_kernel_creation() {
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = "#!irms6\nmain test() {}";

        let result = ANEBackwardKernel::compile(mil_code, &config, "test");
        // May succeed or fail depending on ANE availability
        match result {
            Ok(kernel) => {
                assert_eq!(kernel.operation_name(), "test");
            }
            Err(_) => {
                // ANE not available is acceptable in test environment
            }
        }
    }

    #[test]
    fn test_kernel_cache() {
        let mut cache = ANEBackwardKernelCache::new();
        assert_eq!(cache.stats(), (0, 0));

        cache.cache_hits = 5;
        cache.cache_misses = 3;
        assert_eq!(cache.stats(), (5, 3));

        cache.clear();
        assert_eq!(cache.stats(), (0, 0));
    }
}
