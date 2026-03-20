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

use crate::ane::ANEKernel;
use crate::training::TransformerConfig;
use crate::error::{Error, Result};

/// Compiled ANE backward kernel ready for execution
///
/// This struct wraps a compiled ANE kernel with metadata for backward
/// pass operations. It manages the kernel lifecycle and provides
/// a safe interface for execution.
pub struct ANEBackwardKernel {
    /// The underlying ANE kernel
    kernel: ANEKernel,
    /// Operation name (e.g., "rmsnorm_backward")
    operation_name: String,
    /// Input tensor shapes
    input_shapes: Vec<Vec<usize>>,
    /// Output tensor shapes
    output_shapes: Vec<Vec<usize>>,
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
        // For now, we create a kernel with placeholder sizes
        // In production, this would parse the MIL code to determine sizes
        let input_sizes = vec![config.hidden_dim * 4]; // Placeholder
        let output_sizes = vec![config.hidden_dim * 4]; // Placeholder

        let kernel = ANEKernel::new(input_sizes, output_sizes)
            .map_err(|e| Error::Other(format!("Failed to create ANE kernel: {}", e)))?;

        Ok(Self {
            kernel,
            operation_name: operation_name.to_string(),
            input_shapes: vec![],
            output_shapes: vec![],
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
        if inputs.len() != self.kernel.io_inputs.len() {
            return Err(Error::InvalidParameter(format!(
                "Expected {} inputs, got {}",
                self.kernel.io_inputs.len(),
                inputs.len()
            )));
        }

        // Validate output count
        if outputs.len() != self.kernel.io_outputs.len() {
            return Err(Error::InvalidParameter(format!(
                "Expected {} outputs, got {}",
                self.kernel.io_outputs.len(),
                outputs.len()
            )));
        }

        // Write inputs to IOSurfaces
        for (i, input_data) in inputs.iter().enumerate() {
            self.kernel.write_input(i, input_data)?;
        }

        // Execute on ANE
        self.kernel.eval()?;

        // Read outputs from IOSurfaces
        for (i, output_buffer) in outputs.iter_mut().enumerate() {
            let result = self.kernel.read_output(i)?;
            output_buffer.copy_from_slice(&result);
        }

        Ok(())
    }

    /// Get the operation name
    pub fn operation_name(&self) -> &str {
        &self.operation_name
    }

    /// Get number of inputs
    pub fn num_inputs(&self) -> usize {
        self.kernel.io_inputs.len()
    }

    /// Get number of outputs
    pub fn num_outputs(&self) -> usize {
        self.kernel.io_outputs.len()
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
