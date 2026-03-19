//! ANE executor for running compiled kernels
//!
//! ANEExecutor executes compiled kernels on the Apple Neural Engine,
//! managing input/output data via IOSurface-backed memory.

use crate::sys::{
    ane_bridge_eval, ane_bridge_read_output, ane_bridge_write_input, ANEKernelHandle,
};
use crate::{Error, Result};

/// ANE kernel executor
///
/// ANEExecutor runs compiled kernels on the ANE and manages I/O operations.
/// It borrows the kernel from the compiler and ensures it's not used after
/// the compiler is dropped.
///
/// # Example
///
/// ```no_run
/// # use rustane::wrapper::{ANECompiler, ANETensor};
/// # use rustane::wrapper::ANERuntime;
/// # fn main() -> rustane::Result<()> {
/// let _runtime = ANERuntime::init()?;
/// let mil = "program(1.0) { var _0 = nn.convolution(...) }";
/// let mut compiler = ANECompiler::new();
/// let mut executor = compiler.compile_single(mil, None, &[1024], &[512])?;
///
/// // Write input data
/// let input_data = vec![0.0f32; 256];
/// let input_tensor = ANETensor::from_fp32(input_data, vec![1, 256])?;
/// executor.write_input(0, input_tensor.as_bytes())?;
///
/// // Execute
/// executor.eval()?;
///
/// // Read output
/// let mut output_buf = vec![0u8; 512];
/// executor.read_output(0, &mut output_buf)?;
/// # Ok(())
/// # }
/// ```
pub struct ANEExecutor {
    kernel: *mut ANEKernelHandle,
    input_sizes: Vec<usize>,
    output_sizes: Vec<usize>,
}

impl ANEExecutor {
    /// Create a new executor (internal use only)
    ///
    /// This is called by ANECompiler after successful compilation.
    pub(crate) fn new(
        kernel: *mut ANEKernelHandle,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Self {
        ANEExecutor {
            kernel,
            input_sizes: input_sizes.to_vec(),
            output_sizes: output_sizes.to_vec(),
        }
    }

    /// Execute the compiled kernel on the ANE
    ///
    /// This runs the compiled program on the ANE hardware. Input data must
    /// be written via `write_input()` before calling this method.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Kernel execution fails (hardware error, timeout, etc.)
    /// - Input data hasn't been written
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::{ANECompiler, ANETensor};
    /// # use rustane::wrapper::ANERuntime;
    /// # fn main() -> rustane::Result<()> {
    /// let _runtime = ANERuntime::init()?;
    /// let mil = "program(1.0) { var _0 = nn.convolution(...) }";
    /// let mut compiler = ANECompiler::new();
    /// let mut executor = compiler.compile_single(mil, None, &[1024], &[512])?;
    ///
    /// // Write inputs...
    ///
    /// // Execute
    /// executor.eval()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn eval(&mut self) -> Result<()> {
        if self.kernel.is_null() {
            return Err(Error::ExecutionFailed(
                "Kernel is null (compiler may have been dropped)".to_string(),
            ));
        }

        // SAFETY: eval is safe when:
        // - kernel is valid (checked above)
        // - inputs have been written (user's responsibility)
        let success = unsafe { ane_bridge_eval(self.kernel) };

        if !success {
            return Err(Error::ExecutionFailed(
                "ANE kernel execution failed".to_string(),
            ));
        }

        Ok(())
    }

    /// Write data to an input tensor
    ///
    /// # Arguments
    ///
    /// * `idx` - Input tensor index (0-based)
    /// * `data` - Data to write (must match expected size)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input index is out of bounds
    /// - Data size doesn't match expected size
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::{ANECompiler, ANETensor};
    /// # use rustane::wrapper::ANERuntime;
    /// # fn main() -> rustane::Result<()> {
    /// let _runtime = ANERuntime::init()?;
    /// let mil = "program(1.0) { var _0 = nn.convolution(...) }";
    /// let mut compiler = ANECompiler::new();
    /// let mut executor = compiler.compile_single(mil, None, &[1024], &[512])?;
    ///
    /// let input_data = vec![0u8; 1024];
    /// executor.write_input(0, &input_data)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn write_input(&mut self, idx: usize, data: &[u8]) -> Result<()> {
        // Validate index first (before null check for better error messages)
        let expected_size = self.input_sizes.get(idx).copied().ok_or_else(|| {
            Error::InvalidParameter(format!(
                "Input index {} out of bounds ({} inputs)",
                idx,
                self.input_sizes.len()
            ))
        })?;

        // Validate size
        if data.len() != expected_size {
            return Err(Error::InvalidParameter(format!(
                "Input data size ({}) doesn't match expected size ({}) for input {}",
                data.len(),
                expected_size,
                idx
            )));
        }

        // Check kernel last
        if self.kernel.is_null() {
            return Err(Error::ExecutionFailed(
                "Kernel is null (compiler may have been dropped)".to_string(),
            ));
        }

        // SAFETY: write_input is safe when:
        // - kernel is valid (checked above)
        // - idx is in bounds (checked above)
        // - data pointer is valid for data.len() bytes (ensured by &[u8])
        unsafe {
            ane_bridge_write_input(
                self.kernel,
                idx as i32,
                data.as_ptr() as *const _,
                data.len(),
            );
        }

        Ok(())
    }

    /// Read data from an output tensor
    ///
    /// # Arguments
    ///
    /// * `idx` - Output tensor index (0-based)
    /// * `data` - Buffer to read data into (must match expected size)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Output index is out of bounds
    /// - Buffer size doesn't match expected size
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::{ANECompiler, ANETensor};
    /// # use rustane::wrapper::ANERuntime;
    /// # fn main() -> rustane::Result<()> {
    /// let _runtime = ANERuntime::init()?;
    /// let mil = "program(1.0) { var _0 = nn.convolution(...) }";
    /// let mut compiler = ANECompiler::new();
    /// let mut executor = compiler.compile_single(mil, None, &[1024], &[512])?;
    ///
    /// // Write inputs, eval...
    ///
    /// let mut output_buf = vec![0u8; 512];
    /// executor.read_output(0, &mut output_buf)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_output(&self, idx: usize, data: &mut [u8]) -> Result<()> {
        // Validate index first (before null check for better error messages)
        let expected_size = self.output_sizes.get(idx).copied().ok_or_else(|| {
            Error::InvalidParameter(format!(
                "Output index {} out of bounds ({} outputs)",
                idx,
                self.output_sizes.len()
            ))
        })?;

        // Validate size
        if data.len() != expected_size {
            return Err(Error::InvalidParameter(format!(
                "Output buffer size ({}) doesn't match expected size ({}) for output {}",
                data.len(),
                expected_size,
                idx
            )));
        }

        // Check kernel last
        if self.kernel.is_null() {
            return Err(Error::ExecutionFailed(
                "Kernel is null (compiler may have been dropped)".to_string(),
            ));
        }

        // SAFETY: read_output is safe when:
        // - kernel is valid (checked above)
        // - idx is in bounds (checked above)
        // - data pointer is valid for data.len() bytes (ensured by &mut [u8])
        unsafe {
            ane_bridge_read_output(
                self.kernel,
                idx as i32,
                data.as_mut_ptr() as *mut _,
                data.len(),
            );
        }

        Ok(())
    }

    /// Get the number of inputs
    pub fn num_inputs(&self) -> usize {
        self.input_sizes.len()
    }

    /// Get the number of outputs
    pub fn num_outputs(&self) -> usize {
        self.output_sizes.len()
    }

    /// Get the size of an input tensor in bytes
    pub fn input_size(&self, idx: usize) -> Option<usize> {
        self.input_sizes.get(idx).copied()
    }

    /// Get the size of an output tensor in bytes
    pub fn output_size(&self, idx: usize) -> Option<usize> {
        self.output_sizes.get(idx).copied()
    }
}

// Note: ANEExecutor is !Sync by default due to containing a raw pointer
// This is the desired behavior - executors should not be shared across threads

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_executor_methods() {
        // Create a dummy executor for testing methods
        // Note: We can't actually execute without a real kernel
        let executor = ANEExecutor {
            kernel: ptr::null_mut(),
            input_sizes: vec![1024, 512],
            output_sizes: vec![256],
        };

        assert_eq!(executor.num_inputs(), 2);
        assert_eq!(executor.num_outputs(), 1);
        assert_eq!(executor.input_size(0), Some(1024));
        assert_eq!(executor.output_size(0), Some(256));
        assert_eq!(executor.input_size(5), None);
    }

    #[test]
    fn test_executor_write_input_bounds_check() {
        let mut executor = ANEExecutor {
            kernel: ptr::null_mut(),
            input_sizes: vec![1024],
            output_sizes: vec![256],
        };

        // Valid size
        let data = vec![0u8; 1024];
        let result = executor.write_input(0, &data);
        // Will fail because kernel is null, but not due to bounds
        assert!(matches!(result, Err(Error::ExecutionFailed(_))));

        // Out of bounds
        let data = vec![0u8; 1024];
        let result = executor.write_input(5, &data);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));

        // Wrong size
        let data = vec![0u8; 512];
        let result = executor.write_input(0, &data);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_executor_read_output_bounds_check() {
        let executor = ANEExecutor {
            kernel: ptr::null_mut(),
            input_sizes: vec![1024],
            output_sizes: vec![256],
        };

        // Valid size
        let mut buf = vec![0u8; 256];
        let result = executor.read_output(0, &mut buf);
        // Will fail because kernel is null, but not due to bounds
        assert!(matches!(result, Err(Error::ExecutionFailed(_))));

        // Out of bounds
        let mut buf = vec![0u8; 256];
        let result = executor.read_output(5, &mut buf);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));

        // Wrong size
        let mut buf = vec![0u8; 128];
        let result = executor.read_output(0, &mut buf);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }
}
