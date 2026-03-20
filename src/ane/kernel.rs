//! ANE kernel wrapper for managing compiled models and I/O operations.
//!
//! The ANEKernel struct provides a high-level interface for:
//! - Managing compiled ANE models
//! - Creating and managing IOSurface buffers for inputs and outputs
//! - Writing input data to the kernel
//! - Reading output data from the kernel
//! - Evaluating the kernel on the ANE

use crate::ane::IOSurface;
use crate::error::Result;
use crate::Error;

/// Compiled ANE kernel ready for evaluation
///
/// An ANEKernel manages the lifecycle of a compiled ANE model including
/// its input and output IOSurfaces. It provides methods to write input data,
/// evaluate the kernel, and read output data.
#[derive(Debug)]
pub struct ANEKernel {
    /// Compiled ANE model (opaque handle)
    /// TODO: objc2 reference to _ANEInMemoryModel
    _model: Option<()>,

    /// Input IOSurfaces
    pub io_inputs: Vec<IOSurface>,

    /// Output IOSurfaces
    pub io_outputs: Vec<IOSurface>,

    /// Input tensor sizes in bytes
    pub input_sizes: Vec<usize>,

    /// Output tensor sizes in bytes
    pub output_sizes: Vec<usize>,
}

impl ANEKernel {
    /// Create a new ANEKernel with the specified input and output sizes
    ///
    /// This constructor creates IOSurfaces for all inputs and outputs based on
    /// the provided byte sizes. The surfaces are pre-allocated for efficient
    /// data transfer to the ANE.
    ///
    /// # Arguments
    ///
    /// * `input_sizes` - Vector of input tensor sizes in bytes
    /// * `output_sizes` - Vector of output tensor sizes in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - IOSurface creation fails
    /// - Memory allocation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::ANEKernel;
    ///
    /// let kernel = ANEKernel::new(vec![256, 512], vec![1024])?;
    /// assert_eq!(kernel.io_inputs.len(), 2);
    /// assert_eq!(kernel.io_outputs.len(), 1);
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn new(input_sizes: Vec<usize>, output_sizes: Vec<usize>) -> Result<Self> {
        let mut io_inputs = Vec::new();
        let mut io_outputs = Vec::new();

        // Create IOSurfaces for inputs
        for &size in &input_sizes {
            io_inputs.push(IOSurface::new(size)?);
        }

        // Create IOSurfaces for outputs
        for &size in &output_sizes {
            io_outputs.push(IOSurface::new(size)?);
        }

        Ok(ANEKernel {
            _model: None,
            io_inputs,
            io_outputs,
            input_sizes,
            output_sizes,
        })
    }

    /// Evaluate the kernel on the ANE
    ///
    /// This method triggers the actual computation on the ANE hardware.
    /// Input data must have been written via `write_input()` before calling this.
    /// Output data can be read via `read_output()` after successful evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The ANE runtime is not initialized
    /// - The kernel has not been compiled
    /// - The evaluation fails on the ANE
    /// - The ANE framework is not available
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::ANEKernel;
    ///
    /// let mut kernel = ANEKernel::new(vec![256], vec![512])?;
    /// kernel.write_input(0, &[1.0f32; 64])?;
    /// kernel.eval()?;
    /// let output = kernel.read_output(0)?;
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn eval(&mut self) -> Result<()> {
        // TODO: Implement ANE evaluation via objc2 bindings
        Err(Error::NotImplemented(
            "ANE kernel evaluation not yet implemented".to_string(),
        ))
    }

    /// Write input data to an IOSurface
    ///
    /// Writes f32 data to the specified input IOSurface. The data length (in bytes)
    /// must exactly match the input size specified during kernel creation.
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the input tensor (0 < idx < number of inputs)
    /// * `data` - Slice of f32 data to write
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input index is out of bounds
    /// - The data size doesn't match the expected input size in bytes
    /// - The write operation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::ANEKernel;
    ///
    /// let mut kernel = ANEKernel::new(vec![16], vec![32])?;
    /// let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    /// kernel.write_input(0, &data)?;
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn write_input(&mut self, idx: usize, data: &[f32]) -> Result<()> {
        if idx >= self.io_inputs.len() {
            return Err(Error::InvalidParameter(format!(
                "input index {} out of bounds (have {} inputs)",
                idx,
                self.io_inputs.len()
            )));
        }

        let expected_bytes = self.input_sizes[idx];
        let actual_bytes = data.len() * std::mem::size_of::<f32>();

        if actual_bytes != expected_bytes {
            return Err(Error::Io(format!(
                "input size mismatch: expected {} bytes, got {}",
                expected_bytes, actual_bytes
            )));
        }

        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, actual_bytes) };

        self.io_inputs[idx].write(bytes)?;
        Ok(())
    }

    /// Read output data from an IOSurface
    ///
    /// Reads f32 data from the specified output IOSurface after kernel evaluation.
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the output tensor (0 < idx < number of outputs)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The output index is out of bounds
    /// - The read operation fails
    /// - The data size doesn't match the expected output size
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::ANEKernel;
    ///
    /// let kernel = ANEKernel::new(vec![256], vec![512])?;
    /// let output = kernel.read_output(0)?;
    /// assert_eq!(output.len(), 128); // 512 bytes / 4 bytes per f32
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn read_output(&self, idx: usize) -> Result<Vec<f32>> {
        if idx >= self.io_outputs.len() {
            return Err(Error::InvalidParameter(format!(
                "output index {} out of bounds (have {} outputs)",
                idx,
                self.io_outputs.len()
            )));
        }

        let bytes = self.io_outputs[idx].read()?;
        let expected_bytes = self.output_sizes[idx];

        if bytes.len() != expected_bytes {
            return Err(Error::Io(format!(
                "output size mismatch: expected {} bytes, got {}",
                expected_bytes,
                bytes.len()
            )));
        }

        let floats = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / std::mem::size_of::<f32>(),
            )
        };

        Ok(floats.to_vec())
    }
}

impl Drop for ANEKernel {
    fn drop(&mut self) {
        // TODO: Unload from ANE, release IOSurfaces
        // The IOSurfaces will be automatically dropped when io_inputs and io_outputs
        // are dropped. The ANE model (once implemented) should be unloaded here.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_creation() {
        let kernel = ANEKernel::new(vec![256], vec![512]);
        assert!(kernel.is_ok());
    }

    #[test]
    fn test_kernel_io_surface_counts() {
        let kernel = ANEKernel::new(vec![256, 512], vec![1024, 2048]).unwrap();
        assert_eq!(kernel.io_inputs.len(), 2);
        assert_eq!(kernel.io_outputs.len(), 2);
    }

    #[test]
    fn test_write_input_correct_size() {
        let mut kernel = ANEKernel::new(vec![16], vec![32]).unwrap();
        let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]; // 16 bytes
        assert!(kernel.write_input(0, &data).is_ok());
    }

    #[test]
    fn test_write_input_wrong_size() {
        let mut kernel = ANEKernel::new(vec![16], vec![32]).unwrap();
        let data = vec![1.0f32; 8]; // 32 bytes, doesn't match 16
        assert!(kernel.write_input(0, &data).is_err());
    }

    #[test]
    fn test_write_input_invalid_index() {
        let mut kernel = ANEKernel::new(vec![16], vec![32]).unwrap();
        let data = vec![1.0f32; 4];
        assert!(kernel.write_input(5, &data).is_err());
    }

    #[test]
    fn test_read_output_invalid_index() {
        let kernel = ANEKernel::new(vec![16], vec![32]).unwrap();
        assert!(kernel.read_output(5).is_err());
    }

    #[test]
    fn test_eval_not_implemented() {
        let mut kernel = ANEKernel::new(vec![16], vec![32]).unwrap();
        let result = kernel.eval();
        assert!(result.is_err());
    }
}
