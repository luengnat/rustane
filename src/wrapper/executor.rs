//! ANE executor for running compiled kernels
//!
//! ANEExecutor executes compiled kernels on the Apple Neural Engine,
//! managing input/output data via IOSurface-backed memory.
//!
//! ## Ownership Model
//!
//! ANEExecutor **owns** its kernel handle. When the compiler produces an
//! executor, it transfers ownership of the kernel pointer. The executor's
//! `Drop` implementation calls `ane_bridge_free` to release the ANE resources.
//!
//! This means:
//! - `ANECompiler` no longer frees kernels in its `Drop` (it sets kernel to null after transfer)
//! - `ANEExecutor` can safely outlive the compiler
//! - Cached executors in `KernelCache` / `ANEProgramCache` remain valid indefinitely

use crate::ane::IOSurface;
use crate::sys::{
    ane_bridge_eval, ane_bridge_free, ane_bridge_read_output, ane_bridge_reload_weights,
    ane_bridge_write_input, ANEKernelHandle,
};
use crate::{Error, Result};

/// ANE kernel executor
///
/// ANEExecutor **owns** the compiled kernel handle and is responsible for
/// freeing ANE resources when dropped. It manages I/O via IOSurface-backed memory.
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
    /// Owned kernel handle — freed on drop
    kernel: *mut ANEKernelHandle,
    input_sizes: Vec<usize>,
    output_sizes: Vec<usize>,
    /// IOSurface buffers for inputs (for fp16 conversion)
    io_inputs: Vec<IOSurface>,
    /// IOSurface buffers for outputs (for fp16 conversion)
    io_outputs: Vec<IOSurface>,
}

// ANEExecutor owns its kernel and is safe to send across threads
// (the ANE bridge functions are thread-safe with their own internal locking)
unsafe impl Send for ANEExecutor {}

impl Drop for ANEExecutor {
    fn drop(&mut self) {
        if !self.kernel.is_null() {
            unsafe {
                ane_bridge_free(self.kernel);
            }
            self.kernel = std::ptr::null_mut();
        }
    }
}

impl std::fmt::Debug for ANEExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ANEExecutor")
            .field("kernel_valid", &!self.kernel.is_null())
            .field("input_sizes", &self.input_sizes)
            .field("output_sizes", &self.output_sizes)
            .field("num_io_inputs", &self.io_inputs.len())
            .field("num_io_outputs", &self.io_outputs.len())
            .finish()
    }
}

impl ANEExecutor {
    /// Create a new executor that owns the kernel handle
    ///
    /// Called by ANECompiler after successful compilation.
    /// The compiler transfers ownership of the kernel pointer.
    pub(crate) fn new(
        kernel: *mut ANEKernelHandle,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self> {
        // Create IOSurfaces for inputs and outputs
        let mut io_inputs = Vec::new();
        let mut io_outputs = Vec::new();

        for &size in input_sizes {
            io_inputs.push(IOSurface::new(size)?);
        }

        for &size in output_sizes {
            io_outputs.push(IOSurface::new(size)?);
        }

        Ok(ANEExecutor {
            kernel,
            input_sizes: input_sizes.to_vec(),
            output_sizes: output_sizes.to_vec(),
            io_inputs,
            io_outputs,
        })
    }

    /// Execute the compiled kernel on the ANE
    ///
    /// This runs the compiled program on the ANE hardware. Input data must
    /// be written via `write_input()` before calling this method.
    pub fn eval(&mut self) -> Result<()> {
        if self.kernel.is_null() {
            return Err(Error::ExecutionFailed(
                "Kernel is null (already freed or never compiled)".to_string(),
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

    // -----------------------------------------------------------------------
    // Fast fp16 path (zero extra copy)
    // -----------------------------------------------------------------------

    /// Write fp32 data directly to the ANE input buffer (fp32 → fp16 in-place).
    ///
    /// This is a **single-copy** path: converts fp32→fp16 straight into the
    /// kernel's IOSurface, then copies once into the ANE input.
    ///
    /// For maximum speed use `write_input()` directly with pre-converted fp16 data.
    pub fn write_input_f32(&mut self, idx: usize, data: &[f32]) -> Result<()> {
        let expected_bytes = self.input_sizes.get(idx).copied().ok_or_else(|| {
            Error::InvalidParameter(format!(
                "Input index {} out of bounds ({} inputs)",
                idx,
                self.input_sizes.len()
            ))
        })?;

        let required_bytes = data.len() * 2; // fp32 → fp16
        if required_bytes != expected_bytes {
            return Err(Error::InvalidParameter(format!(
                "Input fp32 data ({}) converts to {} fp16 bytes, expected {} for input {}",
                data.len(),
                required_bytes,
                expected_bytes,
                idx
            )));
        }

        // Convert fp32 → fp16 into the IOSurface (single allocation)
        if idx < self.io_inputs.len() {
            self.io_inputs[idx].write_f32(data)?;
            let fp16_bytes = self.io_inputs[idx].read_vec()?;
            self.write_input(idx, &fp16_bytes)?;
        } else {
            // Fallback: allocate temp buffer if no IOSurface available
            let mut fp16 = vec![0u8; required_bytes];
            // Simple fp32→fp16 conversion inline
            for (i, val) in data.iter().enumerate() {
                let fp16_val = f32_to_fp16_fast(*val);
                let offset = i * 2;
                fp16[offset] = (fp16_val & 0xFF) as u8;
                fp16[offset + 1] = (fp16_val >> 8) as u8;
            }
            self.write_input(idx, &fp16)?;
        }

        Ok(())
    }

    /// Read fp16 output data from the ANE and convert to fp32 (single-copy).
    ///
    /// Reads the raw output bytes, then converts fp16→fp32 in one pass.
    pub fn read_output_f32(&self, idx: usize) -> Result<Vec<f32>> {
        let expected_bytes = self.output_sizes.get(idx).copied().ok_or_else(|| {
            Error::InvalidParameter(format!(
                "Output index {} out of bounds ({} outputs)",
                idx,
                self.output_sizes.len()
            ))
        })?;

        let mut raw_bytes = vec![0u8; expected_bytes];
        self.read_output(idx, &mut raw_bytes)?;

        // Convert fp16 → fp32
        let num_floats = expected_bytes / 2;
        let mut result = Vec::with_capacity(num_floats);
        for i in 0..num_floats {
            let fp16_val = u16::from_le_bytes([raw_bytes[i * 2], raw_bytes[i * 2 + 1]]);
            result.push(fp16_to_f32_fast(fp16_val));
        }

        Ok(result)
    }

    /// Execute with fp32 input, returning fp32 output (optimized single-copy path).
    ///
    /// This replaces the old double-copy `execute_f32` with:
    /// 1. fp32→fp16 conversion directly into IOSurface
    /// 2. Single copy to ANE input
    /// 3. Execute
    /// 4. Single read + fp16→fp32 conversion
    ///
    /// For the fastest possible execution, use `write_input()` + `eval()` + `read_output()`
    /// with pre-converted fp16 data.
    pub fn execute_f32(
        &mut self,
        input_idx: usize,
        output_idx: usize,
        input_data: &[f32],
    ) -> Result<Vec<f32>> {
        self.write_input_f32(input_idx, input_data)?;
        self.eval()?;
        self.read_output_f32(output_idx)
    }

    // -----------------------------------------------------------------------
    // Raw byte I/O (fastest path — caller manages fp16 conversion)
    // -----------------------------------------------------------------------

    /// Write raw bytes directly to an ANE input tensor.
    ///
    /// This is the **zero-conversion** fast path. Callers must provide
    /// data already in the correct format (fp16 for ANE kernels).
    ///
    /// # Arguments
    ///
    /// * `idx` - Input tensor index (0-based)
    /// * `data` - Raw data (must match expected size exactly)
    pub fn write_input(&mut self, idx: usize, data: &[u8]) -> Result<()> {
        let expected_size = self.input_sizes.get(idx).copied().ok_or_else(|| {
            Error::InvalidParameter(format!(
                "Input index {} out of bounds ({} inputs)",
                idx,
                self.input_sizes.len()
            ))
        })?;

        if data.len() != expected_size {
            return Err(Error::InvalidParameter(format!(
                "Input data size ({}) doesn't match expected size ({}) for input {}",
                data.len(),
                expected_size,
                idx
            )));
        }

        if self.kernel.is_null() {
            return Err(Error::ExecutionFailed(
                "Kernel is null (already freed or never compiled)".to_string(),
            ));
        }

        // SAFETY: write_input is safe when kernel is valid and data is valid
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

    /// Read raw bytes from an ANE output tensor.
    ///
    /// This is the **zero-conversion** fast path. Returns raw bytes in
    /// whatever format the kernel produces (typically fp16).
    pub fn read_output(&self, idx: usize, data: &mut [u8]) -> Result<()> {
        let expected_size = self.output_sizes.get(idx).copied().ok_or_else(|| {
            Error::InvalidParameter(format!(
                "Output index {} out of bounds ({} outputs)",
                idx,
                self.output_sizes.len()
            ))
        })?;

        if data.len() != expected_size {
            return Err(Error::InvalidParameter(format!(
                "Output buffer size ({}) doesn't match expected size ({}) for output {}",
                data.len(),
                expected_size,
                idx
            )));
        }

        if self.kernel.is_null() {
            return Err(Error::ExecutionFailed(
                "Kernel is null (already freed or never compiled)".to_string(),
            ));
        }

        // SAFETY: read_output is safe when kernel is valid and data is valid
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

    /// Convenience: read output into a new Vec (raw bytes).
    pub fn read_output_vec(&self, idx: usize) -> Result<Vec<u8>> {
        let size = self.output_sizes.get(idx).copied().ok_or_else(|| {
            Error::InvalidParameter(format!(
                "Output index {} out of bounds ({} outputs)",
                idx,
                self.output_sizes.len()
            ))
        })?;
        let mut buf = vec![0u8; size];
        self.read_output(idx, &mut buf)?;
        Ok(buf)
    }

    // -----------------------------------------------------------------------
    // Weight reload (training fast path)
    // -----------------------------------------------------------------------

    /// Reload weights without recompiling (delta compilation).
    ///
    /// This is the **key ANE training optimization**: instead of recompiling
    /// (~4,200ms), we update weight files on disk and reload (~494ms).
    ///
    /// Call this every training step after the initial compile.
    pub fn reload_weights(&mut self, weight_files: &[(&str, &[u8])]) -> Result<()> {
        if weight_files.is_empty() {
            return Err(Error::InvalidParameter(
                "No weight files provided".to_string(),
            ));
        }

        if self.kernel.is_null() {
            return Err(Error::ExecutionFailed(
                "Kernel is null (already freed or never compiled)".to_string(),
            ));
        }

        // SAFETY: reload_weights is safe when kernel is valid and data is valid
        let success = unsafe { ane_bridge_reload_weights(self.kernel, weight_files) };

        if !success {
            return Err(Error::ExecutionFailed(
                "ANE weight reload failed".to_string(),
            ));
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

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

    /// Check if the kernel handle is still valid
    pub fn is_valid(&self) -> bool {
        !self.kernel.is_null()
    }

    /// Create an executor without a valid kernel (test-only / mock)
    #[cfg(test)]
    pub(crate) fn new_unchecked(
        kernel: *mut ANEKernelHandle,
        input_sizes: Vec<usize>,
        output_sizes: Vec<usize>,
    ) -> Self {
        let io_inputs: Vec<IOSurface> = input_sizes
            .iter()
            .filter_map(|&s| IOSurface::new(s).ok())
            .collect();
        let io_outputs: Vec<IOSurface> = output_sizes
            .iter()
            .filter_map(|&s| IOSurface::new(s).ok())
            .collect();
        Self {
            kernel,
            input_sizes,
            output_sizes,
            io_inputs,
            io_outputs,
        }
    }
}

// ---------------------------------------------------------------------------
// Fast fp16 conversion (inline helpers)
// ---------------------------------------------------------------------------

/// Fast fp16 → f32 conversion
#[inline]
fn fp16_to_f32_fast(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let mut e = 0i32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            let new_exp = 127 - 15 - e + 1;
            f32::from_bits((sign << 31) | ((new_exp as u32) << 23) | ((f & 0x3FF) << 13))
        }
    } else if exp == 0x1F {
        // Inf / NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        // Normal: fp16 exponent is biased by 15, fp32 by 127
        let new_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | ((new_exp as u32) << 23) | (frac << 13))
    }
}

/// Fast f32 → fp16 conversion (clamps to fp16 range)
#[inline]
fn f32_to_fp16_fast(f: f32) -> u16 {
    if f.is_nan() {
        return 0x7FFF; // qNaN
    }
    if f.is_infinite() {
        return if f.is_sign_negative() { 0xFC00 } else { 0x7C00 };
    }

    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = (bits & 0x7FFFFF) as u32;

    if exp == 0 {
        return sign << 15;
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        return (sign << 15) | 0x7C00; // overflow to inf
    }

    if new_exp <= 0 {
        if new_exp < -10 {
            return sign << 15; // underflow to zero
        }
        // Subnormal
        let mant = (frac | 0x800000) >> (1 - new_exp);
        (sign << 15) | ((mant >> 13) as u16)
    } else {
        // Normal
        (sign << 15) | ((new_exp as u16) << 10) | ((frac >> 13) as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_executor_methods() {
        let executor = ANEExecutor::new_unchecked(ptr::null_mut(), vec![1024, 512], vec![256]);

        assert_eq!(executor.num_inputs(), 2);
        assert_eq!(executor.num_outputs(), 1);
        assert_eq!(executor.input_size(0), Some(1024));
        assert_eq!(executor.output_size(0), Some(256));
        assert_eq!(executor.input_size(5), None);
        assert!(!executor.is_valid());
    }

    #[test]
    fn test_executor_write_input_bounds_check() {
        let mut executor = ANEExecutor::new_unchecked(ptr::null_mut(), vec![1024], vec![256]);

        // Valid size but null kernel
        let data = vec![0u8; 1024];
        let result = executor.write_input(0, &data);
        assert!(matches!(result, Err(Error::ExecutionFailed(_))));

        // Out of bounds
        let result = executor.write_input(5, &data);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));

        // Wrong size
        let data = vec![0u8; 512];
        let result = executor.write_input(0, &data);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_executor_read_output_bounds_check() {
        let executor = ANEExecutor::new_unchecked(ptr::null_mut(), vec![1024], vec![256]);

        let mut buf = vec![0u8; 256];
        let result = executor.read_output(0, &mut buf);
        assert!(matches!(result, Err(Error::ExecutionFailed(_))));

        let result = executor.read_output(5, &mut buf);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));

        let mut buf = vec![0u8; 128];
        let result = executor.read_output(0, &mut buf);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_executor_reload_weights_null_kernel() {
        let mut executor = ANEExecutor::new_unchecked(ptr::null_mut(), vec![1024], vec![256]);
        let weights = vec![0u8; 128];
        let result = executor.reload_weights(&[("w.bin", &weights)]);
        assert!(matches!(result, Err(Error::ExecutionFailed(_))));
    }

    #[test]
    fn test_executor_reload_weights_empty_files() {
        let mut executor = ANEExecutor::new_unchecked(ptr::null_mut(), vec![1024], vec![256]);
        let result = executor.reload_weights(&[]);
        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn test_read_output_vec_null_kernel() {
        let executor = ANEExecutor::new_unchecked(ptr::null_mut(), vec![1024], vec![256]);
        let result = executor.read_output_vec(0);
        assert!(matches!(result, Err(Error::ExecutionFailed(_))));
    }

    #[test]
    fn test_fp16_roundtrip() {
        // Test a range of values
        let values: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            100.0,
            -100.0,
            3.14159,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];

        for val in values {
            let fp16 = f32_to_fp16_fast(val);
            let back = fp16_to_f32_fast(fp16);
            if val.is_finite() {
                let rel_err = ((val - back).abs() / val.abs().max(1e-30))
                    .max((back - val).abs() / val.abs().max(1e-30));
                assert!(
                    rel_err < 0.001,
                    "fp16 roundtrip failed for {}: got {} (err={})",
                    val,
                    back,
                    rel_err
                );
            } else {
                assert!(back.is_infinite() || back.is_nan());
            }
        }
    }

    #[test]
    fn test_fp16_subnormal() {
        let tiny = 6.0e-5; // subnormal range for fp16
        let fp16 = f32_to_fp16_fast(tiny);
        let back = fp16_to_f32_fast(fp16);
        // Subnormals lose precision but shouldn't be zero
        assert!(back.abs() > 0.0 || tiny == 0.0);
    }
}
