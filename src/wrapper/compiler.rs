//! ANE compiler for MIL programs
//!
//! ANECompiler compiles MIL (Model Intermediate Language) programs into
//! executable kernels that can run on the Apple Neural Engine.
//!
//! ## Ownership Model
//!
//! After a successful compile, the kernel pointer is **transferred** to the
//! returned `ANEExecutor`. The compiler sets its internal kernel to null so
//! that its `Drop` does not double-free. The executor's `Drop` will free
//! the ANE resources.

use super::executor::ANEExecutor;
use crate::sys::{ane_bridge_compile, ane_bridge_compile_multi_weights, ANEKernelHandle};
use crate::{Error, Result};
use std::ffi::CString;
use std::ptr;

/// MIL program compiler
///
/// ANECompiler compiles MIL text programs into executable kernels that can
/// be run on the ANE. On compile, kernel ownership is transferred to the
/// returned `ANEExecutor`.
///
/// # Example
///
/// ```no_run
/// # use rustane::wrapper::ANECompiler;
/// # use rustane::wrapper::ANERuntime;
/// # fn main() -> rustane::Result<()> {
/// let _runtime = ANERuntime::init()?;
/// let mil_program = r#"
///     program(1.0) {
///         var _0 = nn.convolution(bias=0, groups=1, strides=[1, 1], ...)
///     }
/// "#;
/// let mut compiler = ANECompiler::new();
/// let mut executor = compiler.compile_single(
///     mil_program,
///     None,
///     &[1024],
///     &[512]
/// )?;
///
/// // executor now owns the kernel — compiler can be dropped safely
/// drop(compiler);
/// executor.eval()?;
/// # Ok(())
/// # }
/// ```
pub struct ANECompiler {
    /// Kernel pointer — always null after a successful compile (ownership transferred)
    kernel: *mut ANEKernelHandle,
    input_sizes: Vec<usize>,
    output_sizes: Vec<usize>,
}

impl ANECompiler {
    /// Create a new compiler instance
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::ANECompiler;
    /// let compiler = ANECompiler::new();
    /// ```
    pub fn new() -> Self {
        ANECompiler {
            kernel: ptr::null_mut(),
            input_sizes: Vec::new(),
            output_sizes: Vec::new(),
        }
    }

    /// Compile a MIL program with optional weights
    ///
    /// # Arguments
    ///
    /// * `mil_text` - MIL program as UTF-8 string
    /// * `weight_data` - Optional weight blob (can be None)
    /// * `weight_len` - Length of weight blob in bytes
    /// * `input_sizes` - Array of input tensor sizes in bytes
    /// * `output_sizes` - Array of output tensor sizes in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - MIL text is not valid UTF-8
    /// - Compilation fails (invalid MIL, out of memory, etc.)
    /// - ANE runtime is not initialized
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::ANECompiler;
    /// # use rustane::wrapper::ANERuntime;
    /// # fn main() -> rustane::Result<()> {
    /// let _runtime = ANERuntime::init()?;
    /// let mil = "program(1.0) { var _0 = nn.convolution(...) }";
    /// let mut compiler = ANECompiler::new();
    /// let executor = compiler.compile_single(
    ///     mil,
    ///     None,
    ///     &[1024],
    ///     &[512]
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile_single(
        &mut self,
        mil_text: &str,
        weight_data: Option<&[u8]>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<ANEExecutor> {
        // Free any previous kernel that wasn't transferred
        if !self.kernel.is_null() {
            // This shouldn't happen in normal usage (kernel is nulled after transfer),
            // but handle it defensively
            self.kernel = ptr::null_mut();
        }

        // Convert MIL text to C string
        let mil_cstring = CString::new(mil_text)
            .map_err(|e| Error::CompilationFailed(format!("Invalid MIL text: {}", e)))?;

        // Prepare weight data
        let (weight_ptr, weight_len) = match weight_data {
            Some(data) => (data.as_ptr(), data.len()),
            None => (ptr::null(), 0),
        };

        // Compile
        let kernel = unsafe {
            ane_bridge_compile(
                mil_cstring.as_ptr(),
                mil_text.len(),
                weight_ptr,
                weight_len,
                input_sizes.len() as i32,
                input_sizes.as_ptr(),
                output_sizes.len() as i32,
                output_sizes.as_ptr(),
            )
        };

        if kernel.is_null() {
            return Err(Error::CompilationFailed(
                "ANE compilation returned null kernel".to_string(),
            ));
        }

        // Store sizes (for num_inputs/num_outputs queries)
        self.input_sizes = input_sizes.to_vec();
        self.output_sizes = output_sizes.to_vec();

        // Create executor — kernel ownership is transferred here
        let executor = ANEExecutor::new(kernel, input_sizes, output_sizes)?;

        // Compiler no longer owns the kernel
        self.kernel = ptr::null_mut();

        Ok(executor)
    }

    /// Compile with multiple named weight files
    ///
    /// This is useful for transformer kernels with multiple weight matrices.
    ///
    /// # Arguments
    ///
    /// * `mil_text` - MIL program as UTF-8 string
    /// * `weight_names` - Array of weight file paths (e.g., "@model_path/weights/wq.bin")
    /// * `weight_datas` - Array of weight data pointers
    /// * `weight_lens` - Array of weight data lengths
    /// * `input_sizes` - Array of input tensor sizes in bytes
    /// * `output_sizes` - Array of output tensor sizes in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - MIL text is not valid UTF-8
    /// - Weight arrays have different lengths
    /// - Compilation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use rustane::wrapper::ANECompiler;
    /// # use rustane::wrapper::ANERuntime;
    /// # fn main() -> rustane::Result<()> {
    /// let _runtime = ANERuntime::init()?;
    /// let mil = "program(1.0) { ... }";
    /// let weights_q = vec![0i8; 1024];
    /// let weights_k = vec![0i8; 1024];
    /// let mut compiler = ANECompiler::new();
    /// let executor = compiler.compile_multi(
    ///     mil,
    ///     &["@model_path/weights/wq.bin", "@model_path/weights/wk.bin"],
    ///     &[&weights_q, &weights_k],
    ///     &[1024, 1024],
    ///     &[4096],
    ///     &[512]
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn compile_multi(
        &mut self,
        mil_text: &str,
        weight_names: &[&str],
        weight_datas: &[&[u8]],
        weight_lens: &[usize],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<ANEExecutor> {
        // Validate arrays
        if weight_names.len() != weight_datas.len() || weight_names.len() != weight_lens.len() {
            return Err(Error::InvalidParameter(
                "Weight arrays must have the same length".to_string(),
            ));
        }

        // Free any previous kernel that wasn't transferred
        if !self.kernel.is_null() {
            self.kernel = ptr::null_mut();
        }

        // Convert MIL text to C string
        let mil_cstring = CString::new(mil_text)
            .map_err(|e| Error::CompilationFailed(format!("Invalid MIL text: {}", e)))?;

        // Convert weight names to C strings
        let weight_name_ptrs: Result<Vec<CString>> = weight_names
            .iter()
            .map(|name| {
                CString::new(*name)
                    .map_err(|e| Error::CompilationFailed(format!("Invalid weight name: {}", e)))
            })
            .collect();
        let weight_name_cstrings = weight_name_ptrs?;
        let weight_name_ptrs: Vec<*const i8> =
            weight_name_cstrings.iter().map(|s| s.as_ptr()).collect();

        // Prepare weight data pointers
        let weight_data_ptrs: Vec<*const u8> =
            weight_datas.iter().map(|data| data.as_ptr()).collect();

        // Compile
        let kernel = unsafe {
            ane_bridge_compile_multi_weights(
                mil_cstring.as_ptr(),
                mil_text.len(),
                weight_name_ptrs.as_ptr() as *mut *const i8,
                weight_data_ptrs.as_ptr() as *mut *const u8,
                weight_lens.as_ptr(),
                weight_names.len() as i32,
                input_sizes.len() as i32,
                input_sizes.as_ptr(),
                output_sizes.len() as i32,
                output_sizes.as_ptr(),
            )
        };

        if kernel.is_null() {
            return Err(Error::CompilationFailed(
                "ANE compilation returned null kernel".to_string(),
            ));
        }

        // Store sizes
        self.input_sizes = input_sizes.to_vec();
        self.output_sizes = output_sizes.to_vec();

        // Create executor — kernel ownership is transferred here
        let executor = ANEExecutor::new(kernel, input_sizes, output_sizes)?;

        // Compiler no longer owns the kernel
        self.kernel = ptr::null_mut();

        Ok(executor)
    }

    /// Get the number of inputs for the compiled kernel
    pub fn num_inputs(&self) -> usize {
        self.input_sizes.len()
    }

    /// Get the number of outputs for the compiled kernel
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

impl Default for ANECompiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ANECompiler {
    fn drop(&mut self) {
        // Kernel ownership was transferred to ANEExecutor after compile.
        // self.kernel should always be null here.
        // If it's not (defensive), we still don't free — the executor owns it.
        self.kernel = ptr::null_mut();
    }
}

// ANECompiler is Send but NOT Sync — it's a compile-only tool
unsafe impl Send for ANECompiler {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_creation() {
        let compiler = ANECompiler::new();
        assert_eq!(compiler.num_inputs(), 0);
        assert_eq!(compiler.num_outputs(), 0);
    }

    #[test]
    fn test_compiler_default() {
        let compiler = ANECompiler::default();
        assert_eq!(compiler.num_inputs(), 0);
    }

    #[test]
    fn test_compiler_multi_validation() {
        let mut compiler = ANECompiler::new();
        let mil = "program(1.0) { }";

        // Mismatched array lengths
        let result = compiler.compile_multi(
            mil,
            &["@w1.bin", "@w2.bin"],
            &[&vec![0u8; 1024]],
            &[1024, 1024],
            &[1024],
            &[512],
        );

        assert!(matches!(result, Err(Error::InvalidParameter(_))));
    }
}
