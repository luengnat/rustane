//! MIL Model Loader for ANE Training
//!
//! Loads MIL programs from files and compiles them for ANE execution.

use crate::ane::ANEError;
use crate::wrapper::{ANECompiler, ANEExecutor};
use std::fs;
use std::path::Path;

/// Load a MIL program from file and compile it
///
/// # Arguments
/// * `mil_path` - Path to the .mil file
/// * `input_sizes` - Input tensor sizes in bytes
/// * `output_sizes` - Output tensor sizes in bytes
///
/// # Returns
/// Compiled ANEExecutor ready for execution
///
/// # Example
/// ```no_run
/// use rustane::ane::mil_loader::load_mil_model;
///
/// let executor = load_mil_model(
///     "models/layer0/layer_0_fwd.mil",
///     &[768 * 256 * 2],  // Input: 768 channels, 256 seq, fp16
///     &[768 * 256 * 2],  // Output: same size
/// ).unwrap();
/// ```
pub fn load_mil_model(
    mil_path: impl AsRef<Path>,
    input_sizes: &[usize],
    output_sizes: &[usize],
) -> Result<ANEExecutor, ANEError> {
    // Read MIL file
    let mil_text = fs::read_to_string(mil_path).map_err(|e| ANEError::InvalidShape {
        expected: "read MIL file".to_string(),
        got: e.to_string(),
    })?;

    // Create compiler and compile
    let mut compiler = ANECompiler::new();

    compiler
        .compile_single(&mil_text, None, input_sizes, output_sizes)
        .map_err(|e| ANEError::CompileFailed(e.to_string()))
}

/// Load forward and backward MIL programs for a layer
///
/// # Arguments
/// * `layer_dir` - Directory containing layer_0_fwd.mil and layer_0_bwd.mil
/// * `input_sizes` - Input tensor sizes
/// * `output_sizes` - Output tensor sizes (same for fwd and bwd)
///
/// # Returns
/// Tuple of (forward_executor, backward_executor)
pub fn load_layer_models(
    layer_dir: impl AsRef<Path>,
    input_sizes: &[usize],
    output_sizes: &[usize],
) -> Result<(ANEExecutor, ANEExecutor), ANEError> {
    let dir = layer_dir.as_ref();

    let fwd_path = dir.join("layer_0_fwd.mil");
    let bwd_path = dir.join("layer_0_bwd.mil");

    let fwd = load_mil_model(&fwd_path, input_sizes, output_sizes)?;
    let bwd = load_mil_model(&bwd_path, input_sizes, output_sizes)?;

    Ok((fwd, bwd))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mil_file_exists() {
        // Check that MIL files were created
        let fwd_path = Path::new("models/layer0/layer_0_fwd.mil");
        let bwd_path = Path::new("models/layer0/layer_0_bwd.mil");

        assert!(
            fwd_path.exists(),
            "Forward MIL file should exist: {:?}",
            fwd_path
        );
        assert!(
            bwd_path.exists(),
            "Backward MIL file should exist: {:?}",
            bwd_path
        );
    }

    #[test]
    fn test_load_mil_model_file_not_found() {
        let result = load_mil_model("nonexistent.mil", &[1024], &[512]);
        assert!(result.is_err());
    }
}
