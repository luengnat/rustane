//! ANE-specific error types.

use std::fmt;

/// Errors from ANE operations
#[derive(Debug, Clone)]
pub enum ANEError {
    /// Private ANE framework not found on this system
    FrameworkNotFound,

    /// ANE kernel compilation failed
    CompileFailed(String),

    /// ANE kernel evaluation (forward pass) failed
    EvalFailed(String),

    /// IOSurface creation or operation failed
    IOSurfaceError(String),

    /// Tensor shape mismatch (expected vs got)
    InvalidShape {
        /// Expected tensor/layout description.
        expected: String,
        /// Actual tensor/layout description received at runtime.
        got: String,
    },

    /// Weight blob building failed
    WeightBlobError(String),

    /// Invalid model configuration (e.g. dim not divisible by head_dim)
    ConfigError(String),

    /// HWX file not found in search paths
    HWXNotFound(String),

    /// Invalid HWX file format
    InvalidHWX(String),

    /// IO operation failed
    IOError(String),
}

impl fmt::Display for ANEError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ANEError::FrameworkNotFound => write!(f, "ANE framework not found"),
            ANEError::CompileFailed(msg) => write!(f, "ANE compilation failed: {}", msg),
            ANEError::EvalFailed(msg) => write!(f, "ANE eval failed: {}", msg),
            ANEError::IOSurfaceError(msg) => write!(f, "IOSurface error: {}", msg),
            ANEError::InvalidShape { expected, got } => {
                write!(f, "Shape mismatch: expected {}, got {}", expected, got)
            }
            ANEError::WeightBlobError(msg) => write!(f, "Weight blob error: {}", msg),
            ANEError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            ANEError::HWXNotFound(msg) => write!(f, "HWX not found: {}", msg),
            ANEError::InvalidHWX(msg) => write!(f, "Invalid HWX: {}", msg),
            ANEError::IOError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for ANEError {}

impl From<ANEError> for crate::Error {
    fn from(e: ANEError) -> Self {
        crate::Error::Other(format!("{:?}", e))
    }
}
