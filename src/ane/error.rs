// src/ane/error.rs
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
    InvalidShape { expected: String, got: String },

    /// Weight blob building failed
    WeightBlobError(String),

    /// Invalid model configuration (e.g. dim not divisible by head_dim)
    ConfigError(String),
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
        }
    }
}

impl std::error::Error for ANEError {}

impl From<ANEError> for crate::Error {
    fn from(e: ANEError) -> Self {
        crate::Error::Other(format!("{:?}", e))
    }
}
