//! Error types for Rustane

use std::fmt;

/// Result type alias for Rustane operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in Rustane operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// ANE runtime is not initialized
    NotInitialized,

    /// ANE compilation failed
    CompilationFailed(String),

    /// ANE execution failed
    ExecutionFailed(String),

    /// Invalid tensor shape or dimensions
    InvalidTensorShape(String),

    /// Invalid parameter provided
    InvalidParameter(String),

    /// I/O error (reading/writing tensor data)
    Io(String),

    /// ANE hardware not available or not supported
    HardwareUnavailable(String),

    /// Library linking or loading error
    LibraryError(String),

    /// Feature or method is not implemented yet
    NotImplemented(String),

    /// Graph IR error
    GraphError(String),

    /// Other error
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotInitialized => write!(f, "ANE runtime not initialized"),
            Error::CompilationFailed(msg) => write!(f, "ANE compilation failed: {}", msg),
            Error::ExecutionFailed(msg) => write!(f, "ANE execution failed: {}", msg),
            Error::InvalidTensorShape(msg) => write!(f, "Invalid tensor shape: {}", msg),
            Error::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Error::Io(msg) => write!(f, "I/O error: {}", msg),
            Error::HardwareUnavailable(msg) => write!(f, "ANE hardware unavailable: {}", msg),
            Error::LibraryError(msg) => write!(f, "Library error: {}", msg),
            Error::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            Error::GraphError(msg) => write!(f, "Graph error: {}", msg),
            Error::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

/// Data loader error type
/// Re-exported from data::loader for convenience
pub use crate::data::loader::DataLoaderError;

impl std::error::Error for Error {}

// Implement From for DataLoaderError
impl From<DataLoaderError> for Error {
    fn from(err: DataLoaderError) -> Self {
        Error::Io(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::NotInitialized;
        assert_eq!(format!("{}", err), "ANE runtime not initialized");

        let err = Error::CompilationFailed("test".to_string());
        assert!(format!("{}", err).contains("compilation failed"));
    }
}
