//! Apple Neural Engine internals and low-level helpers.
//!
//! This module contains the `objc2`-based integration with the private
//! `AppleNeuralEngine.framework`. The rest of the crate should generally work
//! through the safe wrappers in [`crate::wrapper`].

/// ANE-specific error types.
pub mod error;
pub(crate) mod blobs;
/// Low-level runtime and compile/load/eval support for the private ANE APIs.
pub mod runtime;
pub(crate) mod sys;
/// IOSurface RAII wrapper for ANE I/O operations.
pub mod io_surface;
/// Weight blob builders for ANE-compatible formats.
pub mod weight_blob;
/// ANE kernel wrapper for managing compiled models and I/O operations.
pub mod kernel;

pub use error::ANEError;
pub use io_surface::IOSurface;
pub use runtime::ANECompileRequest;
pub use weight_blob::WeightBlob;
pub use kernel::ANEKernel;

/// Result type for ANE operations
pub type Result<T> = std::result::Result<T, ANEError>;
