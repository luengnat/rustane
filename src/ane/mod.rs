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

pub use error::ANEError;
pub use io_surface::IOSurface;
pub use runtime::ANECompileRequest;
