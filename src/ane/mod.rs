//! Private ANE implementation details.
//!
//! This module contains the `objc2`-based integration with the private
//! `AppleNeuralEngine.framework`. The rest of the crate should generally work
//! through the safe wrappers in [`crate::wrapper`].

pub mod error;
pub(crate) mod blobs;
pub mod runtime;
pub(crate) mod sys;

pub use error::ANEError;
pub use runtime::ANECompileRequest;
