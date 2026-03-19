//! MIL (Model Intermediate Language) utilities
//!
//! This module provides helpers for constructing MIL programs for the ANE.
//! MIL is Apple's intermediate representation for neural network computations.

pub mod builder;
pub mod programs;
pub mod util;

pub use builder::MILBuilder;
pub use programs::{rmsnorm_mil, LinearLayer};
pub use util::{total_leaked_bytes, WeightBlob};
