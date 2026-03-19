//! Python bindings via PyO3 (feature-gated)
//!
//! Enables loss scaling and gradient accumulation from Python training loops.
//!
//! Usage: python3 -c "import rustane; scaler = rustane.LossScaler(256.0)"
//!
//! Build: PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --lib --features python

#![cfg(feature = "python")]

use pyo3::prelude::*;

use crate::training::grad_accum::GradAccumulator as RustGradAccumulator;
use crate::training::loss_scale::LossScaler as RustLossScaler;

/// Python wrapper for loss scaling
///
/// Manages FP16 training loss scaling and gradient overflow detection.
///
/// # Example (Python)
///
/// ```python
/// import rustane
/// import numpy as np
///
/// scaler = rustane.LossScaler(256.0)
/// loss = 0.5
/// scaled_loss = scaler.scale_loss(loss)
/// print(f"Scaled: {scaled_loss}")
///
/// # After backward pass
/// grads = [1.0, 2.0, 3.0]
/// valid = scaler.update(grads)
/// if not valid:
///     print("Overflow detected!")
/// ```
#[pyclass(name = "LossScaler")]
pub struct PyLossScaler {
    inner: RustLossScaler,
}

#[pymethods]
impl PyLossScaler {
    /// Create a new loss scaler with initial scale
    ///
    /// # Arguments
    ///
    /// * `initial_scale` - Starting scale (typically 256.0 for FP16)
    #[new]
    pub fn new(initial_scale: f32) -> Self {
        PyLossScaler {
            inner: RustLossScaler::new(initial_scale),
        }
    }

    /// Create a loss scaler for typical transformer model
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    ///
    /// # Returns
    ///
    /// Configured LossScaler instance
    #[staticmethod]
    pub fn for_transformer(num_layers: usize) -> Self {
        PyLossScaler {
            inner: RustLossScaler::for_transformer(num_layers),
        }
    }

    /// Scale a loss value before backpropagation
    ///
    /// # Arguments
    ///
    /// * `loss` - Original loss value
    ///
    /// # Returns
    ///
    /// Scaled loss value
    pub fn scale_loss(&self, loss: f32) -> f32 {
        self.inner.scale_loss(loss)
    }

    /// Update scale and check for overflow
    ///
    /// Should be called after computing gradients.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient values as list/array
    ///
    /// # Returns
    ///
    /// True if gradients are valid (no inf/nan), False if overflow detected
    pub fn update(&mut self, grads: Vec<f32>) -> bool {
        self.inner.update(&grads)
    }

    /// Get current loss scale
    ///
    /// # Returns
    ///
    /// Current scale value
    pub fn current_scale(&self) -> f32 {
        self.inner.current_scale()
    }

    /// Unscale gradients
    ///
    /// Returns unscaled gradient values.
    /// Divides all gradients by current scale.
    /// Call this before optimizer step.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient list
    ///
    /// # Returns
    ///
    /// Unscaled gradient values
    pub fn unscale_grads(&self, grads: Vec<f32>) -> Vec<f32> {
        let mut g = grads;
        self.inner.unscale_grads(&mut g);
        g
    }
}

/// Python wrapper for gradient accumulation
///
/// Accumulates gradients over multiple mini-batches for larger effective batch sizes.
///
/// # Example (Python)
///
/// ```python
/// import rustane
///
/// accum = rustane.GradAccumulator(5000, 4)  # 5000 params, 4 accumulation steps
///
/// for i in range(4):
///     loss = forward(batch)
///     backward(loss)
///     grads = get_gradients()
///     accum.accumulate_fp32(grads)
///
/// if accum.is_complete():
///     avg_grads = accum.finalize_averaged()
///     optimizer.step(avg_grads)
///     accum.reset()
/// ```
#[pyclass(name = "GradAccumulator")]
pub struct PyGradAccumulator {
    inner: RustGradAccumulator,
}

#[pymethods]
impl PyGradAccumulator {
    /// Create a new gradient accumulator
    ///
    /// # Arguments
    ///
    /// * `num_params` - Total number of model parameters
    /// * `total_steps` - Number of mini-batches to accumulate before update
    #[new]
    pub fn new(num_params: usize, total_steps: usize) -> Self {
        PyGradAccumulator {
            inner: RustGradAccumulator::new(num_params, total_steps),
        }
    }

    /// Accumulate FP32 gradients
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradient values as list/array
    pub fn accumulate_fp32(&mut self, grads: Vec<f32>) {
        self.inner.accumulate_fp32(&grads, 1.0);
    }

    /// Get accumulated gradients, averaged by step count
    ///
    /// # Returns
    ///
    /// List of averaged gradient values
    pub fn finalize_averaged(&self) -> Vec<f32> {
        self.inner.finalize_averaged()
    }

    /// Get raw accumulated gradients (not averaged)
    ///
    /// # Returns
    ///
    /// List of accumulated gradient values
    pub fn finalize(&self) -> Vec<f32> {
        self.inner.finalize().to_vec()
    }

    /// Reset accumulator for next phase
    ///
    /// Clears all accumulated gradients and resets step counter.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Check if accumulation is complete
    ///
    /// # Returns
    ///
    /// True if current_step >= total_steps
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Get current accumulation step (0-indexed)
    ///
    /// # Returns
    ///
    /// Current step counter
    pub fn current_step(&self) -> usize {
        self.inner.current_step()
    }

    /// Get remaining steps until completion
    ///
    /// # Returns
    ///
    /// Steps remaining (or 0 if already complete)
    pub fn remaining_steps(&self) -> usize {
        self.inner.remaining_steps()
    }

    /// Get number of parameters
    ///
    /// # Returns
    ///
    /// Total parameter count for this accumulator
    pub fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    /// Get total steps configured
    ///
    /// # Returns
    ///
    /// Total accumulation steps
    pub fn total_steps(&self) -> usize {
        self.inner.total_steps()
    }
}

/// Rustane Python module
///
/// FP16 training utilities for ANE-accelerated neural networks.
#[pymodule]
pub fn rustane(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLossScaler>()?;
    m.add_class::<PyGradAccumulator>()?;
    m.add_function(wrap_pyfunction!(total_leaked_bytes, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let docstring = r#"
Rustane: ANE Training Utilities

FP16 training support for Apple Neural Engine integration.

Classes:
    - LossScaler: Dynamic loss scaling to prevent gradient underflow
    - GradAccumulator: Accumulate gradients over multiple mini-batches

Functions:
    - total_leaked_bytes(): Memory leak diagnostic

Example:
    >>> import rustane
    >>> scaler = rustane.LossScaler(256.0)
    >>> scaled = scaler.scale_loss(0.5)
"#;
    m.add("__doc__", docstring)?;

    Ok(())
}

/// Get total memory allocated but not freed (diagnostic)
///
/// Returns the number of bytes in WeightBlobs that have been allocated
/// but not yet freed. In production (with ane_bridge_free_blob), this
/// should be 0. During testing, use this to detect memory leaks.
///
/// # Returns
///
/// Total leaked bytes
#[pyfunction]
pub fn total_leaked_bytes() -> usize {
    crate::total_leaked_bytes()
}
