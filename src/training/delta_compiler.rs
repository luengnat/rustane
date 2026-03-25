//! Delta compilation manager for multi-layer ANE training.
//!
//! [`DeltaCompiler`] owns multiple [`ANEExecutor`] instances (one per compiled program),
//! tracks the ANE compile budget via [`CompileBudgetMonitor`], and provides efficient
//! weight reloading for the training loop.
//!
//! ## Usage
//!
//! ```ignore
//! use rustane::training::delta_compiler::DeltaCompiler;
//! use rustane::wrapper::ANECompiler;
//!
//! let mut dc = DeltaCompiler::new();
//!
//! // Compile a forward pass program
//! let idx = dc.add_program(
//!     &mil_text,
//!     &["@model_path/weights/w.bin"],
//!     &[&weight_blob],
//!     &[weight_blob.len()],
//!     &[input_size],
//!     &[output_size],
//! )?;
//!
//! // Reload weights without recompiling
//! let reload_time = dc.reload_layer(0, &[("w.bin", &new_weights)])?;
//!
//! // Execute
//! dc.executor_mut(0)?.write_input(0, &input_data)?;
//! dc.executor_mut(0)?.eval()?;
//! ```

use crate::ane::runtime::{BudgetStatus, CompileBudgetMonitor};
use crate::wrapper::{ANECompiler, ANEExecutor};
use crate::{Error, Result};
use std::time::{Duration, Instant};

/// A single compiled ANE program with its weight file metadata.
struct CompiledLayer {
    /// Owned executor (owns the ANE kernel handle).
    executor: ANEExecutor,
    /// Weight file paths used during compilation (for reload identification).
    weight_names: Vec<String>,
    /// Number of input tensors.
    num_inputs: usize,
    /// Number of output tensors.
    num_outputs: usize,
}

/// Multi-layer delta compilation manager.
///
/// Compiles ANE programs once, then uses delta compilation (weight reload)
/// for fast weight updates during training. This avoids the ~119 compile
/// limit per process and the ~4,200ms compilation cost.
///
/// Tracks compile budget and emits warnings when approaching the limit.
pub struct DeltaCompiler {
    /// Compiled layers (forward and/or backward programs).
    layers: Vec<CompiledLayer>,
    /// Compile budget monitor.
    monitor: CompileBudgetMonitor,
    /// Compile count at construction time.
    initial_compile_count: i32,
}

impl DeltaCompiler {
    /// Create a new DeltaCompiler.
    ///
    /// Records the current ANE compile count as baseline.
    pub fn new() -> Self {
        let monitor = CompileBudgetMonitor::new();
        let initial_compile_count = monitor.get_compile_count();
        Self {
            layers: Vec::new(),
            monitor,
            initial_compile_count,
        }
    }

    /// Create a DeltaCompiler with a custom compile budget.
    ///
    /// # Arguments
    ///
    /// * `budget` - Maximum number of compiles allowed per process.
    pub fn with_budget(budget: i32) -> Self {
        let monitor = CompileBudgetMonitor::with_budget(budget);
        let initial_compile_count = monitor.get_compile_count();
        Self {
            layers: Vec::new(),
            monitor,
            initial_compile_count,
        }
    }

    /// Compile and add a program (forward or backward).
    ///
    /// Returns the layer index for later reference.
    ///
    /// # Arguments
    ///
    /// * `mil_text` - MIL program source.
    /// * `weight_names` - Weight file paths (e.g., `["@model_path/weights/w.bin"]`).
    /// * `weight_datas` - Weight blob data for each file.
    /// * `weight_lens` - Byte lengths of each weight blob.
    /// * `input_sizes` - Byte sizes of each input tensor.
    /// * `output_sizes` - Byte sizes of each output tensor.
    pub fn add_program(
        &mut self,
        mil_text: &str,
        weight_names: &[&str],
        weight_datas: &[&[u8]],
        weight_lens: &[usize],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<usize> {
        let mut compiler = ANECompiler::new();
        let executor = compiler.compile_multi(
            mil_text,
            weight_names,
            weight_datas,
            weight_lens,
            input_sizes,
            output_sizes,
        )?;

        let num_inputs = input_sizes.len();
        let num_outputs = output_sizes.len();
        let weight_name_strings: Vec<String> = weight_names.iter().map(|s| s.to_string()).collect();

        let idx = self.layers.len();
        self.layers.push(CompiledLayer {
            executor,
            weight_names: weight_name_strings,
            num_inputs,
            num_outputs,
        });

        // Check budget after compile
        if self.check_budget_warning() {
            eprintln!(
                "[DeltaCompiler] WARNING: compile budget low ({} used, {} remaining)",
                self.compile_count(),
                self.remaining_budget()
            );
        }

        Ok(idx)
    }

    /// Reload weights for a specific layer.
    ///
    /// Returns the time taken for the reload operation.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the layer to update.
    /// * `weight_updates` - Pairs of `(weight_file_name, new_weight_data)`.
    pub fn reload_layer(
        &mut self,
        layer_idx: usize,
        weight_updates: &[(&str, &[u8])],
    ) -> Result<Duration> {
        let layer = self.layers.get_mut(layer_idx).ok_or_else(|| {
            Error::InvalidParameter(format!("Invalid layer index: {}", layer_idx))
        })?;

        let start = Instant::now();
        layer.executor.reload_weights(weight_updates)?;
        Ok(start.elapsed())
    }

    /// Reload weights for ALL layers.
    ///
    /// Returns the total time taken for all reloads.
    ///
    /// # Arguments
    ///
    /// * `weight_updates` - Per-layer weight updates. `weight_updates[layer_idx]` contains
    ///   the weight pairs for that layer.
    pub fn reload_all(&mut self, weight_updates: &[Vec<(&str, &[u8])>]) -> Result<Duration> {
        let start = Instant::now();
        for (i, updates) in weight_updates.iter().enumerate() {
            self.reload_layer(i, updates)?;
        }
        Ok(start.elapsed())
    }

    /// Get a reference to the executor for a specific layer.
    pub fn executor(&self, layer_idx: usize) -> Result<&ANEExecutor> {
        self.layers
            .get(layer_idx)
            .map(|l| &l.executor)
            .ok_or_else(|| Error::InvalidParameter(format!("Invalid layer index: {}", layer_idx)))
    }

    /// Get a mutable reference to the executor for a specific layer.
    pub fn executor_mut(&mut self, layer_idx: usize) -> Result<&mut ANEExecutor> {
        self.layers
            .get_mut(layer_idx)
            .map(|l| &mut l.executor)
            .ok_or_else(|| Error::InvalidParameter(format!("Invalid layer index: {}", layer_idx)))
    }

    /// Number of compiled programs.
    pub fn num_programs(&self) -> usize {
        self.layers.len()
    }

    /// Current ANE compile count (process-wide).
    pub fn compile_count(&self) -> i32 {
        self.monitor.get_compile_count()
    }

    /// Number of compiles used by this DeltaCompiler (since construction).
    pub fn compiles_used(&self) -> i32 {
        self.compile_count() - self.initial_compile_count
    }

    /// Remaining compile budget.
    pub fn remaining_budget(&self) -> i32 {
        self.monitor.remaining()
    }

    /// Check if approaching compile limit.
    ///
    /// Returns `true` if in the warning zone (≥90% of budget used).
    /// Also prints a warning to stderr.
    pub fn check_budget_warning(&self) -> bool {
        let in_warning = self.monitor.is_warning_zone();
        if in_warning {
            let status = self.monitor.status();
            eprintln!(
                "[DeltaCompiler] WARNING: compile budget at {:.1}% ({} / {} used, {} remaining)",
                status.percent_used, status.used, status.limit, status.remaining
            );
        }
        in_warning
    }

    /// Check if compile budget is exhausted.
    pub fn is_budget_exhausted(&self) -> bool {
        self.monitor.is_exhausted()
    }

    /// Get full budget status.
    pub fn budget_status(&self) -> BudgetStatus {
        self.monitor.status()
    }

    /// Get weight file names for a layer.
    pub fn weight_names(&self, layer_idx: usize) -> Result<&[String]> {
        self.layers
            .get(layer_idx)
            .map(|l| l.weight_names.as_slice())
            .ok_or_else(|| Error::InvalidParameter(format!("Invalid layer index: {}", layer_idx)))
    }
}

impl Default for DeltaCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_compiler_new() {
        let dc = DeltaCompiler::new();
        assert_eq!(dc.num_programs(), 0);
        assert_eq!(dc.compiles_used(), 0);
    }

    #[test]
    fn test_delta_compiler_with_budget() {
        let dc = DeltaCompiler::with_budget(50);
        assert_eq!(dc.remaining_budget(), 50);
    }

    #[test]
    fn test_invalid_layer_index() {
        let dc = DeltaCompiler::new();
        assert!(dc.executor(0).is_err());
        let mut dc2 = DeltaCompiler::new();
        assert!(dc2.executor_mut(0).is_err());
        assert!(dc2.weight_names(0).is_err());
    }
}
