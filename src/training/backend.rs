//! Training backend implementations.
//!
//! This module provides a small backend abstraction for the training loop.
//! The current implementation is CPU-backed and owns the optimizer split
//! logic used by the Parameter Golf example. Future ANE/GPU backends can
//! implement the same trait without changing the training loop structure.

use crate::error::Result;
use crate::training::{AdamOptimizer, Model, Optimizer, ParameterGroupKind, TransformerANE};

/// Backend interface for applying parameter updates.
pub trait TrainingBackend: Send {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Apply an update to the model parameters.
    fn step(&mut self, model: &mut TransformerANE, grads: &[f32], lr_scale: f32) -> Result<()>;
}

/// CPU training backend with group-specific optimizers.
pub struct CpuTrainingBackend {
    groups: Vec<BackendGroup>,
    name: String,
}

impl CpuTrainingBackend {
    /// Create a CPU backend with the same optimizer split used by train_gpt.
    pub fn new(
        model: &TransformerANE,
        embed_lr: f32,
        head_lr: f32,
        matrix_lr: f32,
        scalar_lr: f32,
        matrix_opt: &str,
        muon_momentum: f32,
        muon_backend_steps: usize,
        muon_nesterov: bool,
    ) -> Self {
        let matrix_opt = matrix_opt.to_ascii_lowercase();
        let groups = model
            .parameter_groups()
            .into_iter()
            .map(|group| {
                let base_lr = match group.kind {
                    ParameterGroupKind::Embedding => embed_lr,
                    ParameterGroupKind::Head => head_lr,
                    ParameterGroupKind::Matrix => matrix_lr,
                    ParameterGroupKind::Scalar => scalar_lr,
                };
                let optimizer = match group.kind {
                    ParameterGroupKind::Matrix if matrix_opt == "muon" => {
                        BackendOptimizer::Muon(MuonOptimizer::new(
                            group.rows,
                            group.cols,
                            muon_momentum,
                            muon_backend_steps,
                            muon_nesterov,
                        ))
                    }
                    _ => BackendOptimizer::Adam(AdamOptimizer::new(
                        group.range.end - group.range.start,
                    )),
                };

                BackendGroup {
                    range: group.range,
                    optimizer,
                    base_lr,
                }
            })
            .collect();

        Self {
            groups,
            name: "cpu".to_string(),
        }
    }
}

impl TrainingBackend for CpuTrainingBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn step(&mut self, model: &mut TransformerANE, grads: &[f32], lr_scale: f32) -> Result<()> {
        let params = model.parameters();
        for group in &mut self.groups {
            let lr = group.base_lr * lr_scale;
            let range = group.range.clone();
            group
                .optimizer
                .step(&grads[range.clone()], &mut params[range], lr)?;
        }
        Ok(())
    }
}

struct BackendGroup {
    range: std::ops::Range<usize>,
    optimizer: BackendOptimizer,
    base_lr: f32,
}

enum BackendOptimizer {
    Adam(AdamOptimizer),
    Muon(MuonOptimizer),
}

impl BackendOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        match self {
            BackendOptimizer::Adam(opt) => opt.step(grads, params, lr),
            BackendOptimizer::Muon(opt) => opt.step(grads, params, lr),
        }
    }
}

struct MuonOptimizer {
    momentum_buffer: Vec<f32>,
    rows: usize,
    cols: usize,
    momentum: f32,
    backend_steps: usize,
    nesterov: bool,
}

impl MuonOptimizer {
    fn new(rows: usize, cols: usize, momentum: f32, backend_steps: usize, nesterov: bool) -> Self {
        Self {
            momentum_buffer: vec![0.0; rows * cols],
            rows,
            cols,
            momentum,
            backend_steps,
            nesterov,
        }
    }

    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        use crate::Error;

        if grads.len() != params.len() || grads.len() != self.momentum_buffer.len() {
            return Err(Error::Other(format!(
                "muon optimizer state mismatch: grads={}, params={}, buffer={}",
                grads.len(),
                params.len(),
                self.momentum_buffer.len()
            )));
        }
        if self.rows == 0 || self.cols == 0 {
            return Err(Error::Other(
                "muon optimizer received empty matrix shape".to_string(),
            ));
        }

        let mut update = vec![0.0f32; grads.len()];
        for i in 0..grads.len() {
            self.momentum_buffer[i] = self.momentum * self.momentum_buffer[i] + grads[i];
            update[i] = if self.nesterov {
                grads[i] + self.momentum * self.momentum_buffer[i]
            } else {
                self.momentum_buffer[i]
            };
        }

        let mut update =
            zero_power_via_newton_schulz5(&update, self.rows, self.cols, self.backend_steps);
        let scale =
            (self.rows.max(self.cols) as f32 / self.rows.min(self.cols).max(1) as f32).sqrt();
        for value in &mut update {
            *value *= scale;
        }
        for (param, delta) in params.iter_mut().zip(update.iter()) {
            *param -= lr * delta;
        }
        Ok(())
    }
}

fn zero_power_via_newton_schulz5(
    gradient: &[f32],
    rows: usize,
    cols: usize,
    steps: usize,
) -> Vec<f32> {
    let mut mat = gradient.to_vec();
    let norm = mat.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-7);
    for value in &mut mat {
        *value /= norm;
    }
    let transposed = rows > cols;
    if transposed {
        mat = transpose_matrix(&mat, rows, cols);
    }
    let (cur_rows, cur_cols) = if transposed {
        (cols, rows)
    } else {
        (rows, cols)
    };
    for _ in 0..steps {
        let xt = transpose_matrix(&mat, cur_rows, cur_cols);
        let a = matmul(&mat, &xt, cur_rows, cur_cols, cur_rows);
        let a2 = matmul(&a, &a, cur_rows, cur_rows, cur_rows);
        let mut b = a.clone();
        for i in 0..b.len() {
            b[i] = -4.7750 * a[i] + 2.0315 * a2[i];
        }
        let bx = matmul(&b, &mat, cur_rows, cur_rows, cur_cols);
        let mut next = mat.clone();
        for i in 0..next.len() {
            next[i] = 3.4445 * mat[i] + bx[i];
        }
        mat = next;
    }

    if transposed {
        transpose_matrix(&mat, cur_rows, cur_cols)
    } else {
        mat
    }
}

fn transpose_matrix(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; matrix.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = matrix[r * cols + c];
        }
    }
    out
}

fn matmul(a: &[f32], b: &[f32], a_rows: usize, a_cols: usize, b_cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; a_rows * b_cols];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let aval = a[i * a_cols + k];
            for j in 0..b_cols {
                out[i * b_cols + j] += aval * b[k * b_cols + j];
            }
        }
    }
    out
}
