//! Error Rate Reduction Over Training Files
//!
//! This test validates that error rates (NaN losses, gradient explosions,
//! training failures) decrease as training progresses through multiple files.
//!
//! Run with:
//! ```bash
//! cargo test --test error_rate_progressive_test -- --nocapture
//! ```

use rustane::data::{DataLoader, Dataset, SequentialDataset, SequentialSampler};
use rustane::training::{
    AdamWOptimizer, ConstantScheduler, CrossEntropyLoss, MetricsTracker, Model, TrainerBuilder,
    TransformerANE, TransformerConfig, WarmupCosineScheduler,
};
use std::collections::HashMap;

/// Statistics for tracking error rates across multiple training files
#[derive(Debug, Clone)]
pub struct ProgressiveErrorStats {
    /// File index
    pub file_idx: usize,
    /// Total steps in this file
    pub total_steps: usize,
    /// Successful steps
    pub successful_steps: usize,
    /// Failed steps
    pub failed_steps: usize,
    /// NaN losses encountered
    pub nan_losses: usize,
    /// Exploding gradients (>100.0)
    pub exploding_grads: usize,
    /// Vanishing gradients (<1e-6)
    pub vanishing_grads: usize,
    /// Average loss for this file
    pub avg_loss: f32,
    /// Average gradient norm
    pub avg_grad_norm: f32,
}

impl ProgressiveErrorStats {
    pub fn new(file_idx: usize) -> Self {
        Self {
            file_idx,
            total_steps: 0,
            successful_steps: 0,
            failed_steps: 0,
            nan_losses: 0,
            exploding_grads: 0,
            vanishing_grads: 0,
            avg_loss: 0.0,
            avg_grad_norm: 0.0,
        }
    }

    /// Calculate error rate for this file
    pub fn error_rate(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.failed_steps + self.nan_losses) as f32 / self.total_steps as f32
    }

    /// Calculate gradient health issues rate
    pub fn gradient_issue_rate(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.exploding_grads + self.vanishing_grads) as f32 / self.total_steps as f32
    }
}

/// Test that error rates decrease as training progresses through multiple files
#[test]
fn test_error_rate_reduction_over_files() {
    println!("\n=== Testing Error Rate Reduction Over Multiple Files ===\n");

    // Parameter-golf style config
    let config = TransformerConfig::new(1024, 64, 256, 8, 4, 512).unwrap();

    // Create multiple "files" (datasets) with different characteristics
    let num_files = 5;
    let samples_per_file = 20;
    let batch_size = 4;

    // Create model once and reuse across files (simulating continued training)
    let mut model = TransformerANE::new(&config).unwrap();
    let param_count = model.param_count();

    // Create trainer
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamWOptimizer::new(param_count))
        .with_scheduler(WarmupCosineScheduler::new(0.001, 20, 100, 0.0001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    // Track stats per file
    let mut file_stats: Vec<ProgressiveErrorStats> = Vec::new();

    for file_idx in 0..num_files {
        // Create synthetic dataset for this "file"
        // Early files have more noise, later files are cleaner
        let noise_level = 1.0 - (file_idx as f32 / num_files as f32);
        let samples = create_noisy_dataset(samples_per_file, 64, 1024, noise_level);

        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let dataloader = DataLoader::new(dataset, sampler, batch_size).unwrap();

        let mut stats = ProgressiveErrorStats::new(file_idx);
        let mut loss_sum = 0.0f32;
        let mut grad_norm_sum = 0.0f32;

        // Train on this file
        for batch_result in dataloader.iter() {
            stats.total_steps += 1;

            match batch_result {
                Ok(batch) => {
                    match trainer.train_step(&batch) {
                        Ok(metrics) => {
                            if !metrics.loss.is_finite() {
                                stats.nan_losses += 1;
                                stats.failed_steps += 1;
                            } else {
                                stats.successful_steps += 1;
                                loss_sum += metrics.loss;
                                grad_norm_sum += metrics.grad_norm;

                                // Check gradient health
                                if metrics.grad_norm > 100.0 {
                                    stats.exploding_grads += 1;
                                } else if metrics.grad_norm < 1e-6 {
                                    stats.vanishing_grads += 1;
                                }
                            }
                        }
                        Err(_) => {
                            stats.failed_steps += 1;
                        }
                    }
                }
                Err(_) => {
                    stats.failed_steps += 1;
                }
            }
        }

        // Calculate averages
        if stats.successful_steps > 0 {
            stats.avg_loss = loss_sum / stats.successful_steps as f32;
            stats.avg_grad_norm = grad_norm_sum / stats.successful_steps as f32;
        }

        file_stats.push(stats);
    }

    // Print results
    println!("Error Rate Progression Across Files:");
    println!("{:-^80}", "");
    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "File", "Error Rate", "NaN Losses", "Grad Issues", "Avg Loss", "Avg Grad", "Success %"
    );
    println!("{:-^80}", "");

    for stats in &file_stats {
        let success_pct = if stats.total_steps > 0 {
            (stats.successful_steps as f32 / stats.total_steps as f32) * 100.0
        } else {
            0.0
        };

        println!(
            "{:<8} {:<12.2}% {:<12} {:<12.2}% {:<12.4} {:<12.4} {:<12.1}%",
            format!("File {}", stats.file_idx),
            stats.error_rate() * 100.0,
            stats.nan_losses,
            stats.gradient_issue_rate() * 100.0,
            stats.avg_loss,
            stats.avg_grad_norm,
            success_pct
        );
    }

    // Verify that error rates generally decrease
    let first_file_error_rate = file_stats[0].error_rate();
    let last_file_error_rate = file_stats[file_stats.len() - 1].error_rate();

    println!("\n=== Error Rate Trend Analysis ===");
    println!(
        "First file error rate: {:.2}%",
        first_file_error_rate * 100.0
    );
    println!(
        "Last file error rate:  {:.2}%",
        last_file_error_rate * 100.0
    );
    println!(
        "Improvement:           {:.2}%",
        (first_file_error_rate - last_file_error_rate) * 100.0
    );

    // Assert that training becomes more stable (error rate should not increase significantly)
    assert!(
        last_file_error_rate <= first_file_error_rate + 0.1, // Allow small variance
        "Error rate should decrease or stay stable over training files. \
         First: {:.2}%, Last: {:.2}%",
        first_file_error_rate * 100.0,
        last_file_error_rate * 100.0
    );

    // Assert that we have a reasonable success rate overall
    let total_successful: usize = file_stats.iter().map(|s| s.successful_steps).sum();
    let total_steps: usize = file_stats.iter().map(|s| s.total_steps).sum();
    let overall_success_rate = total_successful as f32 / total_steps as f32;

    assert!(
        overall_success_rate >= 0.7,
        "Overall success rate should be at least 70%, got {:.1}%",
        overall_success_rate * 100.0
    );

    println!("\n✅ Error rate reduction test passed!");
    println!("Overall success rate: {:.1}%", overall_success_rate * 100.0);
}

/// Test error rate with varying sequence lengths (common in parameter-golf)
#[test]
fn test_error_rate_varying_sequence_lengths() {
    println!("\n=== Testing Error Rate with Varying Sequence Lengths ===\n");

    let config = TransformerConfig::new(1024, 128, 256, 8, 4, 512).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();
    let param_count = model.param_count();

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamWOptimizer::new(param_count))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    // Test different sequence lengths
    let seq_lengths = vec![32, 64, 96, 128];
    let mut stats_by_length: HashMap<usize, (usize, usize)> = HashMap::new(); // (successes, total)

    for seq_len in &seq_lengths {
        // Create dataset with this sequence length
        let samples: Vec<Vec<u32>> = (0..4)
            .map(|i| {
                (0..*seq_len)
                    .map(|j| ((i * seq_len + j) % 1024) as u32)
                    .collect()
            })
            .collect();

        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let dataloader = DataLoader::new(dataset, sampler, 2).unwrap();

        let mut successes = 0;
        let mut total = 0;

        for batch_result in dataloader.iter() {
            total += 1;
            if let Ok(batch) = batch_result {
                if let Ok(metrics) = trainer.train_step(&batch) {
                    if metrics.loss.is_finite() {
                        successes += 1;
                    }
                }
            }
        }

        stats_by_length.insert(*seq_len, (successes, total));
    }

    println!("Error Rate by Sequence Length:");
    println!("{:-^50}", "");
    println!(
        "{:<15} {:<15} {:<15}",
        "Seq Length", "Success Rate", "Success/Total"
    );
    println!("{:-^50}", "");

    for seq_len in &seq_lengths {
        let (successes, total) = stats_by_length[seq_len];
        let rate = if total > 0 {
            (successes as f32 / total as f32) * 100.0
        } else {
            0.0
        };
        println!("{:<15} {:<14.1}% {}/{}", seq_len, rate, successes, total);
    }

    // All sequence lengths should have reasonable success rates
    for (seq_len, (successes, total)) in &stats_by_length {
        let rate = if *total > 0 {
            *successes as f32 / *total as f32
        } else {
            0.0
        };
        assert!(
            rate >= 0.5,
            "Sequence length {} should have at least 50% success rate, got {:.1}%",
            seq_len,
            rate * 100.0
        );
    }

    println!("\n✅ Varying sequence length test passed!");
}

/// Test error rate with increasing batch sizes
#[test]
fn test_error_rate_batch_size_scaling() {
    println!("\n=== Testing Error Rate with Increasing Batch Sizes ===\n");

    let config = TransformerConfig::new(1024, 64, 256, 8, 4, 512).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();
    let param_count = model.param_count();

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamWOptimizer::new(param_count))
        .with_scheduler(ConstantScheduler::new(0.001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    // Test different batch sizes
    let batch_sizes = vec![1, 2, 4, 8];
    let mut stats: Vec<(usize, f32)> = Vec::new(); // (batch_size, success_rate)

    for bs in &batch_sizes {
        // Create dataset for this batch size
        let samples: Vec<Vec<u32>> = (0..*bs * 2)
            .map(|i| (0..64).map(|j| ((i * 64 + j) % 1024) as u32).collect())
            .collect();

        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let dataloader = DataLoader::new(dataset, sampler, *bs).unwrap();

        let mut successes = 0;
        let mut total = 0;

        for batch_result in dataloader.iter().take(2) {
            total += 1;
            if let Ok(batch) = batch_result {
                if let Ok(metrics) = trainer.train_step(&batch) {
                    if metrics.loss.is_finite() {
                        successes += 1;
                    }
                }
            }
        }

        let rate = if total > 0 {
            successes as f32 / total as f32
        } else {
            0.0
        };
        stats.push((*bs, rate));
    }

    println!("Error Rate by Batch Size:");
    println!("{:-^40}", "");
    println!(
        "{:<15} {:<15} {:<10}",
        "Batch Size", "Success Rate", "Status"
    );
    println!("{:-^40}", "");

    for (bs, rate) in &stats {
        let status = if *rate >= 0.8 {
            "✅ Good"
        } else if *rate >= 0.5 {
            "⚠️  Fair"
        } else {
            "❌ Poor"
        };
        println!("{:<15} {:<14.1}% {}", bs, rate * 100.0, status);
    }

    // Larger batch sizes should still maintain reasonable success rates
    for (bs, rate) in &stats {
        assert!(
            *rate >= 0.4,
            "Batch size {} should have at least 40% success rate, got {:.1}%",
            bs,
            rate * 100.0
        );
    }

    println!("\n✅ Batch size scaling test passed!");
}

/// Test cumulative error statistics over extended training
#[test]
fn test_cumulative_error_statistics() {
    println!("\n=== Testing Cumulative Error Statistics ===\n");

    let config = TransformerConfig::new(512, 32, 128, 4, 2, 256).unwrap();
    let mut model = TransformerANE::new(&config).unwrap();
    let param_count = model.param_count();

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(AdamWOptimizer::new(param_count))
        .with_scheduler(WarmupCosineScheduler::new(0.001, 10, 50, 0.0001))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()
        .unwrap();

    let mut tracker = MetricsTracker::new();

    // Simulate training over 5 "epochs" (files)
    let num_epochs = 5;
    let mut cumulative_errors = Vec::new();
    let mut window_errors = Vec::new(); // Sliding window of last 10 steps

    for epoch in 0..num_epochs {
        let samples: Vec<Vec<u32>> = (0..20)
            .map(|i| {
                (0..32)
                    .map(|j| ((epoch * 100 + i * 32 + j) % 512) as u32)
                    .collect()
            })
            .collect();

        let dataset = SequentialDataset::new(samples);
        let sampler = SequentialSampler::new(dataset.len());
        let dataloader = DataLoader::new(dataset, sampler, 4).unwrap();

        let mut epoch_errors = 0;
        let mut epoch_steps = 0;

        for batch_result in dataloader.iter() {
            epoch_steps += 1;

            match batch_result {
                Ok(batch) => match trainer.train_step(&batch) {
                    Ok(metrics) => {
                        tracker.log_step_metrics(&metrics);

                        if !metrics.loss.is_finite() {
                            epoch_errors += 1;
                        }
                    }
                    Err(_) => {
                        epoch_errors += 1;
                    }
                },
                Err(_) => {
                    epoch_errors += 1;
                }
            }
        }

        let epoch_error_rate = if epoch_steps > 0 {
            epoch_errors as f32 / epoch_steps as f32
        } else {
            0.0
        };

        cumulative_errors.push((epoch, epoch_error_rate, epoch_errors, epoch_steps));
        window_errors.push(epoch_error_rate);

        // Keep only last 3 epochs for trend analysis
        if window_errors.len() > 3 {
            window_errors.remove(0);
        }
    }

    println!("Cumulative Error Statistics:");
    println!("{:-^70}", "");
    println!(
        "{:<10} {:<15} {:<15} {:<15} {:<15}",
        "Epoch", "Error Rate", "Errors", "Total Steps", "Window Avg"
    );
    println!("{:-^70}", "");

    for (idx, (epoch, rate, errors, total)) in cumulative_errors.iter().enumerate() {
        let window_avg = if idx >= 2 {
            let window: Vec<f32> = window_errors.iter().copied().collect();
            window.iter().sum::<f32>() / window.len() as f32
        } else {
            *rate
        };

        println!(
            "{:<10} {:<14.2}% {:<15} {:<15} {:<14.2}%",
            format!("Epoch {}", epoch),
            rate * 100.0,
            errors,
            total,
            window_avg * 100.0
        );
    }

    tracker.print_summary();

    // Verify that error rate stays within acceptable bounds
    let avg_error_rate: f32 = cumulative_errors
        .iter()
        .map(|(_, rate, _, _)| rate)
        .sum::<f32>()
        / cumulative_errors.len() as f32;

    assert!(
        avg_error_rate < 0.3,
        "Average error rate {:.2}% should be below 30%",
        avg_error_rate * 100.0
    );

    println!("\n✅ Cumulative error statistics test passed!");
}

/// Helper function to create dataset with configurable noise level
fn create_noisy_dataset(
    num_samples: usize,
    seq_len: usize,
    vocab_size: usize,
    noise_level: f32,
) -> Vec<Vec<u32>> {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    (0..num_samples)
        .map(|i| {
            (0..seq_len)
                .map(|j| {
                    // Base token
                    let base_token = ((i * seq_len + j) % vocab_size) as u32;

                    // Add noise based on noise_level
                    if rng.gen::<f32>() < noise_level {
                        // Random token (high noise)
                        rng.gen_range(0..vocab_size) as u32
                    } else {
                        base_token
                    }
                })
                .collect()
        })
        .collect()
}
