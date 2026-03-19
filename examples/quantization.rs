//! Quantization Example
//!
//! Demonstrates FP32 → INT8 quantization for improved performance.
//! Shows: quantization, dequantization, accuracy vs speed trade-off.

use rustane::{
    init,
    mil::WeightBlob,
    wrapper::{ANECompiler, ANETensor},
};
use std::time::Instant;

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

/// FP32 model weights
struct FP32Model {
    weights: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

/// INT8 quantized model
struct INT8Model {
    weights: Vec<i8>,
    scale: f32,
    zero_point: i8,
    input_size: usize,
    output_size: usize,
}

impl FP32Model {
    fn new(input_size: usize, output_size: usize) -> Self {
        let weights: Vec<f32> = (0..input_size * output_size)
            .map(|i| ((i as f32 * 0.001) % 2.0) - 1.0)
            .collect();

        Self {
            weights,
            input_size,
            output_size,
        }
    }

    /// Quantize FP32 weights to INT8
    fn quantize(&self) -> INT8Model {
        println!("Quantizing FP32 → INT8...");
        println!("  Input: {} × {} matrix", self.input_size, self.output_size);
        println!(
            "  Original range: {:.3} to {:.3}",
            self.weights.iter().cloned().fold(f32::INFINITY, f32::min),
            self.weights
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        );

        // Find min/max for scale calculation
        let min_val = self.weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = self
            .weights
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // Calculate scale and zero point
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i8;
        let zero_point = zero_point.clamp(-128, 127);

        println!("  Scale: {:.6}", scale);
        println!("  Zero point: {}", zero_point);

        // Quantize to INT8
        let quantized: Vec<i8> = self
            .weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round() as i32 + zero_point as i32;
                q.clamp(-128, 127) as i8
            })
            .collect();

        // Calculate quantization error
        let dequantized: Vec<f32> = quantized
            .iter()
            .map(|&q| (q as i32 - zero_point as i32) as f32 * scale)
            .collect();

        let mse: f32 = self
            .weights
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / self.weights.len() as f32;

        println!("  MSE: {:.6}", mse);
        println!("  ✓ Quantized {} weights", quantized.len());

        INT8Model {
            weights: quantized,
            scale,
            zero_point,
            input_size: self.input_size,
            output_size: self.output_size,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.output_size];

        for o in 0..self.output_size {
            for i in 0..self.input_size {
                output[o] += input[i] * self.weights[i * self.output_size + o];
            }
        }

        output
    }
}

impl INT8Model {
    /// Dequantize INT8 back to FP32
    fn dequantize(&self) -> Vec<f32> {
        self.weights
            .iter()
            .map(|&w| ((w as i32 - self.zero_point as i32) as f32 * self.scale))
            .collect()
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Dequantize weights
        let fp32_weights = self.dequantize();

        let mut output = vec![0.0f32; self.output_size];

        for o in 0..self.output_size {
            for i in 0..self.input_size {
                output[o] += input[i] * fp32_weights[i * self.output_size + o];
            }
        }

        output
    }
}

fn benchmark_fp32(
    input_size: usize,
    output_size: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FP32 Benchmark ({} → {})", input_size, output_size);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let model = FP32Model::new(input_size, output_size);
    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01) % 2.0 - 1.0)
        .collect();

    // Warmup
    for _ in 0..WARMUP {
        let _ = model.forward(&input);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = model.forward(&input);
    }
    let duration = start.elapsed();

    let avg_ms = duration.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let throughput = ITERATIONS as f64 / duration.as_secs_f64();

    println!("  Average time: {:.3}ms", avg_ms);
    println!("  Throughput: {:.1} ops/sec", throughput);

    Ok((avg_ms, throughput))
}

fn benchmark_int8(
    input_size: usize,
    output_size: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("INT8 Benchmark ({} → {})", input_size, output_size);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let fp32_model = FP32Model::new(input_size, output_size);
    let int8_model = fp32_model.quantize();

    let input: Vec<f32> = (0..input_size)
        .map(|i| (i as f32 * 0.01) % 2.0 - 1.0)
        .collect();

    // Warmup
    for _ in 0..WARMUP {
        let _ = int8_model.forward(&input);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = int8_model.forward(&input);
    }
    let duration = start.elapsed();

    let avg_ms = duration.as_secs_f64() * 1000.0 / ITERATIONS as f64;
    let throughput = ITERATIONS as f64 / duration.as_secs_f64();

    println!("  Average time: {:.3}ms", avg_ms);
    println!("  Throughput: {:.1} ops/sec", throughput);

    Ok((avg_ms, throughput))
}

fn compare_accuracy(fp32_model: &FP32Model, int8_model: &INT8Model, input: &[f32]) {
    let fp32_output = fp32_model.forward(input);
    let int8_output = int8_model.forward(input);

    // Calculate MAE (Mean Absolute Error)
    let mae: f32 = fp32_output
        .iter()
        .zip(int8_output.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / fp32_output.len() as f32;

    // Calculate cosine similarity
    let dot_product: f32 = fp32_output
        .iter()
        .zip(int8_output.iter())
        .map(|(a, b)| a * b)
        .sum();

    let norm_a: f32 = fp32_output.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = int8_output.iter().map(|x| x * x).sum::<f32>().sqrt();

    let cosine_sim = dot_product / (norm_a * norm_b);

    println!("\n  Accuracy Metrics:");
    println!("    MAE: {:.6}", mae);
    println!("    Cosine Similarity: {:.6}", cosine_sim);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Quantization Example");
    println!("================================\n");

    // Test configurations
    let configs = vec![(256, 512), (512, 1024), (1024, 1024)];

    for (input_size, output_size) in configs {
        println!("\n");
        println!("═══════════════════════════════════════════════════════");
        println!("Configuration: {} × {}", input_size, output_size);
        println!("═══════════════════════════════════════════════════════");

        // Create models
        let fp32_model = FP32Model::new(input_size, output_size);
        let int8_model = fp32_model.quantize();

        // Accuracy comparison
        let test_input: Vec<f32> = (0..input_size)
            .map(|i| (i as f32 * 0.01) % 2.0 - 1.0)
            .collect();

        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("ACCURACY COMPARISON");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        compare_accuracy(&fp32_model, &int8_model, &test_input);

        // Performance benchmarks
        let (fp32_time, fp32_ops) = benchmark_fp32(input_size, output_size)?;
        let (int8_time, int8_ops) = benchmark_int8(input_size, output_size)?;

        // Summary
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("PERFORMANCE SUMMARY");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!(
            "{:<12} {:>12} {:>12} {:>12}",
            "Precision", "Time (ms)", "Ops/sec", "Speedup"
        );
        println!("{}", "-".repeat(50));
        println!(
            "{:<12} {:>12.3} {:>12.1} {:>12}",
            "FP32", fp32_time, fp32_ops, "1.0x"
        );
        println!(
            "{:<12} {:>12.3} {:>12.1} {:>12.1}x",
            "INT8",
            int8_time,
            int8_ops,
            int8_ops / fp32_ops
        );
    }

    // Overall summary
    println!("\n\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("QUANTIZATION INSIGHTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n📊 Benefits of INT8 Quantization:");
    println!("  • 4x smaller model size (1 byte vs 4 bytes per weight)");
    println!("  • Faster computation (integer ops)");
    println!("  • Lower memory bandwidth requirements");
    println!("  • Better cache utilization");

    println!("\n⚠️  Trade-offs:");
    println!("  • Accuracy loss (quantization error)");
    println!("  • Limited dynamic range (256 levels vs 2^32)");
    println!("  • May not work well for all models");
    println!("  • Requires calibration for best results");

    println!("\n💡 When to use INT8:");
    println!("  • Edge devices with limited memory");
    println!("  • Real-time applications (faster inference)");
    println!("  • Large models (4x size reduction)");
    println!("  • Models tolerant to quantization error");

    println!("\n✅ Quantization example completed!");

    Ok(())
}
