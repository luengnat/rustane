//! Delta Compilation Example
//!
//! Demonstrates Orion-style delta compilation: updating weights without recompilation.
//!
//! ## How it works
//!
//! 1. Compile a model once at startup (expensive: ~4,200ms for 60 programs)
//! 2. For weight updates: unload → patch weights on disk → reload (cheap: ~494ms)
//! 3. This is 8.5x faster than full recompilation
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example delta_compilation --release
//! ```

use rustane::ane::WeightBlob;
use rustane::wrapper::{ANECompiler, ANERuntime};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rustane Delta Compilation Demo ===\n");
    println!("Orion's optimization: 8.5x faster weight updates\n");

    // Initialize ANE runtime
    ANERuntime::init()?;
    println!("✅ ANE runtime initialized\n");

    // Define a simple linear layer MIL program
    // The weights are loaded from @model_path/weights/linear.bin
    let mil_program = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 64, 1, 64]> x) {
        tensor<fp16, [64, 64, 1, 1]> W = const()[name=string("W"),
            val=tensor<fp16, [64, 64, 1, 1]>(
                BLOBFILE(path=string("@model_path/weights/linear.bin"), offset=uint64(64))
            )];
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [1, 64, 1, 64]> out = conv(
            dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x
        )[name=string("out")];
    } -> (out);
}"#;

    // Create initial weights (FP32 -> FP16 blob)
    let initial_weights: Vec<f32> = (0..64 * 64)
        .map(|i| ((i % 10) as f32 - 5.0) / 10.0)
        .collect();
    let weight_blob = WeightBlob::from_f32(&initial_weights, 64, 64)?;

    // Compile the kernel with initial weights
    println!("=== Step 1: Initial Compilation ===");
    let start = Instant::now();
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_multi(
        mil_program,
        &["@model_path/weights/linear.bin"],
        &[weight_blob.as_bytes()],
        &[weight_blob.as_bytes().len()],
        &[64 * 64 * 2], // Input: 64x64 FP16
        &[64 * 64 * 2], // Output: 64x64 FP16
    )?;
    let compile_time = start.elapsed();
    println!("✅ Kernel compiled in {:?}\n", compile_time);

    // Simulate training loop with weight updates
    println!("=== Step 2: Training Loop (Weight Updates) ===");

    let mut total_reload_time = std::time::Duration::ZERO;

    for step in 0..5 {
        // Generate new weights (simulating gradient update)
        let new_weights: Vec<f32> = (0..64 * 64)
            .map(|i| {
                let noise = (step as f32 * 0.01) + ((i % 5) as f32 - 2.5) / 100.0;
                initial_weights[i] + noise
            })
            .collect();
        let new_blob = WeightBlob::from_f32(&new_weights, 64, 64)?;

        // Delta compilation: reload weights without recompiling
        let start = Instant::now();
        executor.reload_weights(&[("linear.bin", new_blob.as_bytes())])?;
        let reload_time = start.elapsed();
        total_reload_time += reload_time;

        println!("Step {}: Weight reload took {:?}", step + 1, reload_time);
    }

    let avg_reload = total_reload_time / 5;

    println!("\n=== Performance Summary ===");
    println!("Initial compilation:     {:?}", compile_time);
    println!("Average reload time:     {:?}", avg_reload);
    println!("Estimated speedup:       ~8.5x (vs full recompilation)");
    println!(
        "
Key Benefits:"
    );
    println!("  • Avoids the ~119 ANE compile limit per process");
    println!("  • Enables fast weight updates during SGD/Adam");
    println!("  • Reduces ANE memory leaks from repeated compiles");
    println!("  • Critical for training with frequent weight updates");

    Ok(())
}
