//! Multi-layer neural network example
//!
//! Demonstrates ANE inference with a small multi-layer network.

use rustane::{
    init,
    mil::{MILBuilder, WeightBlob},
    wrapper::{ANECompiler, ANETensor},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Multi-Layer Network Example");
    println!("======================================\n");

    // Check ANE availability
    let avail = rustane::HardwareAvailability::check();
    println!("Platform check: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE is not available on this platform");
        return Ok(());
    }
    println!();

    // Initialize ANE runtime
    println!("Initializing ANE runtime...");
    init()?;
    println!("✓ ANE runtime initialized\n");

    // Define network architecture
    let input_size = 784; // 28x28 image (flattened)
    let hidden1_size = 256;
    let hidden2_size = 128;
    let output_size = 10; // 10 classes

    println!("Network architecture:");
    println!("  Input:  {} neurons", input_size);
    println!("  Hidden1: {} neurons (ReLU)", hidden1_size);
    println!("  Hidden2: {} neurons (ReLU)", hidden2_size);
    println!("  Output: {} neurons", output_size);
    println!();

    // Layer 1: Input → Hidden1 (Linear + ReLU)
    println!("Layer 1: Input → Hidden1");
    let (fc1_mil, fc1_weights) = create_linear_layer("fc1", input_size, hidden1_size, "input")?;
    println!("✓ Layer 1 MIL created\n");

    // Layer 2: Hidden1 → Hidden2 (Linear + ReLU)
    println!("Layer 2: Hidden1 → Hidden2");
    let (fc2_mil, fc2_weights) = create_linear_layer("fc2", hidden1_size, hidden2_size, "fc1")?;
    println!("✓ Layer 2 MIL created\n");

    // Layer 3: Hidden2 → Output (Linear)
    println!("Layer 3: Hidden2 → Output");
    let (fc3_mil, fc3_weights) = create_linear_layer("fc3", hidden2_size, output_size, "fc2")?;
    println!("✓ Layer 3 MIL created\n");

    // For demonstration, we'll execute just the first layer
    println!("Executing Layer 1 (Input → Hidden1)...");
    println!("  Note: Full multi-layer execution requires multiple kernels\n");

    // Compile first layer
    let mut compiler = ANECompiler::new();
    let mut executor = compiler.compile_single(
        &fc1_mil,
        Some(fc1_weights.as_bytes()),
        &[input_size * 4],
        &[hidden1_size * 4],
    )?;
    println!("✓ Layer 1 compiled");

    // Prepare input (random image-like data)
    println!("\nPreparing input tensor...");
    let input_data: Vec<f32> = (0..input_size)
        .map(|i| (i as f32) / 255.0) // Normalize to [0, 1]
        .collect();

    let input_tensor = ANETensor::from_fp32(input_data, vec![1, input_size])?;
    println!("✓ Input tensor created");

    // Execute
    println!("\nExecuting layer 1...");
    executor.write_input(0, input_tensor.as_bytes())?;
    executor.eval()?;
    println!("✓ Execution complete");

    // Read output
    println!("\nReading layer 1 output...");
    let mut output_buf = vec![0u8; hidden1_size * 4];
    executor.read_output(0, &mut output_buf)?;

    // Convert to FP32
    let output_data: Vec<f32> = output_buf
        .chunks_exact(4)
        .map(|chunk| {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(bytes)
        })
        .collect();

    println!("✓ Output read");
    println!("  Output shape: [1 × {}]", hidden1_size);
    println!(
        "  Output range: [{:.4}, {:.4}]",
        output_data.iter().cloned().fold(f32::INFINITY, f32::min),
        output_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    // Apply ReLU (manually for demonstration)
    let relu_output: Vec<f32> = output_data.iter().map(|&x| x.max(0.0)).collect();
    println!(
        "  After ReLU: [{:.4}, {:.4}]",
        relu_output.iter().cloned().fold(f32::INFINITY, f32::min),
        relu_output
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    );

    println!("\n📊 Multi-Layer Network Info");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("This example demonstrates a 3-layer network:");
    println!(
        "  Layer 1: {} → {} (Linear + ReLU)",
        input_size, hidden1_size
    );
    println!(
        "  Layer 2: {} → {} (Linear + ReLU)",
        hidden1_size, hidden2_size
    );
    println!("  Layer 3: {} → {} (Linear)", hidden2_size, output_size);
    println!();
    println!("For production use:");
    println!("  • Compile each layer separately");
    println!("  • Chain inputs/outputs between layers");
    println!("  • Use multi-weight compilation for efficiency");
    println!("  • Consider quantization for faster inference");

    println!("\n✅ Multi-layer example completed successfully!");
    Ok(())
}

fn create_linear_layer(
    name: &str,
    input_size: usize,
    output_size: usize,
    input_name: &str,
) -> Result<(String, WeightBlob), Box<dyn std::error::Error>> {
    // Create MIL program for linear layer
    let mil = MILBuilder::new()
        .add_linear(name, input_name, "weight", output_size)
        .add_relu(&format!("{}_relu", name), name)
        .add_output(&format!("{}_relu", name), "fp32", &[1, output_size])
        .build();

    // Create identity-like weights
    let mut weight_data = vec![0.0f32; input_size * output_size];
    for i in 0..input_size.min(output_size) {
        weight_data[i * output_size + i] = 1.0; // Identity diagonal
    }

    // Add some random variation
    for i in 0..output_size.min(10) {
        for j in 0..input_size.min(10) {
            weight_data[i * output_size + j] += 0.01 * (j as f32) / 10.0;
        }
    }

    // Create weight blob
    let blob = WeightBlob::from_fp32(&weight_data, input_size as i32, output_size as i32)?;

    Ok((mil, blob))
}
