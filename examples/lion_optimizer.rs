//! Example: Lion Optimizer with Sign-Based Updates
//!
//! Demonstrates the Lion optimizer, which uses only the sign of gradients
//! rather than their magnitude.
//!
//! Lion advantages:
//! - Less memory usage (single momentum vector vs Adam's two)
//! - Better generalization in many cases
//! - More stable training for large models
//!
//! Reference: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)

use rustane::training::{AdamOptimizer, LionOptimizer, Optimizer};

fn main() {
    println!("Rustane Lion Optimizer Example");
    println!("==============================\n");

    // Example 1: Basic usage comparison
    println!("Example 1: Lion vs Adam Update Comparison");
    println!("------------------------------------------");
    compare_optimizers();
    println!();

    // Example 2: Sign-based behavior
    println!("Example 2: Sign-Based Gradient Updates");
    println!("--------------------------------------");
    demonstrate_sign_behavior();
    println!();

    // Example 3: Memory efficiency
    println!("Example 3: Memory Efficiency");
    println!("---------------------------");
    demonstrate_memory_efficiency();
    println!();

    // Example 4: Training scenario
    println!("Example 4: Simulated Training Steps");
    println!("------------------------------------");
    simulate_training();
    println!();

    println!("✓ Example completed!");
    println!("\nKey takeaways:");
    println!("  • Lion uses only gradient sign, not magnitude");
    println!("  • Requires less memory than Adam (1 vs 2 momentum vectors)");
    println!("  • Often generalizes better than Adam");
    println!("  • Recommended: β₁=0.9, wd=0.01, lr=3e-4 for transformers");
}

/// Compare Lion and Adam optimizer updates
fn compare_optimizers() {
    let mut lion = LionOptimizer::new(3);
    let mut adam = AdamOptimizer::new(3);

    let mut params_lion = vec![1.0, 1.0, 1.0];
    let mut params_adam = vec![1.0, 1.0, 1.0];
    let grads = vec![0.1, 0.1, 0.1];
    let lr = 0.001;

    println!("Initial parameters: [1.0, 1.0, 1.0]");
    println!("Gradients:           [0.1, 0.1, 0.1]");
    println!("Learning rate:       {}", lr);
    println!();

    // First step
    lion.step(&grads, &mut params_lion, lr).unwrap();
    adam.step(&grads, &mut params_adam, lr).unwrap();

    println!("After step 1:");
    println!("  Lion: params = [{:.6}, {:.6}, {:.6}]", params_lion[0], params_lion[1], params_lion[2]);
    println!(
        "  Adam: params = [{:.6}, {:.6}, {:.6}]",
        params_adam[0], params_adam[1], params_adam[2]
    );

    let diff: f32 = params_lion
        .iter()
        .zip(params_adam.iter())
        .map(|(l, a)| (l - a).abs())
        .sum::<f32>()
        / 3.0;
    println!("  Average difference: {:.6}", diff);
    println!();
    println!("→ Lion updates are more aggressive (sign-based, not magnitude-based)");
}

/// Demonstrate sign-based behavior
fn demonstrate_sign_behavior() {
    let mut lion = LionOptimizer::with_hyperparams(5, 0.9, 0.0); // No weight decay
    let mut params = vec![0.0; 5];
    let lr = 0.01;

    println!("Demonstrating sign-based updates:");
    println!("Initial parameters: [0.0, 0.0, 0.0, 0.0, 0.0]");
    println!();

    // Positive gradient
    let pos_grads = vec![0.001, 0.1, 1.0, 10.0, 100.0]; // Different magnitudes
    lion.step(&pos_grads, &mut params, lr).unwrap();

    println!(
        "After positive gradients [0.001, 0.1, 1.0, 10.0, 100.0]:\n  params = [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
        params[0], params[1], params[2], params[3], params[4]
    );
    println!();
    println!("→ All parameters decreased by same amount (sign(-1) = -1)");
    println!("  Magnitude doesn't matter, only sign!");
}

/// Demonstrate memory efficiency
fn demonstrate_memory_efficiency() {
    println!("Memory comparison (per parameter):");
    println!();
    println!("  Adam:");
    println!("    • m: 4 bytes (f32)");
    println!("    • v: 4 bytes (f32)");
    println!("    • Total: 8 bytes/parameter");
    println!();
    println!("  Lion:");
    println!("    • m: 4 bytes (f32)");
    println!("    • Total: 4 bytes/parameter");
    println!();
    println!("  Savings: 50% less memory for optimizer state");
    println!();

    // Example with realistic model size
    let param_count = 1_000_000; // 1M parameters (small transformer)
    let adam_memory = param_count * 8; // 2 f32 vectors
    let lion_memory = param_count * 4; // 1 f32 vector
    let saved = adam_memory - lion_memory;

    println!("For a 1M parameter model:");
    println!("  Adam:  {:.2} MB", adam_memory as f64 / 1024.0 / 1024.0);
    println!("  Lion:  {:.2} MB", lion_memory as f64 / 1024.0 / 1024.0);
    println!("  Saved: {:.2} MB", saved as f64 / 1024.0 / 1024.0);
}

/// Simulate a simplified training scenario
fn simulate_training() {
    let mut lion = LionOptimizer::new(4);
    let mut adam = AdamOptimizer::new(4);

    let mut params_lion = vec![0.5, 0.5, 0.5, 0.5];
    let mut params_adam = vec![0.5, 0.5, 0.5, 0.5];

    println!("Training simulation (20 steps):");
    println!("Initial parameters: [0.5, 0.5, 0.5, 0.5]");
    println!();

    for step in 1..=20 {
        // Simulated gradients (decreasing over time as model converges)
        let scale = 1.0 / (step as f32);
        let grads: Vec<f32> = (0..4).map(|i| (i as f32 + 1.0) * 0.1 * scale).collect();

        lion.step(&grads, &mut params_lion, 0.001).unwrap();
        adam.step(&grads, &mut params_adam, 0.001).unwrap();

        let avg_lion: f32 = params_lion.iter().sum::<f32>() / 4.0;
        let avg_adam: f32 = params_adam.iter().sum::<f32>() / 4.0;

        if step % 5 == 0 || step == 20 {
            println!(
                "Step {:2}: Lion avg={:.6}, Adam avg={:.6}, Δ={:.6}",
                step,
                avg_lion,
                avg_adam,
                (avg_lion - avg_adam).abs()
            );
        }
    }

    println!();
    println!("→ Lion often converges differently than Adam");
    println!("  Sign-based updates can escape local minima differently");
}
