//! Example: AdamW Optimizer with Decoupled Weight Decay
//!
//! Demonstrates the AdamW optimizer and compares it with standard Adam:
//! 1. Adam: Applies weight decay to gradients (L2 regularization)
//! 2. AdamW: Applies weight decay directly to parameters (decoupled)
//!
//! AdamW is recommended for training transformers as it often improves
//! generalization and produces better results than Adam with L2 regularization.

use rustane::training::{AdamOptimizer, AdamWOptimizer, Optimizer};

fn main() {
    println!("Rustane AdamW Optimizer Example");
    println!("===============================\n");

    // Example 1: Basic usage comparison
    println!("Example 1: Adam vs AdamW Update Comparison");
    println!("-------------------------------------------");
    compare_optimizers();
    println!();

    // Example 2: Weight decay effect
    println!("Example 2: Weight Decay Effect Over Time");
    println!("----------------------------------------");
    demonstrate_weight_decay_effect();
    println!();

    // Example 3: Custom hyperparameters
    println!("Example 3: Custom AdamW Hyperparameters");
    println!("--------------------------------------");
    demonstrate_custom_hyperparams();
    println!();

    // Example 4: Training scenario
    println!("Example 4: Simulated Training Steps");
    println!("------------------------------------");
    simulate_training();
    println!();

    println!("✓ Example completed!");
    println!("\nKey takeaways:");
    println!("  • AdamW applies weight decay directly to parameters (not gradients)");
    println!("  • This decoupling often improves generalization for transformers");
    println!("  • Recommended weight decay: 0.01 for most transformer models");
    println!("  • AdamW reduces parameters even when gradients are zero");
}

/// Compare Adam and AdamW optimizer updates
fn compare_optimizers() {
    let mut adam = AdamOptimizer::with_hyperparams(3, 0.9, 0.999, 1e-8);
    let mut adamw = AdamWOptimizer::with_hyperparams(3, 0.9, 0.999, 1e-8, 0.01);

    let mut params_adam = vec![1.0, 1.0, 1.0];
    let mut params_adamw = vec![1.0, 1.0, 1.0];
    let grads = vec![0.1, 0.1, 0.1];
    let lr = 0.001;

    println!("Initial parameters: [1.0, 1.0, 1.0]");
    println!("Gradients:           [0.1, 0.1, 0.1]");
    println!("Learning rate:       {}", lr);
    println!("Weight decay:        0.01 (AdamW only)");
    println!();

    // First step
    adam.step(&grads, &mut params_adam, lr).unwrap();
    adamw.step(&grads, &mut params_adamw, lr).unwrap();

    println!("After step 1:");
    println!(
        "  Adam:  params = [{:.6}, {:.6}, {:.6}]",
        params_adam[0], params_adam[1], params_adam[2]
    );
    println!(
        "  AdamW: params = [{:.6}, {:.6}, {:.6}] (smaller due to weight decay)",
        params_adamw[0], params_adamw[1], params_adamw[2]
    );

    let diff: f32 = ((params_adam[0] - params_adamw[0]).abs()
        + (params_adam[1] - params_adamw[1]).abs()
        + (params_adam[2] - params_adamw[2]).abs())
        / 3.0;
    println!("  Average difference: {:.6}", diff);
}

/// Demonstrate weight decay effect with zero gradients
fn demonstrate_weight_decay_effect() {
    let mut adamw = AdamWOptimizer::with_hyperparams(5, 0.9, 0.999, 1e-8, 0.1);
    let mut params = vec![1.0; 5];
    let zero_grads = vec![0.0; 5];
    let lr = 0.01;

    println!("Simulating weight decay with ZERO gradients:");
    println!("Initial parameters: [1.0, 1.0, 1.0, 1.0, 1.0]");
    println!("Weight decay: 0.1, Learning rate: 0.01");
    println!();

    for step in 1..=10 {
        adamw.step(&zero_grads, &mut params, lr).unwrap();
        let avg_param: f32 = params.iter().sum::<f32>() / params.len() as f32;
        println!(
            "Step {:2}: Average param = {:.6} (reduced by weight decay)",
            step, avg_param
        );
    }

    println!();
    println!("→ Parameters shrink over time even with zero gradients!");
    println!("  This prevents weights from growing too large and improves generalization.");
}

/// Demonstrate custom hyperparameter configuration
fn demonstrate_custom_hyperparams() {
    println!("Creating AdamW optimizers with different configurations:");
    println!();

    // Default configuration
    let _default = AdamWOptimizer::new(100);
    println!("  Default AdamW:");
    println!("    • Parameters: 100");
    println!("    • β₁ (beta1):     0.9 (default)");
    println!("    • β₂ (beta2):     0.999 (default)");
    println!("    • ε (eps):        1e-8 (default)");
    println!("    • weight_decay:   0.01 (default)");
    println!();

    // Custom configuration for fine-tuning
    let _custom = AdamWOptimizer::with_hyperparams(100, 0.99, 0.9999, 1e-7, 0.001);
    println!("  Fine-tuning AdamW:");
    println!("    • Parameters: 100");
    println!("    • β₁ (beta1):     0.99 (higher momentum)");
    println!("    • β₂ (beta2):     0.9999 (slower decay)");
    println!("    • ε (eps):        1e-7 (more stable)");
    println!("    • weight_decay:   0.001 (lower for fine-tuning)");
    println!();

    // Builder pattern for weight decay
    let with_custom_wd = AdamWOptimizer::new(100).with_weight_decay(0.05);
    println!("  Builder pattern for weight decay:");
    println!("    • weight_decay:   {}", with_custom_wd.weight_decay());
    println!();

    println!("Configuration tips:");
    println!("  • Use higher β₁ (0.99) for more stable training");
    println!("  • Use lower weight decay (0.001-0.01) for fine-tuning");
    println!("  • Use higher weight decay (0.1) for regularization");
}

/// Simulate a simplified training scenario
fn simulate_training() {
    let mut adam = AdamOptimizer::new(4);
    let mut adamw = AdamWOptimizer::new(4);

    let mut params_adam = vec![0.5, 0.5, 0.5, 0.5];
    let mut params_adamw = vec![0.5, 0.5, 0.5, 0.5];

    println!("Training simulation (10 steps):");
    println!("Initial parameters: [0.5, 0.5, 0.5, 0.5]");
    println!();

    for step in 1..=10 {
        // Simulated gradients (decreasing over time as model converges)
        let scale = 1.0 / (step as f32);
        let grads: Vec<f32> = (0..4).map(|i| (i as f32 + 1.0) * 0.1 * scale).collect();

        adam.step(&grads, &mut params_adam, 0.01).unwrap();
        adamw.step(&grads, &mut params_adamw, 0.01).unwrap();

        let avg_adam: f32 = params_adam.iter().sum::<f32>() / 4.0;
        let avg_adamw: f32 = params_adamw.iter().sum::<f32>() / 4.0;

        if step % 2 == 0 || step == 10 {
            let delta: f32 = (avg_adam - avg_adamw).abs();
            println!(
                "Step {:2}: Adam avg={:.6}, AdamW avg={:.6}, Δ={:.6}",
                step, avg_adam, avg_adamw, delta
            );
        }
    }

    println!();
    println!("→ AdamW parameters stay smaller due to decoupled weight decay");
    println!("  This typically leads to better generalization on test data.");
}
