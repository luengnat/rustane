//! Example: Learning rate schedules for training
//!
//! Demonstrates different learning rate scheduling strategies:
//! 1. Constant: Fixed learning rate throughout training
//! 2. Warmup-Linear: Warmup then linear decay
//! 3. Warmup-Cosine: Warmup then cosine annealing
//!
//! Learning rate scheduling is crucial for training stability and convergence.

use rustane::{ConstantScheduler, LRScheduler, WarmupCosineScheduler, WarmupLinearScheduler};

fn main() {
    println!("Rustane Learning Rate Schedule Example");
    println!("=====================================\n");

    // Example 1: Constant learning rate (simple baseline)
    println!("Example 1: Constant Learning Rate");
    println!("--------------------------------");
    let scheduler = ConstantScheduler::new(0.001);
    print_schedule(&scheduler, 10, 5);
    println!();

    // Example 2: Warmup then linear decay
    println!("Example 2: Warmup-Linear Schedule");
    println!("(10 warmup steps, 40 decay steps, peak LR = 0.001)");
    println!("------");
    let scheduler = WarmupLinearScheduler::new(0.001, 10, 50);
    print_schedule(&scheduler, 50, 10);
    println!();

    // Example 3: Warmup then cosine annealing (most common in modern training)
    println!("Example 3: Warmup-Cosine Schedule");
    println!("(10 warmup steps, 40 cosine steps, peak LR = 0.001, min LR = 0.0001)");
    println!("------");
    let scheduler = WarmupCosineScheduler::new(0.001, 10, 50, 0.0001);
    print_schedule(&scheduler, 50, 10);
    println!();

    // Example 4: Realistic larger-scale schedule
    println!("Example 4: Realistic Training Schedule (10k steps total)");
    println!("(500 warmup, 9500 cosine, peak = 1e-4, min = 1e-6)");
    println!("------");
    let scheduler = WarmupCosineScheduler::new(1e-4, 500, 10000, 1e-6);

    println!("Learning rate progression:");
    println!(
        "  Step 0:    LR = {:.6e} (start warmup)",
        scheduler.get_lr(0)
    );
    println!(
        "  Step 250:  LR = {:.6e} (mid warmup)",
        scheduler.get_lr(250)
    );
    println!(
        "  Step 500:  LR = {:.6e} (end warmup, start cosine)",
        scheduler.get_lr(500)
    );
    println!(
        "  Step 2500: LR = {:.6e} (mid training)",
        scheduler.get_lr(2500)
    );
    println!("  Step 5000: LR = {:.6e} (halfway)", scheduler.get_lr(5000));
    println!(
        "  Step 7500: LR = {:.6e} (3/4 done)",
        scheduler.get_lr(7500)
    );
    println!(
        "  Step 9999: LR = {:.6e} (near end)",
        scheduler.get_lr(9999)
    );
    println!();

    // Example 5: Compare schedules
    println!("Example 5: Schedule Comparison");
    println!("-----");
    let steps = vec![0, 5, 10, 25, 50];
    println!("Step | Constant | Warmup-Linear | Warmup-Cosine");
    println!("-----|----------|-------|------");

    let constant = ConstantScheduler::new(0.001);
    let linear = WarmupLinearScheduler::new(0.001, 10, 50);
    let cosine = WarmupCosineScheduler::new(0.001, 10, 50, 0.0001);

    for step in steps {
        let c_lr = format!("{:.6}", constant.get_lr(step));
        let l_lr = format!("{:.6}", linear.get_lr(step));
        let co_lr = format!("{:.6}", cosine.get_lr(step));
        println!("{:4} | {}  | {}    | {}", step, c_lr, l_lr, co_lr);
    }

    println!("\n✓ Example completed!");
    println!("\nKey takeaways:");
    println!("  • Use warmup to stabilize training in first ~5-10% of steps");
    println!("  • Cosine annealing often produces better final results than linear decay");
    println!("  • Learning rate directly impacts convergence speed and stability");
    println!("  • Different models may benefit from different schedules");
}

/// Helper function to print a schedule progression
fn print_schedule(scheduler: &dyn LRScheduler, total_steps: u32, sample_interval: u32) {
    for step in (0..=total_steps).step_by(sample_interval as usize) {
        let lr = scheduler.get_lr(step);
        let lr_str = format!("{:.6}", lr);
        print!("Step {:4}: LR = {}  ", step, lr_str);

        // Visual representation with bar
        let bar_width = (lr * 10000.0) as usize; // Scale for visibility
        print!("[");
        for _ in 0..bar_width.min(40) {
            print!("=");
        }
        println!("]");
    }

    // Print final step if needed
    let final_lr = scheduler.get_lr(total_steps);
    if total_steps % sample_interval != 0 {
        let lr_str = format!("{:.6}", final_lr);
        print!("Step {:4}: LR = {}  ", total_steps, lr_str);
        let bar_width = (final_lr * 10000.0) as usize;
        print!("[");
        for _ in 0..bar_width.min(40) {
            print!("=");
        }
        println!("]");
    }
}
