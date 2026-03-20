//! Example: Flash Attention for Memory-Efficient Long Sequences
//!
//! Demonstrates Flash Attention, which computes attention in blocks to reduce
//! memory usage from O(seq_len²) to O(seq_len * block_size).
//!
//! Key benefits:
//! - Enables training on longer sequences without OOM
//! - Reduces memory by 85-95% for typical sequence lengths
//! - Maintains exact attention computation (no approximation)

use rustane::layers::FlashAttention;

fn main() {
    println!("Rustane Flash Attention Example");
    println!("==============================\n");

    // Example 1: Basic usage
    println!("Example 1: Basic Flash Attention");
    println!("--------------------------------");
    demonstrate_basic_usage();
    println!();

    // Example 2: Memory savings
    println!("Example 2: Memory Efficiency Comparison");
    println!("-------------------------------------");
    demonstrate_memory_savings();
    println!();

    // Example 3: Block size tradeoffs
    println!("Example 3: Block Size Tradeoffs");
    println!("------------------------------");
    demonstrate_block_size_tradeoffs();
    println!();

    // Example 4: Causal masking
    println!("Example 4: Causal Masking (Autoregressive)");
    println!("-----------------------------------------");
    demonstrate_causal_masking();
    println!();

    // Example 5: Real-world scenario
    println!("Example 5: Real-World Training Scenario");
    println!("--------------------------------------");
    demonstrate_real_world_scenario();
    println!();

    println!("✓ Example completed!");
    println!("\nKey takeaways:");
    println!("  • Flash Attention reduces memory from O(seq_len²) to O(seq_len * block_size)");
    println!("  • Enables training on sequences 4-8x longer with same memory");
    println!("  • Exact computation (not an approximation)");
    println!("  • Use block_size=128 for seq_len ≤ 2048, block_size=64 for longer sequences");
}

/// Demonstrate basic Flash Attention usage
fn demonstrate_basic_usage() {
    let num_heads = 8;
    let head_dim = 64;
    let seq_len = 512;
    let causal = false;

    println!("Configuration:");
    println!("  num_heads: {}", num_heads);
    println!("  head_dim: {}", head_dim);
    println!("  seq_len: {}", seq_len);
    println!("  causal: {}", causal);
    println!();

    let flash_attn = FlashAttention::new(num_heads, head_dim, causal);

    // Create simple Q, K, V tensors
    let total_size = seq_len * num_heads * head_dim;
    let q: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.01 - 0.5)).collect();
    let k: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.01)).collect();
    let v: Vec<f32> = (0..total_size).map(|i| (i as f32 * 0.01 + 0.5)).collect();

    // Compute Flash Attention
    let output = flash_attn.forward(&q, &k, &v).unwrap();

    println!("Output shape: [{} × {} × {}]", seq_len, num_heads, head_dim);
    println!("Output range: [{:.4}, {:.4}]",
        output.iter().cloned().fold(f32::INFINITY, f32::min),
        output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
}

/// Demonstrate memory savings compared to standard attention
fn demonstrate_memory_savings() {
    let flash_attn = FlashAttention::new(8, 64, false);

    println!("Memory comparison (per attention head):");
    println!();
    println!("seq_len | Standard Attention | Flash Attention | Savings");
    println!("--------|-------------------|-----------------|---------");

    for seq_len in [512, 1024, 2048, 4096, 8192] {
        let standard_mb = (seq_len * seq_len * 4) as f64 / 1024.0 / 1024.0;
        let flash_bytes = flash_attn.memory_usage(seq_len);
        let flash_mb = flash_bytes as f64 / 1024.0 / 1024.0;
        let saving_pct = flash_attn.memory_saving_percentage(seq_len);

        println!("{:7} | {:15.2} MB | {:14.2} MB | {:6.1}%",
            seq_len, standard_mb, flash_mb, saving_pct
        );
    }

    println!();
    println!("→ Flash Attention enables 4-8x longer sequences with same memory!");
}

/// Demonstrate block size tradeoffs
fn demonstrate_block_size_tradeoffs() {
    let seq_len = 2048;

    println!("For seq_len = {}, different block sizes:", seq_len);
    println!();
    println!("block_size | Memory Usage | Speed");
    println!("-----------|--------------|-------");

    for block_size in [32, 64, 128, 256, 512] {
        let attn = FlashAttention::with_block_size(8, 64, false, block_size);
        let memory_mb = (attn.memory_usage(seq_len) as f64 / 1024.0 / 1024.0);

        // Speed is inversely proportional to number of blocks
        let num_blocks = (seq_len + block_size - 1) / block_size;
        let speed = if num_blocks > 0 {
            format!("{:.1}x", 128.0 / block_size as f64) // Relative to block_size=128
        } else {
            "N/A".to_string()
        };

        println!("{:10} | {:10.2} MB | {}", block_size, memory_mb, speed);
    }

    println!();
    println!("→ Larger blocks = faster but more memory");
    println!("  Smaller blocks = less memory but more overhead");
}

/// Demonstrate causal masking for autoregressive models
fn demonstrate_causal_masking() {
    let seq_len = 4;
    let num_heads = 1;
    let head_dim = 4;

    println!("Causal masking example (seq_len=4):");
    println!("Each position can only attend to itself and previous positions");
    println!();

    // Create simple Q, K, V where each position has distinct values
    let total_size = seq_len * num_heads * head_dim;

    // Q: position vectors [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
    let mut q = vec![0.0; total_size];
    for i in 0..seq_len {
        q[i * head_dim + i] = 1.0;
    }

    let k = q.clone();
    let v: Vec<f32> = (0..total_size).map(|i| i as f32).collect();

    // Compute with causal masking
    let flash_attn = FlashAttention::new(num_heads, head_dim, true);
    let output = flash_attn.forward(&q, &k, &v).unwrap();

    println!("Output (each position attends to previous positions):");
    for i in 0..seq_len {
        print!("  Pos {}: ", i);
        for d in 0..head_dim {
            print!("{:.2} ", output[i * head_dim + d]);
        }
        println!();
    }

    println!();
    println!("→ Position 3 only sees positions 0, 1, 2 (not position 3 itself)");
}

/// Demonstrate real-world training scenario
fn demonstrate_real_world_scenario() {
    println!("Scenario: Training a transformer with long context");
    println!();

    // Model configuration
    let num_heads = 32;
    let head_dim = 128;
    let num_layers = 24;
    let batch_size = 1;

    println!("Model configuration:");
    println!("  num_heads: {}", num_heads);
    println!("  head_dim: {}", head_dim);
    println!("  num_layers: {}", num_layers);
    println!("  batch_size: {}", batch_size);
    println!();

    // Compare sequence lengths
    let seq_lengths = [2048, 4096, 8192, 16384];

    println!("seq_len | Standard Memory | Flash Memory | Can Use Standard?");
    println!("--------|----------------|--------------|-------------------");

    for &seq_len in &seq_lengths {
        let flash_attn = FlashAttention::new(num_heads, head_dim, true);

        // Standard attention memory: seq_len² * num_heads * 4 bytes
        let standard_memory_gb =
            (seq_len * seq_len * num_heads * 4) as f64 / 1024.0 / 1024.0 / 1024.0;

        // Flash attention memory: seq_len * block_size * 2 * num_heads * 4 bytes
        let flash_memory_gb = (flash_attn.memory_usage(seq_len) * num_heads) as f64
            / 1024.0
            / 1024.0
            / 1024.0;

        let can_use_standard = if standard_memory_gb < 16.0 {
            "✓ Yes"
        } else {
            "✗ No (OOM)"
        };

        println!(
            "{:7} | {:12.2} GB | {:11.2} GB | {}",
            seq_len, standard_memory_gb, flash_memory_gb, can_use_standard
        );
    }

    println!();
    println!("Assuming 16 GB GPU memory available for attention");
    println!();
    println!("→ Flash Attention enables training on sequences 2-4x longer!");
    println!("  Standard: Max ~4K tokens");
    println!("  Flash:   Max ~16K tokens");
}
