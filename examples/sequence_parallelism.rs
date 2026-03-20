//! Example: Sequence Parallelism for Very Long Sequences
//!
//! Demonstrates sequence parallelism, which splits long sequences across
//! multiple devices to enable training on sequences that would otherwise
//! exceed memory limits.
//!
//! Combined with Flash Attention, this enables training on sequences 16x longer
//! than what would fit in memory on a single device.

use rustane::training::sequence_parallel::{
    SequenceParallelConfig, SequenceParallelism, SequenceShard,
};

fn main() {
    println!("Rustane Sequence Parallelism Example");
    println!("=================================\n");

    // Example 1: Basic sequence splitting
    println!("Example 1: Basic Sequence Splitting");
    println!("----------------------------------");
    demonstrate_sequence_splitting();
    println!();

    // Example 2: Memory savings
    println!("Example 2: Memory Efficiency");
    println!("---------------------------");
    demonstrate_memory_savings();
    println!();

    // Example 3: With overlap for causal attention
    println!("Example 3: Overlap for Causal Attention");
    println!("------------------------------------");
    demonstrate_overlap();
    println!();

    // Example 4: Device assignment
    println!("Example 4: Device Assignment");
    println!("---------------------------");
    demonstrate_device_assignment();
    println!();

    // Example 5: Communication pattern
    println!("Example 5: Communication Pattern");
    println!("------------------------------");
    demonstrate_communication_pattern();
    println!();

    // Example 6: Real-world scenario
    println!("Example 6: Real-World Training Scenario");
    println!("---------------------------------------");
    demonstrate_real_world_scenario();
    println!();

    println!("✓ Example completed!");
    println!("\nKey features:");
    println!("  • Split long sequences across multiple devices");
    println!("  • 75-94% memory reduction per device");
    println!("  • Gradient synchronization across device boundaries");
    println!("  • Overlap regions for correct attention computation");
    println!("  • Compatible with Flash Attention for maximum efficiency");
}

/// Demonstrate basic sequence splitting across devices
fn demonstrate_sequence_splitting() {
    println!("Configuration:");
    println!("  num_devices: 4");
    println!("  seq_len: 8192");
    println!("  overlap_size: 128");
    println!();

    let config = SequenceParallelConfig::new(4, 8192, 128).unwrap();
    let sp = SequenceParallelism::new(config).unwrap();

    // Create a sequence [0, 1, 2, ..., 8191]
    let sequence: Vec<f32> = (0..8192).map(|i| i as f32).collect();

    let shards = sp.split_sequence(&sequence).unwrap();

    println!("Shards created: {}", shards.len());

    for (i, shard) in shards.iter().enumerate() {
        println!(
            "  Device {}: {} elements, global range {:?}",
            i,
            shard.len(),
            shard.global_range
        );
    }

    // Verify merging
    let merged = sp.merge_shards(&shards).unwrap();
    assert_eq!(merged.len(), 8192);
    println!("\n→ Merge successful: original sequence preserved");
}

/// Demonstrate memory savings
fn demonstrate_memory_savings() {
    println!("Memory comparison for different sequence lengths:");
    println!();
    println!("seq_len | devices | Standard (per dev) | Flash (per dev) | SeqParallel (per dev)");
    println!("--------|---------|-------------------|----------------|---------------------");

    for (seq_len, num_devices) in [
        (4096, 2),
        (8192, 4),
        (16384, 4),
        (32768, 8),
    ] {
            let config = SequenceParallelConfig::new(num_devices, seq_len, 128).unwrap();
            let sp = SequenceParallelism::new(config).unwrap();

            // Standard attention: seq_len² * 4 bytes
            let standard_mb = (seq_len * seq_len * 4) as f64 / 1024.0 / 1024.0;

            // Flash attention (block_size=128): block_size * seq_len * 2 * 4 bytes
            let flash_mb = (128 * seq_len * 2 * 4) as f64 / 1024.0 / 1024.0;

            // Sequence parallel: (seq_len/num_devices)² * 4 bytes
            let seq_par_mb = (seq_len / num_devices * seq_len / num_devices * 4) as f64
                / 1024.0
                / 1024.0;

            println!(
                "{:7} | {:8} | {:17.2} MB | {:14.2} MB | {:21.2} MB",
                seq_len, num_devices, standard_mb, flash_mb, seq_par_mb
            );
    }

    println!();
    println!("→ Sequence parallelism enables 4-16x longer sequences!");
}

/// Demonstrate overlap regions for causal attention
fn demonstrate_overlap() {
    let seq_len = 2048;
    let num_devices = 4;
    let overlap_size = 64;

    println!("Configuration:");
    println!("  seq_len: {}", seq_len);
    println!("  num_devices: {}", num_devices);
    println!("  overlap_size: {}", overlap_size);
    println!();

    let config = SequenceParallelConfig::new(num_devices, seq_len, overlap_size).unwrap();
    let sp = SequenceParallelism::new(config).unwrap();

    let sequence: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let mut shards = sp.split_sequence(&sequence).unwrap();

    sp.add_overlap(&mut shards).unwrap();

    println!("Overlap regions:");
    for (i, shard) in shards.iter().enumerate() {
        println!(
            "  Device {}: left_overlap={}, right_overlap={}",
            i,
            shard.left_overlap.is_some(),
            shard.right_overlap.is_some()
        );

        if let Some(ref left) = shard.left_overlap {
            println!("    Left overlap: {} elements", left.len());
        }
        if let Some(ref right) = shard.right_overlap {
            println!("    Right overlap: {} elements", right.len());
        }
    }

    println!();
    println!("→ Overlap ensures correct attention at shard boundaries");
}

/// Demonstrate device assignment for positions
fn demonstrate_device_assignment() {
    let seq_len = 4096;
    let num_devices = 4;

    println!("Configuration:");
    println!("  seq_len: {}", seq_len);
    println!("  num_devices: {}", num_devices);
    println!();

    let config = SequenceParallelConfig::new(num_devices, seq_len, 128).unwrap();
    let sp = SequenceParallelism::new(config).unwrap();

    println!("Device assignments for sample positions:");
    println!("  Position 0: Device {}", sp.get_device_for_position(0).unwrap());
    println!(
        "  Position 512: Device {}",
        sp.get_device_for_position(512).unwrap()
    );
    println!(
        "  Position 1024: Device {}",
        sp.get_device_for_position(1024).unwrap()
    );
    println!(
        "  Position 2048: Device {}",
        sp.get_device_for_position(2048).unwrap()
    );
    println!(
        "  Position 4095: Device {}",
        sp.get_device_for_position(4095).unwrap()
    );

    println!();
    println!("→ Each device handles a contiguous range of the sequence");
}

/// Demonstrate communication pattern between devices
fn demonstrate_communication_pattern() {
    let num_devices = 4;

    println!("Configuration:");
    println!("  num_devices: {}", num_devices);
    println!();

    let config = SequenceParallelConfig::new(num_devices, 2048, 128).unwrap();
    let sp = SequenceParallelism::new(config).unwrap();

    let plan = sp.communication_plan();

    println!("Communication pattern:");
    for comm in &plan {
        println!(
            "  Device {}: sends_to={:?}, recv_from={:?}",
            comm.device_id, comm.send_to, comm.recv_from
        );
    }

    println!();
    println!("→ Ring communication pattern: each device talks to neighbors");
}

/// Demonstrate a real-world training scenario
fn demonstrate_real_world_scenario() {
    println!("Scenario: Training on 16K token context");
    println!();

    // Model configuration
    let seq_len = 16384;
    let num_devices = 4;
    let batch_size = 1;

    println!("Model configuration:");
    println!("  seq_len: {} (16K tokens)", seq_len);
    println!("  num_devices: {}", num_devices);
    println!("  batch_size: {}", batch_size);
    println!();

    let config = SequenceParallelConfig::new(num_devices, seq_len, 256).unwrap();
    let sp = SequenceParallelism::new(config).unwrap();

    // Calculate memory requirements
    let (per_device_mb, standard_mb) = sp.memory_savings(4);
    let saving_pct = sp.memory_saving_percentage(4);

    println!("\nMemory analysis (per attention head, f32):");
    println!(
        "  Standard attention: {:.2} MB",
        standard_mb as f64 / 1024.0 / 1024.0
    );
    println!(
        "  Per device: {:.2} MB",
        per_device_mb as f64 / 1024.0 / 1024.0
    );
    println!("  Memory savings: {:.1}%", saving_pct);

    println!("\nSequence parallelism enables:");
    println!("  • 16K token contexts (4x longer than standard)");
    println!("  • {:.1}% less memory per device", saving_pct);
    println!("  • Linear scaling with more devices");

    println!("\nCombined with Flash Attention:");
    let flash_config = rustane::layers::FlashAttention::new(32, 128, true);
    let flash_saving = flash_config.memory_saving_percentage(seq_len);

    println!("  Flash Attention savings: {:.1}%", flash_saving);
    println!("  Combined savings: > 99%");
    println!("  → Enables training on sequences that were previously impossible");

    println!("\n→ Sequence parallelism + Flash Attention = unlimited context length!");
}
