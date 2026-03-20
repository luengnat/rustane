//! Example: Chunked Backward Pass for ANE
//!
//! Demonstrates chunked backward pass, which splits the backward computation
//! into smaller chunks to work around ANE's limitation of not supporting
//! multi-input MIL programs.
//!
//! This enables more of the backward pass to run on ANE instead of falling
//! back to CPU.

use rustane::training::chunked_backward::{
    ChunkedBackwardConfig, ChunkedBackwardExecutor, ChunkedBackwardStats,
};

fn main() {
    println!("Rustane Chunked Backward Pass Example");
    println!("=====================================\n");

    // Example 1: Basic chunking configuration
    println!("Example 1: Basic Chunking Configuration");
    println!("--------------------------------------");
    basic_chunking_config();
    println!();

    // Example 2: Memory savings
    println!("Example 2: Memory Efficiency");
    println!("---------------------------");
    memory_efficiency_demo();
    println!();

    // Example 3: Chunk execution plan
    println!("Example 3: Execution Plan");
    println!("-------------------------");
    execution_plan_demo();
    println!();

    // Example 4: ANE vs CPU comparison
    println!("Example 4: ANE vs CPU Execution");
    println!("------------------------------");
    ane_vs_cpu_demo();
    println!();

    // Example 5: Real-world scenario
    println!("Example 5: Real-World Training Scenario");
    println!("---------------------------------------");
    real_world_scenario();
    println!();

    println!("✓ Example completed!");
    println!("\nKey features:");
    println!("  • Split backward pass into smaller chunks");
    println!("  • Each chunk uses single-input ANE kernels");
    println!("  • Reduces activation memory requirements");
    println!("  • Enables more backward work on ANE");
}

/// Demonstrate basic chunking configuration
fn basic_chunking_config() {
    println!("Configuration options:");
    println!();

    let config = ChunkedBackwardConfig::new(4, 1);
    println!("  chunk_size: {}", config.chunk_size);
    println!("  overlap_size: {}", config.overlap_size);
    println!("  use_ane: {}", config.use_ane);
    println!("  max_chunk_memory: {} MB", config.max_chunk_memory / 1024 / 1024);

    println!("\n→ Chunk size determines layers per backward pass");
    println!("→ Overlap ensures gradient continuity across chunks");
}

/// Demonstrate memory efficiency
fn memory_efficiency_demo() {
    println!("Memory comparison for different model sizes:");
    println!();

    for num_layers in [8, 16, 32, 64] {
        let config = ChunkedBackwardConfig::new(4, 1);
        let executor = ChunkedBackwardExecutor::new(config);

        let (chunked, original) = executor.memory_savings(num_layers);
        let savings_pct = executor.memory_saving_percentage(num_layers);

        println!(
            "Layers: {:2} | Original: {:8.2} MB | Chunked: {:8.2} MB | Savings: {:.1}%",
            num_layers,
            original as f64 / 1024.0 / 1024.0,
            chunked as f64 / 1024.0 / 1024.0,
            savings_pct
        );
    }

    println!("\n→ Chunked backward reduces peak memory usage");
}

/// Demonstrate execution plan
fn execution_plan_demo() {
    println!("Execution plan for 16-layer model:");
    println!();

    let config = ChunkedBackwardConfig::new(4, 1);
    let executor = ChunkedBackwardExecutor::new(config);
    let plan = executor.create_execution_plan(16).unwrap();

    println!("Total chunks: {}", plan.num_chunks());
    println!("Total memory: {:.2} MB", plan.total_memory as f64 / 1024.0 / 1024.0);
    println!();

    println!("Chunk breakdown:");
    for (i, chunk) in plan.chunks.iter().enumerate() {
        println!(
            "  Chunk {}: layers {}-{} ({} layers, {} activation keys)",
            i,
            chunk.layer_range.0,
            chunk.layer_range.1 - 1,
            chunk.num_layers(),
            chunk.activation_keys.len()
        );
    }

    println!("\n→ Each chunk processes a subset of layers independently");
}

/// Demonstrate ANE vs CPU execution
fn ane_vs_cpu_demo() {
    println!("Comparing ANE and CPU execution:");
    println!();

    let mut ane_stats = ChunkedBackwardStats::new();
    let mut cpu_stats = ChunkedBackwardStats::new();

    // Simulate stats
    ane_stats.num_chunks = 4;
    ane_stats.ane_chunks = 4;
    ane_stats.total_time_ms = 25.0;
    ane_stats.ane_time_ms = 25.0;

    cpu_stats.num_chunks = 4;
    cpu_stats.cpu_chunks = 4;
    cpu_stats.total_time_ms = 100.0;
    cpu_stats.cpu_time_ms = 100.0;

    println!("ANE Execution:");
    println!("  Chunks: {}", ane_stats.ane_chunks);
    println!("  Time: {:.2} ms", ane_stats.ane_time_ms);
    println!("  Coverage: {:.1}%", ane_stats.ane_coverage());

    println!("\nCPU Execution:");
    println!("  Chunks: {}", cpu_stats.cpu_chunks);
    println!("  Time: {:.2} ms", cpu_stats.cpu_time_ms);

    println!("\nSpeedup: {:.1}x", ane_stats.speedup());

    println!("\n→ ANE execution provides significant speedup");
}

/// Demonstrate real-world training scenario
fn real_world_scenario() {
    println!("Scenario: Training a 32-layer transformer");
    println!();

    let config = ChunkedBackwardConfig::new(4, 1).with_ane(true);
    let mut executor = ChunkedBackwardExecutor::new(config);

    // Simulate storing activations
    println!("Storing activations from forward pass...");
    for layer in 0..32 {
        executor.store_activations(layer, vec![0.0; 512]);
        executor.store_output(layer, vec![0.0; 512]);
    }
    println!("✓ Stored 64 activation tensors (32 layers × 2 per layer)");

    println!("\nCreating execution plan...");
    let plan = executor.create_execution_plan(32).unwrap();
    println!("✓ Created {} chunks for backward pass", plan.num_chunks());

    println!("\nMemory analysis:");
    let (chunked, original) = executor.memory_savings(32);
    let savings_pct = executor.memory_saving_percentage(32);

    println!(
        "  Without chunking: {:.2} MB",
        original as f64 / 1024.0 / 1024.0
    );
    println!(
        "  With chunking: {:.2} MB",
        chunked as f64 / 1024.0 / 1024.0
    );
    println!("  Memory savings: {:.1}%", savings_pct);

    println!("\nBenefits:");
    println!("  • {:.1}% less memory per backward pass", savings_pct);
    println!("  • Enables ANE execution for more operations");
    println!("  • Reduces CPU fallback frequency");
    println!("  • Scales to larger models");

    println!("\n→ Chunked backward makes large model training feasible");
}
