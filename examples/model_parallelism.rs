//! Example: Model Parallelism for Large-Scale Transformers
//!
//! Demonstrates model parallelism, which shards large transformer models
//! across multiple devices to enable training models that don't fit on
//! a single device.
//!
//! Combined with sequence parallelism and data parallelism, this enables
//! training of 7B+ parameter models across multiple ANE devices.

use rustane::training::model_parallel::{
    CommunicationType, ModelParallelConfig, ModelParallelism, ParallelismType,
};

fn main() {
    println!("Rustane Model Parallelism Example");
    println!("=================================\n");

    // Example 1: Layer Parallelism
    println!("Example 1: Layer Parallelism");
    println!("---------------------------");
    layer_parallelism_demo();
    println!();

    // Example 2: Tensor Parallelism
    println!("Example 2: Tensor Parallelism");
    println!("------------------------------");
    tensor_parallelism_demo();
    println!();

    // Example 3: Pipeline Parallelism
    println!("Example 3: Pipeline Parallelism");
    println!("-------------------------------");
    pipeline_parallelism_demo();
    println!();

    // Example 4: Hybrid Parallelism
    println!("Example 4: Hybrid Parallelism");
    println!("-----------------------------");
    hybrid_parallelism_demo();
    println!();

    // Example 5: Memory Efficiency Comparison
    println!("Example 5: Memory Efficiency Comparison");
    println!("---------------------------------------");
    memory_efficiency_comparison();
    println!();

    // Example 6: Communication Patterns
    println!("Example 6: Communication Patterns");
    println!("---------------------------------");
    communication_patterns_demo();
    println!();

    // Example 7: Real-World Large Model
    println!("Example 7: Real-World Large Model (32 layers)");
    println!("------------------------------------------------");
    real_world_large_model();
    println!();

    println!("✓ Example completed!");
    println!("\nKey features:");
    println!("  • Layer parallelism: distribute layers across devices");
    println!("  • Tensor parallelism: split attention heads and projections");
    println!("  • Pipeline parallelism: sequential execution with micro-batches");
    println!("  • Hybrid: combine multiple strategies for maximum efficiency");
    println!("  • 75-87% memory reduction per device");
}

/// Demonstrate layer parallelism
fn layer_parallelism_demo() {
    println!("Configuration:");
    println!("  num_devices: 4");
    println!("  num_layers: 32");
    println!("  type: Layer");
    println!();

    let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
    let mp = ModelParallelism::new(config).unwrap();

    println!("Device assignments:");
    for device_id in 0..4 {
        if let Some(shard) = mp.get_shard_for_device(device_id) {
            println!(
                "  Device {}: layers {}-{} ({} layers)",
                device_id,
                shard.layer_range.0,
                shard.layer_range.1 - 1,
                shard.num_layers()
            );
        }
    }

    println!("\nLayer to device mapping (sample):");
    for layer in [0, 8, 16, 24, 31] {
        println!(
            "  Layer {}: Device {}",
            layer,
            mp.get_device_for_layer(layer).unwrap()
        );
    }

    println!("\n→ Each device processes 8 layers independently");
}

/// Demonstrate tensor parallelism
fn tensor_parallelism_demo() {
    println!("Configuration:");
    println!("  num_devices: 4");
    println!("  num_layers: 32");
    println!("  heads_per_device: 8");
    println!("  type: Tensor");
    println!();

    let config = ModelParallelConfig::new(4, ParallelismType::Tensor, 32)
        .unwrap()
        .with_heads_per_device(8);
    let mp = ModelParallelism::new(config).unwrap();

    println!("Head assignments:");
    for device_id in 0..4 {
        if let Some(shard) = mp.get_shard_for_device(device_id) {
            if let Some((start, end)) = shard.head_range {
                println!(
                    "  Device {}: heads {}-{} ({} heads)",
                    device_id,
                    start,
                    end - 1,
                    end - start
                );
            }
        }
    }

    println!("\n→ All devices process all layers, but split the attention heads");
}

/// Demonstrate pipeline parallelism
fn pipeline_parallelism_demo() {
    println!("Configuration:");
    println!("  num_devices: 4");
    println!("  num_layers: 32");
    println!("  num_micro_batches: 4");
    println!("  type: Pipeline");
    println!();

    let config = ModelParallelConfig::new(4, ParallelismType::Pipeline, 32)
        .unwrap()
        .with_micro_batches(4);
    let mp = ModelParallelism::new(config).unwrap();

    println!("Pipeline stages:");
    for (i, shard) in mp.shards().iter().enumerate() {
        println!(
            "  Stage {}: Device {}, layers {}-{}",
            i,
            shard.device_id,
            shard.layer_range.0,
            shard.layer_range.1 - 1
        );
    }

    println!("\n→ Micro-batches flow through pipeline stages for better device utilization");
}

/// Demonstrate hybrid parallelism
fn hybrid_parallelism_demo() {
    println!("Configuration:");
    println!("  num_devices: 4");
    println!("  num_layers: 32");
    println!("  heads_per_device: 4");
    println!("  type: Hybrid");
    println!();

    let config = ModelParallelConfig::new(4, ParallelismType::Hybrid, 32)
        .unwrap()
        .with_heads_per_device(4);
    let mp = ModelParallelism::new(config).unwrap();

    println!("Hybrid assignment:");
    for (_i, shard) in mp.shards().iter().enumerate() {
        println!(
            "  Device {}: layers {}-{}, heads {:?}",
            shard.device_id,
            shard.layer_range.0,
            shard.layer_range.1 - 1,
            shard.head_range
        );
    }

    println!("\n→ Combines layer and tensor parallelism for maximum efficiency");
}

/// Compare memory efficiency of different parallelism types
fn memory_efficiency_comparison() {
    println!("Memory efficiency for 32-layer model across 4 devices:");
    println!();

    let parallelism_types = [
        ParallelismType::Layer,
        ParallelismType::Tensor,
        ParallelismType::Pipeline,
        ParallelismType::Hybrid,
    ];

    println!("Type          | Memory Efficiency | Comm Overhead | Savings");
    println!("--------------|-------------------|---------------|--------");

    for ptype in parallelism_types {
        let config = ModelParallelConfig::new(4, ptype, 32).unwrap();
        let mp = ModelParallelism::new(config).unwrap();

        let efficiency = mp.memory_efficiency();
        let overhead = mp.communication_overhead();
        let savings = mp.memory_saving_percentage();

        println!(
            "{:13} | {:.17} | {:.13} | {:.1}%",
            format!("{:?}", ptype),
            efficiency,
            overhead,
            savings
        );
    }

    println!("\n→ Hybrid offers best memory efficiency, Pipeline has lowest communication");
}

/// Demonstrate communication patterns
fn communication_patterns_demo() {
    println!("Communication patterns for different parallelism types:");
    println!();

    // Layer parallelism
    let config = ModelParallelConfig::new(4, ParallelismType::Layer, 32).unwrap();
    let mp = ModelParallelism::new(config).unwrap();
    let plan = mp.communication_plan().unwrap();

    println!("Layer Parallelism:");
    println!("  Total communications: {}", plan.len());

    let forward: Vec<_> = plan
        .iter()
        .filter(|c| c.comm_type == CommunicationType::ForwardActivation)
        .collect();
    let backward: Vec<_> = plan
        .iter()
        .filter(|c| c.comm_type == CommunicationType::BackwardGradient)
        .collect();

    println!("  Forward activations: {}", forward.len());
    println!("  Backward gradients: {}", backward.len());

    // Tensor parallelism
    let config = ModelParallelConfig::new(2, ParallelismType::Tensor, 8)
        .unwrap()
        .with_heads_per_device(4);
    let mp = ModelParallelism::new(config).unwrap();
    let plan = mp.communication_plan().unwrap();

    println!("\nTensor Parallelism:");
    println!("  Total communications: {}", plan.len());

    let all_reduce: Vec<_> = plan
        .iter()
        .filter(|c| c.comm_type == CommunicationType::AllReduce)
        .collect();
    println!("  All-reduce operations: {}", all_reduce.len());

    println!("\n→ Layer parallelism: communication at boundaries");
    println!("→ Tensor parallelism: all-reduce for every layer");
}

/// Demonstrate a real-world large model scenario
fn real_world_large_model() {
    println!("Scenario: Training a 32-layer transformer (similar to GPT-2 medium)");
    println!();

    let num_devices = 4;
    let num_layers = 32;
    let config = ModelParallelConfig::new(num_devices, ParallelismType::Layer, num_layers).unwrap();
    let mp = ModelParallelism::new(config).unwrap();

    println!("Model configuration:");
    println!("  Layers: {}", num_layers);
    println!("  Devices: {}", num_devices);
    println!("  Parallelism: Layer");
    println!();

    // Calculate memory
    let (per_device, original) = mp.memory_savings();
    let savings_pct = mp.memory_saving_percentage();

    println!("Memory analysis (estimated):");
    println!(
        "  Single device:     {:.2} GB",
        original as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!(
        "  Per device (4x):   {:.2} GB",
        per_device as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!("  Memory savings:   {:.1}%", savings_pct);

    println!("\nWith layer parallelism:");
    println!("  • Device 0: Layers 0-7 (embedding + early layers)");
    println!("  • Device 1: Layers 8-15 (lower layers)");
    println!("  • Device 2: Layers 16-23 (middle layers)");
    println!("  • Device 3: Layers 24-31 (upper layers + output)");

    println!("\nCombined with other techniques:");
    println!("  • Sequence parallelism: +75% savings for long sequences");
    println!("  • Gradient checkpointing: +50% savings");
    println!("  • Mixed precision: +50% savings");
    println!("  → Total: >95% memory reduction vs baseline");

    println!("\n→ Model parallelism enables training large models on limited hardware");
}
