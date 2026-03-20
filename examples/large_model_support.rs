//! Example: Large Model Support (7B+ Parameters)
//!
//! Demonstrates utilities for training very large transformer models
//! that don't fit in memory on a single device.
//!
//! This includes memory-efficient initialization, parameter sharding,
//! and optimization strategies for models with 7B+ parameters.

use rustane::training::large_models::{
    InitializationStrategy, LargeModelInitializer, ModelPresets, ModelSizeCategory,
    ParameterSharding,
};

fn main() {
    println!("Rustane Large Model Support Example");
    println!("==================================\n");

    // Example 1: Model size categories
    println!("Example 1: Model Size Categories");
    println!("-------------------------------");
    size_categories_demo();
    println!();

    // Example 2: Memory requirements
    println!("Example 2: Memory Requirements");
    println!("-----------------------------");
    memory_requirements_demo();
    println!();

    // Example 3: Optimization techniques
    println!("Example 3: Optimization Techniques");
    println!("----------------------------------");
    optimization_techniques_demo();
    println!();

    // Example 4: Progressive initialization
    println!("Example 4: Progressive Initialization");
    println!("-------------------------------------");
    progressive_initialization_demo();
    println!();

    // Example 5: Parameter sharding
    println!("Example 5: Parameter Sharding");
    println!("------------------------------");
    parameter_sharding_demo();
    println!();

    // Example 6: Model presets
    println!("Example 6: Common Model Presets");
    println!("-------------------------------");
    model_presets_demo();
    println!();

    println!("✓ Example completed!");
    println!("\nKey features:");
    println!("  • Memory-efficient initialization for large models");
    println!("  • Automatic memory requirement calculation");
    println!("  • Progressive layer-by-layer initialization");
    println!("  • Parameter sharding across devices");
    println!("  • Optimization recommendations by model size");
}

/// Demonstrate model size categories
fn size_categories_demo() {
    println!("Model size categories:");
    println!();

    let sizes = [
        (ModelSizeCategory::Small, "500M params"),
        (ModelSizeCategory::Medium, "3B params"),
        (ModelSizeCategory::Large, "10B params"),
        (ModelSizeCategory::XL, "30B params"),
        (ModelSizeCategory::XXL, "100B+ params"),
    ];

    for (category, description) in sizes {
        println!("{:?} - {}", category, description);
        println!("  Description: {}", category.description());
        println!();
    }

    println!("→ Model size determines required optimization techniques");
}

/// Demonstrate memory requirements calculation
fn memory_requirements_demo() {
    println!("Memory requirements for different model sizes:");
    println!();

    let models = [
        ("7B model", ModelPresets::model_7b()),
        ("13B model", ModelPresets::model_13b()),
        ("30B model", ModelPresets::model_30b()),
    ];

    for (name, config) in models {
        let memory = config.calculate_memory_requirements();

        println!("{}:", name);
        println!(
            "  Parameters: {:.2}B",
            memory.parameter_count as f64 / 1e9
        );
        println!("  Parameter memory: {:4} MB", memory.parameter_memory_mb);
        println!("  Optimizer memory: {:4} MB", memory.optimizer_memory_mb);
        println!("  Activation memory: {:4} MB", memory.activation_memory_mb);
        println!("  Gradient memory:   {:4} MB", memory.gradient_memory_mb);
        println!("  Total memory:      {:4} MB", memory.total_memory_mb);
        println!(
            "  Total: {:.2} GB",
            memory.total_memory_mb as f64 / 1024.0
        );
        println!();
    }

    println!("→ Accurate memory planning prevents OOM errors");
}

/// Demonstrate optimization techniques
fn optimization_techniques_demo() {
    println!("Recommended optimization techniques:");
    println!();

    for category in [
        ModelSizeCategory::Small,
        ModelSizeCategory::Medium,
        ModelSizeCategory::Large,
        ModelSizeCategory::XL,
        ModelSizeCategory::XXL,
    ] {
        println!("{:?}:", category);
        let techniques = category.recommended_techniques();
        for technique in techniques {
            println!("  • {}", technique);
        }
        println!();
    }

    println!("→ Techniques scale with model size");
}

/// Demonstrate progressive initialization
fn progressive_initialization_demo() {
    println!("Progressive initialization for 7B model:");
    println!();

    let config = ModelPresets::model_7b();
    let initializer = LargeModelInitializer::new(config)
        .with_strategy(InitializationStrategy::LayerByLayer);

    let progress = initializer.initialize_progressively().unwrap();

    println!("Model configuration:");
    println!("  Total layers: {}", progress.total_layers());
    println!("  Total parameters: {:.2}B", progress.total_params() as f64 / 1e9);
    println!();

    println!("Initialization strategy: Layer-by-Layer");
    println!(
        "  Init memory: {:.2} MB",
        progress.init_memory_mb() as f64
    );
    println!(
        "  Total memory: {:.2} MB",
        progress.total_memory_mb() as f64
    );
    println!(
        "  Memory efficiency: {:.1}%",
        progress.memory_efficiency() * 100.0
    );
    println!();

    println!("→ Progressive initialization reduces peak memory by {:.1}%",
        progress.memory_efficiency() * 100.0);
}

/// Demonstrate parameter sharding
fn parameter_sharding_demo() {
    println!("Parameter sharding for 13B model across 4 devices:");
    println!();

    let config = ModelPresets::model_13b().with_num_devices(4);
    let sharding = ParameterSharding::new(4, &config);

    println!("Sharding configuration:");
    println!("  Number of shards: {}", sharding.num_shards);
    println!();

    for shard_idx in 0..sharding.num_shards {
        let params = sharding.get_shard_params(shard_idx);
        println!("  Shard {}: {} parameters", shard_idx, params.len());

        // Show sample parameters
        if !params.is_empty() {
            println!("    Sample: {}", params[0]);
            if params.len() > 1 {
                println!("            {}", params[1]);
            }
        }
        println!();
    }

    println!("→ Sharding distributes model across multiple devices");
}

/// Demonstrate model presets
fn model_presets_demo() {
    println!("Common model architecture presets:");
    println!();

    // 7B model
    {
        let config = ModelPresets::model_7b();
        let memory = config.calculate_memory_requirements();
        println!("7B (LLaMA-7B):");
        println!(
            "  Parameters: {:.2}B",
            memory.parameter_count as f64 / 1e9
        );
        println!("  Memory: {:.2} GB", memory.total_memory_mb as f64 / 1024.0);
        println!("  Devices: {}", config.num_devices);
        println!("  Mixed precision: {}", config.use_mixed_precision);
        println!(
            "  Gradient checkpointing: {}",
            config.use_gradient_checkpointing
        );
        println!();
    }

    // 13B model
    {
        let config = ModelPresets::model_13b();
        let memory = config.calculate_memory_requirements();
        println!("13B (LLaMA-13B):");
        println!(
            "  Parameters: {:.2}B",
            memory.parameter_count as f64 / 1e9
        );
        println!("  Memory: {:.2} GB", memory.total_memory_mb as f64 / 1024.0);
        println!("  Devices: {}", config.num_devices);
        println!("  Mixed precision: {}", config.use_mixed_precision);
        println!(
            "  Gradient checkpointing: {}",
            config.use_gradient_checkpointing
        );
        println!();
    }

    // 30B model
    {
        let config = ModelPresets::model_30b();
        let memory = config.calculate_memory_requirements();
        println!("30B:");
        println!(
            "  Parameters: {:.2}B",
            memory.parameter_count as f64 / 1e9
        );
        println!("  Memory: {:.2} GB", memory.total_memory_mb as f64 / 1024.0);
        println!("  Devices: {}", config.num_devices);
        println!("  Mixed precision: {}", config.use_mixed_precision);
        println!(
            "  Gradient checkpointing: {}",
            config.use_gradient_checkpointing
        );
        println!();
    }

    // 70B model
    {
        let config = ModelPresets::model_70b();
        let memory = config.calculate_memory_requirements();
        println!("70B (LLaMA-70B):");
        println!(
            "  Parameters: {:.2}B",
            memory.parameter_count as f64 / 1e9
        );
        println!("  Memory: {:.2} GB", memory.total_memory_mb as f64 / 1024.0);
        println!("  Devices: {}", config.num_devices);
        println!("  Mixed precision: {}", config.use_mixed_precision);
        println!(
            "  Gradient checkpointing: {}",
            config.use_gradient_checkpointing
        );
        println!();
    }

    println!("→ Presets provide starting points for common model sizes");
}
