//! Example: RandomSampler for training with shuffled epochs
//!
//! Demonstrates how to use RandomSampler for creating shuffled training batches,
//! which is essential for stochastic gradient descent convergence.

use rustane::{
    Collator, DataLoader, Dataset, PadCollator, RandomSampler, Sampler, SequentialDataset,
    SequentialSampler,
};

fn main() -> rustane::Result<()> {
    println!("Rustane Random Sampling Example");
    println!("===============================\n");

    // Create a simple dataset
    let samples = vec![
        vec![1, 1, 1],  // Sample 0
        vec![2, 2, 2],  // Sample 1
        vec![3, 3, 3],  // Sample 2
        vec![4, 4, 4],  // Sample 3
        vec![5, 5, 5],  // Sample 4
    ];
    let dataset = SequentialDataset::new(samples.clone());
    println!("Dataset: {} samples", dataset.len());
    for (i, s) in samples.iter().enumerate() {
        println!("  Sample {}: {:?}", i, s);
    }
    println!();

    // Example 1: Sequential sampling (deterministic, for validation)
    println!("Example 1: Sequential Sampler (for validation)");
    println!("---------------------------------------------");
    let mut sampler = SequentialSampler::new(dataset.len());
    let indices = sampler.sample();
    println!("Indices: {:?}", indices);
    println!("Order: same every time (deterministic)\n");

    // Example 2: Random sampling with fixed seed (reproducible training)
    println!("Example 2: Random Sampler with seed=42 (reproducible)");
    println!("-----------------------------------------------------");
    let mut sampler1 = RandomSampler::new(dataset.len(), 42);
    let indices1 = sampler1.sample();
    println!("Epoch 1 indices: {:?}", indices1);

    let mut sampler2 = RandomSampler::new(dataset.len(), 42);
    let indices2 = sampler2.sample();
    println!("Epoch 2 indices: {:?}", indices2);
    println!("Note: Same seed produces identical shuffle order\n");

    // Example 3: Different seeds produce different shuffles
    println!("Example 3: Different seeds produce different shuffles");
    println!("-----------------------------------------------------");
    let mut sampler_seed_100 = RandomSampler::new(dataset.len(), 100);
    let indices_100 = sampler_seed_100.sample();
    println!("Seed 100 indices: {:?}", indices_100);

    let mut sampler_seed_200 = RandomSampler::new(dataset.len(), 200);
    let indices_200 = sampler_seed_200.sample();
    println!("Seed 200 indices: {:?}", indices_200);
    println!("Note: Different seeds produce different shuffles\n");

    // Example 4: Simulating training with shuffled batches
    println!("Example 4: Training simulation with shuffled batches");
    println!("---------------------------------------------------");
    let collator = PadCollator::new(3, 0);

    for epoch in 0..3 {
        println!("\nEpoch {}:", epoch);
        let mut sampler = RandomSampler::new(dataset.len(), epoch as u64 + 42);
        let indices = sampler.sample();

        // Simulate batch processing
        let batch_size = 2;
        for batch_idx in (0..dataset.len()).step_by(batch_size) {
            let end = (batch_idx + batch_size).min(dataset.len());
            let batch_indices = &indices[batch_idx..end];

            print!("  Batch indices: {:?} → samples: [", batch_indices);
            for idx in batch_indices {
                print!("{}", idx);
                if idx != batch_indices.last().unwrap() {
                    print!(", ");
                }
            }
            println!("]");
        }
    }

    println!("\n✓ Example completed successfully!");
    println!("\nKey takeaways:");
    println!("  • Use SequentialSampler for validation (consistent order)");
    println!("  • Use RandomSampler for training (shuffled data)");
    println!("  • Same seed ensures reproducible training runs");
    println!("  • Different seeds per epoch prevent overfitting");

    Ok(())
}
