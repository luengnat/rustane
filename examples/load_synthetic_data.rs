//! Example: Loading and batching synthetic data
//!
//! This example demonstrates the basic data loading pipeline:
//! 1. Create an in-memory dataset from synthetic token sequences
//! 2. Create a sequential sampler
//! 3. Create a dataloader with a specific batch size
//! 4. Iterate over batches and print batch information

use rustane::{DataLoader, Dataset, SequentialDataset, SequentialSampler};

fn main() -> rustane::Result<()> {
    println!("Rustane Data Loading Example");
    println!("============================\n");

    // Step 1: Create synthetic dataset
    println!("Creating synthetic dataset...");
    let samples = vec![
        vec![0, 1, 2, 3, 4],
        vec![5, 6, 7, 8, 9],
        vec![10, 11, 12, 13, 14],
        vec![15, 16, 17, 18, 19],
        vec![20, 21, 22, 23, 24],
        vec![25, 26, 27, 28, 29],
    ];

    let dataset = SequentialDataset::new(samples);
    println!("Dataset created with {} samples", dataset.len());
    println!(
        "Sample 0: {:?}",
        dataset.get(0).expect("Failed to get sample 0")
    );
    println!(
        "Sample 5: {:?}\n",
        dataset.get(5).expect("Failed to get sample 5")
    );

    // Step 2: Create sequential sampler
    println!("Creating sequential sampler...");
    let sampler = SequentialSampler::new(dataset.len());
    println!("Sampler created for {} samples\n", dataset.len());

    // Step 3: Create dataloader with batch_size=2
    println!("Creating dataloader with batch_size=2...");
    let dataloader = DataLoader::new(dataset, sampler, 2)?;
    println!("Dataloader created\n");

    // Step 4: Iterate over batches
    println!("Iterating over batches:");
    println!("------------------------");

    for (batch_idx, batch_result) in dataloader.iter().enumerate() {
        let batch = batch_result?;
        let (batch_size, seq_len) = batch.shape();

        println!("Batch {}: shape=({}, {})", batch_idx, batch_size, seq_len);
        println!("  Tokens: {:?}", batch.tokens());

        // Print individual samples within the batch
        for sample_idx in 0..batch_size {
            print!("  Sample {} tokens: [", sample_idx);
            for seq_idx in 0..seq_len {
                if let Some(token) = batch.get(sample_idx, seq_idx) {
                    print!("{}", token);
                    if seq_idx < seq_len - 1 {
                        print!(", ");
                    }
                }
            }
            println!("]");
        }
        println!();
    }

    println!("Example completed successfully!");
    Ok(())
}
