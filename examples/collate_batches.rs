//! Example: Using collators to prepare batches
//!
//! This example demonstrates different batching strategies:
//! 1. Padding sequences to a fixed length
//! 2. Truncating long sequences and padding short ones

use rustane::{
    Collator, DataLoader, Dataset, PadCollator, RandomSampler, SequentialDataset,
    SequentialSampler, TruncateCollator,
};

fn main() -> rustane::Result<()> {
    println!("Rustane Collation Example");
    println!("=========================\n");

    // Create a dataset with variable-length sequences
    println!("Creating dataset with variable-length sequences...");
    let samples = vec![
        vec![10, 11, 12],
        vec![20, 21, 22, 23, 24, 25],
        vec![30, 31],
        vec![40, 41, 42, 43],
        vec![50],
    ];
    let dataset = SequentialDataset::new(samples.clone());
    println!("Dataset created with {} samples", dataset.len());
    for (i, sample) in samples.iter().enumerate() {
        println!("  Sample {}: length={}", i, sample.len());
    }
    println!();

    // Example 1: PadCollator with fixed sequence length
    println!("Example 1: PadCollator (max_len=6, pad_token=0)");
    println!("----------------------------------------------");
    let sampler = SequentialSampler::new(dataset.len());
    let collator = PadCollator::new(6, 0);
    let batch = collator.collate(
        (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect(),
    )?;

    let (batch_size, seq_len) = batch.shape();
    println!("Batch shape: ({}, {})", batch_size, seq_len);
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

    // Example 2: TruncateCollator with truncation and padding
    println!("Example 2: TruncateCollator (max_len=4, pad_token=99)");
    println!("----------------------------------------------------");
    let collator = TruncateCollator::new(4, 99);
    let batch = collator.collate(
        (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect(),
    )?;

    let (batch_size, seq_len) = batch.shape();
    println!("Batch shape: ({}, {})", batch_size, seq_len);
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

    // Example 3: Using collators with DataLoader for batching
    println!("Example 3: Collators with DataLoader");
    println!("-----------------------------------");
    let dataset =
        SequentialDataset::new(vec![vec![1, 2], vec![3, 4, 5, 6], vec![7], vec![8, 9, 10]]);
    let sampler = SequentialSampler::new(4);
    let collator = PadCollator::new(4, 0);

    println!("Creating batch samples with padding...");
    let samples: Vec<_> = (0..4).map(|i| dataset.get(i).unwrap()).collect();
    println!("Raw samples:");
    for (i, s) in samples.iter().enumerate() {
        println!("  {}: {:?}", i, s);
    }

    let batch = collator.collate(samples)?;
    println!("\nAfter collation (PadCollator, max_len=4, pad_token=0):");
    for sample_idx in 0..batch.batch_size() {
        print!("  Sample {}: [", sample_idx);
        for seq_idx in 0..batch.seq_len() {
            if let Some(token) = batch.get(sample_idx, seq_idx) {
                print!("{}", token);
                if seq_idx < batch.seq_len() - 1 {
                    print!(", ");
                }
            }
        }
        println!("]");
    }

    println!();
    println!("Example completed successfully!");
    Ok(())
}
