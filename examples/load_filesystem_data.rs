//! Example: Loading token sequences from files
//!
//! Demonstrates different file format loaders:
//! 1. JSONL format (one JSON array per line)
//! 2. Space-separated format (tokens separated by spaces)
//! 3. Comma-separated format (tokens separated by commas)

use rustane::{
    Collator, DataLoader, Dataset, JsonlDataset, PadCollator, SequentialDataset,
    SequentialSampler, TextDataset,
};
use std::io::Write;
use tempfile::NamedTempFile;

fn main() -> rustane::Result<()> {
    println!("Rustane Filesystem Dataset Example");
    println!("==================================\n");

    // Example 1: Create and load JSONL dataset
    println!("Example 1: JSONL Format");
    println!("----------------------");
    let mut jsonl_file = NamedTempFile::new().map_err(|e| {
        rustane::Error::Io(format!("Failed to create temp file: {}", e))
    })?;

    writeln!(jsonl_file, "[10, 20, 30, 40]").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(jsonl_file, "[11, 21, 31, 41]").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(jsonl_file, "[12, 22, 32, 42]").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;

    let jsonl_dataset = JsonlDataset::load(jsonl_file.path())?;
    println!("Loaded JSONL dataset: {} samples", jsonl_dataset.len());
    for i in 0..jsonl_dataset.len() {
        println!("  Sample {}: {:?}", i, jsonl_dataset.get(i)?);
    }
    println!();

    // Example 2: Create and load space-separated dataset
    println!("Example 2: Space-Separated Format");
    println!("--------------------------------");
    let mut text_file = NamedTempFile::new().map_err(|e| {
        rustane::Error::Io(format!("Failed to create temp file: {}", e))
    })?;

    writeln!(text_file, "1 2 3 4 5").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(text_file, "6 7 8 9 10").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(text_file, "11 12 13").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;

    let text_dataset = TextDataset::load_space_separated(text_file.path())?;
    println!("Loaded space-separated dataset: {} samples", text_dataset.len());
    for i in 0..text_dataset.len() {
        println!("  Sample {}: {:?}", i, text_dataset.get(i)?);
    }
    println!();

    // Example 3: Create and load comma-separated dataset
    println!("Example 3: Comma-Separated Format");
    println!("--------------------------------");
    let mut csv_file = NamedTempFile::new().map_err(|e| {
        rustane::Error::Io(format!("Failed to create temp file: {}", e))
    })?;

    writeln!(csv_file, "100,101,102,103").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(csv_file, "110, 111, 112, 113").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?; // With spaces

    let csv_dataset = TextDataset::load_comma_separated(csv_file.path())?;
    println!("Loaded comma-separated dataset: {} samples", csv_dataset.len());
    for i in 0..csv_dataset.len() {
        println!("  Sample {}: {:?}", i, csv_dataset.get(i)?);
    }
    println!();

    // Example 4: Using filesystem dataset with DataLoader and collator
    println!("Example 4: Full Pipeline (Load → Sample → Batch)");
    println!("-----------------------------------------------");

    // Create a dataset
    let mut data_file = NamedTempFile::new().map_err(|e| {
        rustane::Error::Io(format!("Failed to create temp file: {}", e))
    })?;

    writeln!(data_file, "[1, 2]").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(data_file, "[3, 4, 5, 6]").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(data_file, "[7]").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;
    writeln!(data_file, "[8, 9, 10]").map_err(|e| {
        rustane::Error::Io(format!("Failed to write: {}", e))
    })?;

    let dataset = JsonlDataset::load(data_file.path())?;
    println!("Dataset loaded from file: {} samples", dataset.len());

    // Create sampler and collator
    let sampler = SequentialSampler::new(dataset.len());
    let collator = PadCollator::new(4, 0);

    // Prepare batches
    println!("Creating batches with PadCollator (seq_len=4, pad_token=0):");
    let batch_size = 2;
    for batch_start in (0..dataset.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(dataset.len());
        let samples: Vec<Vec<u32>> = (batch_start..batch_end)
            .map(|i| dataset.get(i).unwrap())
            .collect();

        let batch = collator.collate(samples)?;
        println!(
            "  Batch [{}..{}]: shape={:?}",
            batch_start,
            batch_end,
            batch.shape()
        );
        for sample_idx in 0..batch.batch_size() {
            print!("    Sample {}: [", sample_idx);
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
    }

    println!("\n✓ Example completed successfully!");
    println!("\nSupported formats:");
    println!("  • JSONL: [0, 1, 2, 3] (one JSON array per line)");
    println!("  • Space-separated: 0 1 2 3");
    println!("  • Comma-separated: 0,1,2,3");

    Ok(())
}
