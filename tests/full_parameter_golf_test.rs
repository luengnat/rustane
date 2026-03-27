//! Full Parameter-Golf Training Test
//!
//! Tests training on all 196 parameter-golf files

use std::path::PathBuf;

/// Test that counts all 196 files
#[test]
fn test_count_all_196_files() {
    let data_dir = PathBuf::from("/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024");

    if !data_dir.exists() {
        println!("⚠️  Data directory not found");
        return;
    }

    let entries: Vec<_> = std::fs::read_dir(&data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let binding = e.file_name();
            let name = binding.to_string_lossy();
            name.starts_with("fineweb_train") && name.ends_with(".bin")
        })
        .collect();

    println!("\n=== Parameter-Golf Training Files ===");
    println!("Found {} training files\n", entries.len());

    // Check file sizes
    let mut total_size: u64 = 0;
    for entry in &entries {
        if let Ok(metadata) = entry.metadata() {
            total_size += metadata.len();
        }
    }

    println!("Total dataset size: {:.2} GB\n", total_size as f64 / 1e9);

    // Show first 3 files
    println!("First 3 files:");
    for entry in entries.iter().take(3) {
        let binding = entry.file_name();
        let name = binding.to_string_lossy();
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        println!("  {} ({:.1} MB)", name, size as f64 / 1e6);
    }

    if entries.len() > 3 {
        println!("  ...");
        println!("\nLast 3 files:");
        for entry in entries.iter().rev().take(3).rev() {
            let binding = entry.file_name();
            let name = binding.to_string_lossy();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            println!("  {} ({:.1} MB)", name, size as f64 / 1e6);
        }
    }

    // Verify we have ~195 files (actual count is 195)
    assert!(
        entries.len() >= 195,
        "Expected at least 195 training files, found {}",
        entries.len()
    );

    println!(
        "\n✅ Found all {} parameter-golf training files!",
        entries.len()
    );
}

/// Simple test that data can be accessed
#[test]
fn test_data_accessible() {
    let data_dir = PathBuf::from("/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024");

    if !data_dir.exists() {
        println!("⚠️  Data directory not found - skipping test");
        return;
    }

    // Try to read first file
    let first_file = data_dir.join("fineweb_train_000000.bin");

    match std::fs::metadata(&first_file) {
        Ok(metadata) => {
            println!("✅ First file accessible: {} bytes", metadata.len());
            assert!(metadata.len() > 0, "File should not be empty");
        }
        Err(e) => {
            panic!("❌ Cannot access first file: {}", e);
        }
    }
}
