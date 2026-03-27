//! Data Pattern Analysis for Rustane Training
//!
//! Analyzes parameter-golf data to extract training-relevant insights:
//! - Shard header format (256 int32 header + uint16 tokens)
//! - Token distribution statistics for learning rate tuning
//! - Sequence length distribution for batch sizing
//! - Validation set characteristics
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example data_patterns --release
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

/// Shard header magic number from train_gpt.py
const SHARD_MAGIC: i32 = 20240520;
/// Shard header version
const SHARD_VERSION: i32 = 1;
/// Header size in int32s
const HEADER_SIZE: usize = 256;

/// Parse shard header
fn parse_shard_header(path: &str) -> std::io::Result<(i32, i32, i32)> {
    let mut file = File::open(path)?;
    let mut header = [0i32; HEADER_SIZE];

    // Read header bytes
    let mut header_bytes = [0u8; HEADER_SIZE * 4];
    file.read_exact(&mut header_bytes)?;

    // Convert to i32 (little-endian as per numpy "<i4")
    for i in 0..HEADER_SIZE {
        header[i] = i32::from_le_bytes([
            header_bytes[i * 4],
            header_bytes[i * 4 + 1],
            header_bytes[i * 4 + 2],
            header_bytes[i * 4 + 3],
        ]);
    }

    Ok((header[0], header[1], header[2]))
}

/// Load tokens from shard (with proper header skipping)
fn load_shard_tokens(path: &str, count: usize) -> std::io::Result<Vec<u16>> {
    let mut file = File::open(path)?;

    // Skip header (256 * 4 = 1024 bytes)
    file.seek(SeekFrom::Start(HEADER_SIZE as u64 * 4))?;

    let mut tokens = Vec::with_capacity(count);
    let mut buffer = [0u8; 2];

    for _ in 0..count {
        file.read_exact(&mut buffer)?;
        tokens.push(u16::from_le_bytes(buffer));
    }

    Ok(tokens)
}

/// Analyze sequence lengths between BOS tokens
fn analyze_sequence_lengths(tokens: &[u16], bos_token: u16) -> Vec<usize> {
    let mut lengths = Vec::new();
    let mut current_len = 0;

    for &t in tokens {
        if t == bos_token {
            if current_len > 0 {
                lengths.push(current_len);
            }
            current_len = 0;
        } else {
            current_len += 1;
        }
    }

    if current_len > 0 {
        lengths.push(current_len);
    }

    lengths
}

/// Compute percentile
fn percentile(sorted: &[usize], p: f64) -> usize {
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[idx]
}

/// Analyze token co-occurrence patterns
fn analyze_co_occurrence(tokens: &[u16], window: usize) -> HashMap<(u16, u16), usize> {
    let mut cooccur = HashMap::new();

    for i in 0..tokens.len().saturating_sub(window) {
        for j in 1..=window.min(tokens.len() - i - 1) {
            let key = (tokens[i].min(tokens[i + j]), tokens[i].max(tokens[i + j]));
            *cooccur.entry(key).or_insert(0) += 1;
        }
    }

    cooccur
}

fn main() -> std::io::Result<()> {
    println!("=== Data Pattern Analysis for Rustane ===\n");
    println!("Extracting training-relevant insights from parameter-golf data\n");

    let data_dir = PathBuf::from(
        std::env::var("PARAMETER_GOLF_DATA")
            .unwrap_or_else(|_| "/Users/nat/dev/parameter-golf/data".to_string()),
    );

    let dataset_dir = data_dir.join("datasets/fineweb10B_sp1024");

    if !dataset_dir.exists() {
        println!("Dataset not found at {:?}", dataset_dir);
        return Ok(());
    }

    // Get list of shards
    let mut entries: Vec<_> = std::fs::read_dir(&dataset_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "bin")
                .unwrap_or(false)
        })
        .collect();
    entries.sort_by_key(|e| e.path());

    println!("1. SHARD STRUCTURE ANALYSIS");
    println!("   Total shards: {}", entries.len());

    // Analyze first few shard headers
    println!("\n   Shard header validation:");
    for entry in entries.iter().take(3) {
        match parse_shard_header(entry.path().to_str().unwrap()) {
            Ok((magic, version, num_tokens)) => {
                let valid = magic == SHARD_MAGIC && version == SHARD_VERSION;
                println!(
                    "      {}: magic={} version={} tokens={} {}",
                    entry.path().file_name().unwrap().to_string_lossy(),
                    magic,
                    version,
                    num_tokens,
                    if valid { "✓" } else { "✗ INVALID" }
                );
            }
            Err(e) => {
                println!(
                    "      {}: Error - {}",
                    entry.path().file_name().unwrap().to_string_lossy(),
                    e
                );
            }
        }
    }

    // Compute total dataset size
    let total_tokens: u64 = entries
        .iter()
        .filter_map(|e| parse_shard_header(e.path().to_str().unwrap()).ok())
        .map(|(_, _, n)| n as u64)
        .sum();

    println!("\n   Dataset totals:");
    println!("      Total tokens: {}", total_tokens);
    println!("      Total bytes: {} MB", total_tokens * 2 / (1024 * 1024));
    println!("      Expected tokens (manifest): 19,535,223,186");

    // Sample analysis from first shard
    let first_shard = entries.first().unwrap();
    println!(
        "\n2. TOKEN DISTRIBUTION (sample: {})",
        first_shard.path().display()
    );

    let sample = load_shard_tokens(first_shard.path().to_str().unwrap(), 500_000)?;
    println!("   Loaded {} tokens\n", sample.len());

    // Token frequency
    let mut freq = HashMap::new();
    for &t in &sample {
        *freq.entry(t).or_insert(0) += 1;
    }

    // Rank-frequency analysis (Zipf's law)
    let mut freqs: Vec<_> = freq.values().collect();
    freqs.sort_by(|a, b| b.cmp(a));

    println!("   Zipf's Law Analysis:");
    println!("      Unique tokens: {}", freq.len());
    println!("      Most common token: {} occurrences", freqs[0]);
    println!(
        "      10th most common: {} occurrences",
        freqs.get(9).unwrap_or(&&0)
    );
    println!(
        "      100th most common: {} occurrences",
        freqs.get(99).unwrap_or(&&0)
    );

    // Power law exponent estimation
    if freqs.len() > 100 {
        let rank1 = *freqs[0] as f64;
        let rank10 = *freqs[9] as f64;
        let estimated_alpha = (rank1 / rank10).ln() / 10.0_f64.ln();
        println!("      Estimated Zipf exponent (α): {:.3}", estimated_alpha);
        println!("         (α ≈ 1.0 is typical for natural language)");
    }

    // Sequence length analysis
    println!("\n3. SEQUENCE LENGTH DISTRIBUTION (BOS-delimited)");
    let seq_lengths = analyze_sequence_lengths(&sample, 1); // BOS=1

    if !seq_lengths.is_empty() {
        let mut sorted_lengths = seq_lengths.clone();
        sorted_lengths.sort();

        let mean = sorted_lengths.iter().sum::<usize>() as f64 / sorted_lengths.len() as f64;
        let median = percentile(&sorted_lengths, 50.0);
        let p95 = percentile(&sorted_lengths, 95.0);
        let p99 = percentile(&sorted_lengths, 99.0);
        let max = sorted_lengths.last().unwrap_or(&0);

        println!("   Sequences analyzed: {}", sorted_lengths.len());
        println!("   Mean length: {:.1} tokens", mean);
        println!("   Median length: {} tokens", median);
        println!("   95th percentile: {} tokens", p95);
        println!("   99th percentile: {} tokens", p99);
        println!("   Maximum length: {} tokens", max);

        // Length bucket distribution
        println!("\n   Length buckets:");
        let buckets = [
            (0..32, "1-31"),
            (32..64, "32-63"),
            (64..128, "64-127"),
            (128..256, "128-255"),
            (256..512, "256-511"),
            (512..1024, "512-1023"),
            (1024..usize::MAX, "1024+"),
        ];

        for (range, label) in buckets.iter() {
            let count = seq_lengths.iter().filter(|&&l| range.contains(&l)).count();
            let pct = count as f64 / seq_lengths.len() as f64 * 100.0;
            println!("      {:>8}: {:5} ({:.1}%)", label, count, pct);
        }

        // Implications for training
        println!("\n   Training implications:");
        if median < 512 {
            println!(
                "      - Median sequence ({}) < 512: consider shorter context windows",
                median
            );
        }
        if p95 > 1024 {
            println!(
                "      - 95th percentile ({}) > 1024: may need attention mask tuning",
                p95
            );
        }
        if p99 > 2048 {
            println!(
                "      - Long tail (p99={}) suggests Flash Attention benefit",
                p99
            );
        }
    }

    // Co-occurrence analysis
    println!("\n4. TOKEN CO-OCCURRENCE (window=5)");
    let cooccur = analyze_co_occurrence(&sample, 5);
    let mut cooccur_vec: Vec<_> = cooccur.iter().collect();
    cooccur_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("   Unique co-occurring pairs: {}", cooccur.len());
    println!("   Top 10 pairs:");
    for ((t1, t2), count) in cooccur_vec.iter().take(10) {
        println!("      ({:4}, {:4}): {}", t1, t2, *count);
    }

    // Learning rate implications
    println!("\n5. LEARNING RATE IMPLICATIONS");
    let unique_ratio = freq.len() as f64 / 1024.0 * 100.0;
    println!(
        "   Vocab utilization: {:.1}% ({} / 1024 tokens used)",
        unique_ratio,
        freq.len()
    );

    if unique_ratio > 80.0 {
        println!("   - High vocab utilization: embedding LR should be higher");
    } else if unique_ratio < 50.0 {
        println!("   - Low vocab utilization: many tokens are rare, consider lower embedding LR");
    }

    // Entropy for loss baseline
    let entropy: f64 = freq
        .values()
        .map(|&count| {
            let p = count as f64 / sample.len() as f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum();

    let max_entropy = (freq.len() as f64).log2();
    println!("\n6. BASELINE METRICS");
    println!("   Token entropy: {:.4} bits", entropy);
    println!("   Max entropy: {:.4} bits", max_entropy);
    println!(
        "   Expected cross-entropy loss (uniform model): {:.4}",
        (1.0 / freq.len() as f64).ln().abs()
    );
    println!(
        "   Expected cross-entropy loss (optimal model): {:.4}",
        entropy / std::f64::consts::LN_2
    );

    // Comparison to train_gpt.py defaults
    println!("\n7. COMPARISON TO train_gpt.py DEFAULTS");
    println!("   train_gpt.py default config:");
    println!("      - Model dim: 416, Heads: 8, KV heads: 4");
    println!("      - Layers: 11, MLP mult: 2");
    println!("      - Train seq len: 1024, Batch tokens: 524,288");
    println!("      - Warmup: 20 steps, Iterations: 20,000");

    // Capture median for comparison (from sequence analysis above)
    let training_median = if !seq_lengths.is_empty() {
        let mut sorted = seq_lengths.clone();
        sorted.sort();
        percentile(&sorted, 50.0)
    } else {
        0
    };

    println!("\n   Recommended adjustments based on data:");
    if training_median < 256 && training_median > 0 {
        println!(
            "      - Consider train_seq_len=512 (median seq is {}) for efficiency",
            training_median
        );
    }
    if unique_ratio < 90.0 {
        println!("      - Vocab is underutilized: could reduce to 512 or use larger vocab");
    }
    println!("      - Batch size should accommodate median seq * num_gpus");

    println!("\n=== Analysis Complete ===");

    Ok(())
}
