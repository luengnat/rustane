//! Token Analysis Example
//!
//! Analyzes parameter-golf data to learn patterns:
//! - Token frequency distribution
//! - Bigram/trigram statistics
//! - Sequence patterns
//! - Decodes tokens back to text using SentencePiece
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example analyze_tokens --release
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

/// Load tokens from a .bin file
fn load_tokens(path: &str, count: usize) -> std::io::Result<Vec<u16>> {
    let mut file = File::open(path)?;
    let mut tokens = Vec::with_capacity(count);
    let mut buffer = [0u8; 2];

    for _ in 0..count {
        file.read_exact(&mut buffer)?;
        tokens.push(u16::from_le_bytes(buffer));
    }

    Ok(tokens)
}

/// Compute token frequency
fn compute_frequency(tokens: &[u16]) -> HashMap<u16, usize> {
    let mut freq = HashMap::new();
    for &t in tokens {
        *freq.entry(t).or_insert(0) += 1;
    }
    freq
}

/// Compute bigram frequency
fn compute_bigrams(tokens: &[u16]) -> HashMap<(u16, u16), usize> {
    let mut bigrams = HashMap::new();
    for window in tokens.windows(2) {
        *bigrams.entry((window[0], window[1])).or_insert(0) += 1;
    }
    bigrams
}

/// Compute trigram frequency
fn compute_trigrams(tokens: &[u16]) -> HashMap<(u16, u16, u16), usize> {
    let mut trigrams = HashMap::new();
    for window in tokens.windows(3) {
        *trigrams
            .entry((window[0], window[1], window[2]))
            .or_insert(0) += 1;
    }
    trigrams
}

/// Find sequences between special tokens
fn extract_sequences(tokens: &[u16], start_token: u16, end_token: u16) -> Vec<Vec<u16>> {
    let mut sequences = Vec::new();
    let mut current = Vec::new();

    for &t in tokens {
        if t == start_token {
            if !current.is_empty() {
                sequences.push(std::mem::take(&mut current));
            }
        } else if t == end_token && !current.is_empty() {
            sequences.push(std::mem::take(&mut current));
        } else {
            current.push(t);
        }
    }

    if !current.is_empty() {
        sequences.push(current);
    }

    sequences
}

/// Decode tokens using SentencePiece model
fn decode_tokens(tokens: &[u16], model_path: &str) -> Result<String, String> {
    // Use sentencepiece Rust bindings if available, otherwise show raw tokens
    let python_script = format!(
        r#"
import sentencepiece
sp = sentencepiece.SentencePieceProcessor()
sp.Load('{}')
tokens = {:?}
decoded = sp.DecodeIds(list(tokens))
print(decoded)
"#,
        model_path, tokens
    );

    std::fs::write("/tmp/decode_tokens.py", &python_script)
        .map_err(|e| format!("Failed to write temp file: {}", e))?;

    let output = std::process::Command::new("python3")
        .arg("/tmp/decode_tokens.py")
        .output()
        .map_err(|e| format!("Failed to run python: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

fn main() -> std::io::Result<()> {
    println!("=== Token Analysis: Learning from Parameter-Golf Data ===\n");

    let data_dir = PathBuf::from(
        std::env::var("PARAMETER_GOLF_DATA")
            .unwrap_or_else(|_| "/Users/nat/dev/parameter-golf/data".to_string()),
    );

    let dataset_dir = data_dir.join("datasets/fineweb10B_sp1024");
    let tokenizer_model = data_dir.join("tokenizers/fineweb_1024_bpe.model");

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

    println!("Available shards: {}", entries.len());

    // Analyze first shard (100M tokens) - sample 1M for speed
    let first_shard = entries.first().unwrap();
    println!("\nAnalyzing: {}", first_shard.path().display());

    let sample_size = 1_000_000;
    println!("Loading {} tokens...", sample_size);
    let tokens = load_tokens(first_shard.path().to_str().unwrap(), sample_size)?;
    println!("Loaded {} tokens\n", tokens.len());

    // 1. Token frequency analysis
    println!("1. TOKEN FREQUENCY DISTRIBUTION");
    let freq = compute_frequency(&tokens);
    let mut freq_vec: Vec<_> = freq.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("   Unique tokens: {}", freq.len());
    println!("   Top 20 most frequent:");
    for (token, count) in freq_vec.iter().take(20) {
        let pct = (**count as f64 / tokens.len() as f64) * 100.0;
        println!("      Token {:4}: {:6} ({:.2}%)", token, **count, pct);
    }

    // Check special tokens
    println!("\n   Special tokens:");
    let count_0 = *freq.get(&0).unwrap_or(&0);
    let count_1 = *freq.get(&1).unwrap_or(&0);
    let count_2 = *freq.get(&2).unwrap_or(&0);
    println!(
        "      Token 0 (unknown/padding): {} ({:.2}%)",
        count_0,
        (count_0 as f64 / tokens.len() as f64) * 100.0
    );
    println!(
        "      Token 1 (BOS): {} ({:.2}%)",
        count_1,
        (count_1 as f64 / tokens.len() as f64) * 100.0
    );
    println!(
        "      Token 2 (EOS): {} ({:.2}%)",
        count_2,
        (count_2 as f64 / tokens.len() as f64) * 100.0
    );

    // 2. Bigram analysis
    println!("\n2. BIGRAM ANALYSIS");
    let bigrams = compute_bigrams(&tokens);
    let mut bigram_vec: Vec<_> = bigrams.iter().collect();
    bigram_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("   Unique bigrams: {}", bigrams.len());
    println!("   Top 20 most frequent bigrams:");
    for ((t1, t2), count) in bigram_vec.iter().take(20) {
        let pct = (**count as f64 / (tokens.len() - 1) as f64) * 100.0;
        println!("      ({:4}, {:4}): {:6} ({:.2}%)", t1, t2, **count, pct);
    }

    // 3. Trigram analysis
    println!("\n3. TRIGRAM ANALYSIS");
    let trigrams = compute_trigrams(&tokens);
    let mut trigram_vec: Vec<_> = trigrams.iter().collect();
    trigram_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("   Unique trigrams: {}", trigrams.len());
    println!("   Top 10 most frequent trigrams:");
    for ((t1, t2, t3), count) in trigram_vec.iter().take(10) {
        let pct = (**count as f64 / (tokens.len() - 2) as f64) * 100.0;
        println!(
            "      ({:4}, {:4}, {:4}): {:6} ({:.2}%)",
            t1, t2, t3, **count, pct
        );
    }

    // 4. Sequence analysis (between BOS tokens)
    println!("\n4. SEQUENCE ANALYSIS");
    let sequences = extract_sequences(&tokens, 1, 1); // BOS to BOS
    println!("   Sequences found (BOS to BOS): {}", sequences.len());

    if !sequences.is_empty() {
        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let avg_len = lengths.iter().sum::<usize>() as f64 / lengths.len() as f64;
        let min_len = lengths.iter().min().unwrap_or(&0);
        let max_len = lengths.iter().max().unwrap_or(&0);

        println!("   Average sequence length: {:.1} tokens", avg_len);
        println!("   Min sequence length: {}", min_len);
        println!("   Max sequence length: {}", max_len);

        // Show first few sequences (raw tokens)
        println!("\n   First 3 sequences (raw tokens):");
        for (i, seq) in sequences.iter().take(3).enumerate() {
            let preview: Vec<u16> = seq.iter().take(30).copied().collect();
            println!(
                "      Seq {}: {:?}{}",
                i + 1,
                preview,
                if seq.len() > 30 { "..." } else { "" }
            );
        }
    }

    // 5. Decode sample sequences
    println!("\n5. DECODED TEXT SAMPLES");
    if tokenizer_model.exists() {
        println!("   Using tokenizer: {:?}", tokenizer_model);

        // Decode first 100 tokens
        let first_100 = &tokens[..100.min(tokens.len())];
        println!("\n   First 100 tokens decoded:");
        match decode_tokens(first_100, tokenizer_model.to_str().unwrap()) {
            Ok(decoded) => {
                // Truncate long output for display
                let display = if decoded.len() > 200 {
                    format!("{}...", &decoded[..200])
                } else {
                    decoded
                };
                println!("      {}", display.replace('\n', " "));
            }
            Err(e) => {
                println!("      Decode failed: {}", e);
                println!("      Raw tokens: {:?}", first_100);
            }
        }

        // Decode a high-frequency bigram context
        if bigram_vec.len() > 5 {
            let (t1, t2) = *bigram_vec[5].0;
            // Find context around this bigram
            for i in 0..tokens.len().saturating_sub(2) {
                if tokens[i] == t1 && tokens[i + 1] == t2 {
                    let start = i.saturating_sub(5);
                    let end = (i + 15).min(tokens.len());
                    let context = &tokens[start..end];
                    println!("\n   Context around bigram ({}, {}):", t1, t2);
                    match decode_tokens(context, tokenizer_model.to_str().unwrap()) {
                        Ok(decoded) => {
                            let display = decoded.replace('\n', " ").replace('\r', " ");
                            println!("      {}", display);
                        }
                        Err(_) => {
                            println!("      Raw: {:?}", context);
                        }
                    }
                    break;
                }
            }
        }
    } else {
        println!("   Tokenizer model not found, showing raw tokens only");
    }

    // 6. Entropy analysis
    println!("\n6. ENTROPY ANALYSIS");
    let entropy: f64 = freq
        .values()
        .map(|&count| {
            let p = count as f64 / tokens.len() as f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum();
    println!("   Shannon entropy: {:.4} bits/token", entropy);
    println!(
        "   Max entropy (uniform): {:.4} bits/token",
        (freq.len() as f64).log2()
    );
    println!(
        "   Entropy efficiency: {:.2}%",
        (entropy / (freq.len() as f64).log2()) * 100.0
    );

    println!("\n=== Analysis Complete ===");
    println!("\nKey findings:");
    println!("- Token distribution follows Zipf's law (few very common, many rare)");
    println!("- BOS tokens mark document/sentence boundaries");
    println!("- Bigram patterns reveal common word pairs and phrases");
    println!("- Entropy indicates information density of the tokenized text");

    Ok(())
}
