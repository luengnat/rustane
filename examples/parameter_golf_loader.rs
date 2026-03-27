//! Parameter-Golf Data Loader Example
//!
//! Demonstrates streaming data loading from parameter-golf shards:
//! - TokenStream: sequential reading with wrap-around
//! - DistributedTokenLoader: multi-rank batch generation
//! - Shard header validation
//!
//! ## Usage
//!
//! ```bash
//! # Simple streaming
//! cargo run --example parameter_golf_loader --release
//!
//! # With custom data path
//! PARAMETER_GOLF_DATA=/path/to/data cargo run --example parameter_golf_loader --release
//! ```

use rustane::data::{load_shard, BatchConfig, DistributedTokenLoader, ShardHeader, TokenStream};
use std::path::PathBuf;

fn main() -> rustane::Result<()> {
    println!("=== Parameter-Golf Data Loader Demo ===\n");

    let data_dir = PathBuf::from(
        std::env::var("PARAMETER_GOLF_DATA")
            .unwrap_or_else(|_| "/Users/nat/dev/parameter-golf/data".to_string()),
    );

    let pattern = data_dir
        .join("datasets/fineweb10B_sp1024/fineweb_train_*.bin")
        .to_string_lossy()
        .to_string();

    println!("Data pattern: {}", pattern);

    // 1. Shard header validation
    println!("\n1. Shard Header Validation");
    let shards = glob::glob(&pattern)
        .map_err(|e| rustane::Error::Io(format!("Glob error: {}", e)))?
        .filter_map(|r| r.ok())
        .take(3)
        .collect::<Vec<_>>();

    for shard_path in &shards {
        match ShardHeader::from_file(shard_path) {
            Ok(header) => {
                println!(
                    "   {}: magic={} version={} tokens={}",
                    shard_path.file_name().unwrap().to_string_lossy(),
                    header.magic,
                    header.version,
                    header.num_tokens
                );
            }
            Err(e) => {
                println!("   {}: Error - {}", shard_path.display(), e);
            }
        }
    }

    // 2. Load a single shard
    println!("\n2. Load Single Shard");
    if let Some(first_shard) = shards.first() {
        let tokens = load_shard(first_shard)?;
        println!(
            "   Loaded {} tokens from {}",
            tokens.len(),
            first_shard.display()
        );
        println!("   First 20 tokens: {:?}", &tokens[..20.min(tokens.len())]);

        // Count special tokens
        let bos_count = tokens.iter().filter(|&&t| t == 1).count();
        let eos_count = tokens.iter().filter(|&&t| t == 2).count();
        let pad_count = tokens.iter().filter(|&&t| t == 0).count();
        println!(
            "   BOS (1): {}, EOS (2): {}, PAD (0): {}",
            bos_count, eos_count, pad_count
        );
    }

    // 3. TokenStream: sequential streaming
    println!("\n3. TokenStream Demo (sequential streaming)");
    let mut stream = TokenStream::new(&pattern)?;
    println!("   Total shards: {}", stream.num_shards());
    println!("   Total tokens: {}", stream.total_tokens()?);

    // Take a sample of tokens
    let sample = stream.take(1000)?;
    println!("   Sampled {} tokens", sample.len());
    println!(
        "   Position after take: shard {}, offset {}",
        stream.position().0,
        stream.position().1
    );

    // 4. DistributedTokenLoader: batch generation
    println!("\n4. DistributedTokenLoader Demo (batch generation)");

    // Simulate single-rank training (world_size=1)
    let config = BatchConfig::new(
        8192, // global_batch_tokens
        1024, // seq_len
        1,    // grad_accum_steps
        1,    // world_size
        0,    // rank
    );

    let mut loader = DistributedTokenLoader::new(&pattern, config.clone())?;
    println!(
        "   Config: batch_tokens={}, seq_len={}, grad_accum={}",
        config.global_batch_tokens, config.seq_len, config.grad_accum_steps
    );
    println!(
        "   Tokens per rank per accum: {}",
        config.tokens_per_rank_accum()
    );

    // Get a few batches
    for i in 0..3 {
        match loader.next_batch() {
            Ok((input, target)) => {
                println!(
                    "   Batch {}: input_shape=[{}, {}], target_shape=[{}, {}]",
                    i + 1,
                    input.len() / config.seq_len,
                    config.seq_len,
                    target.len() / config.seq_len,
                    config.seq_len
                );

                // Verify x,y shift relationship
                if input.len() == target.len() && input.len() > 0 {
                    println!(
                        "      First 5 input tokens:  {:?}",
                        &input[..5.min(input.len())]
                    );
                    println!(
                        "      First 5 target tokens: {:?}",
                        &target[..5.min(target.len())]
                    );
                }
            }
            Err(e) => {
                println!("   Batch {}: Error - {}", i + 1, e);
                break;
            }
        }
    }

    // 5. Multi-rank simulation
    println!("\n5. Multi-Rank Simulation (world_size=4)");
    let multi_rank_config = BatchConfig::new(
        32768, // global_batch_tokens
        1024,  // seq_len
        8,     // grad_accum_steps
        4,     // world_size
        0,     // rank (we simulate rank 0)
    );

    println!(
        "   Tokens per rank per accum: {}",
        multi_rank_config.tokens_per_rank_accum()
    );
    println!(
        "   Span per rank (+1 for x,y): {}",
        multi_rank_config.span_per_rank()
    );

    let mut multi_loader = DistributedTokenLoader::new(&pattern, multi_rank_config.clone())?;
    if let Ok((input, target)) = multi_loader.next_batch() {
        println!(
            "   Batch for rank 0: {} input tokens, {} target tokens",
            input.len(),
            target.len()
        );
    }

    // 6. Performance stats
    println!("\n6. Performance Notes");
    println!("   - TokenStream reads shards sequentially, wraps around forever");
    println!("   - No shuffling, no sampling (deterministic like train_gpt.py)");
    println!("   - Each rank gets a disjoint slice of the token stream");
    println!("   - +1 token span allows x,y construction via shift");

    println!("\n=== Demo Complete ===");
    println!("\nNext steps:");
    println!("1. Integrate DistributedTokenLoader with training loop");
    println!("2. Use gradient accumulation for larger effective batches");
    println!("3. Scale to multi-GPU with DDP (one loader per rank)");

    Ok(())
}
