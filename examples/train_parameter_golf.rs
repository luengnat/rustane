//! Train a transformer on Parameter-Golf FineWeb data using ANE/CPU.
//!
//! Loads the pre-tokenized FineWeb 10B shards from `~/dev/parameter-golf/data/datasets/fineweb10B_sp1024/`
//! and trains a small GPT model using the rustane training pipeline.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example train_parameter_golf
//!
//! # Quick smoke test (2 shards, 50 steps):
//! cargo run --example train_parameter_golf -- --max-shards 2 --steps 50
//!
//! # Custom architecture:
//! cargo run --example train_parameter_golf -- --layers 8 --dim 256 --heads 8 --hidden-mult 4
//! ```

use rustane::{
    data::{DataLoader, Dataset, RandomSampler, ShardTokenDataset},
    training::{
        AdamWOptimizer, CrossEntropyLoss, TrainerBuilder, TransformerANE, TransformerConfig,
        WarmupLinearScheduler,
    },
    Result,
};
use std::path::PathBuf;
use std::time::Instant;

/// Architecture presets matching parameter-golf submissions
struct ArchPreset {
    name: &'static str,
    layers: usize,
    dim: usize,
    heads: usize,
    hidden_mult: usize, // hidden_dim = dim * hidden_mult
}

const PRESETS: &[ArchPreset] = &[
    ArchPreset {
        name: "baseline-6l-256d",
        layers: 6,
        dim: 256,
        heads: 8,
        hidden_mult: 2,
    },
    ArchPreset {
        name: "compact-12l-208d",
        layers: 12,
        dim: 208,
        heads: 8,
        hidden_mult: 4,
    },
    ArchPreset {
        name: "wide-8l-384d",
        layers: 8,
        dim: 384,
        heads: 8,
        hidden_mult: 2,
    },
    ArchPreset {
        name: "deep-15l-192d",
        layers: 15,
        dim: 192,
        heads: 6,
        hidden_mult: 3,
    },
];

fn find_preset(name: &str) -> Option<&'static ArchPreset> {
    PRESETS.iter().find(|p| p.name == name)
}

fn default_data_dir() -> PathBuf {
    dirs_home()
        .join("dev")
        .join("parameter-golf")
        .join("data")
        .join("datasets")
        .join("fineweb10B_sp1024")
}

fn dirs_home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
}

fn main() -> Result<()> {
    // Parse args from environment or command-line
    let args = parse_args();

    println!("=== Parameter-Golf ANE Training ===");
    println!();

    // ── Step 1: Load data ─────────────────────────────────────────────────
    println!("Step 1: Loading FineWeb shards");
    println!("------------------------------");

    let shard_pattern = args
        .data_dir
        .join("fineweb_train_*.bin")
        .to_string_lossy()
        .to_string();

    println!("  Pattern:   {}", shard_pattern);
    println!("  Max shards: {:?}", args.max_shards);
    println!("  Seq len:   {}", args.seq_len);

    let load_start = Instant::now();
    let dataset = ShardTokenDataset::from_pattern(&shard_pattern, args.seq_len, args.max_shards)?;
    let load_time = load_start.elapsed();

    println!(
        "  Loaded:    {} shards, {} tokens, {} samples",
        dataset.shard_count(),
        dataset.total_tokens(),
        dataset.len()
    );
    println!("  Time:      {:.2}s", load_time.as_secs_f64());
    println!();

    // ── Step 2: Configure model ───────────────────────────────────────────
    println!("Step 2: Model Configuration");
    println!("---------------------------");

    let (layers, dim, heads, hidden_mult) = if let Some(preset_name) = &args.preset {
        let preset =
            find_preset(preset_name).unwrap_or_else(|| panic!("Unknown preset: {}", preset_name));
        println!("  Preset:    {}", preset.name);
        (preset.layers, preset.dim, preset.heads, preset.hidden_mult)
    } else {
        (args.layers, args.dim, args.heads, args.hidden_mult)
    };

    let vocab_size = args.vocab_size;
    let hidden_dim = dim * hidden_mult;

    println!("  Vocab:     {}", vocab_size);
    println!("  Layers:    {}", layers);
    println!("  Dim:       {}", dim);
    println!("  Heads:     {}", heads);
    println!("  Head dim:  {}", dim / heads);
    println!("  Hidden:    {} ({}x)", hidden_dim, hidden_mult);
    println!("  Seq len:   {}", args.seq_len);

    let config = TransformerConfig::new(vocab_size, dim, hidden_dim, heads, layers, args.seq_len)?;

    println!(
        "  Params:    {:.3}M",
        config.param_count() as f64 / 1_000_000.0
    );
    println!();

    // ── Step 3: Create model ──────────────────────────────────────────────
    println!("Step 3: Model Initialization");
    println!("-----------------------------");

    let mut model = TransformerANE::new(&config)?;
    println!(
        "  TransformerANE initialized: {:.3}M parameters",
        config.param_count() as f64 / 1_000_000.0
    );
    println!();

    // ── Step 4: Setup trainer ─────────────────────────────────────────────
    println!("Step 4: Trainer Setup");
    println!("---------------------");

    let total_steps = args.steps;
    let warmup_steps = (total_steps as f32 * args.warmup_frac) as u32;
    let lr = args.lr;
    let weight_decay = args.weight_decay;
    let batch_size = args.batch_size;
    let grad_clip = args.grad_clip;

    println!("  Optimizer:       AdamW (lr={}, wd={})", lr, weight_decay);
    println!("  Scheduler:       Linear warmup ({} steps)", warmup_steps);
    println!("  Total steps:     {}", total_steps);
    println!("  Batch size:      {}", batch_size);
    println!("  Grad clip norm:  {}", grad_clip);
    println!("  Loss fn:         CrossEntropy");

    let optimizer =
        AdamWOptimizer::with_hyperparams(config.param_count(), 0.9, 0.999, 1e-8, weight_decay);
    let scheduler = WarmupLinearScheduler::new(lr, warmup_steps, total_steps as u32);

    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(optimizer)
        .with_scheduler(scheduler)
        .with_loss_fn(CrossEntropyLoss::new())
        .with_grad_clip_norm(grad_clip)
        .build()?;

    println!();

    // ── Step 5: Training loop ─────────────────────────────────────────────
    println!("Step 5: Training");
    println!("----------------");
    println!(
        "{:>6} │ {:>10} │ {:>10} │ {:>12} │ {:>8}",
        "step", "loss", "grad_norm", "lr", "ms/step"
    );
    println!(
        "{}─┼─{}─┼─{}─┼─{}─┼─{}",
        "─".repeat(6),
        "─".repeat(10),
        "─".repeat(10),
        "─".repeat(12),
        "─".repeat(8)
    );

    let sampler = RandomSampler::new(dataset.len(), 42);
    let dataloader = DataLoader::new(dataset, sampler, batch_size)?;

    let train_start = Instant::now();
    let mut step_times = Vec::new();
    let mut last_loss = f32::NAN;

    for (step, batch_result) in dataloader.iter().enumerate() {
        if step >= total_steps {
            break;
        }

        let step_start = Instant::now();
        let batch = batch_result?;
        let metrics = trainer.train_step(&batch)?;
        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        step_times.push(step_ms);

        last_loss = metrics.loss;

        let log_every = (total_steps / 20).max(1);
        if step % log_every == 0 || step == total_steps - 1 {
            println!(
                "{:>6} │ {:>10.4} │ {:>10.6} │ {:>12.8} │ {:>7.1}ms",
                step, metrics.loss, metrics.grad_norm, metrics.learning_rate, step_ms
            );
        }
    }

    let total_time = train_start.elapsed();
    let avg_step_ms: f64 = if step_times.is_empty() {
        0.0
    } else {
        step_times.iter().sum::<f64>() / step_times.len() as f64
    };

    println!();
    println!("=== Summary ===");
    println!("  Steps completed:  {}", step_times.len());
    println!("  Final loss:       {:.4}", last_loss);
    println!("  Avg step time:    {:.1}ms", avg_step_ms);
    println!("  Total time:       {:.1}s", total_time.as_secs_f64());
    println!(
        "  Tokens trained:   {:.2}M",
        step_times.len() as f64 * batch_size as f64 * args.seq_len as f64 / 1_000_000.0
    );
    println!(
        "  Throughput:       {:.0} tokens/sec",
        step_times.len() as f64 * batch_size as f64 * args.seq_len as f64
            / total_time.as_secs_f64()
    );

    Ok(())
}

struct CliArgs {
    data_dir: PathBuf,
    max_shards: Option<usize>,
    seq_len: usize,
    vocab_size: usize,
    layers: usize,
    dim: usize,
    heads: usize,
    hidden_mult: usize,
    preset: Option<String>,
    steps: usize,
    lr: f32,
    weight_decay: f32,
    batch_size: usize,
    grad_clip: f32,
    warmup_frac: f32,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();

    let mut cli = CliArgs {
        data_dir: default_data_dir(),
        max_shards: None,
        seq_len: 256,
        vocab_size: 1024,
        layers: 6,
        dim: 256,
        heads: 8,
        hidden_mult: 2,
        preset: None,
        steps: 100,
        lr: 1e-3,
        weight_decay: 0.01,
        batch_size: 4,
        grad_clip: 1.0,
        warmup_frac: 0.1,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                i += 1;
                cli.data_dir = PathBuf::from(&args[i]);
            }
            "--max-shards" => {
                i += 1;
                cli.max_shards = Some(args[i].parse().expect("--max-shards must be an integer"));
            }
            "--seq-len" => {
                i += 1;
                cli.seq_len = args[i].parse().expect("--seq-len must be an integer");
            }
            "--vocab" => {
                i += 1;
                cli.vocab_size = args[i].parse().expect("--vocab must be an integer");
            }
            "--layers" => {
                i += 1;
                cli.layers = args[i].parse().expect("--layers must be an integer");
            }
            "--dim" => {
                i += 1;
                cli.dim = args[i].parse().expect("--dim must be an integer");
            }
            "--heads" => {
                i += 1;
                cli.heads = args[i].parse().expect("--heads must be an integer");
            }
            "--hidden-mult" => {
                i += 1;
                cli.hidden_mult = args[i].parse().expect("--hidden-mult must be an integer");
            }
            "--preset" => {
                i += 1;
                cli.preset = Some(args[i].clone());
            }
            "--steps" => {
                i += 1;
                cli.steps = args[i].parse().expect("--steps must be an integer");
            }
            "--lr" => {
                i += 1;
                cli.lr = args[i].parse().expect("--lr must be a float");
            }
            "--weight-decay" => {
                i += 1;
                cli.weight_decay = args[i].parse().expect("--weight-decay must be a float");
            }
            "--batch-size" => {
                i += 1;
                cli.batch_size = args[i].parse().expect("--batch-size must be an integer");
            }
            "--grad-clip" => {
                i += 1;
                cli.grad_clip = args[i].parse().expect("--grad-clip must be a float");
            }
            "--warmup-frac" => {
                i += 1;
                cli.warmup_frac = args[i].parse().expect("--warmup-frac must be a float");
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    cli
}

fn print_help() {
    println!(
        r#"Train a transformer on Parameter-Golf FineWeb data.

USAGE:
    cargo run --example train_parameter_golf [OPTIONS]

OPTIONS:
    --data-dir DIR        Path to shard directory (default: ~/dev/parameter-golf/data/datasets/fineweb10B_sp1024)
    --max-shards N        Limit shards loaded (default: all 195)
    --seq-len N           Sequence length (default: 256)
    --vocab N             Vocabulary size (default: 1024)
    --layers N            Number of transformer layers (default: 6)
    --dim N               Model dimension (default: 256)
    --heads N             Number of attention heads (default: 8)
    --hidden-mult N       hidden_dim = dim * N (default: 2)
    --preset NAME         Use a named architecture preset
    --steps N             Training steps (default: 100)
    --lr F                Learning rate (default: 0.001)
    --weight-decay F      AdamW weight decay (default: 0.01)
    --batch-size N        Batch size in samples (default: 4)
    --grad-clip F         Gradient clip norm (default: 1.0)
    --warmup-frac F       Warmup fraction of total steps (default: 0.1)
    -h, --help            Print this help

PRESETS:
    baseline-6l-256d      6 layers, dim=256, 8 heads, 2x FFN
    compact-12l-208d      12 layers, dim=208, 8 heads, 4x FFN
    wide-8l-384d          8 layers, dim=384, 8 heads, 2x FFN
    deep-15l-192d         15 layers, dim=192, 6 heads, 3x FFN

EXAMPLES:
    # Quick smoke test
    cargo run --example train_parameter_golf -- --max-shards 2 --steps 20

    # Full training with compact architecture
    cargo run --example train_parameter_golf -- --preset compact-12l-208d --steps 500

    # Custom architecture
    cargo run --example train_parameter_golf -- --layers 10 --dim 256 --heads 8 --hidden-mult 4
"#
    );
}
