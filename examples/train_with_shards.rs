//! Example: Training on sharded data with gradient accumulation.
//!
//! Demonstrates:
//! - Loading FineWeb binary shards from disk
//! - Streaming shards through DataLoader
//! - Chunking batches and training with gradient accumulation
//! - Integration with parameter-golf datasets
//!
//! Note: The SimpleModel is intentionally minimal to show the data pipeline.
//! For realistic training, integrate with actual transformer models.

use rustane::data::{
    Batch, DataLoader, Dataset, JsonlDataset, RandomSampler, ShardConfig, ShardedDataLoader,
};
use rustane::error::Result;
use rustane::training::{CrossEntropyLoss, Model, Optimizer, TrainerBuilder};
use rustane::wrapper::ANETensor;
use rustane::ConstantScheduler;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("Rustane Sharded Training Example");
    println!("================================\n");

    let mut args: Vec<String> = env::args().skip(1).collect();

    // Extract --steps argument if present
    let max_steps = if let Some(pos) = args.iter().position(|arg| arg == "--steps") {
        args.remove(pos);
        if let Some(count_str) = args.get(pos) {
            let count = count_str.parse::<usize>()
                .map_err(|_| rustane::Error::Other("--steps requires a number argument".to_string()))?;
            args.remove(pos);
            count
        } else {
            return Err(rustane::Error::Other("--steps requires a number argument".to_string()));
        }
    } else {
        std::env::var("TRAIN_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3)
    };

    let source_paths = resolve_source_paths(&args)?;
    if let Some(bin_files) = resolve_fineweb_bin_files(&source_paths)? {
        println!("Mode: FineWeb binary shards streamed directly");
        println!("Source count: {}", bin_files.len());
        println!("Max steps: {}\n", max_steps);

        let mut model = SimpleModel::new(32);
        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.5))
            .with_scheduler(ConstantScheduler::new(0.5))
            .with_loss_fn(CrossEntropyLoss::new())
            .build()?;

        println!("Step | Shard | Loss    | Grad Norm | LR       | Time(ms)");
        println!("-----|-------|---------|-----------|----------|----------");

        let mut step_count = 0usize;
        let start_time = std::time::Instant::now();

        for (step_idx, file) in bin_files.iter().enumerate() {
            if step_count >= max_steps {
                break;
            }

            let step_start = std::time::Instant::now();

            // Load large chunk from shard file (65536 tokens ≈ 128 sequences of 512)
            let batch = load_fineweb_batch(file, 65536, 512)?;

            // Process in large chunks for gradient accumulation
            let chunk_size = 8192;  // tokens per chunk
            let chunks = batch.into_chunks(chunk_size)?;

            // Train with gradient accumulation across chunks
            let metrics = trainer.train_accumulated_steps(
                chunks.into_iter().map(Ok),
                8,  // accumulate over 8 chunks at a time
            )?;

            let elapsed = step_start.elapsed().as_millis();

            println!(
                "{:4} | {:5} | {:.5} | {:.5}    | {:.6} | {:>8}",
                step_count, step_idx, metrics.loss, metrics.grad_norm, metrics.learning_rate, elapsed
            );
            step_count += 1;
        }

        let total_time = start_time.elapsed().as_secs_f32();
        println!("\nTotal training time: {:.2}s", total_time);

        println!("\n✓ Training completed!");
        return Ok(());
    }

    let (shard_dir, using_real_data): (PathBuf, bool) = match source_paths.as_slice() {
        [] => {
            println!("Mode: synthetic demo shards\n");
            (create_demo_shards()?, false)
        }
        [path] if path.is_dir() && contains_shard_jsonl(path)? => {
            println!("Mode: direct shard directory");
            println!("Source: {}\n", path.display());
            (path.clone(), true)
        }
        [path] if path.is_dir() => {
            println!("Mode: source directory converted to shards");
            println!("Source: {}\n", path.display());
            (create_demo_shards_from_dir(path)?, true)
        }
        [path] => {
            println!("Mode: text source converted to shards");
            println!("Source: {}\n", path.display());
            (create_demo_shards_from_text(path)?, true)
        }
        _ => {
            println!("Mode: multiple source files converted to demo shards");
            println!("Source count: {}\n", source_paths.len());
            (create_demo_shards_from_bin_files(&source_paths)?, true)
        }
    };
    let shard_pattern = format!("{}/shard_*.jsonl", shard_dir.display());
    let config = ShardConfig::new(shard_pattern, 256)?;
    let loader = ShardedDataLoader::new(&config)?;

    println!("Discovered {} shard(s)\n", loader.shard_count());

    let mut model = SimpleModel::new(32);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.5))
        .with_scheduler(ConstantScheduler::new(0.5))
        .with_loss_fn(CrossEntropyLoss::new())
        .build()?;

    println!("Step | Shard | Loss    | Grad Norm | LR");
    println!("-----|-------|---------|-----------|--------");

    let batch_size = 2;
    let chunk_tokens = if using_real_data { 16 } else { 8 };
    let mut step = 0usize;
    for (shard_idx, shard_path) in loader.iter_shards() {
        if step >= max_steps {
            break;
        }

        let dataset = JsonlDataset::load(&shard_path)?;
        let sampler = RandomSampler::new(dataset.len(), 42);
        let dataloader = DataLoader::new(dataset, sampler, batch_size)?;

        let batch = match dataloader.iter().next() {
            Some(batch_result) => batch_result?,
            None => continue,
        };

        let chunks = batch.into_chunks(chunk_tokens)?;
        let metrics = trainer.train_accumulated_steps(chunks.into_iter().map(Ok), 2)?;

        println!(
            "{:4} | {:5} | {:.5} | {:.5}    | {:.6}",
            step, shard_idx, metrics.loss, metrics.grad_norm, metrics.learning_rate
        );
        step += 1;

        println!("Processed shard: {}", shard_path.display());
    }

    println!("\n✓ Training completed!");
    Ok(())
}

fn create_demo_shards() -> Result<PathBuf> {
    let shard_dir = env::temp_dir().join(format!("rustane-shards-{}", std::process::id()));
    fs::create_dir_all(&shard_dir).map_err(|e| rustane::Error::Io(e.to_string()))?;

    write_shard(
        &shard_dir.join("shard_000.jsonl"),
        &[
            vec![0, 1, 2, 3, 4, 5, 6, 7],
            vec![8, 9, 10, 11, 12, 13, 14, 15],
            vec![16, 17, 18, 19, 20, 21, 22, 23],
            vec![24, 25, 26, 27, 28, 29, 30, 31],
            vec![32, 33, 34, 35, 36, 37, 38, 39],
            vec![40, 41, 42, 43, 44, 45, 46, 47],
            vec![48, 49, 50, 51, 52, 53, 54, 55],
            vec![56, 57, 58, 59, 60, 61, 62, 63],
        ],
    )?;

    write_shard(
        &shard_dir.join("shard_001.jsonl"),
        &[
            vec![64, 65, 66, 67, 68, 69, 70, 71],
            vec![72, 73, 74, 75, 76, 77, 78, 79],
            vec![80, 81, 82, 83, 84, 85, 86, 87],
            vec![88, 89, 90, 91, 92, 93, 94, 95],
            vec![96, 97, 98, 99, 100, 101, 102, 103],
            vec![104, 105, 106, 107, 108, 109, 110, 111],
            vec![112, 113, 114, 115, 116, 117, 118, 119],
            vec![120, 121, 122, 123, 124, 125, 126, 127],
        ],
    )?;

    Ok(shard_dir)
}

fn create_demo_shards_from_text(source: &Path) -> Result<PathBuf> {
    let text = fs::read_to_string(source).map_err(|e| rustane::Error::Io(e.to_string()))?;
    let tokens: Vec<u32> = text
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .map(|b| (b as u32) % 256)
        .collect();

    if tokens.is_empty() {
        return Err(rustane::Error::Other(format!(
            "source file {} did not contain usable text",
            source.display()
        )));
    }

    let seq_len = 16;
    let samples_per_shard = 8;
    let shard_dir = env::temp_dir().join(format!("rustane-real-shards-{}", std::process::id()));
    fs::create_dir_all(&shard_dir).map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut samples = Vec::new();
    let mut pos = 0usize;
    while pos < tokens.len() {
        let end = (pos + seq_len).min(tokens.len());
        let mut sample = tokens[pos..end].to_vec();
        if sample.len() < seq_len {
            sample.resize(seq_len, 0);
        }
        samples.push(sample);
        pos = end;
    }

    for (shard_idx, shard_samples) in samples.chunks(samples_per_shard).enumerate() {
        let shard_path = shard_dir.join(format!("shard_{:03}.jsonl", shard_idx));
        write_shard(&shard_path, shard_samples)?;
    }

    Ok(shard_dir)
}

fn load_fineweb_batch(file: &Path, max_tokens: usize, seq_len: usize) -> Result<Batch> {
    let tokens = load_fineweb_tokens_prefix(file, max_tokens)?;
    let usable = (tokens.len() / seq_len) * seq_len;
    if usable == 0 {
        return Err(rustane::Error::Other(format!(
            "binary shard {} did not contain enough tokens for seq_len={}",
            file.display(),
            seq_len
        )));
    }

    let tokens = tokens[..usable].to_vec();
    let batch_size = usable / seq_len;
    Batch::new(tokens, batch_size, seq_len)
}

fn create_demo_shards_from_bin_files(files: &[PathBuf]) -> Result<PathBuf> {
    let shard_dir = env::temp_dir().join(format!("rustane-fineweb-shards-{}", std::process::id()));
    fs::create_dir_all(&shard_dir).map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut samples = Vec::new();
    for file in files.iter().take(3) {
        let tokens = load_fineweb_tokens_prefix(file, 256)?;
        if tokens.len() < 16 {
            continue;
        }

        let seq_len = 16;
        let mut pos = 0usize;
        while pos < tokens.len() {
            let end = (pos + seq_len).min(tokens.len());
            let mut sample = tokens[pos..end].to_vec();
            if sample.len() < seq_len {
                sample.resize(seq_len, 0);
            }
            samples.push(sample);
            pos = end;
        }
    }

    if samples.is_empty() {
        return Err(rustane::Error::Other(
            "binary shard files did not yield any usable samples".to_string(),
        ));
    }

    for (shard_idx, shard_samples) in samples.chunks(8).enumerate() {
        let shard_path = shard_dir.join(format!("shard_{:03}.jsonl", shard_idx));
        write_shard(&shard_path, shard_samples)?;
    }

    Ok(shard_dir)
}

fn load_fineweb_tokens_prefix(file: &Path, max_tokens: usize) -> Result<Vec<u32>> {
    let mut f = File::open(file).map_err(|e| rustane::Error::Io(e.to_string()))?;
    let mut header_buf = [0u8; 256 * 4];
    f.read_exact(&mut header_buf)
        .map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut header = [0i32; 256];
    for (idx, chunk) in header_buf.chunks_exact(4).enumerate() {
        header[idx] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    if header[0] != 20240520 || header[1] != 1 {
        return Err(rustane::Error::Other(format!(
            "unexpected FineWeb shard header for {}",
            file.display()
        )));
    }

    let num_tokens = header[2].max(0) as usize;
    let count = num_tokens.min(max_tokens);
    let mut token_buf = vec![0u8; count * 2];
    f.seek(SeekFrom::Start((256 * 4) as u64))
        .map_err(|e| rustane::Error::Io(e.to_string()))?;
    f.read_exact(&mut token_buf)
        .map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut tokens = Vec::with_capacity(count);
    for chunk in token_buf.chunks_exact(2) {
        tokens.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
    }
    Ok(tokens)
}

fn create_demo_shards_from_dir(source_dir: &Path) -> Result<PathBuf> {
    let mut text = String::new();
    collect_text_files(source_dir, &mut text)?;
    if text.is_empty() {
        return Err(rustane::Error::Other(format!(
            "source directory {} did not contain readable text files",
            source_dir.display()
        )));
    }

    let tokens: Vec<u32> = text
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .map(|b| (b as u32) % 256)
        .collect();

    if tokens.is_empty() {
        return Err(rustane::Error::Other(format!(
            "source directory {} did not contain usable text",
            source_dir.display()
        )));
    }

    let seq_len = 16;
    let samples_per_shard = 8;
    let shard_dir = env::temp_dir().join(format!("rustane-real-dir-shards-{}", std::process::id()));
    fs::create_dir_all(&shard_dir).map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut samples = Vec::new();
    let mut pos = 0usize;
    while pos < tokens.len() {
        let end = (pos + seq_len).min(tokens.len());
        let mut sample = tokens[pos..end].to_vec();
        if sample.len() < seq_len {
            sample.resize(seq_len, 0);
        }
        samples.push(sample);
        pos = end;
    }

    for (shard_idx, shard_samples) in samples.chunks(samples_per_shard).enumerate() {
        let shard_path = shard_dir.join(format!("shard_{:03}.jsonl", shard_idx));
        write_shard(&shard_path, shard_samples)?;
    }

    Ok(shard_dir)
}

fn collect_text_files(dir: &Path, out: &mut String) -> Result<()> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| rustane::Error::Io(e.to_string()))? {
        entries.push(entry.map_err(|e| rustane::Error::Io(e.to_string()))?);
    }
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_text_files(&path, out)?;
            continue;
        }

        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let is_text_like = matches!(
            ext.as_str(),
            "txt" | "log" | "md" | "json" | "jsonl" | "toml" | "py" | "rs" | "csv"
        );
        if !is_text_like {
            continue;
        }

        let contents = fs::read_to_string(&path).map_err(|e| rustane::Error::Io(e.to_string()))?;
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&contents);
    }

    Ok(())
}

fn contains_shard_jsonl(dir: &Path) -> Result<bool> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| rustane::Error::Io(e.to_string()))? {
        entries.push(entry.map_err(|e| rustane::Error::Io(e.to_string()))?);
    }

    Ok(entries.iter().any(|entry| {
        entry
            .path()
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with("shard_") && name.ends_with(".jsonl"))
            .unwrap_or(false)
    }))
}

fn resolve_fineweb_bin_files(source_paths: &[PathBuf]) -> Result<Option<Vec<PathBuf>>> {
    if source_paths.is_empty() {
        return Ok(None);
    }

    if source_paths.len() == 1 && source_paths[0].is_dir() {
        let files = collect_bin_files(&source_paths[0])?;
        if files.is_empty() {
            return Ok(None);
        }
        return Ok(Some(files));
    }

    if source_paths.iter().all(|path| path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("bin")) {
        let mut files = source_paths.to_vec();
        files.sort();
        files.dedup();
        return Ok(Some(files));
    }

    Ok(None)
}

fn collect_bin_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| rustane::Error::Io(e.to_string()))? {
        let entry = entry.map_err(|e| rustane::Error::Io(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn resolve_source_paths(args: &[String]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for arg in args {
        if arg.contains('*') || arg.contains('?') || arg.contains('[') {
            for entry in glob::glob(arg).map_err(|e| rustane::Error::Other(e.to_string()))? {
                let path = entry.map_err(|e| rustane::Error::Other(e.to_string()))?;
                paths.push(path);
            }
        } else {
            paths.push(PathBuf::from(arg));
        }
    }

    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn write_shard(path: &Path, samples: &[Vec<u32>]) -> Result<()> {
    let mut file = File::create(path).map_err(|e| rustane::Error::Io(e.to_string()))?;
    for sample in samples {
        let line =
            serde_json::to_string(sample).map_err(|e| rustane::Error::Other(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| rustane::Error::Io(e.to_string()))?;
    }
    Ok(())
}

/// Simple logits-based model: directly learns per-token-per-prediction logits
pub struct SimpleModel {
    /// Logits: [vocab_size] values that directly predict probabilities
    /// For simplicity, uses same logits for all tokens in a batch
    logits: Vec<f32>,
    vocab_size: usize,
    /// Store last batch tokens for gradient computation
    last_tokens: Option<Vec<u32>>,
    /// Store last expanded logits for gradient computation
    last_expanded_logits: Option<Vec<f32>>,
}

impl SimpleModel {
    fn new(_hidden_dim: usize) -> Self {
        let vocab_size = 1024; // SentencePiece sp1024

        // Initialize logits to uniform distribution (will give loss = ln(vocab_size))
        let logits = vec![0.0f32; vocab_size];

        Self {
            logits,
            vocab_size,
            last_tokens: None,
            last_expanded_logits: None,
        }
    }
}

impl Model for SimpleModel {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        let tokens = batch.tokens();
        let num_tokens = tokens.len();
        let vocab_size = self.vocab_size;

        // Store tokens for backward pass
        self.last_tokens = Some(tokens.to_vec());

        // Expanded logits: replicate learned logits for each token
        // (simplified version that allows proper gradient flow)
        let mut expanded_logits = Vec::with_capacity(num_tokens * vocab_size);
        for _ in 0..num_tokens {
            expanded_logits.extend_from_slice(&self.logits);
        }

        // Store for backward pass
        self.last_expanded_logits = Some(expanded_logits.clone());

        ANETensor::from_fp32(expanded_logits, vec![num_tokens, self.vocab_size])
    }

    fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
        // REAL gradient computation: dL/dlogits = softmax(logits) - one_hot(target)
        // This is the actual gradient from cross-entropy loss!
        //
        // KEY: Different gradient for each logit enables softmax to change, loss to decrease
        // Without this (all same gradients), softmax is invariant and loss is stuck

        let tokens = self.last_tokens.as_ref().ok_or_else(|| {
            rustane::Error::Other("No batch tokens stored".to_string())
        })?;

        let expanded_logits = self.last_expanded_logits.as_ref().ok_or_else(|| {
            rustane::Error::Other("No logits stored".to_string())
        })?;

        let mut grad_sum = vec![0.0f32; self.vocab_size];
        let num_tokens = tokens.len();

        // For each position, compute gradient = softmax - one_hot(target)
        for pos in 0..num_tokens {
            let logits_at_pos = &expanded_logits[pos * self.vocab_size..(pos + 1) * self.vocab_size];

            // Compute softmax with numerical stability
            let max_logit = logits_at_pos.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits_at_pos.iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f32 = exp_logits.iter().sum();

            // Get target token for next-token prediction
            let target_idx = if pos + 1 < tokens.len() {
                (tokens[pos + 1] as usize).min(self.vocab_size - 1)
            } else {
                (tokens[pos] as usize).min(self.vocab_size - 1)
            };

            // Compute gradient: softmax - one_hot(target)
            for pred_id in 0..self.vocab_size {
                let softmax_val = exp_logits[pred_id] / sum_exp;
                let target_val = if pred_id == target_idx { 1.0 } else { 0.0 };
                grad_sum[pred_id] += (softmax_val - target_val) / num_tokens as f32;
            }
        }

        // Scale gradients for stable training (larger scale = faster learning)
        let grads = grad_sum.iter().map(|&g| g * 1.0).collect();

        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.logits
    }

    fn param_count(&self) -> usize {
        self.logits.len()
    }
}

struct SimpleOptimizer {
    _lr: f32,
}

impl SimpleOptimizer {
    fn new(lr: f32) -> Self {
        Self { _lr: lr }
    }
}

impl Optimizer for SimpleOptimizer {
    fn step(&mut self, grads: &[f32], params: &mut [f32], lr: f32) -> Result<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= lr * grad;
        }
        Ok(())
    }
}
