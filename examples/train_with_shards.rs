//! Example: Training on sharded data with gradient accumulation.
//!
//! Demonstrates:
//! - Loading FineWeb binary shards from disk
//! - Streaming shards through DataLoader
//! - Chunking batches and training with gradient accumulation
//! - Integration with parameter-golf datasets
//!
//! This example now uses a real tiny causal-attention model with manual
//! backward gradients so the training loop exercises actual learning.

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

        let mut model = TinyAttentionLanguageModel::new(1024, 32, 512);
        let mut trainer = TrainerBuilder::new(&mut model)
            .with_optimizer(SimpleOptimizer::new(0.001))
            .with_scheduler(ConstantScheduler::new(0.001))
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

    let mut model = TinyAttentionLanguageModel::new(1024, 32, 512);
    let mut trainer = TrainerBuilder::new(&mut model)
        .with_optimizer(SimpleOptimizer::new(0.001))
        .with_scheduler(ConstantScheduler::new(0.001))
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

#[derive(Clone, Debug)]
struct TinyAttentionCache {
    tokens: Vec<u32>,
    batch_size: usize,
    seq_len: usize,
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    context: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<Vec<Vec<f32>>>,
}

#[derive(Clone, Debug)]
struct ParamRanges {
    token_embed: std::ops::Range<usize>,
    pos_embed: std::ops::Range<usize>,
    wq: std::ops::Range<usize>,
    wk: std::ops::Range<usize>,
    wv: std::ops::Range<usize>,
    wo: std::ops::Range<usize>,
}

/// Small causal-attention language model with real next-token gradients.
///
/// This is intentionally small enough to train on CPU while still exercising
/// token embeddings, QKV projections, causal attention, and output projection.
pub struct TinyAttentionLanguageModel {
    params: Vec<f32>,
    ranges: ParamRanges,
    vocab_size: usize,
    d_model: usize,
    max_seq_len: usize,
    cache: Option<TinyAttentionCache>,
}

impl TinyAttentionLanguageModel {
    fn new(vocab_size: usize, d_model: usize, max_seq_len: usize) -> Self {
        let token_embed = vocab_size * d_model;
        let pos_embed = max_seq_len * d_model;
        let wq = d_model * d_model;
        let wk = d_model * d_model;
        let wv = d_model * d_model;
        let wo = d_model * vocab_size;
        let total = token_embed + pos_embed + 3 * d_model * d_model + wo;

        let mut params = vec![0.0f32; total];
        for (i, p) in params.iter_mut().enumerate() {
            let scale = 0.02f32;
            *p = ((i as f32) * 0.013).sin() * scale;
        }

        let ranges = ParamRanges {
            token_embed: 0..token_embed,
            pos_embed: token_embed..token_embed + pos_embed,
            wq: token_embed + pos_embed..token_embed + pos_embed + wq,
            wk: token_embed + pos_embed + wq..token_embed + pos_embed + wq + wk,
            wv: token_embed + pos_embed + wq + wk..token_embed + pos_embed + wq + wk + wv,
            wo: token_embed + pos_embed + 3 * d_model * d_model..total,
        };

        Self {
            params,
            ranges,
            vocab_size,
            d_model,
            max_seq_len,
            cache: None,
        }
    }

    fn slice(&self, r: &std::ops::Range<usize>) -> &[f32] {
        &self.params[r.clone()]
    }

    fn softmax(values: &[f32]) -> Vec<f32> {
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = values.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            for x in &mut exps {
                *x /= sum;
            }
        }
        exps
    }

    fn matmul_row(input: &[f32], weights: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; out_dim];
        for o in 0..out_dim {
            let mut sum = 0.0f32;
            for i in 0..in_dim {
                sum += input[i] * weights[i * out_dim + o];
            }
            out[o] = sum;
        }
        out
    }

    fn add_outer(grad: &mut [f32], input: &[f32], output_grad: &[f32], in_dim: usize, out_dim: usize) {
        for i in 0..in_dim {
            for o in 0..out_dim {
                grad[i * out_dim + o] += input[i] * output_grad[o];
            }
        }
    }

    fn matmul_row_t(output_grad: &[f32], weights: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; in_dim];
        for i in 0..in_dim {
            let mut sum = 0.0f32;
            for o in 0..out_dim {
                sum += output_grad[o] * weights[i * out_dim + o];
            }
            out[i] = sum;
        }
        out
    }
}

impl Model for TinyAttentionLanguageModel {
    fn forward(&mut self, batch: &Batch) -> Result<ANETensor> {
        let tokens = batch.tokens();
        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();
        if seq_len < 2 {
            return Err(rustane::Error::Other("seq_len must be at least 2".to_string()));
        }
        if seq_len > self.max_seq_len {
            return Err(rustane::Error::Other(format!(
                "seq_len {} exceeds max_seq_len {}",
                seq_len, self.max_seq_len
            )));
        }
        if tokens.len() != batch_size * seq_len {
            return Err(rustane::Error::Other("batch token shape mismatch".to_string()));
        }

        let d = self.d_model;
        let vocab = self.vocab_size;
        let scale = 1.0f32 / (d as f32).sqrt();
        let token_embed = self.slice(&self.ranges.token_embed);
        let pos_embed = self.slice(&self.ranges.pos_embed);
        let wq = self.slice(&self.ranges.wq);
        let wk = self.slice(&self.ranges.wk);
        let wv = self.slice(&self.ranges.wv);
        let wo = self.slice(&self.ranges.wo);

        let mut x = vec![0.0f32; batch_size * seq_len * d];
        let mut q = vec![0.0f32; batch_size * seq_len * d];
        let mut k = vec![0.0f32; batch_size * seq_len * d];
        let mut v = vec![0.0f32; batch_size * seq_len * d];
        let mut context = vec![0.0f32; batch_size * (seq_len - 1) * d];
        let mut logits = vec![0.0f32; batch_size * (seq_len - 1) * vocab];
        let mut probs = vec![vec![vec![]; seq_len - 1]; batch_size];

        for b in 0..batch_size {
            for t in 0..seq_len {
                let token_id = tokens[b * seq_len + t] as usize % vocab;
                let x_offset = (b * seq_len + t) * d;
                let embed_offset = token_id * d;
                let pos_offset = t * d;
                for i in 0..d {
                    x[x_offset + i] = token_embed[embed_offset + i] + pos_embed[pos_offset + i];
                }
                let x_row = &x[x_offset..x_offset + d];
                let q_row = Self::matmul_row(x_row, wq, d, d);
                let k_row = Self::matmul_row(x_row, wk, d, d);
                let v_row = Self::matmul_row(x_row, wv, d, d);
                q[x_offset..x_offset + d].copy_from_slice(&q_row);
                k[x_offset..x_offset + d].copy_from_slice(&k_row);
                v[x_offset..x_offset + d].copy_from_slice(&v_row);
            }

            for t in 0..(seq_len - 1) {
                let q_row = &q[(b * seq_len + t) * d..(b * seq_len + t + 1) * d];
                let mut scores = vec![0.0f32; t + 1];
                for j in 0..=t {
                    let k_row = &k[(b * seq_len + j) * d..(b * seq_len + j + 1) * d];
                    let mut score = 0.0f32;
                    for i in 0..d {
                        score += q_row[i] * k_row[i];
                    }
                    scores[j] = score * scale;
                }
                let p = Self::softmax(&scores);
                probs[b][t] = p.clone();

                let mut ctx = vec![0.0f32; d];
                for j in 0..=t {
                    let v_row = &v[(b * seq_len + j) * d..(b * seq_len + j + 1) * d];
                    for i in 0..d {
                        ctx[i] += p[j] * v_row[i];
                    }
                }
                let ctx_offset = (b * (seq_len - 1) + t) * d;
                context[ctx_offset..ctx_offset + d].copy_from_slice(&ctx);

                let row = Self::matmul_row(&ctx, wo, d, vocab);
                let logit_offset = (b * (seq_len - 1) + t) * vocab;
                logits[logit_offset..logit_offset + vocab].copy_from_slice(&row);
            }
        }

        self.cache = Some(TinyAttentionCache {
            tokens: tokens.to_vec(),
            batch_size,
            seq_len,
            x,
            q,
            k,
            v,
            context,
            logits: logits.clone(),
            probs,
        });

        ANETensor::from_fp32(logits, vec![batch_size * (seq_len - 1), vocab])
    }

    fn backward(&mut self, _loss: f32) -> Result<Vec<f32>> {
        let cache = self
            .cache
            .as_ref()
            .ok_or_else(|| rustane::Error::Other("No forward cache available".to_string()))?;

        let d = self.d_model;
        let vocab = self.vocab_size;
        let seq_len = cache.seq_len;
        let batch_size = cache.batch_size;
        let scale = 1.0f32 / (d as f32).sqrt();
        let num_rows = batch_size * (seq_len - 1);

        let mut grads = vec![0.0f32; self.params.len()];
        let wq = self.slice(&self.ranges.wq).to_vec();
        let wk = self.slice(&self.ranges.wk).to_vec();
        let wv = self.slice(&self.ranges.wv).to_vec();
        let wo = self.slice(&self.ranges.wo).to_vec();

        for b in 0..batch_size {
            let mut d_q = vec![vec![0.0f32; d]; seq_len - 1];
            let mut d_k = vec![vec![0.0f32; d]; seq_len];
            let mut d_v = vec![vec![0.0f32; d]; seq_len];

            for t in 0..(seq_len - 1) {
                let row_idx = b * (seq_len - 1) + t;
                let logits = &cache.logits[row_idx * vocab..(row_idx + 1) * vocab];
                let mut p = Self::softmax(logits);
                let target = cache.tokens[b * seq_len + t + 1] as usize % vocab;
                p[target] -= 1.0;
                let inv_rows = 1.0 / num_rows as f32;
                for x in &mut p {
                    *x *= inv_rows;
                }

                let ctx = &cache.context[row_idx * d..(row_idx + 1) * d];
                let grad_wo = &mut grads[self.ranges.wo.clone()];
                Self::add_outer(grad_wo, ctx, &p, d, vocab);

                let d_ctx = Self::matmul_row_t(&p, &wo, d, vocab);
                let probs = &cache.probs[b][t];

                let mut dot_terms = vec![0.0f32; t + 1];
                for j in 0..=t {
                    let v_row = &cache.v[(b * seq_len + j) * d..(b * seq_len + j + 1) * d];
                    let mut dot = 0.0f32;
                    for i in 0..d {
                        dot += d_ctx[i] * v_row[i];
                    }
                    dot_terms[j] = dot;
                    for i in 0..d {
                        d_v[j][i] += probs[j] * d_ctx[i];
                    }
                }

                let mut weighted_dot = 0.0f32;
                for j in 0..=t {
                    weighted_dot += dot_terms[j] * probs[j];
                }

                let q_row = &cache.q[(b * seq_len + t) * d..(b * seq_len + t + 1) * d];
                for j in 0..=t {
                    let dscore = probs[j] * (dot_terms[j] - weighted_dot) * scale;
                    let k_row = &cache.k[(b * seq_len + j) * d..(b * seq_len + j + 1) * d];
                    for i in 0..d {
                        d_q[t][i] += dscore * k_row[i];
                        d_k[j][i] += dscore * q_row[i];
                    }
                }
            }

            for t in 0..(seq_len - 1) {
                let x_row = &cache.x[(b * seq_len + t) * d..(b * seq_len + t + 1) * d];
                let token_idx = cache.tokens[b * seq_len + t] as usize % vocab;
                let pos_idx = t;

                {
                    let grad_wq = &mut grads[self.ranges.wq.clone()];
                    Self::add_outer(grad_wq, x_row, &d_q[t], d, d);
                }
                {
                    let grad_wk = &mut grads[self.ranges.wk.clone()];
                    Self::add_outer(grad_wk, x_row, &d_k[t], d, d);
                }
                {
                    let grad_wv = &mut grads[self.ranges.wv.clone()];
                    Self::add_outer(grad_wv, x_row, &d_v[t], d, d);
                }

                let dx_q = Self::matmul_row_t(&d_q[t], &wq, d, d);
                let dx_k = Self::matmul_row_t(&d_k[t], &wk, d, d);
                let dx_v = Self::matmul_row_t(&d_v[t], &wv, d, d);
                for i in 0..d {
                    let dx = dx_q[i] + dx_k[i] + dx_v[i];
                    grads[self.ranges.token_embed.start + token_idx * d + i] += dx;
                    grads[self.ranges.pos_embed.start + pos_idx * d + i] += dx;
                }
            }
        }

        Ok(grads)
    }

    fn parameters(&mut self) -> &mut [f32] {
        &mut self.params
    }

    fn param_count(&self) -> usize {
        self.params.len()
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
