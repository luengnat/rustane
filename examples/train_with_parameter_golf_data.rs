//! Train Rustane on Parameter Golf FineWeb shards.
//!
//! This example mirrors the `train_gpt.py` data contract:
//! - `fineweb_train_*.bin` / `fineweb_val_*.bin` shard files
//! - 256-int32 header
//! - little-endian `u16` token payload
//! - SentencePiece BPB accounting from the tokenizer model

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use glob::glob;
use rustane::data::{Batch, Dataset, DataLoader, SequentialSampler};
use rustane::error::Result;
use rustane::training::{
    AdamOptimizer, ConstantScheduler, CrossEntropyLoss, LossFn, Model, Optimizer,
    WarmupCosineScheduler, WarmupLinearScheduler,
};
use rustane::training::{TransformerANE, TransformerConfig};
use sentencepiece_model::SentencePieceModel;

fn main() -> Result<()> {
    let data_path = std::env::var("DATA_PATH")
        .unwrap_or_else(|_| "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024".to_string());
    let tokenizer_path = std::env::var("TOKENIZER_PATH")
        .unwrap_or_else(|_| "/Users/nat/dev/parameter-golf/data/tokenizers/fineweb_1024_bpe.model".to_string());

    let vocab_size = std::env::var("VOCAB_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024);
    let seq_len = std::env::var("TRAIN_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(512);
    let batch_size = std::env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16);
    let chunk_tokens = std::env::var("CHUNK_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048);
    let train_steps = std::env::var("TRAIN_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20);
    let train_log_every = std::env::var("TRAIN_LOG_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let val_loss_every = std::env::var("VAL_LOSS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let max_shards = std::env::var("MAX_SHARDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2);
    let val_max_shards = std::env::var("VAL_MAX_SHARDS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let val_max_batches = std::env::var("VAL_MAX_BATCHES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);
    let val_batch_size = std::env::var("VAL_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(batch_size);
    let grad_clip_norm = std::env::var("GRAD_CLIP_NORM")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.0);
    let lr_scheduler = std::env::var("LR_SCHEDULER").unwrap_or_else(|_| "cosine".to_string());
    let peak_lr = std::env::var("PEAK_LR")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.001);
    let min_lr = std::env::var("MIN_LR")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1e-5);
    let warmup_steps = std::env::var("WARMUP_STEPS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or((train_steps / 10).max(1) as u32);

    let train_pattern = format!("{}/fineweb_train_*.bin", data_path);
    let val_pattern = format!("{}/fineweb_val_*.bin", data_path);
    let accum_steps = (batch_size * seq_len).div_ceil(chunk_tokens).max(1);

    println!("Rustane + Parameter-Golf FineWeb Training Example");
    println!("==================================================\n");
    println!("Configuration:");
    println!("  Data path:      {}", data_path);
    println!("  Train pattern:  {}", train_pattern);
    println!("  Val pattern:    {}", val_pattern);
    println!("  Tokenizer:      {}", tokenizer_path);
    println!("  Vocab size:     {}", vocab_size);
    println!("  Sequence len:   {}", seq_len);
    println!("  Batch size:     {}", batch_size);
    println!("  Chunk tokens:   {}", chunk_tokens);
    println!("  Accum steps:    {}", accum_steps);
    println!("  Train steps:    {}", train_steps);
    println!("  Train log every: {}", train_log_every);
    println!("  Val loss every:  {}", val_loss_every);
    println!("  Max shards:      {}", max_shards);
    println!("  Val shards:      {}", val_max_shards);
    println!("  Val batch size:  {}", val_batch_size);
    println!("  Val batches:     {}\n", val_max_batches);
    println!("  LR scheduler:    {}", lr_scheduler);
    println!("  Peak LR:         {}", peak_lr);
    println!("  Min LR:          {}", min_lr);
    println!("  Warmup steps:    {}", warmup_steps);
    println!("  Grad clip norm:  {}\n", grad_clip_norm);

    let tokenizer_stats = SentencePieceStats::load(&tokenizer_path)?;
    let train_dataset = FineWebSequenceDataset::load(&train_pattern, seq_len, max_shards)?;
    let val_dataset = FineWebSequenceDataset::load(&val_pattern, seq_len, val_max_shards)?;

    println!(
        "Loaded {} training sequences and {} validation sequences\n",
        train_dataset.len(),
        val_dataset.len()
    );

    let train_sampler = SequentialSampler::new(train_dataset.len());
    let mut train_iter = DataLoader::new(train_dataset.clone(), train_sampler, batch_size)?.iter();

    let config = TransformerConfig::new(vocab_size, 256, 512, 4, 2, seq_len)?;
    let mut model = TransformerANE::new(&config)?;
    let param_count = model.param_count();
    let mut optimizer = AdamOptimizer::new(param_count);
    let scheduler = build_scheduler(
        &lr_scheduler,
        peak_lr,
        warmup_steps,
        train_steps as u32,
        min_lr,
    );
    let loss_fn = CrossEntropyLoss::new();

    println!("Step | Train Loss | Grad Norm  | LR       | Validation");
    println!("-----|------------|------------|----------|----------------");

    let mut total_train_batches = 0usize;
    let mut last_val_loss = None;
    let mut last_val_bpb = None;
    let train_start = Instant::now();

    for step in 0..train_steps {
        let batch = match train_iter.next() {
            Some(batch) => batch?,
            None => {
                train_iter = DataLoader::new(
                    train_dataset.clone(),
                    SequentialSampler::new(train_dataset.len()),
                    batch_size,
                )?
                .iter();
                train_iter
                    .next()
                    .ok_or_else(|| rustane::Error::Other("training dataset produced no batch".to_string()))??
            }
        };

        let mut total_loss = 0.0f32;
        let mut total_grad_norm = 0.0f32;
        let mut chunk_count = 0usize;
        let scale = 1.0 / accum_steps as f32;
        let mut accum_grads = vec![0.0f32; param_count];
        for chunk_result in batch.into_chunks(chunk_tokens)? {
            let chunk = chunk_result;
            let logits = model.forward(&chunk)?;
            let loss = loss_fn.compute(&logits, &chunk)?;
            let grads = model.backward_with_batch(&chunk, loss)?;
            if grads.len() != param_count {
                return Err(rustane::Error::Other(format!(
                    "gradient count {} != param count {}",
                    grads.len(),
                    param_count
                )));
            }
            let grad_norm = l2_norm(&grads);
            if !grad_norm.is_finite() {
                return Err(rustane::Error::Other(format!("invalid grad norm: {grad_norm}")));
            }
            let mut clipped = grads;
            if grad_clip_norm > 0.0 && grad_norm > grad_clip_norm {
                let clip_scale = grad_clip_norm / grad_norm;
                for g in &mut clipped {
                    *g *= clip_scale;
                }
            }
            for (dst, src) in accum_grads.iter_mut().zip(clipped.iter()) {
                *dst += src * scale;
            }
            total_loss += loss * scale;
            total_grad_norm = grad_norm;
            chunk_count += 1;
        }
        if chunk_count != accum_steps {
            return Err(rustane::Error::Other(format!(
                "expected {} chunks, got {}",
                accum_steps, chunk_count
            )));
        }
        let learning_rate = scheduler.get_lr(step as u32);
        optimizer.step(&accum_grads, model.parameters(), learning_rate)?;
        let metrics = SimpleStepMetrics {
            loss: total_loss,
            grad_norm: total_grad_norm,
            learning_rate,
        };
        total_train_batches += 1;

        let should_validate = val_loss_every > 0 && (step == 0 || (step + 1) % val_loss_every == 0 || step + 1 == train_steps);
        let mut val_note = String::from("-");
        if should_validate {
            let (val_loss, val_bpb, val_tokens) = evaluate_validation(
                &mut model,
                &val_dataset,
                val_max_batches,
                val_batch_size,
                chunk_tokens,
                &tokenizer_stats,
            )?;
            last_val_loss = Some(val_loss);
            last_val_bpb = Some(val_bpb);
            val_note = format!("val_loss:{:.4} val_bpb:{:.4} tokens:{}", val_loss, val_bpb, val_tokens);
        }

        if train_log_every > 0 && (step < 10 || (step + 1) % train_log_every == 0 || step + 1 == train_steps) {
            println!(
                "{:>4} | {:>10.6} | {:>10.6} | {:>8.6} | {}",
                step, metrics.loss, metrics.grad_norm, metrics.learning_rate, val_note
            );
        }
    }

    println!("\n✓ Training completed successfully");
    println!("  Training batches: {}", total_train_batches);
    let elapsed_ms = train_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Elapsed time:     {:.0} ms", elapsed_ms);
    if total_train_batches > 0 {
        println!("  Avg step time:    {:.2} ms", elapsed_ms / total_train_batches as f64);
    }
    if let Some(loss) = last_val_loss {
        println!("  Final val loss:   {:.6}", loss);
    }
    if let Some(bpb) = last_val_bpb {
        println!("  Final val BPB:    {:.6}", bpb);
    }

    Ok(())
}

fn evaluate_validation(
    model: &mut TransformerANE,
    dataset: &FineWebSequenceDataset,
    max_batches: usize,
    batch_size: usize,
    chunk_tokens: usize,
    tokenizer_stats: &SentencePieceStats,
) -> Result<(f32, f32, usize)> {
    let sampler = SequentialSampler::new(dataset.len());
    let val_loader = DataLoader::new(dataset.clone(), sampler, batch_size)?;
    let mut val_iter = val_loader.iter();
    let loss_fn = CrossEntropyLoss::new();
    let mut total_loss = 0.0f32;
    let mut total_positions = 0usize;
    let mut total_targets = 0usize;
    let mut total_bytes = 0usize;

    for _ in 0..max_batches {
        let batch = match val_iter.next() {
            Some(batch) => batch?,
            None => break,
        };

        for chunk in batch.into_chunks(chunk_tokens)? {
            let logits = model.forward(&chunk)?;
            let loss = loss_fn.compute(&logits, &chunk)?;
            let chunk_targets = chunk.batch_size() * chunk.seq_len().saturating_sub(1);
            let chunk_bytes = tokenizer_stats.byte_count_for_batch(&chunk)?;
            total_loss += loss * chunk_targets as f32;
            total_positions += chunk_targets;
            total_targets += chunk_targets;
            total_bytes += chunk_bytes;
        }
    }

    if total_positions == 0 || total_bytes == 0 {
        return Err(rustane::Error::Other(
            "validation produced zero positions".to_string(),
        ));
    }

    let avg_loss = total_loss / total_positions as f32;
    let tokens_per_byte = total_targets as f32 / total_bytes as f32;
    let val_bpb = (avg_loss / std::f32::consts::LN_2) * tokens_per_byte;
    Ok((avg_loss, val_bpb, total_targets))
}

fn build_scheduler(
    name: &str,
    peak_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
    min_lr: f32,
) -> Box<dyn rustane::training::LRScheduler> {
    match name.to_ascii_lowercase().as_str() {
        "linear" => Box::new(WarmupLinearScheduler::new(peak_lr, warmup_steps, total_steps)),
        "constant" => Box::new(ConstantScheduler::new(peak_lr)),
        _ => Box::new(WarmupCosineScheduler::new(peak_lr, warmup_steps, total_steps, min_lr)),
    }
}

struct SimpleStepMetrics {
    loss: f32,
    grad_norm: f32,
    learning_rate: f32,
}

fn l2_norm(grads: &[f32]) -> f32 {
    grads.iter().map(|g| g * g).sum::<f32>().sqrt()
}

#[derive(Clone)]
struct FineWebSequenceDataset {
    samples: Vec<Vec<u32>>,
}

impl FineWebSequenceDataset {
    fn load(pattern: &str, seq_len: usize, max_shards: usize) -> Result<Self> {
        let mut shard_paths: Vec<PathBuf> = glob(pattern)
            .map_err(|e| rustane::Error::Other(format!("invalid glob pattern: {e}")))?
            .filter_map(|entry| entry.ok())
            .collect();
        shard_paths.sort();

        if shard_paths.is_empty() {
            return Err(rustane::Error::Other(format!(
                "no shard files found matching pattern: {}",
                pattern
            )));
        }

        let mut tokens = Vec::new();
        for path in shard_paths.into_iter().take(max_shards) {
            tokens.extend(load_fineweb_shard_tokens(&path)?);
        }

        let usable = (tokens.len() / seq_len) * seq_len;
        if usable == 0 {
            return Err(rustane::Error::Other(format!(
                "FineWeb shards did not contain enough tokens for seq_len={seq_len}"
            )));
        }

        let samples = tokens[..usable]
            .chunks(seq_len)
            .map(|chunk| chunk.to_vec())
            .collect();
        Ok(Self { samples })
    }
}

impl Dataset for FineWebSequenceDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> Result<Vec<u32>> {
        self.samples.get(idx).cloned().ok_or_else(|| {
            rustane::Error::InvalidParameter(format!(
                "dataset index out of bounds: {} >= {}",
                idx,
                self.samples.len()
            ))
        })
    }
}

fn load_fineweb_shard_tokens(path: &Path) -> Result<Vec<u32>> {
    let mut file = File::open(path).map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut header_buf = [0u8; 256 * 4];
    file.read_exact(&mut header_buf)
        .map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut header = [0i32; 256];
    for (idx, chunk) in header_buf.chunks_exact(4).enumerate() {
        header[idx] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    if header[0] != 20240520 || header[1] != 1 {
        return Err(rustane::Error::Other(format!(
            "unexpected FineWeb shard header for {}",
            path.display()
        )));
    }

    let num_tokens = header[2].max(0) as usize;
    let mut token_buf = vec![0u8; num_tokens * 2];
    file.seek(SeekFrom::Start((256 * 4) as u64))
        .map_err(|e| rustane::Error::Io(e.to_string()))?;
    file.read_exact(&mut token_buf)
        .map_err(|e| rustane::Error::Io(e.to_string()))?;

    let mut tokens = Vec::with_capacity(num_tokens);
    for chunk in token_buf.chunks_exact(2) {
        tokens.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
    }
    Ok(tokens)
}

struct SentencePieceStats {
    base_bytes: Vec<u32>,
    has_leading_space: Vec<bool>,
    is_boundary_token: Vec<bool>,
}

impl SentencePieceStats {
    fn load(path: &str) -> Result<Self> {
        let model = SentencePieceModel::from_file(path)
            .map_err(|e| rustane::Error::Other(format!("failed to load tokenizer model {}: {}", path, e)))?;

        let vocab_size = model.pieces().len();
        let mut base_bytes = vec![0u32; vocab_size];
        let mut has_leading_space = vec![false; vocab_size];
        let mut is_boundary_token = vec![true; vocab_size];

        for (idx, piece) in model.pieces().iter().enumerate() {
            let text = piece
                .piece
                .as_ref()
                .ok_or_else(|| rustane::Error::Other(format!("token {} missing piece string", idx)))?;
            let ty = piece.r#type.unwrap_or(1);
            let is_control = ty == 3 || ty == 2 || ty == 5;
            let is_byte = ty == 6;
            let is_unknown = ty == 2;
            let is_unused = ty == 5;

            if is_control || is_unknown || is_unused {
                is_boundary_token[idx] = true;
                continue;
            }

            is_boundary_token[idx] = false;
            if is_byte {
                base_bytes[idx] = 1;
                continue;
            }

            let mut piece_text = text.clone();
            if let Some(stripped) = piece_text.strip_prefix('▁') {
                has_leading_space[idx] = true;
                piece_text = stripped.to_string();
            }
            base_bytes[idx] = piece_text.as_bytes().len() as u32;
        }

        Ok(Self {
            base_bytes,
            has_leading_space,
            is_boundary_token,
        })
    }

    fn byte_count_for_batch(&self, batch: &Batch) -> Result<usize> {
        let mut bytes = 0usize;
        let batch_size = batch.batch_size();
        let seq_len = batch.seq_len();

        for sample_idx in 0..batch_size {
            let sample_offset = sample_idx * seq_len;
            for pos in 0..seq_len.saturating_sub(1) {
                let prev_id = batch.tokens()[sample_offset + pos] as usize;
                let tgt_id = batch.tokens()[sample_offset + pos + 1] as usize;
                let base = *self.base_bytes.get(tgt_id).ok_or_else(|| {
                    rustane::Error::Other(format!("token id {} out of bounds for tokenizer", tgt_id))
                })? as usize;
                let has_space = *self.has_leading_space.get(tgt_id).ok_or_else(|| {
                    rustane::Error::Other(format!("token id {} out of bounds for tokenizer", tgt_id))
                })?;
                let boundary = *self.is_boundary_token.get(prev_id).ok_or_else(|| {
                    rustane::Error::Other(format!("token id {} out of bounds for tokenizer", prev_id))
                })?;
                bytes += base + if has_space && !boundary { 1 } else { 0 };
            }
        }
        Ok(bytes)
    }
}
