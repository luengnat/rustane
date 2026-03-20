//! Model Serialization Example
//!
//! Demonstrates saving and loading TransformerANE model weights for deployment.
//! Shows: weight extraction, binary serialization, loading, round-trip verification.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example model_serialization
//! ```

use rustane::training::{Model, TransformerANE, TransformerConfig};
use rustane::data::Batch;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

const MODEL_DIR: &str = "/tmp/rustane_models";
const MODEL_NAME: &str = "transformer_ane";

/// Model metadata for versioning
#[derive(Debug, Clone)]
struct ModelMetadata {
    name: String,
    version: String,
    created_at: String,
    vocab_size: usize,
    dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_layers: usize,
    seq_len: usize,
    param_count: usize,
}

/// Save TransformerANE weights to disk
fn save_model(
    model: &mut TransformerANE,
    name: &str,
    meta: &ModelMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Saving Model: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let model_path = Path::new(MODEL_DIR).join(name);
    fs::create_dir_all(&model_path)?;

    // Extract parameters as contiguous f32 slice
    let params: Vec<f32> = model.parameters().to_vec();
    println!("  Parameters: {}", params.len());

    // Serialize as raw FP32 binary (little-endian)
    let weights_path = model_path.join("weights.bin");
    let mut file = fs::File::create(&weights_path)?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(params.as_ptr() as *const u8, params.len() * 4)
    };
    file.write_all(bytes)?;
    println!("  ✓ Weights → {} ({} bytes)", weights_path.display(), bytes.len());

    // Save metadata as JSON using serde_json::json! macro
    let metadata_path = model_path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&serde_json::json!({
        "name": meta.name,
        "version": meta.version,
        "created_at": meta.created_at,
        "config": {
            "vocab_size": meta.vocab_size,
            "dim": meta.dim,
            "hidden_dim": meta.hidden_dim,
            "n_heads": meta.n_heads,
            "n_layers": meta.n_layers,
            "seq_len": meta.seq_len,
        },
        "param_count": meta.param_count,
    }))?;
    fs::write(&metadata_path, metadata_json)?;
    println!("  ✓ Metadata → {}", metadata_path.display());

    println!("  ✓ Model saved to {}", model_path.display());
    Ok(())
}

/// Load TransformerANE weights from disk
fn load_model(
    name: &str,
) -> Result<(Vec<f32>, ModelMetadata), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading Model: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let model_path = Path::new(MODEL_DIR).join(name);
    if !model_path.exists() {
        return Err(format!("Model not found at: {}", model_path.display()).into());
    }

    // Load metadata
    let metadata_path = model_path.join("metadata.json");
    let metadata_json = fs::read_to_string(&metadata_path)?;
    let v: serde_json::Value = serde_json::from_str(&metadata_json)?;
    let cfg = &v["config"];
    let meta = ModelMetadata {
        name: v["name"].as_str().unwrap_or("").to_string(),
        version: v["version"].as_str().unwrap_or("").to_string(),
        created_at: v["created_at"].as_str().unwrap_or("").to_string(),
        vocab_size: cfg["vocab_size"].as_u64().unwrap_or(0) as usize,
        dim: cfg["dim"].as_u64().unwrap_or(0) as usize,
        hidden_dim: cfg["hidden_dim"].as_u64().unwrap_or(0) as usize,
        n_heads: cfg["n_heads"].as_u64().unwrap_or(0) as usize,
        n_layers: cfg["n_layers"].as_u64().unwrap_or(0) as usize,
        seq_len: cfg["seq_len"].as_u64().unwrap_or(0) as usize,
        param_count: v["param_count"].as_u64().unwrap_or(0) as usize,
    };

    println!("  Name:    {}", meta.name);
    println!("  Version: {}", meta.version);
    println!("  Config:  vocab={} dim={} heads={} layers={}",
        meta.vocab_size, meta.dim, meta.n_heads, meta.n_layers);

    // Load weights binary
    let weights_path = model_path.join("weights.bin");
    let mut file = fs::File::open(&weights_path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let weights: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    println!("  ✓ Loaded {} parameters ({} bytes)", weights.len(), bytes.len());
    Ok((weights, meta))
}

/// Verify round-trip: apply loaded weights to new model and compare a forward pass
fn verify_round_trip(
    original: &mut TransformerANE,
    loaded_weights: &[f32],
    config: &TransformerConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Verifying Round-Trip");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create a fresh model and inject loaded weights
    let mut restored = TransformerANE::new(config)?;
    let params = restored.parameters();
    assert_eq!(params.len(), loaded_weights.len(),
        "Parameter count mismatch: {} vs {}", params.len(), loaded_weights.len());
    params.copy_from_slice(loaded_weights);
    println!("  ✓ Weights injected into fresh model");

    // Run the same forward pass on both models
    let tokens: Vec<u32> = (0..config.seq_len as u32).map(|i| i % config.vocab_size as u32).collect();
    let batch = Batch::new(tokens, 1, config.seq_len)?;

    let original_out = original.forward(&batch)?;
    let restored_out = restored.forward(&batch)?;

    assert_eq!(original_out.num_elements(), restored_out.num_elements(),
        "Output shape mismatch");

    // Check all outputs match within floating-point precision
    let orig_data = original_out.to_vec_f32();
    let rest_data = restored_out.to_vec_f32();
    let max_diff = orig_data.iter().zip(rest_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("  Max output difference: {:.2e}", max_diff);
    if max_diff < 1e-5 {
        println!("  ✓ Round-trip verification PASSED (max diff < 1e-5)");
    } else {
        println!("  ✗ Round-trip verification FAILED (max diff = {:.2e})", max_diff);
        return Err("Round-trip verification failed".into());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - ANE Model Serialization Example");
    println!("============================================\n");

    // 1. Configure and initialize TransformerANE model
    let config = TransformerConfig::new(
        512,  // vocab_size
        128,  // dim
        256,  // hidden_dim
        4,    // n_heads
        2,    // n_layers
        64,   // seq_len
    )?;

    println!("Step 1: Initializing TransformerANE...");
    let mut model = TransformerANE::new(&config)?;
    println!("  ✓ Model ready ({} parameters)", config.param_count());

    // 2. Build metadata
    let created_at = {
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!("unix:{}", secs)
    };
    let meta = ModelMetadata {
        name: MODEL_NAME.to_string(),
        version: "1.0.0".to_string(),
        created_at,
        vocab_size: config.vocab_size,
        dim: config.dim,
        hidden_dim: config.hidden_dim,
        n_heads: config.n_heads,
        n_layers: config.n_layers,
        seq_len: config.seq_len,
        param_count: config.param_count(),
    };

    // 3. Save model
    println!("\nStep 2: Saving model weights...");
    save_model(&mut model, MODEL_NAME, &meta)?;

    // 4. Load model
    println!("\nStep 3: Loading model weights...");
    let (loaded_weights, loaded_meta) = load_model(MODEL_NAME)?;
    assert_eq!(loaded_weights.len(), config.param_count());
    println!("  ✓ Version: {}", loaded_meta.version);

    // 5. Round-trip verification
    println!("\nStep 4: Round-trip verification...");
    verify_round_trip(&mut model, &loaded_weights, &config)?;

    // 6. File format summary
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FILE FORMAT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n  {}/", MODEL_NAME);
    println!("    ├── metadata.json   # Config, version, param count");
    println!("    └── weights.bin     # Raw FP32 (little-endian)");
    println!("\n  Weight layout: contiguous f32 parameter vector");
    println!("  Size: {} params × 4 bytes = {} KB",
        config.param_count(), config.param_count() * 4 / 1024);

    println!("\n✅ ANE model serialization example completed!");
    Ok(())
}
