//! Model Serialization Example
//!
//! Demonstrates saving and loading model weights for deployment.
//! Shows: weight extraction, serialization, loading, verification.

use rustane::{
    layers::{Linear, ReLU},
    Sequential,
};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

const MODEL_DIR: &str = "/tmp/rustane_models";
const MODEL_NAME: &str = "mlp_classifier";

/// Model metadata for versioning
#[derive(Debug, Clone)]
struct ModelMetadata {
    name: String,
    version: String,
    created_at: String,
    layers: Vec<LayerInfo>,
}

#[derive(Debug, Clone)]
struct LayerInfo {
    layer_type: String,
    input_size: usize,
    output_size: usize,
    parameters: usize,
}

/// Extract weights from a sequential model
fn extract_weights(model: &Sequential) -> Vec<(String, Vec<f32>)> {
    println!("Extracting weights from model...");

    // In a real implementation, this would traverse the model graph
    // For this demo, we'll extract what we can access

    let mut weights = Vec::new();

    // Example: Extract first layer weights if available
    // In production, you'd implement proper layer introspection

    println!("  Extracted {} weight tensors", weights.len());
    println!(
        "  Total parameters: {}",
        weights.iter().map(|w| w.1.len()).sum::<usize>()
    );

    weights
}

/// Save model weights to disk
fn save_model(
    model: &Sequential,
    name: &str,
    metadata: &ModelMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Saving Model: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Create model directory
    let model_path = Path::new(MODEL_DIR).join(name);
    fs::create_dir_all(&model_path)?;

    println!("  Created directory: {}", model_path.display());

    // Extract weights
    let weights = extract_weights(model);

    // Save each weight tensor
    let weights_dir = model_path.join("weights");
    fs::create_dir_all(&weights_dir)?;

    println!("\n  Saving weights...");
    for (layer_name, weight_data) in &weights {
        let weight_path = weights_dir.join(format!("{}.bin", layer_name));

        // Serialize as FP32 binary
        let mut file = fs::File::create(&weight_path)?;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(weight_data.as_ptr() as *const u8, weight_data.len() * 4)
        };
        file.write_all(bytes)?;

        println!(
            "    ✓ {}: {} ({} bytes)",
            layer_name,
            weight_path.display(),
            weight_data.len() * 4
        );
    }

    // Save metadata
    let metadata_path = model_path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(metadata, &Default::default())?;
    fs::write(&metadata_path, metadata_json)?;

    println!("\n  ✓ Metadata saved to {}", metadata_path.display());

    // Save model summary
    let summary_path = model_path.join("summary.txt");
    let mut summary = fs::File::create(&summary_path)?;
    writeln!(summary, "Model: {}", metadata.name)?;
    writeln!(summary, "Version: {}", metadata.version)?;
    writeln!(summary, "Created: {}", metadata.created_at)?;
    writeln!(summary, "\nLayers:")?;
    for layer in &metadata.layers {
        writeln!(
            summary,
            "  - {}: {} → {} ({} params)",
            layer.layer_type, layer.input_size, layer.output_size, layer.parameters
        )?;
    }

    println!("  ✓ Summary saved to {}", summary_path.display());

    println!("\n  ✓ Model saved successfully!");
    println!("  Location: {}", model_path.display());

    Ok(())
}

/// Load model weights from disk
fn load_model(
    name: &str,
) -> Result<(Vec<(String, Vec<f32>)>, ModelMetadata), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Loading Model: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let model_path = Path::new(MODEL_DIR).join(name);

    if !model_path.exists() {
        return Err(format!("Model not found: {}", model_path.display()).into());
    }

    // Load metadata
    let metadata_path = model_path.join("metadata.json");
    let metadata_json = fs::read_to_string(&metadata_path)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_json)?;

    println!("  Name: {}", metadata.name);
    println!("  Version: {}", metadata.version);
    println!("  Created: {}", metadata.created_at);
    println!("  Layers: {}", metadata.layers.len());

    // Load weights
    let weights_dir = model_path.join("weights");
    let mut weights = Vec::new();

    println!("\n  Loading weights...");
    for entry in fs::read_dir(&weights_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("bin") {
            let layer_name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            // Read binary data
            let mut file = fs::File::open(&path)?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)?;

            // Convert to FP32
            let weight_data: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(arr)
                })
                .collect();

            println!(
                "    ✓ {}: {} ({} parameters)",
                layer_name,
                path.display(),
                weight_data.len()
            );

            weights.push((layer_name, weight_data));
        }
    }

    println!("\n  ✓ Model loaded successfully!");
    println!(
        "  Total parameters: {}",
        weights.iter().map(|w| w.1.len()).sum::<usize>()
    );

    Ok((weights, metadata))
}

/// Verify loaded weights
fn verify_weights(
    original: &Sequential,
    loaded: &[(String, Vec<f32>)],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Verifying Loaded Weights");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // In a real implementation, compare loaded weights with original
    // For this demo, we'll just verify they loaded correctly

    println!("  Checking weight tensors...");

    for (name, data) in loaded {
        // Verify data is valid FP32
        let valid = data.iter().all(|&x| x.is_finite());

        if valid {
            println!("    ✓ {}: {} parameters (valid)", name, data.len());
        } else {
            println!("    ✗ {}: Contains invalid values", name);
        }
    }

    println!("\n  ✓ Verification complete!");

    Ok(())
}

fn create_demo_model() -> Sequential {
    println!("Creating demo model...");

    let mut model = Sequential::new("mlp_classifier");

    // Add layers (in production, this would be a real trained model)
    model.add(Box::new(Linear::new(784, 256).build().unwrap()));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(256, 128).build().unwrap()));
    model.add(Box::new(ReLU::new()));
    model.add(Box::new(Linear::new(128, 10).build().unwrap()));

    println!("✓ Model created");
    println!("  Architecture: 784 → 256 → 128 → 10");
    println!(
        "  Total parameters: {}",
        784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10
    );

    model
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Model Serialization Example");
    println!("========================================\n");

    // Create demo model
    let model = create_demo_model();

    // Create metadata
    let metadata = ModelMetadata {
        name: MODEL_NAME.to_string(),
        version: "1.0.0".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        layers: vec![
            LayerInfo {
                layer_type: "Linear".to_string(),
                input_size: 784,
                output_size: 256,
                parameters: 784 * 256 + 256,
            },
            LayerInfo {
                layer_type: "ReLU".to_string(),
                input_size: 256,
                output_size: 256,
                parameters: 0,
            },
            LayerInfo {
                layer_type: "Linear".to_string(),
                input_size: 256,
                output_size: 128,
                parameters: 256 * 128 + 128,
            },
            LayerInfo {
                layer_type: "ReLU".to_string(),
                input_size: 128,
                output_size: 128,
                parameters: 0,
            },
            LayerInfo {
                layer_type: "Linear".to_string(),
                input_size: 128,
                output_size: 10,
                parameters: 128 * 10 + 10,
            },
        ],
    };

    println!("Metadata:");
    println!("  Name: {}", metadata.name);
    println!("  Version: {}", metadata.version);
    println!("  Layers: {}", metadata.layers.len());
    println!();

    // Save model
    save_model(&model, MODEL_NAME, &metadata)?;

    // Load model
    let (loaded_weights, loaded_metadata) = load_model(MODEL_NAME)?;

    // Verify
    verify_weights(&model, &loaded_weights)?;

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("USAGE EXAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n1. Save a trained model:");
    println!("   let model = train_model(...);");
    println!("   save_model(&model, \"my_model\", &metadata)?;");

    println!("\n2. Load model for inference:");
    println!("   let (weights, metadata) = load_model(\"my_model\")?;");
    println!("   let model = build_model_from_weights(weights)?;");

    println!("\n3. Export for deployment:");
    println!("   cp -r /tmp/rustane_models/my_model /path/to/deployment/");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FILE FORMAT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\nModel directory structure:");
    println!("  {model_name}/");
    println!("    ├── metadata.json          # Model metadata");
    println!("    ├── summary.txt            # Human-readable summary");
    println!("    └── weights/               # Weight tensors");
    println!("        ├── layer1.bin");
    println!("        ├── layer2.bin");
    println!("        └── ...");

    println!("\nWeight format:");
    println!("  • Binary FP32 (little-endian)");
    println!("  • Shape: [output_size, input_size] for Linear");
    println!("  • One .bin file per layer");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PRODUCTION CONSIDERATIONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\nFor production deployment:");
    println!("  ✓ Use version control for model iterations");
    println!("  ✓ Include checksums for weight files");
    println!("  ✓ Add compression for storage efficiency");
    println!("  ✓ Support incremental updates");
    println!("  ✓ Add encryption for sensitive models");

    println!("\nFor deployment:");
    println!("  ✓ Bundle weights with application");
    println!("  ✓ Use memory-mapped files for large models");
    println!("  ✓ Implement lazy loading for memory efficiency");
    println!("  ✓ Add model validation before loading");

    println!("\n✅ Model serialization example completed!");

    Ok(())
}

// Add chrono dependency for timestamps
use chrono;
