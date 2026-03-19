//! BERT-like Encoder Example
//!
//! Demonstrates a BERT-style encoder for sequence classification using
//! multi-head attention without causal masking.

use rustane::layers::{
    Linear, MultiHeadAttentionBuilder, RMSNormBuilder, ReLU, Sequential, SequentialBuilder,
    SwiGLUBuilder,
};

// Model configuration
const VOCAB_SIZE: usize = 1000;
const EMBED_DIM: usize = 256;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS; // 64
const FF_HIDDEN: usize = EMBED_DIM * 4; // 4x expansion (SwiGLU)
const NUM_CLASSES: usize = 4;
const MAX_SEQ_LEN: usize = 128;
const BATCH_SIZE: usize = 2;

/// BERT-style encoder layer
///
/// Architecture:
/// 1. Multi-Head Attention (non-causal, bidirectional)
/// 2. RMSNorm
/// 3. SwiGLU Feed-Forward Network
/// 4. RMSNorm
///
/// This differs from GPT in that attention is bidirectional (no causal mask).
struct BERTEncoderLayer {
    attention: MultiHeadAttentionBuilder,
    norm1: RMSNormBuilder,
    ffn: Sequential,
    norm2: RMSNormBuilder,
}

impl BERTEncoderLayer {
    fn new() -> Self {
        Self {
            attention: MultiHeadAttentionBuilder::new(EMBED_DIM, NUM_HEADS)
                .with_causal(false) // BERT is bidirectional
                .with_name("bert_attention"),
            norm1: RMSNormBuilder::new(EMBED_DIM).with_name("bert_norm1"),
            norm2: RMSNormBuilder::new(EMBED_DIM).with_name("bert_norm2"),
            ffn: Sequential::new("bert_ffn")
                .add(Box::new(
                    SwiGLUBuilder::new(EMBED_DIM)
                        .with_multiplier(4)
                        .build()
                        .unwrap(),
                ))
                .add(Box::new(ReLU::new()))
                .build(),
        }
    }
}

/// Simple token embedding (learnable)
struct TokenEmbedding {
    weights: Vec<f32>,
}

impl TokenEmbedding {
    fn new() -> Self {
        // Initialize with random weights
        let std_dev = (2.0 / (VOCAB_SIZE + EMBED_DIM) as f32).sqrt();
        let weights: Vec<f32> = (0..VOCAB_SIZE * EMBED_DIM)
            .map(|_| (rand::random::<f32>() * 2.0 - 1.0) * std_dev)
            .collect();

        Self { weights }
    }

    fn forward(&self, tokens: &[usize]) -> Vec<f32> {
        let seq_len = tokens.len();
        let mut output = vec![0.0f32; seq_len * EMBED_DIM];

        for (pos, &token_id) in tokens.iter().enumerate() {
            if token_id < VOCAB_SIZE {
                let offset = token_id * EMBED_DIM;
                for i in 0..EMBED_DIM {
                    output[pos * EMBED_DIM + i] = self.weights[offset + i];
                }
            }
        }

        output
    }
}

/// Position encoding (sinusoidal, like original Transformer)
fn positional_encoding(seq_len: usize, embed_dim: usize) -> Vec<f32> {
    let mut encoding = vec![0.0f32; seq_len * embed_dim];

    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let dim = i as f32;
            let position = pos as f32;

            // Original Transformer formula
            let div_term = std::f32::consts::PI
                / (10000.0f32).powf(2.0 * (dim / 2.0) as f32 / embed_dim as f32);

            encoding[pos * embed_dim + i] = if i % 2 == 0 {
                (position * div_term).sin()
            } else {
                (position * div_term).cos()
            };
        }
    }

    encoding
}

/// BERT-style classifier
struct BERTClassifier {
    embedding: TokenEmbedding,
    encoder_layers: Vec<BERTEncoderLayer>,
    classifier: rustane::layers::Linear,
}

impl BERTClassifier {
    fn new(num_layers: usize) -> Self {
        let mut encoder_layers = Vec::new();
        for _ in 0..num_layers {
            encoder_layers.push(BERTEncoderLayer::new());
        }

        Self {
            embedding: TokenEmbedding::new(),
            encoder_layers,
            classifier: Linear::new(EMBED_DIM, NUM_CLASSES).build().unwrap(),
        }
    }

    fn forward_cpu(&self, tokens: &[usize]) -> (Vec<f32>, Vec<f32>) {
        // Embedding
        let token_emb = self.embedding.forward(tokens);
        let pos_emb = positional_encoding(tokens.len(), EMBED_DIM);

        // Combine embeddings
        let mut hidden = vec![0.0f32; tokens.len() * EMBED_DIM];
        for i in 0..hidden.len() {
            hidden[i] = token_emb[i] + pos_emb[i];
        }

        println!("Input shape: [{} × {}]", tokens.len(), EMBED_DIM);

        // Encoder layers (simplified - CPU only)
        for (layer_idx, _layer) in self.encoder_layers.iter().enumerate() {
            println!("  Encoder layer {}", layer_idx);

            // In a full implementation, this would run the attention + FFN
            // For this demo, we just apply a simple transformation
            for val in hidden.iter_mut() {
                *val = (*val).tanh(); // Non-linearity
            }
        }

        // Pooling (mean pool)
        let mut pooled = vec![0.0f32; EMBED_DIM];
        for i in 0..EMBED_DIM {
            let mut sum = 0.0f32;
            for j in 0..tokens.len() {
                sum += hidden[j * EMBED_DIM + i];
            }
            pooled[i] = sum / tokens.len() as f32;
        }

        println!("Pooled shape: [{}]", EMBED_DIM);

        // Classification
        let mut logits = vec![0.0f32; NUM_CLASSES];
        let classifier_weights = vec![1.0f32; EMBED_DIM * NUM_CLASSES]; // Identity-like
        let classifier_bias = vec![0.0f32; NUM_CLASSES];

        for i in 0..NUM_CLASSES {
            for j in 0..EMBED_DIM {
                logits[i] += pooled[j] * classifier_weights[j * NUM_CLASSES + i];
            }
            logits[i] += classifier_bias[i];
        }

        println!("Logits shape: [{}]", NUM_CLASSES);

        // Softmax
        let mut exp_sum = 0.0f32;
        for &logit in &logits {
            exp_sum += logit.exp();
        }

        let mut probs = vec![0.0f32; NUM_CLASSES];
        for (i, &logit) in logits.iter().enumerate() {
            probs[i] = logit.exp() / exp_sum;
        }

        (logits, probs)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - BERT-like Encoder Example");
    println!("======================================\n");

    // Model configuration
    let num_layers = 2;

    println!("Model Configuration:");
    println!("  Vocabulary: {}", VOCAB_SIZE);
    println!("  Embed dim: {}", EMBED_DIM);
    println!("  Heads: {}", NUM_HEADS);
    println!("  FF hidden: {}", FF_HIDDEN);
    println!("  Classes: {}", NUM_CLASSES);
    println!("  Encoder layers: {}", num_layers);
    println!(
        "  Parameters: ~{}",
        VOCAB_SIZE * EMBED_DIM +           // Embedding
        num_layers * (                      // Encoder layers
            NUM_HEADS * EMBED_DIM * HEAD_DIM * 4 +  // Attention (Q,K,V,O)
            EMBED_DIM * FF_HIDDEN * 3 +             // FFN (SwiGLU: gate, up, down)
            EMBED_DIM * 2                          // 2 RMSNorm layers
        ) +
        EMBED_DIM * NUM_CLASSES + NUM_CLASSES // Classifier
    );
    println!();

    // Create model
    println!("Creating BERT classifier...");
    let model = BERTClassifier::new(num_layers);
    println!("✓ Model created\n");

    // Example input: 2 sequences, 8 tokens each
    let batch1 = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let batch2 = vec![15, 25, 35, 45, 55, 65, 75, 85];

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Processing Batch 1");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let (logits1, probs1) = model.forward_cpu(&batch1);

    println!("\nOutput (Batch 1):");
    println!("  Logits: {:?}", logits1);
    println!("  Probabilities: {:?}", probs1);
    println!(
        "  Predicted class: {}",
        probs1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Processing Batch 2");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let (logits2, probs2) = model.forward_cpu(&batch2);

    println!("\nOutput (Batch 2):");
    println!("  Logits: {:?}", logits2);
    println!("  Probabilities: {:?}", probs2);
    println!(
        "  Predicted class: {}",
        probs2
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    );

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Architecture Details");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n1. EMBEDDING LAYER");
    println!("   Token embedding → Positional encoding → Sum");
    println!("   Shape: [seq_len × {}]", EMBED_DIM);

    println!("\n2. ENCODER LAYERS ({} layers)", num_layers);
    println!("   Each layer:");
    println!("   - Multi-Head Attention (bidirectional)");
    println!("     * {} heads, {} embed_dim", NUM_HEADS, EMBED_DIM);
    println!("     * NO causal mask (can attend to all tokens)");
    println!("   - RMSNorm");
    println!("   - SwiGLU Feed-Forward");
    println!("     * {} → {} → {}", EMBED_DIM, FF_HIDDEN, EMBED_DIM);
    println!("   - RMSNorm");

    println!("\n3. CLASSIFICATION HEAD");
    println!("   - Mean pooling (average over sequence)");
    println!("   - Linear layer → Softmax");
    println!("   - Output: [{}]", NUM_CLASSES);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Key Differences from GPT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\nGPT (Decoder-only):");
    println!("  - Causal attention (can't see future)");
    println!("  - Autoregressive generation");
    println!("  - Used for: Text generation");

    println!("\nBERT (Encoder-only):");
    println!("  - Bidirectional attention (sees entire sequence)");
    println!("  - No generation (encoder only)");
    println!("  - Used for: Classification, NER, QA");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("ANE Integration Notes");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\nFor ANE acceleration:");
    println!("  1. Use scaled_dot_product_attention operation");
    println!("  2. Set causal=false for bidirectional attention");
    println!("  3. Pre-compile MIL for fixed sequence lengths");
    println!("  4. See causal_attention.rs for working example");

    println!("\nCurrent limitation:");
    println!("  - This example uses CPU for simplicity");
    println!("  - Full ANE integration requires MIL compilation");
    println!("  - RMSNorm and SwiGLU need ANE operation support");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Extension Ideas");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\nTo extend this example:");
    println!("  1. Add more encoder layers (6-12 for BERT-base)");
    println!("  2. Implement layer normalization before each sub-layer");
    println!("  3. Add dropout for regularization");
    println!("  4. Load pre-trained embeddings");
    println!("  5. Implement masked language modeling (MLM)");
    println!("  6. Add next sentence prediction (NSP)");
    println!("  7. Integrate with ANE for attention acceleration");

    println!("\n✅ BERT encoder example completed!");
    println!("\nNote: This is a simplified CPU implementation demonstrating");
    println!("the architecture. For production use, integrate with ANE");
    println!("using scaled_dot_product_attention for attention layers.");

    Ok(())
}
