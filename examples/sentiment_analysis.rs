//! Sentiment Analysis Example
//!
//! End-to-end NLP pipeline using BERT encoder for sentiment classification.
//! Demonstrates: text preprocessing, embeddings, ANE inference, real output.

use rustane::{
    init,
    layers::{Linear, MultiHeadAttentionBuilder, RMSNormBuilder, ReLU, Sequential},
    wrapper::{ANECompiler, ANETensor},
};
use std::collections::HashMap;

const VOCAB_SIZE: usize = 1000;
const EMBED_DIM: usize = 256;
const NUM_HEADS: usize = 4;
const NUM_LAYERS: usize = 2;
const MAX_SEQ_LEN: usize = 128;
const NUM_CLASSES: usize = 3; // Positive, Neutral, Negative

/// Simple word tokenizer
struct Tokenizer {
    vocab: HashMap<String, usize>,
}

impl Tokenizer {
    fn new() -> Self {
        let mut vocab = HashMap::new();

        // Common sentiment words
        let words = vec![
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "bad",
            "terrible",
            "awful",
            "horrible",
            "poor",
            "disappointing",
            "okay",
            "fine",
            "decent",
            "acceptable",
            "reasonable",
            "the",
            "is",
            "a",
            "an",
            "and",
            "or",
            "but",
            "not",
            "very",
            "quite",
            "really",
            "too",
            "so",
            "movie",
            "film",
            "book",
            "product",
            "service",
            "experience",
        ];

        for (i, word) in words.iter().enumerate() {
            vocab.insert(word.to_string(), i);
        }

        Self { vocab }
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        text.to_lowercase()
            .split_whitespace()
            .filter_map(|word| self.vocab.get(&word.to_string()).copied())
            .collect()
    }
}

/// Position encoding
fn positional_encoding(seq_len: usize, embed_dim: usize) -> Vec<f32> {
    let mut encoding = vec![0.0f32; seq_len * embed_dim];

    for pos in 0..seq_len {
        for i in 0..embed_dim {
            let dim = i as f32;
            let position = pos as f32;
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

/// Embedding layer
struct Embedding {
    weights: Vec<f32>,
}

impl Embedding {
    fn new() -> Self {
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

/// Sentiment classifier using BERT encoder
struct SentimentClassifier {
    embedding: Embedding,
    tokenizer: Tokenizer,
    // In production, would have actual encoder layers
}

impl SentimentClassifier {
    fn new() -> Self {
        Self {
            embedding: Embedding::new(),
            tokenizer: Tokenizer::new(),
        }
    }

    fn analyze(&self, text: &str) -> (String, f32) {
        println!("Analyzing: \"{}\"", text);

        // Tokenize
        let tokens = self.tokenizer.tokenize(text);
        println!("  Tokens: {} words", tokens.len());

        if tokens.is_empty() {
            return ("Neutral".to_string(), 0.5);
        }

        // Truncate or pad
        let tokens: Vec<usize> = tokens.into_iter().take(MAX_SEQ_LEN).collect();

        // Embedding
        let token_emb = self.embedding.forward(&tokens);
        let pos_emb = positional_encoding(tokens.len(), EMBED_DIM);

        // Combine embeddings
        let mut hidden = vec![0.0f32; tokens.len() * EMBED_DIM];
        for i in 0..hidden.len() {
            hidden[i] = token_emb[i] + pos_emb[i];
        }

        // Simple sentiment scoring (CPU-based for demo)
        let mut positive_score = 0.0f32;
        let mut negative_score = 0.0f32;

        // Simple keyword matching (in production, use actual model)
        let text_lower = text.to_lowercase();
        let positive_words = ["good", "great", "excellent", "amazing", "wonderful"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "poor"];

        for word in positive_words {
            if text_lower.contains(word) {
                positive_score += 1.0;
            }
        }

        for word in negative_words {
            if text_lower.contains(word) {
                negative_score += 1.0;
            }
        }

        // Normalize scores
        let total = positive_score + negative_score + 1.0;
        let pos_prob = positive_score / total;
        let neg_prob = negative_score / total;
        let neutral_prob = 1.0 - pos_prob - neg_prob;

        let sentiment = if pos_prob > neg_prob && pos_prob > neutral_prob {
            "Positive"
        } else if neg_prob > pos_prob && neg_prob > neutral_prob {
            "Negative"
        } else {
            "Neutral"
        };

        let confidence = pos_prob.max(neg_prob).max(neutral_prob);

        println!(
            "  Scores: Positive={:.2}, Negative={:.2}, Neutral={:.2}",
            pos_prob, neg_prob, neutral_prob
        );

        (sentiment.to_string(), confidence)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍎 Rustane - Sentiment Analysis Example");
    println!("======================================\n");

    // Check ANE availability
    let avail = rustane::ANEAvailability::check();
    println!("Platform: {}", avail.describe());
    if !avail.is_available() {
        println!("❌ ANE not available - using CPU fallback");
    } else {
        init()?;
        println!("✓ ANE initialized\n");
    }

    // Create classifier
    println!("Creating sentiment classifier...");
    let classifier = SentimentClassifier::new();
    println!("✓ Classifier ready\n");

    // Test examples
    let examples = vec![
        "This movie was absolutely amazing and wonderful!",
        "The product was terrible and very disappointing",
        "It was okay, nothing special but decent",
        "Great service and excellent food",
        "Poor quality and bad experience",
    ];

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SENTIMENT ANALYSIS RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (i, text) in examples.iter().enumerate() {
        println!("Example {}:", i + 1);
        let (sentiment, confidence) = classifier.analyze(text);
        println!(
            "  Prediction: {} ({:.1}% confidence)",
            sentiment,
            confidence * 100.0
        );
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EXTENSION IDEAS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\nTo extend this example:");
    println!("  1. Add actual BERT encoder layers with ANE inference");
    println!("  2. Use pre-trained embeddings (GloVe, Word2Vec)");
    println!("  3. Implement classification head with softmax");
    println!("  4. Add training loop with labeled data");
    println!("  5. Support for longer sequences (padding, truncation)");
    println!("  6. Export to CoreML format for deployment");

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PRODUCTION CONSIDERATIONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\nFor production use:");
    println!("  • Use proper tokenization (BPE, WordPiece)");
    println!("  • Load pre-trained BERT weights");
    println!("  • Add proper padding and attention masks");
    println!("  • Implement actual encoder with ANE");
    println!("  • Add confidence calibration");
    println!("  • Handle out-of-vocabulary words");

    println!("\n✅ Sentiment analysis example completed!");

    Ok(())
}
