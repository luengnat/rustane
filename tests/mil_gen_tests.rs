//! Tests for MIL code generation
//!
//! Validates that the MILGenerator correctly creates MIL code for
//! attention and feed-forward network operations.

#[test]
fn test_mil_generator_creation() {
    use rustane::layers::MILGenerator;
    use rustane::training::TransformerConfig;

    let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
    let gen = MILGenerator::new(&config);

    assert_eq!(gen.config().dim, 256);
    assert_eq!(gen.config().n_heads, 8);
}

#[test]
fn test_mil_attention_forward_generation() {
    use rustane::layers::MILGenerator;
    use rustane::training::TransformerConfig;

    let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
    let gen = MILGenerator::new(&config);

    let mil = gen.gen_attention_forward();

    // Check for required MIL keywords
    assert!(mil.contains("func "));
    assert!(mil.contains("cast"));
    assert!(mil.len() > 100); // Should be substantial
}

#[test]
fn test_mil_ffn_forward_generation() {
    use rustane::layers::MILGenerator;
    use rustane::training::TransformerConfig;

    let config = TransformerConfig::new(4096, 256, 768, 8, 6, 512).unwrap();
    let gen = MILGenerator::new(&config);

    let mil = gen.gen_ffn_forward();

    // Check for required MIL keywords
    assert!(mil.contains("func "));
    assert!(mil.contains("cast"));
    assert!(mil.len() > 100); // Should be substantial
}
