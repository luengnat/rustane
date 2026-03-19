//! Debug MIL generation

use rustane::layers::{Layer, MultiHeadAttentionBuilder};

fn main() {
    let mha = MultiHeadAttentionBuilder::new(64, 4).build().unwrap();

    println!("=== Method Generated MIL ===");
    let mil1 = mha.build_sdpa_mil_program(1, 8);
    println!("{}", mil1);

    println!("\n=== Working MIL from causal_attention ===");
    let mil2 = r#"program(1.3)
[buildInfo = dict<string, string>({"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremlc-component-milinternal", ""}, {"coremltools-version", "9.0"})]
{
    func main<ios18>(tensor<fp16, [1, 4, 8, 16]> q, tensor<fp16, [1, 4, 8, 16]> k, tensor<fp16, [1, 4, 8, 16]> v) {
        tensor<fp16, [1, 4, 8, 16]> att = scaled_dot_product_attention(query = q, key = k, value = v)[name = string("sdpa")];
    } -> (att);
}
"#;

    println!("{}", mil2);

    println!("\n=== Comparison ===");
    if mil1.trim() == mil2.trim() {
        println!("✓ MIL programs match!");
    } else {
        println!("❌ MIL programs differ!");
        println!("\nMethod output:");
        println!("{}", mil1);
        println!("\nExpected output:");
        println!("{}", mil2);
    }
}
