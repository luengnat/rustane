use rustane::*;
use rustane::training::TransformerConfig;
use rustane::layers::backward::{RMSNormBackwardGen, BackwardMILGenerator};

fn main() -> Result<()> {
    let config = TransformerConfig::tiny();
    println!("Config: dim={}, seq_len={}", config.dim, config.seq_len);
    
    let gen = RMSNormBackwardGen::new();
    let mil = gen.generate(&config)?;
    
    println!("=== RMSNorm Backward MIL ===");
    println!("{}", mil);
    println!("=== End MIL ===");
    
    // Try to compile
    let d_out_bytes = config.seq_len * config.dim * 4;
    let x_bytes = config.seq_len * config.dim * 4;
    let w_bytes = config.dim * 4;
    
    let req = ane::ANECompileRequest::new(
        &mil,
        vec![d_out_bytes, x_bytes, w_bytes],  // 3 inputs
        vec![config.seq_len * config.dim * 4, config.dim * 4],  // 2 outputs
    );
    
    match req.compile() {
        Ok(_) => println!("SUCCESS: MIL compiled on ANE"),
        Err(e) => println!("FAILED: {}", e),
    }
    
    Ok(())
}
