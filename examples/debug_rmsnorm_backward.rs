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
    
    Ok(())
}
