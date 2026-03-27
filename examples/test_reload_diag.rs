//! Quick diagnostic: does reload_weights actually change ANE output?
use rustane::ane::WeightBlob;
use rustane::mil::programs::conv1x1_mil;
use rustane::wrapper::ANECompiler;

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    rustane::init()?;
    println!("=== Reload Diagnostics ===\n");

    let dim = 64;
    let seq = 16;
    let mil = conv1x1_mil(seq, dim, dim);
    let input_size = dim * seq * 4;
    let output_size = dim * seq * 4;

    // Two very different weight sets
    let w_a: Vec<f32> = vec![0.1; dim * dim];
    let w_b: Vec<f32> = vec![0.9; dim * dim];
    let blob_a = WeightBlob::from_f32(&w_a, dim, dim)?;
    let blob_b = WeightBlob::from_f32(&w_b, dim, dim)?;

    let input: Vec<f32> = vec![0.5; dim * seq];
    let input_bytes = f32_to_bytes(&input);

    // Compile with weights A
    let mut compiler = ANECompiler::new();
    let mut exec = compiler.compile_multi(
        &mil,
        &["@model_path/weights/weight.bin"],
        &[blob_a.as_bytes()],
        &[blob_a.as_bytes().len()],
        &[input_size],
        &[output_size],
    )?;

    // Eval with weights A
    exec.write_input(0, &input_bytes)?;
    exec.eval()?;
    let mut buf_a = vec![0u8; output_size];
    exec.read_output(0, &mut buf_a)?;
    let out_a = bytes_to_f32(&buf_a);

    // Reload with weights B
    exec.reload_weights(&[("@model_path/weights/weight.bin", blob_b.as_bytes())])?;

    // Eval after reload
    exec.write_input(0, &input_bytes)?;
    exec.eval()?;
    let mut buf_b = vec![0u8; output_size];
    exec.read_output(0, &mut buf_b)?;
    let out_b = bytes_to_f32(&buf_b);

    println!("Output A (weights=0.1): first 5 = {:?}", &out_a[..5]);
    println!(
        "Output B after reload (weights=0.9): first 5 = {:?}",
        &out_b[..5]
    );

    let diff_count = out_a
        .iter()
        .zip(out_b.iter())
        .filter(|(a, b)| (*a - *b).abs() > 0.001)
        .count();
    println!(
        "Values differing after reload: {}/{}",
        diff_count,
        out_a.len()
    );

    // Fresh compile with B weights
    println!("\n--- Fresh compile with weights B ---");
    let mut compiler2 = ANECompiler::new();
    let mut exec2 = compiler2.compile_multi(
        &mil,
        &["@model_path/weights/weight.bin"],
        &[blob_b.as_bytes()],
        &[blob_b.as_bytes().len()],
        &[input_size],
        &[output_size],
    )?;
    exec2.write_input(0, &input_bytes)?;
    exec2.eval()?;
    let mut buf_b2 = vec![0u8; output_size];
    exec2.read_output(0, &mut buf_b2)?;
    let out_b2 = bytes_to_f32(&buf_b2);
    println!("Output B fresh compile: first 5 = {:?}", &out_b2[..5]);

    let diff_ab = out_a
        .iter()
        .zip(out_b2.iter())
        .filter(|&(a, b)| (a - b).abs() > 0.001)
        .count();
    println!("A vs B fresh differ: {}/{}", diff_ab, out_a.len());

    Ok(())
}
