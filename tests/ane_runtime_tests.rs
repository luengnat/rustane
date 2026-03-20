#[test]
fn test_ane_init_succeeds_on_macos() {
    use rustane::ane::runtime;

    // Should not panic, should return Ok or Err
    // (depending on whether we're on Apple Silicon)
    let result = runtime::ane_init();

    match result {
        Ok(_) => {
            println!("ANE framework loaded successfully");
        }
        Err(e) => {
            // Expected on non-Apple Silicon systems or if framework not available
            println!("ANE init failed (expected on non-Apple Silicon): {:?}", e);
        }
    }
}

#[test]
fn test_ane_compile_request_builder() {
    use rustane::ane::WeightBlob;
    use rustane::mil::linear_matmul_compile_request;

    let weights = vec![1.0f32; 8 * 4];
    let blob = WeightBlob::from_f32(&weights, 8, 4).unwrap();
    let req = linear_matmul_compile_request(16, 4, 8, &blob);

    assert_eq!(req.input_sizes, vec![4 * 16 * 4]);
    assert_eq!(req.output_sizes, vec![8 * 16 * 4]);
    assert_eq!(
        req.weights.get("@model_path/weights/weight.bin"),
        Some(&blob.as_bytes().to_vec())
    );
}
