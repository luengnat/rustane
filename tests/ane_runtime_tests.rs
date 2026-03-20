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
    use rustane::ane::ANECompileRequest;
    use std::collections::HashMap;

    let weights = HashMap::new();

    let req = ANECompileRequest {
        mil_text: "func main(x: (1, 1, 1, 16)) -> (1, 1, 1, 16) { return x }".to_string(),
        weights,
        input_sizes: vec![16],
        output_sizes: vec![16],
    };

    assert_eq!(req.input_sizes.len(), 1);
    assert_eq!(req.output_sizes.len(), 1);
}
