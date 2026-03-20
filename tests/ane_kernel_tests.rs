/// Tests for ANEKernel struct and I/O operations
use rustane::ane::ANEKernel;

#[test]
fn test_ane_kernel_creation() {
    // Kernel creation with basic input/output sizes
    let input_sizes = vec![256, 512];
    let output_sizes = vec![1024];

    let kernel = ANEKernel::new(input_sizes.clone(), output_sizes.clone());

    assert!(kernel.is_ok());
    let kernel = kernel.unwrap();
    assert_eq!(kernel.input_sizes.len(), 2);
    assert_eq!(kernel.output_sizes.len(), 1);
}

#[test]
fn test_ane_kernel_input_output_surfaces_created() {
    let input_sizes = vec![128];
    let output_sizes = vec![256];

    let kernel = ANEKernel::new(input_sizes, output_sizes);
    assert!(kernel.is_ok());

    let kernel = kernel.unwrap();
    assert_eq!(kernel.io_inputs.len(), 1);
    assert_eq!(kernel.io_outputs.len(), 1);
}

#[test]
fn test_ane_kernel_write_input_valid_size() {
    let input_sizes = vec![16];
    let output_sizes = vec![32];

    let mut kernel = ANEKernel::new(input_sizes, output_sizes).unwrap();

    // 4 f32 values = 16 bytes
    let data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let result = kernel.write_input(0, &data);

    assert!(result.is_ok());
}

#[test]
fn test_ane_kernel_write_input_size_mismatch() {
    let input_sizes = vec![16];
    let output_sizes = vec![32];

    let mut kernel = ANEKernel::new(input_sizes, output_sizes).unwrap();

    // 8 f32 values = 32 bytes (doesn't match 16 byte input)
    let data = vec![1.0f32; 8];
    let result = kernel.write_input(0, &data);

    assert!(result.is_err());
    match result {
        Err(rustane::Error::Io(_)) => {}, // Expected
        _ => panic!("Expected size mismatch error"),
    }
}

#[test]
fn test_ane_kernel_write_input_invalid_index() {
    let input_sizes = vec![16];
    let output_sizes = vec![32];

    let mut kernel = ANEKernel::new(input_sizes, output_sizes).unwrap();

    let data = vec![1.0f32; 4];
    let result = kernel.write_input(5, &data);

    assert!(result.is_err());
}

#[test]
fn test_ane_kernel_read_output_valid() {
    let input_sizes = vec![16];
    let output_sizes = vec![32]; // 8 f32 values

    let mut kernel = ANEKernel::new(input_sizes, output_sizes).unwrap();

    // Write some data to the output surface first
    let write_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32];
    kernel.io_outputs[0].write(&unsafe {
        std::slice::from_raw_parts(
            write_data.as_ptr() as *const u8,
            write_data.len() * std::mem::size_of::<f32>(),
        )
    }).unwrap();

    let result = kernel.read_output(0);
    assert!(result.is_ok());

    let output = result.unwrap();
    assert_eq!(output.len(), 8);
    assert_eq!(output[0], 1.0f32);
    assert_eq!(output[7], 8.0f32);
}

#[test]
fn test_ane_kernel_read_output_invalid_index() {
    let input_sizes = vec![16];
    let output_sizes = vec![32];

    let kernel = ANEKernel::new(input_sizes, output_sizes).unwrap();

    let result = kernel.read_output(5);
    assert!(result.is_err());
}

#[test]
fn test_ane_kernel_eval_not_initialized() {
    let input_sizes = vec![16];
    let output_sizes = vec![32];

    let mut kernel = ANEKernel::new(input_sizes, output_sizes).unwrap();

    // eval() should fail when kernel not initialized
    let result = kernel.eval();
    assert!(result.is_err());
}

#[test]
fn test_ane_kernel_multiple_inputs_outputs() {
    let input_sizes = vec![64, 128, 256];
    let output_sizes = vec![512, 1024];

    let kernel = ANEKernel::new(input_sizes.clone(), output_sizes.clone()).unwrap();

    assert_eq!(kernel.io_inputs.len(), 3);
    assert_eq!(kernel.io_outputs.len(), 2);
    assert_eq!(kernel.input_sizes, input_sizes);
    assert_eq!(kernel.output_sizes, output_sizes);
}

#[test]
fn test_ane_kernel_empty_inputs_and_outputs() {
    let kernel = ANEKernel::new(vec![], vec![]);
    assert!(kernel.is_ok());

    let kernel = kernel.unwrap();
    assert_eq!(kernel.io_inputs.len(), 0);
    assert_eq!(kernel.io_outputs.len(), 0);
}
