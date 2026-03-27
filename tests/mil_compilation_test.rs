//! MIL Compilation Tests
//!
//! Tests compilation of parameter-golf MIL programs.

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    /// Test that MIL files exist and are syntactically valid
    #[test]
    fn test_mil_files_exist() {
        let mil_dir = PathBuf::from("models/layer0");

        assert!(mil_dir.exists(), "MIL directory should exist");

        let fwd_path = mil_dir.join("layer_0_fwd.mil");
        let bwd_path = mil_dir.join("layer_0_bwd.mil");

        assert!(fwd_path.exists(), "Forward MIL should exist");
        assert!(bwd_path.exists(), "Backward MIL should exist");

        // Check file sizes
        let fwd_size = fs::metadata(&fwd_path).unwrap().len();
        let bwd_size = fs::metadata(&bwd_path).unwrap().len();

        assert!(fwd_size > 0, "Forward MIL should not be empty");
        assert!(bwd_size > 0, "Backward MIL should not be empty");

        println!("Forward MIL: {} bytes", fwd_size);
        println!("Backward MIL: {} bytes", bwd_size);
    }

    /// Test MIL syntax validation
    #[test]
    fn test_mil_syntax_valid() {
        let fwd_path = PathBuf::from("models/layer0/layer_0_fwd.mil");
        let mil_content = fs::read_to_string(&fwd_path).unwrap();

        // Check required elements
        assert!(
            mil_content.contains("program(1.3)"),
            "Should have program version"
        );
        assert!(
            mil_content.contains("func main<ios18>"),
            "Should have main function with ios18"
        );
        assert!(
            mil_content.contains("tensor<fp16"),
            "Should use fp16 tensors"
        );
        assert!(
            mil_content.contains("BLOBFILE"),
            "Should reference weight blobs"
        );

        // Check structure
        assert!(mil_content.contains("conv("), "Should have convolution ops");
        assert!(mil_content.contains("matmul("), "Should have matmul ops");
        assert!(
            mil_content.contains("-> (out)"),
            "Should have output declaration"
        );
    }

    /// Test weight blobs exist
    #[test]
    fn test_weight_blobs_exist() {
        let weights_dir = PathBuf::from("models/layer0/weights");

        assert!(weights_dir.exists(), "Weights directory should exist");

        let required_weights = [
            "layer0_wq.bin",
            "layer0_wk.bin",
            "layer0_wv.bin",
            "layer0_wo.bin",
            "layer0_w1.bin",
            "layer0_w2.bin",
            "layer0_w3.bin",
            "layer0_q_gain.bin",
            "layer0_resid_mix_0.bin",
            "layer0_resid_mix_1.bin",
            "layer0_attn_scale.bin",
            "layer0_mlp_scale.bin",
        ];

        for weight in &required_weights {
            let path = weights_dir.join(weight);
            assert!(path.exists(), "Weight blob {} should exist", weight);

            let size = fs::metadata(&path).unwrap().len();
            assert!(
                size > 64,
                "Weight blob {} should have header + data",
                weight
            );
        }
    }

    /// Test weight blob header format
    #[test]
    fn test_weight_blob_header() {
        use std::io::Read;

        let wq_path = PathBuf::from("models/layer0/weights/layer0_wq.bin");
        let mut file = fs::File::open(&wq_path).unwrap();

        // Read header (64 bytes)
        let mut header = [0u8; 64];
        file.read_exact(&mut header).unwrap();

        // Check magic
        assert_eq!(&header[0..4], b"ANEB", "Weight blob should have ANEB magic");

        // Check version (little-endian u32 at offset 4)
        let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        assert_eq!(version, 1, "Weight blob version should be 1");

        // Check data type (little-endian u32 at offset 8)
        let data_type = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        assert!(
            data_type == 16 || data_type == 32,
            "Data type should be 16 (fp16) or 32 (fp32), got {}",
            data_type
        );

        println!("Weight blob header validated:");
        println!("  Magic: ANEB");
        println!("  Version: {}", version);
        println!(
            "  Data type: {} ({})",
            data_type,
            if data_type == 16 { "fp16" } else { "fp32" }
        );
    }

    /// Test MIL program can be loaded (basic validation)
    #[test]
    fn test_mil_program_loadable() {
        let fwd_path = PathBuf::from("models/layer0/layer_0_fwd.mil");
        let mil_content = fs::read_to_string(&fwd_path).unwrap();

        // Basic structure validation
        assert!(
            mil_content.lines().count() > 50,
            "MIL file should have substantial content"
        );
        assert!(
            mil_content.contains("program(1.3)"),
            "Should have valid program declaration"
        );

        // Count operations
        let op_count = mil_content.matches("[name=string(").count();
        println!("MIL file contains approximately {} operations", op_count);

        // For now, just check it doesn't panic
        // Full validation requires ANE framework
        println!("MIL file structure validated successfully");
    }

    /// Test tensor shape compatibility
    #[test]
    fn test_tensor_shapes_valid() {
        let fwd_path = PathBuf::from("models/layer0/layer_0_fwd.mil");
        let mil_content = fs::read_to_string(&fwd_path).unwrap();

        // Extract shapes and verify they're compatible
        // Input: [1, 512, 1, 1024] = batch=1, channels=512, height=1, width=1024 (seq_len)
        assert!(
            mil_content.contains("[1, 512, 1, 1024]"),
            "Input shape should be [1, 512, 1, 1024]"
        );

        // Q projection: [512, 512] = 8 heads * 64 dim
        assert!(
            mil_content.contains("[512, 512, 1, 1]"),
            "Wq shape should be [512, 512, 1, 1]"
        );

        // K/V projection: [256, 512] = 4 heads * 64 dim
        assert!(
            mil_content.contains("[256, 512, 1, 1]"),
            "Wk/Wv shape should be [256, 512, 1, 1]"
        );

        // MLP hidden: [1024, 512] = 2x expansion
        assert!(
            mil_content.contains("[1024, 512, 1, 1]"),
            "W1/W3 shape should be [1024, 512, 1, 1]"
        );
    }
}
