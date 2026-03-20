//! RMSNorm backward pass MIL generator
//!
//! Generates MIL code for Root Mean Square Layer Normalization backward pass.

use super::{validate_mil_structure, BackwardMILGenerator};
use crate::ane::Result;
use crate::training::TransformerConfig;

/// MIL generator for RMSNorm backward pass
#[derive(Debug)]
pub struct RMSNormBackwardGen;

impl RMSNormBackwardGen {
    /// Create new RMSNorm backward MIL generator
    pub fn new() -> Self {
        RMSNormBackwardGen
    }

    /// Generate MIL code for RMSNorm backward operation
    fn generate_mil_code(&self, config: &TransformerConfig) -> String {
        let dim = config.dim;
        let seq_len = config.seq_len;
        let inv_dim = 1.0f32 / dim as f32;
        let eps = 1e-6f32;

        let mut mil = String::new();
        mil.push_str("program(1.3)\n");
        mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n");
        mil.push_str("{\n");
        mil.push_str(&format!(
            "    func main<ios18>(tensor<fp32, [1, {dim}, 1, {seq_len}]> d_out, tensor<fp32, [1, {dim}, 1, {seq_len}]> x, tensor<fp32, [1, {dim}, 1, 1]> w) {{\n"
        ));

        // 1. Compute inv_rms for each position
        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> x_sq = mul(x=x, y=x)[name = string(\"x_sq\")];\n"));
        mil.push_str("        tensor<int32, [1]> rax1 = const()[name = string(\"rax1\"), val=tensor<int32, [1]>([1])];\n");
        mil.push_str("        bool kd = const()[name = string(\"kd\"), val=bool(true)];\n");
        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> mean_sq_unscaled = reduce_sum(x=x_sq, axes=rax1, keep_dims=kd)[name = string(\"ms_un\")];\n"));
        mil.push_str(&format!(
            "        fp32 inv_dim = const()[name = string(\"inv_dim\"), val=fp32({inv_dim:.8})];\n"
        ));
        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> mean_sq = mul(x=mean_sq_unscaled, y=inv_dim)[name = string(\"ms\")];\n"));
        mil.push_str(&format!(
            "        fp32 eps = const()[name = string(\"eps\"), val=fp32({eps:.8})];\n"
        ));
        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> mean_sq_eps = add(x=mean_sq, y=eps)[name = string(\"ms_eps\")];\n"));
        mil.push_str("        fp32 nhalf = const()[name = string(\"nhalf\"), val=fp32(-0.5)];\n");
        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> inv_rms = pow(x=mean_sq_eps, y=nhalf)[name = string(\"inv_rms\")];\n"));

        // 2. Compute norm_x
        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> norm_x = mul(x=x, y=inv_rms)[name = string(\"norm_x\")];\n"));

        // 3. Compute weight gradient (dw)
        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> dw_prod = mul(x=d_out, y=norm_x)[name = string(\"dw_prod\")];\n"));
        mil.push_str("        tensor<int32, [1]> rax3 = const()[name = string(\"rax3\"), val=tensor<int32, [1]>([3])];\n");
        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, 1]> dw = reduce_sum(x=dw_prod, axes=rax3, keep_dims=kd)[name = string(\"dw\")];\n"));

        // 4. Compute input gradient (d_x)
        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> d_out_w = mul(x=d_out, y=w)[name = string(\"d_out_w\")];\n"));
        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> term1 = mul(x=d_out_w, y=inv_rms)[name = string(\"term1\")];\n"));

        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> d_out_w_x = mul(x=d_out_w, y=x)[name = string(\"d_out_w_x\")];\n"));
        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> dot_prod = reduce_sum(x=d_out_w_x, axes=rax1, keep_dims=kd)[name = string(\"dot\")];\n"));

        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> inv_rms_sq = mul(x=inv_rms, y=inv_rms)[name = string(\"inv_rms_sq\")];\n"));
        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> inv_rms_cube = mul(x=inv_rms_sq, y=inv_rms)[name = string(\"inv_rms_cube\")];\n"));

        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> scalar_term = mul(x=dot_prod, y=inv_rms_cube)[name = string(\"s_term\")];\n"));
        mil.push_str(&format!("        tensor<fp32, [1, 1, 1, {seq_len}]> scalar_term_scaled = mul(x=scalar_term, y=inv_dim)[name = string(\"s_term_s\")];\n"));

        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> term2 = mul(x=scalar_term_scaled, y=x)[name = string(\"term2\")];\n"));
        mil.push_str(&format!("        tensor<fp32, [1, {dim}, 1, {seq_len}]> d_x = sub(x=term1, y=term2)[name = string(\"d_x\")];\n"));

        mil.push_str("    } -> (d_x, dw);\n");
        mil.push_str("}\n");

        mil
    }

    /// Helper function to calculate input size in bytes
    pub fn input_bytes(config: &TransformerConfig) -> usize {
        let seq_len = config.seq_len;
        let dim = config.dim;

        // d_out + x + w
        (seq_len * dim + seq_len * dim + dim) * 4
    }

    /// Helper function to calculate output sizes in bytes
    pub fn output_sizes(config: &TransformerConfig) -> Vec<usize> {
        let seq_len = config.seq_len;
        let dim = config.dim;

        vec![
            seq_len * dim * 4, // d_x
            dim * 4,           // dw
        ]
    }

    /// Helper function to run RMSNorm backward on ANE
    pub fn run_on_ane(
        config: &TransformerConfig,
        d_out: &[f32],
        x: &[f32],
        w: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let gen = RMSNormBackwardGen;
        let mil_code = gen.generate_mil_code(config);

        let input_bytes = Self::input_bytes(config);
        let output_sizes = Self::output_sizes(config);

        // Pack inputs as bytes
        let mut packed_input = Vec::with_capacity(input_bytes);
        for &val in d_out.iter() {
            packed_input.extend_from_slice(&val.to_le_bytes());
        }
        for &val in x.iter() {
            packed_input.extend_from_slice(&val.to_le_bytes());
        }
        for &val in w.iter() {
            packed_input.extend_from_slice(&val.to_le_bytes());
        }

        let request =
            crate::ane::ANECompileRequest::new(&mil_code, vec![input_bytes], output_sizes.clone());
        let mut executor = request
            .compile()
            .map_err(|e| crate::ane::ANEError::EvalFailed(e.to_string()))?;

        executor
            .write_input(0, &packed_input)
            .map_err(|e| crate::ane::ANEError::EvalFailed(e.to_string()))?;
        executor
            .eval()
            .map_err(|e| crate::ane::ANEError::EvalFailed(e.to_string()))?;

        let d_x_len = config.seq_len * config.hidden_dim;
        let dw_len = config.hidden_dim;

        let mut d_x_bytes = vec![0u8; d_x_len * 4];
        let mut dw_bytes = vec![0u8; dw_len * 4];

        executor
            .read_output(0, &mut d_x_bytes)
            .map_err(|e| crate::ane::ANEError::EvalFailed(e.to_string()))?;
        executor
            .read_output(1, &mut dw_bytes)
            .map_err(|e| crate::ane::ANEError::EvalFailed(e.to_string()))?;

        let d_x: Vec<f32> = d_x_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let dw: Vec<f32> = dw_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok((d_x, dw))
    }
}

impl Default for RMSNormBackwardGen {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardMILGenerator for RMSNormBackwardGen {
    fn generate(&self, config: &TransformerConfig) -> Result<String> {
        Ok(self.generate_mil_code(config))
    }

    fn validate(&self, config: &TransformerConfig) -> Result<()> {
        let mil_code = self.generate(config)?;
        validate_mil_structure(
            &mil_code,
            "rmsnorm_backward",
            &["d_out", "x", "w"],
            &["d_x", "dw"],
        )?;

        Ok(())
    }

    fn operation_name(&self) -> &'static str {
        "rmsnorm_backward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_backward_gen_creation() {
        let gen = RMSNormBackwardGen::new();
        assert_eq!(gen.operation_name(), "rmsnorm_backward");
    }

    #[test]
    fn test_rmsnorm_backward_generate_mil() {
        let gen = RMSNormBackwardGen::new();
        let config = TransformerConfig::new(256, 128, 256, 4, 2, 64).unwrap();
        let mil_code = gen.generate(&config).unwrap();

        assert!(mil_code.contains("program(1.3)"));
    }
}
