//! ANE-Compatible MIL Templates
//!
//! These MIL programs use only ANE-supported operations.
//! Based on error analysis: ANE fails with "InvalidMILProgram" for complex ops.

/// Build ANE-compatible MIL for given dimensions
pub struct ANEMILBuilder;

impl ANEMILBuilder {
    /// Create element-wise add MIL
    pub fn add(size: usize) -> String {
        format!(
            r#"main add(a: tensor<{}xf32>, b: tensor<{}xf32>) -> (c: tensor<{}xf32>) {{
    let c = a + b;
    return (c);
}}"#,
            size, size, size
        )
    }

    /// Create element-wise multiply MIL
    pub fn mul(size: usize, scale: f32) -> String {
        format!(
            r#"main mul(x: tensor<{}xf32>) -> (y: tensor<{}xf32>) {{
    let scale = {};
    let y = x * scale;
    return (y);
}}"#,
            size, size, scale
        )
    }

    /// Create matrix multiplication MIL
    ///
    /// # Arguments
    /// * `batch` - Batch size (rows of x)
    /// * `in_features` - Input dimension (cols of x, rows of w)
    /// * `out_features` - Output dimension (cols of w)
    pub fn matmul(batch: usize, in_features: usize, out_features: usize) -> String {
        format!(
            r#"main matmul(x: tensor<{}x{}xf32>) -> (y: tensor<{}x{}xf32>) {{
    let w = const_tensor<{}x{}xf32>(@w.bin);
    let y = matmul(x, w);
    return (y);
}}"#,
            batch, in_features, batch, out_features, in_features, out_features
        )
    }

    /// Create ReLU MIL
    pub fn relu(size: usize) -> String {
        format!(
            r#"main relu(x: tensor<{}xf32>) -> (y: tensor<{}xf32>) {{
    let zero = 0.0;
    let y = max(x, zero);
    return (y);
}}"#,
            size, size
        )
    }

    /// Create residual connection MIL
    pub fn residual(size: usize) -> String {
        format!(
            r#"main residual(x: tensor<{}xf32>, residual: tensor<{}xf32>) -> (y: tensor<{}xf32>) {{
    let y = x + residual;
    return (y);
}}"#,
            size, size, size
        )
    }

    /// Create a simple linear layer (matmul only, no bias)
    pub fn linear_simple(
        batch_size: usize,
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> String {
        let total_rows = batch_size * seq_len;
        format!(
            r#"main linear(
    x: tensor<{}x{}xf32>
) -> (
    y: tensor<{}x{}xf32>
) {{
    let w = const_tensor<{}x{}xf32>(@w.bin);
    let y = matmul(x, w);
    return (y);
}}"#,
            total_rows, in_dim, total_rows, out_dim, in_dim, out_dim
        )
    }

    /// Create a chain of operations (if ANE supports it)
    /// Note: Complex chains often fail on ANE
    pub fn linear_with_relu(batch: usize, in_dim: usize, out_dim: usize) -> String {
        format!(
            r#"main linear_relu(
    x: tensor<{}x{}xf32>
) -> (
    y: tensor<{}x{}xf32>
) {{
    let w = const_tensor<{}x{}xf32>(@w.bin);
    let z = matmul(x, w);
    let zero = 0.0;
    let y = max(z, zero);
    return (y);
}}"#,
            batch, in_dim, batch, out_dim, in_dim, out_dim
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ane_mil_add() {
        let mil = ANEMILBuilder::add(512);
        assert!(mil.contains("main add"));
        assert!(mil.contains("tensor<512xf32>"));
        println!("Add MIL:\n{}\n", mil);
    }

    #[test]
    fn test_ane_mil_matmul() {
        let mil = ANEMILBuilder::matmul(4, 512, 512);
        assert!(mil.contains("main matmul"));
        assert!(mil.contains("const_tensor"));
        assert!(mil.contains("matmul(x, w)"));
        println!("Matmul MIL:\n{}\n", mil);
    }

    #[test]
    fn test_ane_mil_linear() {
        let mil = ANEMILBuilder::linear_simple(2, 256, 512, 512);
        assert!(mil.contains("main linear"));
        println!("Linear MIL:\n{}\n", mil);
    }

    #[test]
    fn test_ane_mil_relu() {
        let mil = ANEMILBuilder::relu(416);
        assert!(mil.contains("main relu"));
        assert!(mil.contains("max(x, zero)"));
        println!("ReLU MIL:\n{}\n", mil);
    }
}
