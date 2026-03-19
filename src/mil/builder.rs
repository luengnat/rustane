//! MIL program builder
//!
//! Provides a fluent interface for constructing MIL programs.
//! Generates program(1.3) format with proper function signatures.

use std::collections::HashMap;

/// MIL program builder
///
/// Generates program(1.3) [buildInfo] { func main<ios18>(...) -> (...); }
/// format compatible with modern ANE compilation.
///
/// # Example
///
/// ```
/// # use rustane::mil::MILBuilder;
/// let mil = MILBuilder::new()
///     .add_input("input", "fp32", &[1, 1, 256])
///     .add_output("output", "fp32", &[1, 1, 512])
///     .add_matmul("out", "input", "weight", false)
///     .build();
/// ```
pub struct MILBuilder {
    inputs: Vec<(String, String, Vec<usize>)>, // (name, dtype, shape)
    outputs: Vec<(String, String, Vec<usize>)>, // (name, dtype, shape)
    operations: Vec<String>,
    build_info: HashMap<String, String>,
}

impl MILBuilder {
    /// Create a new MIL builder for program(1.3) format
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::MILBuilder;
    /// let builder = MILBuilder::new();
    /// ```
    pub fn new() -> Self {
        let mut build_info = HashMap::new();
        build_info.insert("target_os".to_string(), "ios".to_string());
        build_info.insert("target_version".to_string(), "18".to_string());

        MILBuilder {
            inputs: Vec::new(),
            outputs: Vec::new(),
            operations: Vec::new(),
            build_info,
        }
    }

    /// Add metadata to buildInfo
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::MILBuilder;
    /// let builder = MILBuilder::new()
    ///     .with_build_info("author", "rustane");
    /// ```
    pub fn with_build_info(mut self, key: &str, value: &str) -> Self {
        self.build_info.insert(key.to_string(), value.to_string());
        self
    }

    /// Add an input tensor with dtype
    ///
    /// # Arguments
    ///
    /// * `name` - Input tensor name
    /// * `dtype` - Data type ("fp32", "fp16", "int32", etc.)
    /// * `shape` - Tensor shape
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::MILBuilder;
    /// let builder = MILBuilder::new()
    ///     .add_input("data", "fp32", &[1, 3, 224, 224]);
    /// ```
    pub fn add_input(mut self, name: &str, dtype: &str, shape: &[usize]) -> Self {
        self.inputs
            .push((name.to_string(), dtype.to_string(), shape.to_vec()));
        self
    }

    /// Add an output tensor with dtype
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::MILBuilder;
    /// let builder = MILBuilder::new()
    ///     .add_output("output", "fp32", &[1, 512]);
    /// ```
    pub fn add_output(mut self, name: &str, dtype: &str, shape: &[usize]) -> Self {
        self.outputs
            .push((name.to_string(), dtype.to_string(), shape.to_vec()));
        self
    }

    /// Add a convolution operation (legacy, for backward compat)
    ///
    /// # Arguments
    ///
    /// * `name` - Operation output name
    /// * `input_name` - Input tensor name
    /// * `weight_name` - Weight tensor name
    /// * `output_channels` - Number of output channels
    /// * `kernel_size` - Kernel height and width
    /// * `strides` - Stride height and width
    pub fn add_convolution(
        mut self,
        name: &str,
        input_name: &str,
        weight_name: &str,
        output_channels: usize,
        kernel_size: [usize; 2],
        strides: [usize; 2],
    ) -> Self {
        let op = format!(
            "var {} = nn.convolution(bias=false, groups=1, input_name=\"{}\", kernel_sizes=[{}, {}], name=\"conv\", output_channels={}, pad_type=\"valid\", strides=[{}, {}], weight_name=\"{}\", padding_top=0, padding_bottom=0, padding_left=0, padding_right=0)",
            name, input_name, kernel_size[0], kernel_size[1], output_channels, strides[0], strides[1], weight_name
        );
        self.operations.push(op);
        self
    }

    /// Add a linear (fully connected) layer via matmul
    ///
    /// Implements y = x @ W where x[..., in] and W[in, out]
    /// Uses matmul operation for non-square projections.
    pub fn add_linear(
        mut self,
        name: &str,
        input_name: &str,
        weight_name: &str,
        _output_channels: usize,
    ) -> Self {
        let op = format!(
            "var {} = mb.matmul(x=\"{}\", y=\"{}\", transpose_y=false, name=\"linear\")",
            name, input_name, weight_name
        );
        self.operations.push(op);
        self
    }

    /// Add a matmul operation
    ///
    /// # Arguments
    ///
    /// * `name` - Output tensor name
    /// * `x` - First input tensor name
    /// * `y` - Second input tensor name
    /// * `transpose_y` - Whether to transpose y before matmul
    pub fn add_matmul(mut self, name: &str, x: &str, y: &str, transpose_y: bool) -> Self {
        let op = format!(
            "var {} = mb.matmul(x=\"{}\", y=\"{}\", transpose_y={}, name=\"matmul\")",
            name, x, y, transpose_y
        );
        self.operations.push(op);
        self
    }

    /// Add a scaled dot-product attention operation
    ///
    /// # Arguments
    ///
    /// * `name` - Output tensor name
    /// * `q` - Query tensor name
    /// * `k` - Key tensor name
    /// * `v` - Value tensor name
    pub fn add_sdpa(mut self, name: &str, q: &str, k: &str, v: &str) -> Self {
        let op = format!(
            "var {} = mb.scaled_dot_product_attention(query=\"{}\", key=\"{}\", value=\"{}\", scale=1.0, name=\"sdpa\")",
            name, q, k, v
        );
        self.operations.push(op);
        self
    }

    /// Add a cast operation
    ///
    /// # Arguments
    ///
    /// * `name` - Output tensor name
    /// * `input` - Input tensor name
    /// * `dtype` - Target data type
    pub fn add_cast(mut self, name: &str, input: &str, dtype: &str) -> Self {
        let op = format!(
            "var {} = mb.cast(x=\"{}\", dtype=\"{}\", name=\"cast\")",
            name, input, dtype
        );
        self.operations.push(op);
        self
    }

    /// Add a concat operation
    ///
    /// # Arguments
    ///
    /// * `name` - Output tensor name
    /// * `inputs` - Input tensor names
    /// * `axis` - Concatenation axis
    pub fn add_concat(mut self, name: &str, inputs: &[&str], axis: i32) -> Self {
        let inputs_str = inputs
            .iter()
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(", ");
        let op = format!(
            "var {} = mb.concat(values=[{}], axis={}, name=\"concat\")",
            name, inputs_str, axis
        );
        self.operations.push(op);
        self
    }

    /// Add a ReLU activation
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::MILBuilder;
    /// let builder = MILBuilder::new()
    ///     .add_relu("relu1", "conv1");
    /// ```
    pub fn add_relu(mut self, name: &str, input_name: &str) -> Self {
        let op = format!(
            "var {} = mb.relu(x=\"{}\", name=\"relu\")",
            name, input_name
        );
        self.operations.push(op);
        self
    }

    /// Build the MIL program string (program 1.3 format)
    ///
    /// Generates:
    /// ```text
    /// program(1.3) [buildInfo=dict<string, string>(...)] {
    ///   func main<ios18>(input: fp32[1, 256], ...) -> (output: fp32[1, 512], ...) {
    ///     var ... = ...
    ///     return (...);
    ///   }
    /// }
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::mil::MILBuilder;
    /// let mil = MILBuilder::new()
    ///     .add_input("input", "fp32", &[1, 256])
    ///     .add_output("output", "fp32", &[1, 512])
    ///     .add_matmul("out", "input", "weight", false)
    ///     .build();
    /// ```
    pub fn build(self) -> String {
        // Build buildInfo dict
        let build_info_items: Vec<String> = self
            .build_info
            .iter()
            .map(|(k, v)| format!("\"{}\" : \"{}\"", k, v))
            .collect();
        let build_info_str = build_info_items.join(", ");

        let mut mil = format!(
            "program(1.3) [buildInfo=dict<string, string>({})] {{\n",
            build_info_str
        );

        // Build function signature
        let input_sig = self
            .inputs
            .iter()
            .map(|(name, dtype, shape)| {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                format!("{}: {}[{}]", name, dtype, shape_str.join(", "))
            })
            .collect::<Vec<_>>()
            .join(", ");

        let output_sig = self
            .outputs
            .iter()
            .map(|(name, dtype, shape)| {
                let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
                format!("{}: {}[{}]", name, dtype, shape_str.join(", "))
            })
            .collect::<Vec<_>>()
            .join(", ");

        mil.push_str(&format!(
            "  func main<ios18>({}) -> ({}) {{\n",
            input_sig, output_sig
        ));

        // Add operations
        for op in &self.operations {
            mil.push_str("    ");
            mil.push_str(op);
            mil.push_str(";\n");
        }

        // Add return statement
        let return_names: Vec<String> = self
            .outputs
            .iter()
            .map(|(name, _, _)| name.clone())
            .collect();
        mil.push_str(&format!("    return ({});\n", return_names.join(", ")));

        mil.push_str("  }\n");
        mil.push('}');
        mil
    }
}

impl Default for MILBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = MILBuilder::new();
        assert_eq!(builder.inputs.len(), 0);
    }

    #[test]
    fn test_builder_default() {
        let builder = MILBuilder::default();
        assert_eq!(builder.inputs.len(), 0);
    }

    #[test]
    fn test_builder_simple() {
        let mil = MILBuilder::new()
            .add_input("input", "fp32", &[1, 256])
            .add_output("output", "fp32", &[1, 512])
            .build();

        assert!(mil.contains("program(1.3)"));
        assert!(mil.contains("func main<ios18>"));
        assert!(mil.contains("input: fp32[1, 256]"));
        assert!(mil.contains("output: fp32[1, 512]"));
        assert!(mil.contains("return (output)"));
    }

    #[test]
    fn test_builder_matmul() {
        let mil = MILBuilder::new()
            .add_input("x", "fp32", &[1, 256])
            .add_input("w", "fp32", &[256, 512])
            .add_output("out", "fp32", &[1, 512])
            .add_matmul("out", "x", "w", false)
            .build();

        assert!(mil.contains("mb.matmul"));
        assert!(mil.contains("x=\"x\""));
        assert!(mil.contains("y=\"w\""));
        assert!(mil.contains("transpose_y=false"));
    }

    #[test]
    fn test_builder_sdpa() {
        let mil = MILBuilder::new()
            .add_input("q", "fp32", &[1, 8, 256, 64])
            .add_input("k", "fp32", &[1, 8, 256, 64])
            .add_input("v", "fp32", &[1, 8, 256, 64])
            .add_output("out", "fp32", &[1, 8, 256, 64])
            .add_sdpa("out", "q", "k", "v")
            .build();

        assert!(mil.contains("mb.scaled_dot_product_attention"));
        assert!(mil.contains("query=\"q\""));
        assert!(mil.contains("key=\"k\""));
        assert!(mil.contains("value=\"v\""));
    }

    #[test]
    fn test_builder_cast() {
        let mil = MILBuilder::new()
            .add_input("x", "fp32", &[1, 256])
            .add_output("out", "fp16", &[1, 256])
            .add_cast("out", "x", "fp16")
            .build();

        assert!(mil.contains("mb.cast"));
        assert!(mil.contains("x=\"x\""));
        assert!(mil.contains("dtype=\"fp16\""));
    }

    #[test]
    fn test_builder_concat() {
        let mil = MILBuilder::new()
            .add_input("a", "fp32", &[1, 128])
            .add_input("b", "fp32", &[1, 128])
            .add_output("out", "fp32", &[1, 256])
            .add_concat("out", &["a", "b"], 1)
            .build();

        assert!(mil.contains("mb.concat"));
        assert!(mil.contains("values=[\"a\", \"b\"]"));
        assert!(mil.contains("axis=1"));
    }

    #[test]
    fn test_builder_relu() {
        let mil = MILBuilder::new()
            .add_input("x", "fp32", &[1, 256])
            .add_output("out", "fp32", &[1, 256])
            .add_relu("out", "x")
            .build();

        assert!(mil.contains("mb.relu"));
        assert!(mil.contains("x=\"x\""));
    }

    #[test]
    fn test_builder_with_build_info() {
        let mil = MILBuilder::new()
            .with_build_info("author", "rustane")
            .with_build_info("version", "1.0")
            .add_input("x", "fp32", &[1, 256])
            .add_output("out", "fp32", &[1, 256])
            .build();

        assert!(mil.contains("author"));
        assert!(mil.contains("rustane"));
        assert!(mil.contains("version"));
    }
}
