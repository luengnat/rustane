//! ANE MIL Generator - Compatible with working ANE project syntax
//!
//! Generates MIL in the format that ANE actually accepts:
//! func main<ios18>(tensor<fp16, [1, C, 1, S]> x) { ... } -> (out);
//!
//! # Important Constraints
//!
//! ANE has strict size requirements (from testing and Apple documentation):
//!
//! ## Minimum Requirements
//! - Minimum tensor size: 32x32 (1024 elements)
//! - Tensors smaller than 32x32 will fail with "Program Inference error"
//!
//! ## Maximum Requirements  
//! - Maximum tensor dimension: 16,384 elements total
//! - Larger tensors will fail to compile with ANECCompile() FAILED
//!
//! ## Alignment
//! - Weight blobs must be aligned to 64-byte boundaries (offset=uint64(64))
//! - General alignment: 16-byte boundaries
//!
//! ## Format Preferences
//! - Tensor shapes: Powers of 2 (2, 4, 8, 16, 32, 64, etc.)
//! - Sizes: Multiples of 16 (16, 32, 48, 64, etc.)
//! - Data layout: NCHW [1, C, 1, S] for 1D sequences
//! - Data type: fp16 preferred
//!
//! # Examples
//!
//! Valid sizes for transformer operations:
//! - 512x32 = 16,384 elements (MAX)
//! - 256x64 = 16,384 elements (MAX)
//! - 768x16 = 12,288 elements (OK)
//! - 1024x16 = 16,384 elements (MAX)
//!
//! Invalid sizes:
//! - 4x4 = 16 elements (too small, fails eval)
//! - 768x256 = 196,608 elements (too large, fails compile)

#![allow(dead_code)]

use std::fmt::Write;

/// Minimum dimension size for ANE operations
pub const ANE_MIN_DIMENSION: usize = 32;

/// Minimum total elements for ANE operations
/// Tensors smaller than this will fail with "Program Inference error"
pub const ANE_MIN_ELEMENTS: usize = 32 * 32;

/// Maximum total elements for ANE operations
/// Tensors larger than this will fail to compile
pub const ANE_MAX_ELEMENTS: usize = 16384;

/// Weight blob alignment in bytes
pub const ANE_WEIGHT_ALIGNMENT: usize = 64;

/// ANE MIL tensor type
#[derive(Debug, Clone)]
pub enum ANETensorType {
    FP16,
    FP32,
    Int32,
}

impl ANETensorType {
    pub fn mil_type(&self) -> &str {
        match self {
            ANETensorType::FP16 => "fp16",
            ANETensorType::FP32 => "fp32",
            ANETensorType::Int32 => "int32",
        }
    }
}

/// ANE MIL shape (4D layout: [N, C, H, W])
///
/// For 1D sequences, use NCHW format with H=1: [1, channels, 1, seq_len]
#[derive(Debug, Clone)]
pub struct ANEShape {
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl ANEShape {
    pub fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self { n, c, h, w }
    }

    /// Create shape for 1D sequences: [1, channels, 1, seq_len]
    pub fn seq(channels: usize, seq_len: usize) -> Self {
        Self::new(1, channels, 1, seq_len)
    }

    pub fn mil_string(&self) -> String {
        format!("[{}, {}, {}, {}]", self.n, self.c, self.h, self.w)
    }

    /// Returns the total number of elements in this shape
    pub fn num_elements(&self) -> usize {
        self.n * self.c * self.h * self.w
    }

    /// Returns true if this shape meets ANE minimum size requirements
    ///
    /// ANE requires minimum tensor size of 32x32 (1024 elements).
    /// Tensors smaller than this will fail with "Program Inference error".
    pub fn is_ane_compatible(&self) -> bool {
        let elements = self.num_elements();
        elements >= ANE_MIN_ELEMENTS && elements <= ANE_MAX_ELEMENTS
    }

    /// Get compatibility error message if shape is not ANE-compatible
    pub fn compatibility_error(&self) -> Option<String> {
        let elements = self.num_elements();
        if elements < ANE_MIN_ELEMENTS {
            Some(format!(
                "Tensor too small: {} elements (minimum: {})",
                elements, ANE_MIN_ELEMENTS
            ))
        } else if elements > ANE_MAX_ELEMENTS {
            Some(format!(
                "Tensor too large: {} elements (maximum: {})",
                elements, ANE_MAX_ELEMENTS
            ))
        } else {
            None
        }
    }

    /// Returns recommended dimensions that are ANE-compatible
    ///
    /// For small tensors, this pads to meet minimum requirements.
    /// For large tensors, this would need tiling (not implemented here).
    pub fn to_ane_compatible(&self) -> Self {
        if self.is_ane_compatible() {
            return self.clone();
        }

        let elements = self.num_elements();

        if elements == 0 {
            return Self::new(1, 32, 1, 32);
        }

        if elements < ANE_MIN_ELEMENTS {
            // Scale up to meet minimum
            let scale = ((ANE_MIN_ELEMENTS as f64) / (elements as f64))
                .sqrt()
                .ceil() as usize;
            let scale = scale.max(1);

            Self::new(
                self.n,
                (self.c * scale).max(32),
                self.h,
                (self.w * scale).max(32),
            )
        } else {
            // Too large - would need tiling strategy
            // For now, return as-is and let caller handle
            self.clone()
        }
    }

    /// Returns padding info for ANE-compatible execution
    ///
    /// If the tensor is too small, returns (padded_shape, actual_elements, padding_elements)
    /// If already compatible, returns (same_shape, all_elements, 0)
    pub fn get_padding_info(&self) -> (Self, usize, usize) {
        if self.is_ane_compatible() {
            return (self.clone(), self.num_elements(), 0);
        }

        let elements = self.num_elements();

        if elements < ANE_MIN_ELEMENTS {
            let padded = self.to_ane_compatible();
            let padded_elements = padded.num_elements();
            (padded, elements, padded_elements - elements)
        } else {
            // Too large - no padding, needs tiling
            (self.clone(), elements, 0)
        }
    }

    /// Check if this shape is valid for ANE
    ///
    /// Returns (is_valid, error_message)
    pub fn validate_for_ane(&self) -> (bool, Option<String>) {
        if let Some(error) = self.compatibility_error() {
            (false, Some(error))
        } else {
            (true, None)
        }
    }
}

/// ANE MIL operation
#[derive(Debug, Clone)]
pub struct ANEOp {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub output_shape: ANEShape,
    pub dtype: ANETensorType,
}

impl ANEOp {
    pub fn new(name: &str, op_type: &str, output_shape: ANEShape) -> Self {
        Self {
            name: name.to_string(),
            op_type: op_type.to_string(),
            inputs: Vec::new(),
            output_shape,
            dtype: ANETensorType::FP16,
        }
    }

    pub fn with_dtype(mut self, dtype: ANETensorType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn with_input(mut self, input: &str) -> Self {
        self.inputs.push(input.to_string());
        self
    }

    pub fn to_mil(&self) -> String {
        let shape_str = self.output_shape.mil_string();
        let dtype = self.dtype.mil_type();

        match self.op_type.as_str() {
            "mul" => {
                format!(
                    "        tensor<{}, {}> {} = mul(x={},y={})[name=string(\"{}\")];",
                    dtype, shape_str, self.name, self.inputs[0], self.inputs[1], self.name
                )
            }
            "add" => {
                format!(
                    "        tensor<{}, {}> {} = add(x={},y={})[name=string(\"{}\")];",
                    dtype, shape_str, self.name, self.inputs[0], self.inputs[1], self.name
                )
            }
            "const" => {
                format!(
                    "        {} {} = const()[name=string(\"{}\"), val={}({})];",
                    dtype, self.name, self.name, dtype, self.inputs[0]
                )
            }
            _ => format!(
                "        // Unsupported op: {} -> {}",
                self.op_type, self.name
            ),
        }
    }
}

/// ANE MIL program builder
#[derive(Debug, Clone)]
pub struct ANEMILProgram {
    input_name: String,
    input_shape: ANEShape,
    input_dtype: ANETensorType,
    operations: Vec<ANEOp>,
    output_name: String,
}

impl ANEMILProgram {
    pub fn new(input_name: &str, input_shape: ANEShape) -> Self {
        Self {
            input_name: input_name.to_string(),
            input_shape,
            input_dtype: ANETensorType::FP16,
            operations: Vec::new(),
            output_name: "out".to_string(),
        }
    }

    pub fn with_dtype(mut self, dtype: ANETensorType) -> Self {
        self.input_dtype = dtype;
        self
    }

    pub fn add_op(mut self, op: ANEOp) -> Self {
        self.operations.push(op);
        self
    }

    pub fn set_output(mut self, name: &str) -> Self {
        self.output_name = name.to_string();
        self
    }

    /// Validate that all tensor shapes are ANE-compatible
    pub fn validate_shapes(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check input shape
        if let Some(error) = self.input_shape.compatibility_error() {
            errors.push(format!("Input shape: {}", error));
        }

        // Check all operation output shapes
        for op in &self.operations {
            if let Some(error) = op.output_shape.compatibility_error() {
                errors.push(format!("Op '{}' output: {}", op.name, error));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn generate(&self) -> String {
        let mut mil = String::new();

        // Program header with buildInfo (required by ANE)
        writeln!(&mut mil, "program(1.3)").unwrap();
        writeln!(&mut mil, "[buildInfo = dict<string, string>({{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, {{\"coremlc-version\", \"3505.4.1\"}}, {{\"coremltools-component-milinternal\", \"\"}}, {{\"coremltools-version\", \"9.0\"}}}})]").unwrap();
        writeln!(&mut mil, "{{").unwrap();

        // Function header
        let dtype = self.input_dtype.mil_type();
        let shape = self.input_shape.mil_string();
        writeln!(
            &mut mil,
            "    func main<ios18>(tensor<{}, {}> {}) {{",
            dtype, shape, self.input_name
        )
        .unwrap();

        // Operations
        for op in &self.operations {
            writeln!(&mut mil, "{}", op.to_mil()).unwrap();
        }

        // Function footer
        writeln!(&mut mil, "    }} -> ({});", self.output_name).unwrap();

        // Program footer
        writeln!(&mut mil, "}}").unwrap();

        mil
    }

    /// Generate and validate - returns (mil, validation_errors)
    pub fn generate_validated(&self) -> (String, Vec<String>) {
        let errors = match self.validate_shapes() {
            Ok(()) => Vec::new(),
            Err(e) => e,
        };
        (self.generate(), errors)
    }
}

/// Generate simple operations
pub struct ANEMILOps;

impl ANEMILOps {
    /// Element-wise multiply: out = a * b
    pub fn mul(name: &str, a: &str, b: &str, shape: ANEShape) -> ANEOp {
        ANEOp::new(name, "mul", shape).with_input(a).with_input(b)
    }

    /// Element-wise add: out = a + b
    pub fn add(name: &str, a: &str, b: &str, shape: ANEShape) -> ANEOp {
        ANEOp::new(name, "add", shape).with_input(a).with_input(b)
    }

    /// Constant value
    pub fn constant(name: &str, value: &str, dtype: ANETensorType) -> ANEOp {
        ANEOp::new(name, "const", ANEShape::new(1, 1, 1, 1))
            .with_dtype(dtype)
            .with_input(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_validation() {
        // Too small
        let small = ANEShape::new(1, 4, 1, 4);
        assert!(!small.is_ane_compatible());
        assert!(small.compatibility_error().is_some());

        // Just right
        let medium = ANEShape::new(1, 32, 1, 32);
        assert!(medium.is_ane_compatible());
        assert!(medium.compatibility_error().is_none());

        // Too large
        let large = ANEShape::new(1, 768, 1, 256);
        assert!(!large.is_ane_compatible());
        assert!(large.compatibility_error().is_some());

        // At maximum
        let max = ANEShape::new(1, 512, 1, 32);
        assert!(max.is_ane_compatible());
    }

    #[test]
    fn test_program_validation() {
        let program = ANEMILProgram::new("x", ANEShape::seq(4, 4)).set_output("x");

        let (mil, errors) = program.generate_validated();
        assert!(!errors.is_empty());
        assert!(mil.contains("program(1.3)"));
    }
}
