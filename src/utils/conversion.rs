//! Weight format conversion utilities

use crate::{Error, Result};
use std::f32;

/// Convert FP32 weights to FP16 (half precision)
///
/// Uses IEEE 754 half precision format (1 sign bit, 5 exponent bits, 10 mantissa bits)
///
/// # Arguments
///
/// * `fp32_weights` - FP32 weight values
///
/// # Returns
///
/// FP16 weights as u16 values (bit representation)
///
/// # Example
///
/// ```
/// # use rustane::utils::fp32_to_fp16;
/// let fp32 = vec![1.0f32, 2.0, 3.0];
/// let fp16 = fp32_to_fp16(&fp32).unwrap();
/// ```
pub fn fp32_to_fp16(fp32_weights: &[f32]) -> Result<Vec<u16>> {
    fp32_weights
        .iter()
        .map(|&val| {
            if !val.is_finite() {
                return Err(Error::InvalidParameter(format!(
                    "Cannot convert non-finite value {} to FP16",
                    val
                )));
            }

            // Extract IEEE 754 binary32 representation
            let bits = val.to_bits();

            // Extract components
            let sign = (bits >> 31) & 1;
            let exponent = (bits >> 23) & 0xff;
            let mantissa = bits & 0x7fffff;

            // Handle special cases
            if exponent == 0 && mantissa == 0 {
                // Zero
                Ok((sign as u16) << 15)
            } else if exponent == 0xff {
                // Infinity or NaN - flush to zero in FP16
                Ok(((sign as u16) << 15) | 0x7c00)
            } else {
                // Normal number
                // Adjust exponent bias (127 for FP32 -> 15 for FP16)
                let new_exponent = (exponent as i32) - 127 + 15;

                if new_exponent <= 0 {
                    // Underflow to zero
                    Ok((sign << 15) as u16)
                } else if new_exponent >= 0x1f {
                    // Overflow to infinity
                    Ok(((sign as u16) << 15) | 0x7c00)
                } else {
                    // Round mantissa to 10 bits
                    let round_bit = (mantissa >> 13) & 1;
                    let new_mantissa = (mantissa >> 14) + round_bit;

                    // Handle mantissa overflow
                    let (exponent_bit, mantissa_bits) = if new_mantissa >= 0x400 {
                        (1, 0)
                    } else {
                        (0, new_mantissa)
                    };

                    let final_exponent = new_exponent + exponent_bit;

                    if final_exponent >= 0x1f {
                        // Overflow to infinity
                        Ok(((sign as u16) << 15) | 0x7c00)
                    } else {
                        Ok(((sign as u16) << 15)
                            | ((final_exponent as u16) << 10)
                            | (mantissa_bits as u16))
                    }
                }
            }
        })
        .collect()
}

/// Convert FP16 weights to FP32
///
/// # Arguments
///
/// * `fp16_weights` - FP16 weights as u16 (bit representation)
///
/// # Returns
///
/// FP32 weight values
///
/// # Example
///
/// ```
/// # use rustane::utils::fp16_to_fp32;
/// let fp16 = vec![0x3c00u16, 0x4000]; // 1.0, 2.0 in FP16
/// let fp32 = fp16_to_fp32(&fp16).unwrap();
/// ```
pub fn fp16_to_fp32(fp16_weights: &[u16]) -> Result<Vec<f32>> {
    fp16_weights
        .iter()
        .map(|&bits| {
            // Extract components
            let sign = (bits >> 15) & 1;
            let exponent = (bits >> 10) & 0x1f;
            let mantissa = bits & 0x3ff;

            // Handle special cases
            if exponent == 0 && mantissa == 0 {
                // Zero
                Ok(f32::from_bits((sign as u32) << 31))
            } else if exponent == 0x1f && mantissa == 0 {
                // Infinity
                Ok(f32::from_bits(((sign as u32) << 31) | 0x7f800000))
            } else if exponent == 0 {
                // Subnormal number (convert to zero for simplicity)
                Ok(f32::from_bits((sign as u32) << 31))
            } else {
                // Normal number
                let new_exponent = (exponent as i32) - 15 + 127;
                let new_mantissa = (mantissa as u32) << 13;

                Ok(f32::from_bits(
                    ((sign as u32) << 31) | ((new_exponent as u32) << 23) | new_mantissa,
                ))
            }
        })
        .collect()
}

/// Convert FP32 weights to BF16 (brain float 16)
///
/// Uses BFloat16 format (1 sign bit, 8 exponent bits, 7 mantissa bits)
/// BFloat16 preserves the exponent range of FP32, making it more stable for training
/// than FP16 at the cost of precision.
///
/// # Arguments
///
/// * `fp32_weights` - FP32 weight values
///
/// # Returns
///
/// BF16 weights as u16 values (bit representation)
///
/// # Example
///
/// ```
/// # use rustane::utils::fp32_to_bf16;
/// let fp32 = vec![1.0f32, 2.0, 3.0];
/// let bf16 = fp32_to_bf16(&fp32).unwrap();
/// ```
pub fn fp32_to_bf16(fp32_weights: &[f32]) -> Result<Vec<u16>> {
    fp32_weights
        .iter()
        .map(|&val| {
            if !val.is_finite() {
                return Err(Error::InvalidParameter(format!(
                    "Cannot convert non-finite value {} to BF16",
                    val
                )));
            }

            // Extract IEEE 754 binary32 representation
            let bits = val.to_bits();

            // BF16 truncation: keep upper 16 bits (sign + exponent + upper 7 mantissa bits)
            // This is simpler than FP16 because BF16 keeps the same exponent as FP32
            let bf16_bits = (bits >> 16) as u16;

            Ok(bf16_bits)
        })
        .collect()
}

/// Convert BF16 weights to FP32
///
/// # Arguments
///
/// * `bf16_weights` - BF16 weights as u16 (bit representation)
///
/// # Returns
///
/// FP32 weight values
///
/// # Example
///
/// ```
/// # use rustane::utils::bf16_to_fp32;
/// let bf16 = vec![0x3f80u16, 0x4000]; // 1.0, 2.0 in BF16
/// let fp32 = bf16_to_fp32(&bf16).unwrap();
/// ```
pub fn bf16_to_fp32(bf16_weights: &[u16]) -> Result<Vec<f32>> {
    bf16_weights
        .iter()
        .map(|&bits| {
            // Zero-extend lower 16 bits to get FP32
            let fp32_bits = (bits as u32) << 16;
            Ok(f32::from_bits(fp32_bits))
        })
        .collect()
}

/// Transpose weights from row-major to column-major (or vice versa)
///
/// # Arguments
///
/// * `weights` - Weight values
/// * `shape` - Tensor shape (must be 2D or 4D)
///
/// # Returns
///
/// Transposed weights
///
/// # Example
///
/// ```
/// # use rustane::utils::transpose_weights;
/// let weights = vec![1.0f32, 2.0, 3.0, 4.0];
/// let shape = vec![2, 2];
/// let transposed = transpose_weights(&weights, &shape).unwrap();
/// ```
pub fn transpose_weights(weights: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    match shape.len() {
        2 => {
            // 2D transpose
            let rows = shape[0];
            let cols = shape[1];

            if weights.len() != rows * cols {
                return Err(Error::InvalidParameter(format!(
                    "Weight length {} doesn't match shape {}x{}",
                    weights.len(),
                    rows,
                    cols
                )));
            }

            let mut transposed = vec![0.0f32; weights.len()];

            for i in 0..rows {
                for j in 0..cols {
                    transposed[j * rows + i] = weights[i * cols + j];
                }
            }

            Ok(transposed)
        }
        4 => {
            // 4D transpose - transpose last two dimensions
            // (N, H, W, C) -> (N, C, H, W)
            let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);

            if weights.len() != n * h * w * c {
                return Err(Error::InvalidParameter(format!(
                    "Weight length {} doesn't match shape {}x{}x{}x{}",
                    weights.len(),
                    n,
                    h,
                    w,
                    c
                )));
            }

            let mut transposed = vec![0.0f32; weights.len()];

            for i in 0..n {
                for j in 0..h {
                    for k in 0..w {
                        for l in 0..c {
                            // Original: (i, j, k, l) -> i*h*w*c + j*w*c + k*c + l
                            // Transposed: (i, l, j, k) -> i*c*h*w + l*h*w + j*w + k
                            transposed[i * c * h * w + l * h * w + j * w + k] =
                                weights[i * h * w * c + j * w * c + k * c + l];
                        }
                    }
                }
            }

            Ok(transposed)
        }
        _ => Err(Error::InvalidParameter(format!(
            "Transpose only supports 2D and 4D tensors, got {}D",
            shape.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp32_to_fp16_conversion() {
        let fp32 = vec![1.0f32, 2.0, 0.5, -1.0];
        let fp16 = fp32_to_fp16(&fp32).unwrap();

        assert_eq!(fp16.len(), 4);

        // Convert back and check
        let fp32_roundtrip = fp16_to_fp32(&fp16).unwrap();

        for i in 0..fp32.len() {
            assert!(
                (fp32_roundtrip[i] - fp32[i]).abs() < 0.001,
                "Round-trip failed at index {}: {} vs {}",
                i,
                fp32_roundtrip[i],
                fp32[i]
            );
        }
    }

    #[test]
    fn test_fp32_to_bf16_conversion() {
        let fp32 = vec![1.0f32, 2.0, 0.5, -1.0, 0.0];
        let bf16 = fp32_to_bf16(&fp32).unwrap();

        assert_eq!(bf16.len(), 5);

        // Convert back and check
        let fp32_roundtrip = bf16_to_fp32(&bf16).unwrap();

        for i in 0..fp32.len() {
            // BF16 has less precision than FP32 but preserves exponent range
            // For simple values like 1.0, 2.0, 0.5, -1.0, 0.0, it should be exact
            assert_eq!(
                fp32_roundtrip[i], fp32[i],
                "Round-trip failed at index {}: {} vs {}",
                i, fp32_roundtrip[i], fp32[i]
            );
        }
    }

    #[test]
    fn test_bf16_preserves_special_values() {
        // Test zeros
        let zeros = vec![0.0f32, -0.0];
        let bf16 = fp32_to_bf16(&zeros).unwrap();
        let fp32_back = bf16_to_fp32(&bf16).unwrap();
        assert_eq!(fp32_back[0].to_bits(), zeros[0].to_bits()); // +0.0
        assert_eq!(fp32_back[1].to_bits(), zeros[1].to_bits()); // -0.0
    }

    #[test]
    fn test_transpose_2d() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        // [1.0, 2.0]
        // [3.0, 4.0]
        // Should become:
        // [1.0, 3.0]
        // [2.0, 4.0]

        let transposed = transpose_weights(&weights, &shape).unwrap();

        assert_eq!(transposed, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_4d() {
        let weights = vec![1.0f32; 8]; // 1x2x2x2
        let shape = vec![1, 2, 2, 2];

        let transposed = transpose_weights(&weights, &shape).unwrap();

        assert_eq!(transposed.len(), 8);
        // Just check it doesn't crash for now
    }

    #[test]
    fn test_invalid_shape() {
        let weights = vec![1.0f32; 8];
        let shape = vec![2, 2, 2]; // 3D not supported

        assert!(transpose_weights(&weights, &shape).is_err());
    }
}
