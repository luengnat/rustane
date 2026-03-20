//! Type-safe tensor data wrapper
//!
//! ANETensor provides safe, type-checked access to tensor data that will be
//! passed to the ANE via IOSurface-backed memory.

use crate::Error;
use std::fmt;

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    /// 16-bit floating point (ANE native format)
    FP16,
    /// 32-bit floating point
    FP32,
    /// 8-bit signed integer (quantized)
    INT8,
}

impl TensorDType {
    /// Size in bytes of each element
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorDType::FP16 => 2,
            TensorDType::FP32 => 4,
            TensorDType::INT8 => 1,
        }
    }
}

/// Tensor shape dimensions
pub type TensorShape = Vec<usize>;

/// Type-safe tensor data wrapper
///
/// ANETensor provides safe access to tensor data with type and shape validation.
/// It owns its data and ensures proper memory management.
///
/// # Example
///
/// ```
/// use rustane::wrapper::tensor::{ANETensor, TensorDType};
///
/// // Create a 2x3 tensor with FP32 data
/// let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let tensor = ANETensor::new(data, vec![2, 3], TensorDType::FP32).unwrap();
/// assert_eq!(tensor.num_elements(), 6);
/// ```
#[derive(Clone)]
pub struct ANETensor {
    data: Vec<u8>,
    shape: TensorShape,
    dtype: TensorDType,
}

impl ANETensor {
    /// Create a new tensor from raw bytes
    ///
    /// # Arguments
    ///
    /// * `data` - Raw tensor data as bytes
    /// * `shape` - Tensor dimensions (e.g., [batch, channels, height, width])
    /// * `dtype` - Data type of tensor elements
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data length doesn't match shape * dtype size
    /// - Shape is empty
    /// - Shape contains zero dimensions
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::tensor::{ANETensor, TensorDType};
    /// let data: Vec<u8> = vec![0, 0, 128, 63]; // [1.0f32] in little-endian bytes
    /// let tensor = ANETensor::new(data, vec![1, 1, 1, 1], TensorDType::FP32).unwrap();
    /// ```
    pub fn new(
        data: Vec<u8>,
        shape: TensorShape,
        dtype: TensorDType,
    ) -> Result<Self, crate::Error> {
        // Validate shape
        if shape.is_empty() {
            return Err(Error::InvalidTensorShape(
                "Shape cannot be empty".to_string(),
            ));
        }

        if shape.iter().any(|&dim| dim == 0) {
            return Err(Error::InvalidTensorShape(
                "Shape cannot contain zero dimensions".to_string(),
            ));
        }

        // Validate data length
        let num_elements = shape.iter().product::<usize>();
        let expected_bytes = num_elements * dtype.size_bytes();

        if data.len() != expected_bytes {
            return Err(Error::InvalidTensorShape(format!(
                "Data length ({}) does not match shape {:?} with dtype {:?} (expected {} bytes)",
                data.len(),
                shape,
                dtype,
                expected_bytes
            )));
        }

        Ok(ANETensor { data, shape, dtype })
    }

    /// Create a tensor from FP32 data
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::tensor::ANETensor;
    /// let data = vec![1.0f32, 2.0, 3.0, 4.0];
    /// let tensor = ANETensor::from_fp32(data, vec![2, 2]).unwrap();
    /// ```
    pub fn from_fp32(data: Vec<f32>, shape: TensorShape) -> Result<Self, crate::Error> {
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
            .to_vec()
        };
        Self::new(bytes, shape, TensorDType::FP32)
    }

    /// Create a tensor from FP16 data
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::tensor::ANETensor;
    /// let data = vec![0x3c00u16]; // 1.0 in FP16
    /// let tensor = ANETensor::from_fp16(data, vec![1, 1]).unwrap();
    /// ```
    pub fn from_fp16(data: Vec<u16>, shape: TensorShape) -> Result<Self, crate::Error> {
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<u16>(),
            )
            .to_vec()
        };
        Self::new(bytes, shape, TensorDType::FP16)
    }

    /// Create a tensor from INT8 data
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::tensor::ANETensor;
    /// let data = vec![1i8, 2, 3, 4];
    /// let tensor = ANETensor::from_int8(data, vec![2, 2]).unwrap();
    /// ```
    pub fn from_int8(data: Vec<i8>, shape: TensorShape) -> Result<Self, crate::Error> {
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<i8>(),
            )
            .to_vec()
        };
        Self::new(bytes, shape, TensorDType::INT8)
    }

    /// Get the tensor shape
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::tensor::ANETensor;
    /// let tensor = ANETensor::from_fp32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the tensor data type
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    /// Get the total number of elements in the tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use rustane::wrapper::tensor::ANETensor;
    /// let tensor = ANETensor::from_fp32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// assert_eq!(tensor.num_elements(), 4);
    /// ```
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the total size in bytes
    pub fn num_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the raw byte data
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Get a mutable reference to the raw byte data
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Convert to raw byte vector (consumes the tensor)
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Convert to Vec<f32> (if dtype is FP32)
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not FP32.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        assert_eq!(
            self.dtype,
            TensorDType::FP32,
            "to_vec_f32 called on non-FP32 tensor"
        );
        self.data
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}

impl fmt::Debug for ANETensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ANETensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("num_bytes", &self.data.len())
            .finish()
    }
}

impl AsRef<[u8]> for ANETensor {
    fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl AsMut<[u8]> for ANETensor {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation_fp32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = ANETensor::from_fp32(data.clone(), vec![2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.dtype(), TensorDType::FP32);
        assert_eq!(tensor.num_elements(), 4);
        assert_eq!(tensor.num_bytes(), 16);
    }

    #[test]
    fn test_tensor_empty_shape() {
        let data = vec![1.0f32];
        let result = ANETensor::from_fp32(data, vec![]);
        assert!(matches!(result, Err(Error::InvalidTensorShape(_))));
    }

    #[test]
    fn test_tensor_zero_dimension() {
        let data = vec![1.0f32];
        let result = ANETensor::from_fp32(data, vec![1, 0]);
        assert!(matches!(result, Err(Error::InvalidTensorShape(_))));
    }

    #[test]
    fn test_tensor_size_mismatch() {
        let data = vec![1.0f32, 2.0]; // 2 elements
        let result = ANETensor::from_fp32(data, vec![2, 2]); // expects 4 elements
        assert!(matches!(result, Err(Error::InvalidTensorShape(_))));
    }

    #[test]
    fn test_tensor_as_bytes() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = ANETensor::from_fp32(data, vec![2, 2]).unwrap();
        assert_eq!(tensor.as_bytes().len(), 16);
    }

    #[test]
    fn test_tensor_dtype_sizes() {
        assert_eq!(TensorDType::FP16.size_bytes(), 2);
        assert_eq!(TensorDType::FP32.size_bytes(), 4);
        assert_eq!(TensorDType::INT8.size_bytes(), 1);
    }
}
