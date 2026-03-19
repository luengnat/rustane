//! Weight loading and saving utilities

use crate::layers::Shape;
use crate::{Error, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Load weights from a binary file
///
/// # Arguments
///
/// * `path` - Path to weight file
///
/// # Returns
///
/// A tuple of (weights, shape) where weights is a Vec<f32> and shape is the tensor shape
///
/// # Example
///
/// ```no_run
/// # use rustane::utils::load_weights_from_file;
/// let (weights, shape) = load_weights_from_file("model_weights.bin").unwrap();
/// ```
pub fn load_weights_from_file(path: &Path) -> Result<(Vec<f32>, Shape)> {
    let file =
        File::open(path).map_err(|e| Error::Io(format!("Failed to open weight file: {}", e)))?;

    let mut reader = BufReader::new(file);

    // Read ndims
    let mut ndims_bytes = [0u8; 4];
    reader
        .read_exact(&mut ndims_bytes)
        .map_err(|e| Error::Io(format!("Failed to read ndims: {}", e)))?;
    let ndims = u32::from_le_bytes(ndims_bytes);

    // Read shape dimensions
    let mut shape = Vec::new();
    for _ in 0..ndims {
        let mut dim_bytes = [0u8; 4];
        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| Error::Io(format!("Failed to read dim: {}", e)))?;
        let dim = u32::from_le_bytes(dim_bytes);
        shape.push(dim as usize);
    }

    // Read weight data
    let num_weights = shape.iter().product::<usize>();
    let mut weights_bytes = vec![0u8; num_weights * 4];
    reader
        .read_exact(&mut weights_bytes)
        .map_err(|e| Error::Io(format!("Failed to read weights: {}", e)))?;

    let weights: Vec<f32> = weights_bytes
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect();

    Ok((weights, shape))
}

/// Save weights to a binary file
///
/// # Arguments
///
/// * `path` - Path to save weights
/// * `weights` - Weight values
/// * `shape` - Tensor shape
///
/// # Example
///
/// ```no_run
/// # use rustane::utils::save_weights_to_file;
/// # let weights = vec![0.5f32; 1000];
/// # let shape = vec![10, 10, 10, 1];
/// save_weights_to_file("model_weights.bin", &weights, &shape).unwrap();
/// ```
pub fn save_weights_to_file(path: &Path, weights: &[f32], shape: &Shape) -> Result<()> {
    let file = File::create(path)
        .map_err(|e| Error::Io(format!("Failed to create weight file: {}", e)))?;

    let mut writer = BufWriter::new(file);

    // Write shape metadata
    let ndims = shape.len() as u32;
    writer
        .write_all(&ndims.to_le_bytes())
        .map_err(|e| Error::Io(format!("Failed to write ndims: {}", e)))?;

    for dim in shape {
        let dim_bytes = (*dim as u32).to_le_bytes();
        writer
            .write_all(&dim_bytes)
            .map_err(|e| Error::Io(format!("Failed to write dim: {}", e)))?;
    }

    // Write weight data
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * 4) };
    writer
        .write_all(bytes)
        .map_err(|e| Error::Io(format!("Failed to write weights: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_save_load_roundtrip() {
        let weights = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_weights.bin");

        save_weights_to_file(&path, &weights, &shape).unwrap();
        let (loaded_weights, loaded_shape) = load_weights_from_file(&path).unwrap();

        assert_eq!(loaded_weights, weights);
        assert_eq!(loaded_shape, shape);
    }

    #[test]
    fn test_load_weights_4d() {
        let weights = vec![1.0f32; 24]; // 2x2x2x3
        let shape = vec![2, 2, 2, 3];

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_4d.bin");

        save_weights_to_file(&path, &weights, &shape).unwrap();
        let (loaded_weights, loaded_shape) = load_weights_from_file(&path).unwrap();

        assert_eq!(loaded_weights.len(), 24);
        assert_eq!(loaded_shape, vec![2, 2, 2, 3]);
    }
}
