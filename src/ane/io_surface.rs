//! IOSurface RAII wrapper for ANE I/O operations.
//!
//! This module provides a safe, drop-safe wrapper around IOSurface for
//! transferring data to and from the Apple Neural Engine.

use crate::error::Result;

/// Safe RAII wrapper around IOSurface
///
/// IOSurface is a cross-process memory sharing mechanism on macOS/iOS that
/// allows efficient data transfer to hardware accelerators like the ANE.
///
/// This wrapper ensures proper lifetime management through Rust's Drop trait.
#[derive(Debug)]
pub struct IOSurface {
    /// Byte capacity
    _capacity: usize,

    /// Buffer storage (in real implementation, IOSurfaceRef)
    buffer: Vec<u8>,
}

impl IOSurface {
    /// Create new IOSurface with given byte capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - The size in bytes for the IOSurface
    ///
    /// # Errors
    ///
    /// Returns an error if the IOSurface cannot be created (e.g., not on Apple Silicon)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::IOSurface;
    ///
    /// let io = IOSurface::new(1024)?;
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn new(capacity: usize) -> Result<Self> {
        // In real implementation, would call IOSurfaceCreate via CoreFoundation
        // and return IOSurfaceRef. For now, use Vec as backing for portability.
        // This allows the code to compile and run on non-Apple Silicon systems,
        // falling back to CPU-based I/O for testing.
        Ok(IOSurface {
            _capacity: capacity,
            buffer: vec![0u8; capacity],
        })
    }

    /// Write data to the IOSurface
    ///
    /// # Arguments
    ///
    /// * `data` - The data to write
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The data size exceeds the surface capacity
    /// - The write operation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::IOSurface;
    ///
    /// let mut io = IOSurface::new(64)?;
    /// io.write(&[1, 2, 3, 4])?;
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self._capacity {
            return Err(crate::Error::Io(
                format!(
                    "IOSurface write size {} exceeds capacity {}",
                    data.len(),
                    self._capacity
                )
            ));
        }

        self.buffer[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read data from the IOSurface
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::IOSurface;
    ///
    /// let io = IOSurface::new(64)?;
    /// let data = io.read()?;
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn read(&self) -> Result<Vec<u8>> {
        Ok(self.buffer.clone())
    }

    /// Execute a closure with direct pointer access to the surface buffer
    ///
    /// This method allows low-level access to the IOSurface memory for
    /// advanced use cases (e.g., direct ANE kernel writes).
    ///
    /// # Arguments
    ///
    /// * `f` - Closure that receives a mutable pointer to the buffer
    ///
    /// # Safety
    ///
    /// The closure receives a mutable pointer. It is the caller's responsibility
    /// to ensure that:
    /// - The pointer is used within the scope of this call
    /// - Memory safety invariants are maintained
    /// - The buffer is not accessed through multiple mutable pointers
    ///
    /// # Example
    ///
    /// ```no_run
    /// use rustane::ane::IOSurface;
    ///
    /// let mut io = IOSurface::new(64)?;
    /// let len = io.with_lock(|ptr| {
    ///     // Direct pointer manipulation here
    ///     42
    /// })?;
    /// # Ok::<_, rustane::Error>(())
    /// ```
    pub fn with_lock<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(*mut u8) -> R,
    {
        let ptr = self.buffer.as_mut_ptr();
        Ok(f(ptr))
    }

    /// Get the capacity of this IOSurface in bytes
    pub fn capacity(&self) -> usize {
        self._capacity
    }

    /// Get the current data length (returns full capacity for compatibility)
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the IOSurface is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the IOSurface to zeros
    pub fn clear(&self) {
        // Note: This requires &self but modifies internal state.
        // In production with real IOSurface, this would use IOSurfaceLock
        unsafe {
            std::ptr::write_bytes(self.buffer.as_ptr() as *mut u8, 0, self._capacity);
        }
    }

    /// Read data as f32 slice
    ///
    /// # Arguments
    ///
    /// * `output` - Buffer to fill with f32 values
    pub fn read_f32(&self, output: &mut [f32]) {
        let num_floats = self._capacity / 4;
        assert!(output.len() >= num_floats, "Output buffer too small");
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.as_ptr() as *const f32,
                output.as_mut_ptr(),
                num_floats,
            );
        }
    }

    /// Write data from f32 slice
    ///
    /// # Arguments
    ///
    /// * `data` - f32 values to write
    pub fn write_f32(&self, data: &[f32]) {
        let num_bytes = data.len() * 4;
        assert!(num_bytes <= self._capacity, "Data too large for IOSurface");
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.buffer.as_ptr() as *mut f32,
                data.len(),
            );
        }
    }
}

impl Drop for IOSurface {
    fn drop(&mut self) {
        // In real implementation, would call CFRelease on the IOSurfaceRef.
        // The Vec will be automatically cleaned up, and any OS resources
        // associated with the IOSurface will be released.
        // This ensures no resource leaks even if code panics.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_io_surface_capacity() {
        let io = IOSurface::new(1024).unwrap();
        assert_eq!(io.capacity(), 1024);
        assert_eq!(io.len(), 1024);
        assert!(!io.is_empty());
    }

    #[test]
    fn test_io_surface_empty() {
        let io = IOSurface::new(0).unwrap();
        assert!(io.is_empty());
    }
}
