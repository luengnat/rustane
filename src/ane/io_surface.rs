//! IOSurface RAII wrapper for ANE I/O operations.
//!
//! This module provides a safe, drop-safe wrapper around IOSurface for
//! transferring data to and from the Apple Neural Engine.
//!
//! On Apple platforms, this creates real IOSurface via CoreFoundation.
//! On other platforms, it falls back to regular memory allocation.

use crate::error::Result;

#[cfg(target_vendor = "apple")]
pub mod sys {

    use std::ffi::c_void;

    // CoreFoundation types
    #[repr(C)]
    pub struct __CFDictionary(c_void);
    pub type CFDictionaryRef = *const __CFDictionary;

    #[repr(C)]
    pub struct __CFString(c_void);
    pub type CFStringRef = *const __CFString;

    #[repr(C)]
    pub struct __CFNumber(c_void);
    pub type CFNumberRef = *const __CFNumber;

    #[repr(C)]
    pub struct __IOSurface(c_void);
    pub type IOSurfaceRef = *mut __IOSurface;

    // CFRelease - releases any CoreFoundation object
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFRelease(cf: *const c_void);
    }

    // IOSurface framework
    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        pub fn IOSurfaceCreate(properties: CFDictionaryRef) -> IOSurfaceRef;
        pub fn IOSurfaceLock(buffer: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceUnlock(buffer: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
        pub fn IOSurfaceGetBaseAddress(buffer: IOSurfaceRef) -> *mut c_void;
        pub fn IOSurfaceGetAllocSize(buffer: IOSurfaceRef) -> usize;
        pub fn IOSurfaceIncrementUseCount(buffer: IOSurfaceRef);
        pub fn IOSurfaceDecrementUseCount(buffer: IOSurfaceRef);
    }

    // CFDictionaryCreate - we need this to create the properties dict
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        pub fn CFDictionaryCreate(
            allocator: *const c_void,
            keys: *const *const c_void,
            values: *const *const c_void,
            num_values: usize,
            key_callback: *const c_void,
            value_callback: *const c_void,
        ) -> CFDictionaryRef;

        pub fn CFNumberCreate(
            allocator: *const c_void,
            type_: i32, // CFNumberType
            value_ptr: *const c_void,
        ) -> CFNumberRef;

        pub fn CFStringCreateWithBytes(
            allocator: *const c_void,
            bytes: *const u8,
            num_bytes: isize,
            encoding: u32,
            is_external_representation: bool,
        ) -> CFStringRef;
    }

    // CFNumber types
    pub const K_CF_NUMBER_SINT32_TYPE: i32 = 3;
    pub const K_CF_NUMBER_SINT64_TYPE: i32 = 4;

    // NSString/CFString constants (we'll use string literals instead)
    pub const K_IOSURFACE_WIDTH: &str = "IOSurfaceWidth";
    pub const K_IOSURFACE_HEIGHT: &str = "IOSurfaceHeight";
    pub const K_IOSURFACE_BYTES_PER_ELEMENT: &str = "IOSurfaceBytesPerElement";
    pub const K_IOSURFACE_BYTES_PER_ROW: &str = "IOSurfaceBytesPerRow";
    pub const K_IOSURFACE_ALLOC_SIZE: &str = "IOSurfaceAllocSize";
    pub const K_IOSURFACE_PIXEL_FORMAT: &str = "IOSurfacePixelFormat";

    // Lock options
    pub const IOSURFACE_LOCK_READ_ONLY: u32 = 0x0000_0001;

    /// Create a CFString from a Rust string
    pub unsafe fn cf_string(s: &str) -> CFStringRef {
        CFStringCreateWithBytes(
            std::ptr::null(),
            s.as_ptr(),
            s.len() as isize,
            0x0800_0100, // kCFStringEncodingUTF8
            false,
        )
    }

    /// Create a CFNumber from a usize
    pub unsafe fn cf_number_usize(n: usize) -> CFNumberRef {
        let n = n as i64;
        CFNumberCreate(
            std::ptr::null(),
            K_CF_NUMBER_SINT64_TYPE,
            &n as *const i64 as *const c_void,
        )
    }

    /// Create IOSurface properties dictionary
    pub unsafe fn create_iosurface_properties(size: usize) -> CFDictionaryRef {
        let keys: [*const c_void; 6] = [
            cf_string(K_IOSURFACE_WIDTH) as *const c_void,
            cf_string(K_IOSURFACE_HEIGHT) as *const c_void,
            cf_string(K_IOSURFACE_BYTES_PER_ELEMENT) as *const c_void,
            cf_string(K_IOSURFACE_BYTES_PER_ROW) as *const c_void,
            cf_string(K_IOSURFACE_ALLOC_SIZE) as *const c_void,
            cf_string(K_IOSURFACE_PIXEL_FORMAT) as *const c_void,
        ];

        let values: [*const c_void; 6] = [
            cf_number_usize(size) as *const c_void,
            cf_number_usize(1) as *const c_void,
            cf_number_usize(1) as *const c_void,
            cf_number_usize(size) as *const c_void,
            cf_number_usize(size) as *const c_void,
            cf_number_usize(0) as *const c_void,
        ];

        CFDictionaryCreate(
            std::ptr::null(),
            keys.as_ptr(),
            values.as_ptr(),
            6,
            std::ptr::null(), // kCFTypeDictionaryKeyCallBacks
            std::ptr::null(), // kCFTypeDictionaryValueCallBacks
        )
    }
}

#[cfg(target_vendor = "apple")]
use sys::*;

/// Safe RAII wrapper around IOSurface
///
/// On Apple platforms, this creates a real IOSurface via CoreFoundation.
/// On other platforms, it falls back to regular memory allocation.
pub struct IOSurface {
    /// Byte capacity
    capacity: usize,

    /// Platform-specific storage
    #[cfg(target_vendor = "apple")]
    surface: *mut sys::__IOSurface,

    /// Fallback storage for non-Apple platforms
    #[cfg(not(target_vendor = "apple"))]
    buffer: Vec<u8>,
}

// SAFETY: IOSurface uses thread-safe IOSurface handles on Apple platforms
unsafe impl Send for IOSurface {}
unsafe impl Sync for IOSurface {}

impl IOSurface {
    /// Create new IOSurface with given byte capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - The size in bytes for the IOSurface
    ///
    /// # Errors
    ///
    /// Returns an error if the IOSurface cannot be created
    pub fn new(capacity: usize) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            unsafe {
                let props = create_iosurface_properties(capacity);
                if props.is_null() {
                    return Err(crate::Error::Io(
                        "Failed to create IOSurface properties".to_string(),
                    ));
                }

                let surface = IOSurfaceCreate(props);
                CFRelease(props as *const _);

                if surface.is_null() {
                    return Err(crate::Error::Io("Failed to create IOSurface".to_string()));
                }

                // Increment use count to prevent premature cleanup
                IOSurfaceIncrementUseCount(surface);

                Ok(IOSurface { capacity, surface })
            }
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(IOSurface {
                capacity,
                buffer: vec![0u8; capacity],
            })
        }
    }

    /// Lock the IOSurface for reading
    ///
    /// Returns a pointer to the buffer memory that's valid while locked.
    /// The pointer is only valid until `unlock_read()` is called.
    pub fn lock_read(&self) -> Result<*const u8> {
        #[cfg(target_vendor = "apple")]
        unsafe {
            let result =
                IOSurfaceLock(self.surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut());
            if result != 0 {
                return Err(crate::Error::Io(
                    "Failed to lock IOSurface for reading".to_string(),
                ));
            }
            Ok(IOSurfaceGetBaseAddress(self.surface) as *const u8)
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(self.buffer.as_ptr())
        }
    }

    /// Unlock the IOSurface after reading
    pub fn unlock_read(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        unsafe {
            let result =
                IOSurfaceUnlock(self.surface, IOSURFACE_LOCK_READ_ONLY, std::ptr::null_mut());
            if result != 0 {
                return Err(crate::Error::Io(
                    "Failed to unlock IOSurface after reading".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Lock the IOSurface for writing
    ///
    /// Returns a pointer to the buffer memory that's valid while locked.
    /// The pointer is only valid until `unlock_write()` is called.
    pub fn lock_write(&self) -> Result<*mut u8> {
        #[cfg(target_vendor = "apple")]
        unsafe {
            let result = IOSurfaceLock(self.surface, 0, std::ptr::null_mut());
            if result != 0 {
                return Err(crate::Error::Io(
                    "Failed to lock IOSurface for writing".to_string(),
                ));
            }
            Ok(IOSurfaceGetBaseAddress(self.surface) as *mut u8)
        }

        #[cfg(not(target_vendor = "apple"))]
        {
            Ok(self.buffer.as_ptr() as *mut u8)
        }
    }

    /// Unlock the IOSurface after writing
    pub fn unlock_write(&self) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        unsafe {
            let result = IOSurfaceUnlock(self.surface, 0, std::ptr::null_mut());
            if result != 0 {
                return Err(crate::Error::Io(
                    "Failed to unlock IOSurface after writing".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Write data to the IOSurface
    ///
    /// This locks the surface, copies data, and unlocks.
    pub fn write(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.capacity {
            return Err(crate::Error::Io(format!(
                "IOSurface write size {} exceeds capacity {}",
                data.len(),
                self.capacity
            )));
        }

        let ptr = self.lock_write()?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        self.unlock_write()?;
        Ok(())
    }

    /// Read all data from the IOSurface into a new Vec
    ///
    /// Convenience method that allocates and returns a Vec.
    pub fn read_vec(&self) -> Result<Vec<u8>> {
        let mut dest = vec![0u8; self.capacity];
        self.read(&mut dest)?;
        Ok(dest)
    }

    /// Read data from the IOSurface
    ///
    /// This locks the surface, copies data, and unlocks.
    pub fn read(&self, dest: &mut [u8]) -> Result<()> {
        if dest.len() > self.capacity {
            return Err(crate::Error::Io(format!(
                "IOSurface read size {} exceeds capacity {}",
                dest.len(),
                self.capacity
            )));
        }

        let ptr = self.lock_read()?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, dest.as_mut_ptr(), dest.len());
        }
        self.unlock_read()?;
        Ok(())
    }

    /// Get the capacity of this IOSurface in bytes
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear the IOSurface to zeros
    pub fn clear(&self) -> Result<()> {
        let ptr = self.lock_write()?;
        unsafe {
            std::ptr::write_bytes(ptr, 0, self.capacity);
        }
        self.unlock_write()
    }

    /// Get the actual allocated size from the IOSurface
    #[cfg(target_vendor = "apple")]
    pub fn alloc_size(&self) -> usize {
        unsafe { IOSurfaceGetAllocSize(self.surface) }
    }

    #[cfg(not(target_vendor = "apple"))]
    pub fn alloc_size(&self) -> usize {
        self.capacity
    }

    /// Read data as f32 slice (converts fp16 → fp32)
    ///
    /// # Safety
    /// The IOSurface must contain valid fp16 data
    pub fn read_f32(&self, output: &mut [f32]) -> Result<()> {
        let num_floats = self.capacity / 2; // fp16 = 2 bytes
        if output.len() < num_floats {
            return Err(crate::Error::Io(format!(
                "Output buffer too small: {} < {}",
                output.len(),
                num_floats
            )));
        }

        let ptr = self.lock_read()?;
        unsafe {
            let fp16_ptr = ptr as *const u16; // fp16 as u16
            for i in 0..num_floats {
                output[i] = fp16_to_f32(*fp16_ptr.add(i));
            }
        }
        self.unlock_read()?;
        Ok(())
    }

    /// Write data from f32 slice (converts fp32 → fp16)
    ///
    /// # Safety
    /// This clamps values to fp16 range
    pub fn write_f32(&self, data: &[f32]) -> Result<()> {
        let num_bytes = data.len() * 2; // fp16 = 2 bytes per element
        if num_bytes > self.capacity {
            return Err(crate::Error::Io(format!(
                "Data too large: {} bytes > {} capacity",
                num_bytes, self.capacity
            )));
        }

        let ptr = self.lock_write()?;
        unsafe {
            let fp16_ptr = ptr as *mut u16;
            for i in 0..data.len() {
                *fp16_ptr.add(i) = f32_to_fp16(data[i]);
            }
        }
        self.unlock_write()?;
        Ok(())
    }
}

#[cfg(target_vendor = "apple")]
impl Drop for IOSurface {
    fn drop(&mut self) {
        unsafe {
            IOSurfaceDecrementUseCount(self.surface);
            CFRelease(self.surface as *const _);
        }
    }
}

impl std::fmt::Debug for IOSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IOSurface")
            .field("capacity", &self.capacity)
            .field("alloc_size", &self.alloc_size())
            .field(
                "platform",
                &if cfg!(target_vendor = "apple") {
                    "Apple (real IOSurface)"
                } else {
                    "fallback (Vec)"
                },
            )
            .finish()
    }
}

/// Convert fp16 (u16 representation) to f32
fn fp16_to_f32(fp16: u16) -> f32 {
    // Simple conversion - for production, use hardware intrinsics or a library
    // This is a basic implementation
    let sign = (fp16 >> 15) as i32;
    let exponent = ((fp16 >> 10) & 0x1f) as i32;
    let mantissa = (fp16 & 0x3ff) as i32;

    if exponent == 0 {
        // Subnormal
        if mantissa == 0 {
            return if sign != 0 { -0.0 } else { 0.0 };
        }
        let val = (mantissa as f32) * (2.0f32.powi(-24));
        return if sign != 0 { -val } else { val };
    }

    if exponent == 0x1f {
        // Infinity or NaN
        if mantissa == 0 {
            return if sign != 0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }
        return f32::NAN;
    }

    // Normal
    let val = (1.0 + (mantissa as f32) / 1024.0) * (2.0f32.powi(exponent - 15));
    if sign != 0 {
        -val
    } else {
        val
    }
}

/// Convert f32 to fp16 (u16 representation)
fn f32_to_fp16(f: f32) -> u16 {
    // Simple conversion - clamps to fp16 range
    // For production, use hardware intrinsics or a library like half-rs
    if f.is_nan() {
        return 0x7fff; // qNaN
    }
    if f.is_infinite() {
        return if f.is_sign_negative() { 0xfc00 } else { 0x7c00 };
    }

    // Clamp to fp16 range
    let f = f.clamp(-65504.0, 65504.0);

    if f == 0.0 {
        return if f.is_sign_negative() { 0x8000 } else { 0 };
    }

    // Basic conversion: extract sign, exponent, mantissa
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    // Handle subnormal numbers
    if exponent == 0 {
        return (sign << 15) as u16;
    }

    // Handle normal numbers
    let new_exponent = exponent - 127 + 15;

    if new_exponent >= 31 {
        // Overflow to infinity
        return (sign << 15) | 0x7C00;
    }

    if new_exponent <= 0 {
        // Underflow to subnormal or zero
        if new_exponent < -10 {
            return (sign << 15) as u16;
        }
        // Subnormal
        let mant = (mantissa | 0x800000) >> (1 - new_exponent);
        return (sign << 15) | ((mant >> 13) as u16);
    }

    // Normal number
    let new_mantissa = (mantissa >> 13) as u16;
    ((sign << 15) as u16) | ((new_exponent as u16) << 10) | new_mantissa
}

#[cfg(test)]
mod tests {
    use crate::ane::*;

    #[test]
    fn test_iosurface_capacity() {
        let io = IOSurface::new(1024).unwrap();
        assert_eq!(io.capacity(), 1024);
        assert!(io.alloc_size() >= 1024);
    }

    #[test]
    fn test_iosurface_write_read() {
        let io = IOSurface::new(64).unwrap();
        let data = vec![1u8, 2, 3, 4, 5];

        io.write(&data).unwrap();

        let mut read = vec![0u8; 5];
        io.read(&mut read).unwrap();

        assert_eq!(data, read);
    }

    #[test]
    fn test_iosurface_write_read_f32() {
        let io = IOSurface::new(32).unwrap(); // 16 floats
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();

        io.write_f32(&data).unwrap();

        let mut read = vec![0.0f32; 16];
        io.read_f32(&mut read).unwrap();

        // Note: fp16 conversion loses precision
        for i in 0..16 {
            assert!((data[i] - read[i]).abs() < 0.01);
        }
    }
}
