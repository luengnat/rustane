//! Raw ANE FFI bindings and runtime integration layer.
//!
//! This module re-exports the public items from the runtime module
//! to maintain compatibility with the wrapper layer's import paths.

use core::ffi::c_int;
use core::ffi::c_void;

pub(crate) use super::runtime::{
    ANEKernelHandle, ane_bridge_compile, ane_bridge_compile_multi_weights, ane_bridge_eval,
    ane_bridge_free, ane_bridge_get_compile_count, ane_bridge_init, ane_bridge_read_output,
    ane_bridge_reset_compile_count, ane_bridge_write_input,
};

// Wrapper functions that delegate to blobs module functions
pub(crate) unsafe fn ane_bridge_build_weight_blob(
    src: *const f32,
    rows: c_int,
    cols: c_int,
    out_len: *mut usize,
) -> *mut u8 {
    super::blobs::build_weight_blob(src, rows, cols, out_len)
}

pub(crate) unsafe fn ane_bridge_build_weight_blob_transposed(
    src: *const f32,
    rows: c_int,
    cols: c_int,
    out_len: *mut usize,
) -> *mut u8 {
    super::blobs::build_weight_blob_transposed(src, rows, cols, out_len)
}

pub(crate) unsafe fn ane_bridge_free_blob(ptr: *mut c_void) {
    super::blobs::free_blob(ptr)
}
