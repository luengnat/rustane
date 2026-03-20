//! ANE weight/blob builders.

use core::ffi::{c_int, c_void};
use core::ptr;
use libc::{calloc, free};
use std::slice;

const FP16_BLOB_HEADER_SIZE: usize = 128;
const INT8_BLOB_HEADER_SIZE: usize = 64;

pub(crate) fn encode_fp16_blob_from_f32(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    encode_fp16_blob_impl(weights, rows, cols, false)
}

pub(crate) fn encode_fp16_blob_from_f32_transposed(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    encode_fp16_blob_impl(weights, rows, cols, true)
}

pub(crate) fn encode_fp16_blob_from_f16(
    weights: &[half::f16],
    rows: usize,
    cols: usize,
) -> Vec<u8> {
    let mut buf = vec![0u8; FP16_BLOB_HEADER_SIZE + rows * cols * 2];
    fill_fp16_header(&mut buf, rows * cols * 2);
    let fp16 = bytemuck_cast_slice_mut_u16(&mut buf[FP16_BLOB_HEADER_SIZE..]);
    for (dst, src) in fp16.iter_mut().zip(weights.iter().copied()) {
        *dst = src.to_bits();
    }
    buf
}

pub(crate) fn encode_int8_blob(weights: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    let mut buf = vec![0u8; INT8_BLOB_HEADER_SIZE + rows * cols];
    fill_int8_header(&mut buf);
    for (dst, src) in buf[INT8_BLOB_HEADER_SIZE..]
        .iter_mut()
        .zip(weights.iter().copied())
    {
        *dst = src as u8;
    }
    buf
}

pub(crate) fn quantize_f32_per_row(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut scales = Vec::with_capacity(rows);
    let mut quantized = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let row_start = row * cols;
        let row_end = row_start + cols;
        let row_weights = &weights[row_start..row_end];
        let max_abs = row_weights
            .iter()
            .map(|w| w.abs())
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let scale = max_abs / 127.0;
        scales.push(scale);

        for &w in row_weights {
            quantized.push((w / scale).round().clamp(-127.0, 127.0) as i8);
        }
    }

    (quantized, scales)
}

pub(crate) unsafe fn build_weight_blob(
    src: *const f32,
    rows: c_int,
    cols: c_int,
    out_len: *mut usize,
) -> *mut u8 {
    build_fp16_blob(src, rows, cols, out_len, false)
}

pub(crate) unsafe fn build_weight_blob_transposed(
    src: *const f32,
    rows: c_int,
    cols: c_int,
    out_len: *mut usize,
) -> *mut u8 {
    build_fp16_blob(src, rows, cols, out_len, true)
}

unsafe fn build_fp16_blob(
    src: *const f32,
    rows: c_int,
    cols: c_int,
    out_len: *mut usize,
    transpose: bool,
) -> *mut u8 {
    if src.is_null() || out_len.is_null() || rows <= 0 || cols <= 0 {
        return ptr::null_mut();
    }

    let rows = rows as usize;
    let cols = cols as usize;
    let src = unsafe { slice::from_raw_parts(src, rows * cols) };
    let blob = if transpose {
        encode_fp16_blob_from_f32_transposed(src, rows, cols)
    } else {
        encode_fp16_blob_from_f32(src, rows, cols)
    };

    let total = blob.len();
    let buf = unsafe { calloc(total, 1) as *mut u8 };
    if buf.is_null() {
        return ptr::null_mut();
    }
    unsafe { ptr::copy_nonoverlapping(blob.as_ptr(), buf, total) };

    unsafe { *out_len = total };
    buf
}

#[allow(dead_code)]
pub(crate) unsafe fn build_weight_blob_int8(
    src: *const i8,
    rows: c_int,
    cols: c_int,
    out_len: *mut usize,
) -> *mut u8 {
    if src.is_null() || out_len.is_null() || rows <= 0 || cols <= 0 {
        return ptr::null_mut();
    }
    let rows = rows as usize;
    let cols = cols as usize;
    let weights = unsafe { slice::from_raw_parts(src, rows * cols) };
    let blob = encode_int8_blob(weights, rows, cols);
    let total = blob.len();
    let buf = unsafe { calloc(total, 1) as *mut u8 };
    if buf.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        ptr::copy_nonoverlapping(blob.as_ptr(), buf, total);
        *out_len = total;
    }
    buf
}

#[allow(dead_code)]
pub(crate) unsafe fn build_weight_blob_quantized(
    src: *const f32,
    rows: c_int,
    cols: c_int,
    out_scale: *mut f32,
    out_len: *mut usize,
) -> *mut u8 {
    if src.is_null() || out_scale.is_null() || out_len.is_null() || rows <= 0 || cols <= 0 {
        return ptr::null_mut();
    }

    let rows = rows as usize;
    let cols = cols as usize;
    let src_slice = unsafe { slice::from_raw_parts(src, rows * cols) };
    let (quantized, scales) = quantize_f32_per_row(src_slice, rows, cols);
    unsafe { *out_scale = scales.iter().copied().fold(0.0f32, f32::max) };
    unsafe { build_weight_blob_int8(quantized.as_ptr(), rows as c_int, cols as c_int, out_len) }
}

pub(crate) unsafe fn free_blob(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe { free(ptr) };
    }
}

fn encode_fp16_blob_impl(weights: &[f32], rows: usize, cols: usize, transpose: bool) -> Vec<u8> {
    let mut buf = vec![0u8; FP16_BLOB_HEADER_SIZE + rows * cols * 2];
    fill_fp16_header(&mut buf, rows * cols * 2);
    let fp16 = bytemuck_cast_slice_mut_u16(&mut buf[FP16_BLOB_HEADER_SIZE..]);

    if transpose {
        for i in 0..rows {
            for j in 0..cols {
                fp16[j * rows + i] = half::f16::from_f32(weights[i * cols + j]).to_bits();
            }
        }
    } else {
        for (dst, src) in fp16.iter_mut().zip(weights.iter().copied()) {
            *dst = half::f16::from_f32(src).to_bits();
        }
    }

    buf
}

fn fill_fp16_header(buf: &mut [u8], payload_size: usize) {
    buf[0] = 0x01;
    buf[4] = 0x02;
    buf[64] = 0xEF;
    buf[65] = 0xBE;
    buf[66] = 0xAD;
    buf[67] = 0xDE;
    buf[68] = 0x01;
    buf[72..76].copy_from_slice(&(payload_size as u32).to_le_bytes());
    buf[80..84].copy_from_slice(&(FP16_BLOB_HEADER_SIZE as u32).to_le_bytes());
}

fn fill_int8_header(buf: &mut [u8]) {
    buf[0] = 0xEF;
    buf[1] = 0xBE;
    buf[2] = 0xAD;
    buf[3] = 0xDE;
    buf[4] = 0x01;
    buf[10] = 0x08;
}

fn bytemuck_cast_slice_mut_u16(bytes: &mut [u8]) -> &mut [u16] {
    let len = bytes.len() / 2;
    unsafe { slice::from_raw_parts_mut(bytes.as_mut_ptr().cast::<u16>(), len) }
}
