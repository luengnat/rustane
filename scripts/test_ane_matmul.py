#!/usr/bin/env python3
"""
Working ANE Training - Using libane_bridge directly

This script uses ctypes to call the ANE bridge directly,
following the ane-lora-training approach.
"""

import ctypes
import numpy as np
import struct
import tempfile
import os
from pathlib import Path
import time

# Load ANE bridge
ANE_BRIDGE_PATH = (
    Path(__file__).parent.parent / "target" / "debug" / "libane_bridge.dylib"
)

# Build info from ane-lora-training (required by ANE compiler)
BUILD_INFO = (
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]'
)


def load_ane_bridge():
    """Load the ANE bridge library."""
    if not ANE_BRIDGE_PATH.exists():
        raise RuntimeError(f"ANE bridge not found at {ANE_BRIDGE_PATH}")

    lib = ctypes.CDLL(str(ANE_BRIDGE_PATH))

    # Define function signatures
    lib.ane_bridge_init.restype = ctypes.c_int
    lib.ane_bridge_init.argtypes = []

    lib.ane_bridge_compile.restype = ctypes.c_void_p
    lib.ane_bridge_compile.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,  # MIL text
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,  # weights
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),  # num_inputs, input_sizes
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),  # num_outputs, output_sizes
    ]

    lib.ane_bridge_write_input.restype = None
    lib.ane_bridge_write_input.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,  # kernel, input_idx
        ctypes.c_char_p,
        ctypes.c_size_t,  # data, size
    ]

    lib.ane_bridge_eval.restype = ctypes.c_int
    lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]

    lib.ane_bridge_read_output.restype = ctypes.c_int
    lib.ane_bridge_read_output.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,  # kernel, output_idx
        ctypes.c_char_p,
        ctypes.c_size_t,  # data, size
    ]

    lib.ane_bridge_free.restype = None
    lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

    return lib


def create_weight_blob(weights_f32):
    """Create ANE weight blob: 128-byte header + fp16 data."""
    fp16_data = weights_f32.astype(np.float16).tobytes()
    wsize = len(fp16_data)

    header = bytearray(128)
    # Global header [0:64]
    header[0:4] = struct.pack("<I", 1)  # version
    header[4:8] = struct.pack("<I", 2)  # type = fp16
    # Chunk header [64:128]
    header[64:68] = struct.pack("<I", 0xDEADBEEF)  # magic
    header[68:72] = struct.pack("<I", 1)  # chunk_count
    struct.pack_into("<I", header, 72, wsize)  # data_size
    struct.pack_into("<I", header, 80, 128)  # data_offset

    return bytes(header) + fp16_data


def gen_conv_mil(in_ch, out_ch, spatial):
    """Generate MIL program for 1x1 conv."""
    return f"""program(1.3)
{BUILD_INFO}
{{
    func main<ios18>(tensor<fp32, [1, {in_ch}, 1, {spatial}]> x) {{
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, {in_ch}, 1, {spatial}]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        tensor<fp16, [{out_ch}, {in_ch}, 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [{out_ch}, {in_ch}, 1, 1]>(BLOBFILE(path = string("@model_path/weight.bin"), offset = uint64(128)))];
        tensor<fp16, [1, {out_ch}, 1, {spatial}]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, {out_ch}, 1, {spatial}]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    }} -> (y);
}}
"""


def ane_matmul(lib, W, x):
    """
    Compute y = x @ W.T using ANE 1x1 conv.

    Args:
        lib: ANE bridge library
        W: [out_features, in_features] weight matrix
        x: [batch, in_features] input

    Returns:
        y: [batch, out_features] output
    """
    out_ch, in_ch = W.shape
    orig_spatial = x.shape[0]  # batch size

    # Pad spatial to multiple of 16 (ANE requirement)
    spatial = ((orig_spatial + 15) // 16) * 16
    if spatial > orig_spatial:
        x_padded = np.zeros((spatial, in_ch), dtype=np.float32)
        x_padded[:orig_spatial] = x
        x = x_padded

    # Create weight blob
    W_4d = W.reshape(out_ch, in_ch, 1, 1).astype(np.float32)
    weight_blob = create_weight_blob(W_4d)

    # Generate MIL
    mil_text = gen_conv_mil(in_ch, out_ch, spatial)
    mil_bytes = mil_text.encode("utf-8")

    # Prepare arguments
    wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)
    in_sz = (ctypes.c_size_t * 1)(1 * in_ch * 1 * spatial * 4)  # fp32
    out_sz = (ctypes.c_size_t * 1)(1 * out_ch * 1 * spatial * 4)

    # Compile kernel
    kernel = lib.ane_bridge_compile(
        (ctypes.c_uint8 * len(mil_bytes))(*mil_bytes),
        len(mil_bytes),
        wb,
        len(weight_blob),
        1,
        in_sz,
        1,
        out_sz,
    )

    if not kernel:
        raise RuntimeError(
            f"ANE compile failed: W[{out_ch},{in_ch}] x[{orig_spatial},{in_ch}]"
        )

    try:
        # Write input
        x_4d = x.reshape(1, in_ch, 1, spatial).astype(np.float32)
        x_buf = x_4d.tobytes()
        lib.ane_bridge_write_input(
            kernel, 0, ctypes.c_char_p(x_buf), ctypes.c_size_t(len(x_buf))
        )

        # Execute
        result = lib.ane_bridge_eval(kernel)
        if result != 0:
            raise RuntimeError(f"ANE eval failed with code {result}")

        # Read output
        output_buf = ctypes.create_string_buffer(out_ch * spatial * 4)  # fp32
        lib.ane_bridge_read_output(
            kernel, 0, output_buf, ctypes.c_size_t(len(output_buf))
        )

        # Reshape and truncate
        y = np.frombuffer(output_buf.raw, dtype=np.float32).reshape(out_ch, spatial).T
        return y[:orig_spatial]

    finally:
        lib.ane_bridge_free(kernel)


def test_ane_matmul():
    """Test ANE matrix multiplication."""
    print("Loading ANE bridge...")
    lib = load_ane_bridge()

    print("Initializing ANE...")
    result = lib.ane_bridge_init()
    if result != 0:
        raise RuntimeError(f"ANE init failed: {result}")

    print("\nTesting ANE matmul...")

    # Test parameters
    batch = 16  # Must be >= 16 and multiple of 16 for ANE
    in_features = 512
    out_features = 512

    # Create test matrices
    W = np.random.randn(out_features, in_features).astype(np.float32) * 0.01
    x = np.random.randn(batch, in_features).astype(np.float32) * 0.01

    print(f"Weight shape: {W.shape}")
    print(f"Input shape: {x.shape}")

    # Compute on ANE
    start = time.time()
    y_ane = ane_matmul(lib, W, x)
    ane_time = time.time() - start

    print(f"ANE result shape: {y_ane.shape}")
    print(f"ANE time: {ane_time * 1000:.2f} ms")

    # Compute on CPU for verification
    start = time.time()
    y_cpu = x @ W.T
    cpu_time = time.time() - start

    print(f"CPU time: {cpu_time * 1000:.2f} ms")

    # Check accuracy
    max_error = np.max(np.abs(y_ane - y_cpu))
    print(f"Max error vs CPU: {max_error:.6f}")

    if max_error < 0.01:
        print("✅ ANE matmul working correctly!")
        return True
    else:
        print(f"⚠️  Large error: {max_error}")
        return False


if __name__ == "__main__":
    try:
        test_ane_matmul()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
