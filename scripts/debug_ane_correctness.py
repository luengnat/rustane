#!/usr/bin/env python3
"""
Debug ANE computation correctness by comparing with CPU reference.
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
import tempfile
import os

# Load bridge
lib = ctypes.CDLL(str(Path("target/debug/libane_bridge.dylib").absolute()))
lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_free.restype = None
lib.ane_bridge_eval.restype = ctypes.c_bool
lib.ane_bridge_write_input.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
]
lib.ane_bridge_read_output.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
]

print("Init:", lib.ane_bridge_init())


def build_weight_blob(weights_f32):
    """Build ANE weight blob with proper headers."""
    wsize = len(weights_f32) * 2
    total = 128 + wsize
    buf = bytearray(total)
    buf[0] = 0x01
    buf[4] = 0x02
    buf[64] = 0xEF
    buf[65] = 0xBE
    buf[66] = 0xAD
    buf[67] = 0xDE
    buf[68] = 0x01
    struct.pack_into("<I", buf, 72, wsize)
    struct.pack_into("<I", buf, 80, 128)
    fp16 = np.frombuffer(buf[128:], dtype=np.float16)
    fp16[:] = weights_f32.astype(np.float16)
    return bytes(buf)


def cpu_conv1x1(x, weight):
    """
    CPU reference 1x1 conv.
    x: (B, C_in, 1, S)
    weight: (C_out, C_in, 1, 1)
    out: (B, C_out, 1, S)
    """
    B, C_in, H, S = x.shape
    C_out = weight.shape[0]

    # Reshape: (B, C_in, 1, S) -> (B*S, C_in)
    x_reshaped = x.transpose(0, 3, 1, 2).reshape(-1, C_in)
    # Weight: (C_out, C_in, 1, 1) -> (C_out, C_in) -> (C_in, C_out)
    w_reshaped = weight.reshape(C_out, C_in).T

    # Matmul
    out = x_reshaped @ w_reshaped

    # Reshape back: (B*S, C_out) -> (B, C_out, 1, S)
    out = out.reshape(B, S, C_out, 1).transpose(0, 2, 3, 1)
    return out


# Test with small deterministic values
print("=" * 60)
print("ANE vs CPU Correctness Test")
print("=" * 60)

B, C_in, C_out, S = 1, 4, 8, 16

# Create simple input
x = np.arange(B * C_in * S).reshape(B, C_in, 1, S).astype(np.float32) * 0.1

# Create simple weights (not too small)
W = np.ones((C_out, C_in, 1, 1), dtype=np.float32) * 0.5

print(f"\nInput shape: {x.shape}")
print(f"Weight shape: {W.shape}")
print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Weight values: {W[0, 0, 0, 0]:.2f}")

# CPU reference
cpu_out = cpu_conv1x1(x, W)
print(f"\nCPU output shape: {cpu_out.shape}")
print(f"CPU output range: [{cpu_out.min():.2f}, {cpu_out.max():.2f}]")
print(f"CPU output sample: {cpu_out[0, :3, 0, :3]}")

# Build MIL and compile for ANE
build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]'

mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, {C_in}, 1, {S}]> x) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [{C_out}, {C_in}, 1, 1]> W = const()[name=string("W"), val=tensor<fp16, [{C_out}, {C_in}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        tensor<fp16, [1, {C_out}, 1, {S}]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("conv")];
    }} -> (y);
}}"""

mil = mil_text.encode("utf-8")

# Build weight blob
weight_flat = W.reshape(C_out, C_in).ravel()
weight_blob = build_weight_blob(weight_flat)

print(f"\nWeight blob size: {len(weight_blob)} bytes")

# Compile
in_sz = (ctypes.c_size_t * 1)(C_in * S * 2)
out_sz = (ctypes.c_size_t * 1)(C_out * S * 2)
wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

print("Compiling ANE kernel...")
kernel = lib.ane_bridge_compile(
    mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
)

if not kernel:
    print("❌ Compilation failed")
else:
    print("✅ Compilation succeeded")

    # Run ANE
    x_fp16 = x.astype(np.float16)
    lib.ane_bridge_write_input(
        ctypes.c_void_p(kernel), 0, x_fp16.ctypes.data, x_fp16.nbytes
    )

    success = lib.ane_bridge_eval(ctypes.c_void_p(kernel))
    if success:
        print("✅ ANE evaluation succeeded")

        ane_out_fp16 = np.empty((1, C_out, 1, S), dtype=np.float16)
        lib.ane_bridge_read_output(
            ctypes.c_void_p(kernel), 0, ane_out_fp16.ctypes.data, ane_out_fp16.nbytes
        )
        ane_out = ane_out_fp16.astype(np.float32)

        print(f"\nANE output shape: {ane_out.shape}")
        print(f"ANE output range: [{ane_out.min():.2f}, {ane_out.max():.2f}]")
        print(f"ANE output sample: {ane_out[0, :3, 0, :3]}")

        # Compare
        diff = np.abs(cpu_out - ane_out)
        max_diff = diff.max()
        mean_diff = diff.mean()

        print(f"\n" + "=" * 60)
        print("Comparison:")
        print(f"  Max absolute diff: {max_diff:.6f}")
        print(f"  Mean absolute diff: {mean_diff:.6f}")
        print(f"  CPU output unique values: {len(np.unique(cpu_out))}")
        print(f"  ANE output unique values: {len(np.unique(ane_out))}")
        print(f"  ANE all zeros: {np.allclose(ane_out, 0)}")

        if max_diff < 0.01:
            print("  ✅ PASS: ANE matches CPU reference")
        else:
            print("  ❌ FAIL: ANE output differs from CPU")
            print(f"\n  CPU values:\n{cpu_out[0, :3, 0, :5]}")
            print(f"\n  ANE values:\n{ane_out[0, :3, 0, :5]}")
    else:
        print("❌ ANE evaluation failed")

    lib.ane_bridge_free(ctypes.c_void_p(kernel))

print("\n" + "=" * 60)
