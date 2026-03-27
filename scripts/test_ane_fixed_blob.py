#!/usr/bin/env python3
"""
Fixed weight blob builder matching bridge implementation.
"""

import ctypes
import struct
import numpy as np
from pathlib import Path

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


def build_weight_blob_fixed(weights_f32):
    """
    Build ANE weight blob matching bridge implementation.

    Key difference: Use direct byte writing instead of numpy float16,
    which may have different byte order or rounding.
    """
    rows, cols = (
        weights_f32.shape if len(weights_f32.shape) == 2 else (1, len(weights_f32))
    )
    ws = rows * cols * 2  # fp16 = 2 bytes per element
    tot = 128 + ws
    buf = bytearray(tot)

    # Global header (64 bytes)
    buf[0] = 1  # version
    buf[4] = 2  # type

    # Chunk header (64 bytes at offset 64)
    buf[64] = 0xEF
    buf[65] = 0xBE
    buf[66] = 0xAD
    buf[67] = 0xDE  # magic
    buf[68] = 1  # chunk count
    struct.pack_into("<I", buf, 72, ws)  # data size
    struct.pack_into("<I", buf, 80, 128)  # data offset

    # Convert float32 to float16 manually at byte level
    # Use numpy for conversion but ensure correct byte order
    flat = weights_f32.astype(np.float16).ravel()

    # Copy bytes directly (not through numpy view)
    data_bytes = flat.tobytes()
    buf[128 : 128 + len(data_bytes)] = data_bytes

    return bytes(buf)


# Test with exact ANE project dimensions
DIM = 768
SEQ = 256

print(f"\nTesting with fixed blob builder")
print("=" * 60)

# Create weight matrix (same as ANE project test)
W = np.random.randn(DIM, DIM).astype(np.float32) * 0.02

# Build blob using fixed method
weight_blob = build_weight_blob_fixed(W)
print(f"Weight blob size: {len(weight_blob)} bytes")

# Build MIL
build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]'

mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, {DIM}, 1, {SEQ}]> x) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [{DIM},{DIM},1,1]> Wq = const()[name=string("Wq"), val=tensor<fp16, [{DIM},{DIM},1,1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        tensor<fp16, [1,{DIM},1,{SEQ}]> q = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string("q")];
    }} -> (q);
}}"""

mil = mil_text.encode("utf-8")

in_sz = (ctypes.c_size_t * 1)(DIM * SEQ * 2)
out_sz = (ctypes.c_size_t * 1)(DIM * SEQ * 2)
wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

print("Compiling...")
kernel = lib.ane_bridge_compile(
    mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
)

if not kernel:
    print("❌ Compilation failed")
else:
    print("✅ Compilation succeeded")

    # Create input
    x = np.random.randn(1, DIM, 1, SEQ).astype(np.float32) * 0.1
    x_fp16 = x.astype(np.float16)

    lib.ane_bridge_write_input(
        ctypes.c_void_p(kernel), 0, x_fp16.ctypes.data, x_fp16.nbytes
    )

    print("Evaluating...")
    success = lib.ane_bridge_eval(ctypes.c_void_p(kernel))

    if success:
        print("✅ Evaluation succeeded")

        out_fp16 = np.empty((1, DIM, 1, SEQ), dtype=np.float16)
        lib.ane_bridge_read_output(
            ctypes.c_void_p(kernel), 0, out_fp16.ctypes.data, out_fp16.nbytes
        )
        out = out_fp16.astype(np.float32)

        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")

        if np.allclose(out, 0):
            print("❌ Still all zeros")
        else:
            print("✅ Output has values!")

            # Compare with CPU reference
            x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, DIM)
            W_reshaped = W.T
            cpu_out = (
                (x_reshaped @ W_reshaped).reshape(1, SEQ, DIM, 1).transpose(0, 2, 3, 1)
            )

            diff = np.abs(out - cpu_out)
            print(f"\nCPU comparison:")
            print(f"  Max diff: {diff.max():.6f}")
            print(f"  Mean diff: {diff.mean():.6f}")

            if diff.max() < 0.01:
                print("  ✅ ANE matches CPU!")
            else:
                print("  ❌ ANE differs from CPU")
    else:
        print("❌ Evaluation failed")

    lib.ane_bridge_free(ctypes.c_void_p(kernel))

print("=" * 60)
