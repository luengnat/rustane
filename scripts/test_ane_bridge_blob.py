#!/usr/bin/env python3
"""
Test using bridge's built-in blob builder.
"""

import ctypes
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

# Use bridge's blob builder
lib.ane_bridge_build_weight_blob.restype = ctypes.POINTER(ctypes.c_uint8)

print("Init:", lib.ane_bridge_init())

# Test dimensions
DIM = 768
SEQ = 256

print(f"\nTesting with built-in blob builder")
print("=" * 60)

# Create weight matrix
W = np.random.randn(DIM, DIM).astype(np.float32) * 0.02

# Build blob using bridge
out_len = ctypes.c_size_t()
blob_ptr = lib.ane_bridge_build_weight_blob(
    W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), DIM, DIM, ctypes.byref(out_len)
)

print(f"Blob size: {out_len.value} bytes")

# Copy blob data
weight_blob = bytes(ctypes.string_at(blob_ptr, out_len.value))

# Try to free (use libc free)
libc = ctypes.CDLL(None)
libc.free(blob_ptr)

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
        print(f"Output mean: {out.mean():.4f}")

        if np.allclose(out, 0):
            print("⚠️  Output is all zeros")
        else:
            print("✅ Output has values!")
    else:
        print("❌ Evaluation failed")

    lib.ane_bridge_free(ctypes.c_void_p(kernel))

print("=" * 60)
