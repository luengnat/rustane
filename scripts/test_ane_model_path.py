#!/usr/bin/env python3
"""Test ANE with correct @model_path format matching the working ANE project"""

import ctypes
import struct
import numpy as np
from pathlib import Path
import os

lib = ctypes.CDLL(str(Path("target/debug/libane_bridge.dylib").absolute()))
lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_free.restype = None

print("Init:", lib.ane_bridge_init())


def build_weight_blob(weights_f32):
    """Build ANE weight blob with 128-byte header (64 global + 64 chunk)"""
    wsize = len(weights_f32) * 2
    total = 128 + wsize
    buf = bytearray(total)
    # Global header (64 bytes)
    buf[0] = 0x01  # version
    buf[4] = 0x02  # type
    # Chunk header (64 bytes at offset 64)
    buf[64] = 0xEF
    buf[65] = 0xBE
    buf[66] = 0xAD
    buf[67] = 0xDE  # magic
    buf[68] = 0x01  # chunk count
    struct.pack_into("<I", buf, 72, wsize)  # data size
    struct.pack_into("<I", buf, 80, 128)  # data offset
    # Convert to fp16
    fp16 = np.frombuffer(buf[128:], dtype=np.float16)
    fp16[:] = weights_f32.astype(np.float16)
    return bytes(buf)


# Generate random weights
W = np.random.randn(16 * 16).astype(np.float32) * 0.01
weight_blob = build_weight_blob(W)

# Build MIL using @model_path format (like working ANE project)
build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]'

mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, 16, 1, 16]> x) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [16, 16, 1, 1]> W = const()[name=string("W"), val=tensor<fp16, [16, 16, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        tensor<fp16, [1, 16, 1, 16]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("conv")];
    }} -> (y);
}}"""

mil = mil_text.encode("utf-8")
print(f"MIL ({len(mil)} bytes):")
print("-" * 40)
print(mil_text)
print("-" * 40)

in_sz = (ctypes.c_size_t * 1)(16 * 16 * 2)
out_sz = (ctypes.c_size_t * 1)(16 * 16 * 2)

wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

print("\nTesting ANE conv with @model_path format...")
kernel = lib.ane_bridge_compile(
    mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
)

if kernel:
    print("✅ CONV WORKS!")
    lib.ane_bridge_free(ctypes.c_void_p(kernel))
else:
    print("❌ Conv compilation failed")
    print("\nTrying simpler identity operation...")

    # Test identity as baseline
    mil_simple = b"""program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 16, 1, 16]> x) {
        tensor<fp16, [1, 16, 1, 16]> y = identity(x=x)[name=string("id")];
    } -> (y);
}"""

    kernel_simple = lib.ane_bridge_compile(
        mil_simple, len(mil_simple), None, 0, 1, in_sz, 1, out_sz
    )

    if kernel_simple:
        print("✅ Identity works (ANE is functional)")
        lib.ane_bridge_free(ctypes.c_void_p(kernel_simple))
    else:
        print("❌ Even identity fails (ANE issue)")
