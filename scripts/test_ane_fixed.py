#!/usr/bin/env python3
"""Test ANE with exact format from ANE project"""

import ctypes
import struct
import numpy as np
from pathlib import Path
import tempfile
import os

lib = ctypes.CDLL(str(Path("target/debug/libane_bridge.dylib").absolute()))
lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_free.restype = None

print("Init:", lib.ane_bridge_init())


# Build weight blob EXACTLY like ANE project
def build_weight_blob(weights_f32):
    wsize = len(weights_f32) * 2
    total = 64 + 64 + wsize
    buf = bytearray(total)
    buf[0] = 0x01
    buf[4] = 0x02
    chunk = buf[64:]
    chunk[0] = 0xEF
    chunk[1] = 0xBE
    chunk[2] = 0xAD
    chunk[3] = 0xDE
    chunk[4] = 0x01
    struct.pack_into("<I", chunk, 8, wsize)
    struct.pack_into("<I", chunk, 16, 128)
    fp16 = np.frombuffer(buf[128:], dtype=np.float16)
    fp16[:] = weights_f32.astype(np.float16)
    return bytes(buf)


W = np.random.randn(16, 16, 1, 1).astype(np.float32) * 0.01
weight_blob = build_weight_blob(W.flatten())

tmp_path = tempfile.mktemp(suffix=".bin")
with open(tmp_path, "wb") as f:
    f.write(weight_blob)

# Build MIL
build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]'

mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp32, [1, 16, 1, 16]> x) {{
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
        tensor<fp16, [1, 16, 1, 16]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
        tensor<fp16, [16, 16, 1, 1]> W = const()[name = string("W"), 
            val = tensor<fp16, [16, 16, 1, 1]>(BLOBFILE(path = string("{tmp_path}"), offset = uint64(64)))];
        tensor<fp16, [1, 16, 1, 16]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
        tensor<fp32, [1, 16, 1, 16]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
    }} -> (y);
}}"""

mil = mil_text.encode("utf-8")
print(f"MIL: {len(mil)} bytes")

in_sz = (ctypes.c_size_t * 1)(16 * 16 * 4)
out_sz = (ctypes.c_size_t * 1)(16 * 16 * 4)

wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

print("Testing ANE project format...")
kernel = lib.ane_bridge_compile(
    mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
)

if kernel:
    print("✅ CONV WORKS with ANE project format!")
    lib.ane_bridge_free(ctypes.c_void_p(kernel))
else:
    print("❌ Still fails")

os.unlink(tmp_path)
