#!/usr/bin/env python3
"""
Test ANE evaluation with different dimensions.
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


def build_weight_blob(weights_f32):
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


def test_conv(C_in, C_out, S):
    """Test conv and evaluate"""
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

    # Use non-zero weights
    W = np.ones((C_out, C_in), dtype=np.float32) * 0.1
    weight_blob = build_weight_blob(W.ravel())

    in_sz = (ctypes.c_size_t * 1)(C_in * S * 2)
    out_sz = (ctypes.c_size_t * 1)(C_out * S * 2)
    wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

    kernel = lib.ane_bridge_compile(
        mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
    )

    if not kernel:
        return "compile_fail"

    # Create input (non-zero)
    x = np.ones((1, C_in, 1, S), dtype=np.float32) * 0.5
    x_fp16 = x.astype(np.float16)

    lib.ane_bridge_write_input(
        ctypes.c_void_p(kernel), 0, x_fp16.ctypes.data, x_fp16.nbytes
    )
    success = lib.ane_bridge_eval(ctypes.c_void_p(kernel))

    if not success:
        lib.ane_bridge_free(ctypes.c_void_p(kernel))
        return "eval_fail"

    # Read output
    out_fp16 = np.empty((1, C_out, 1, S), dtype=np.float16)
    lib.ane_bridge_read_output(
        ctypes.c_void_p(kernel), 0, out_fp16.ctypes.data, out_fp16.nbytes
    )
    out = out_fp16.astype(np.float32)

    lib.ane_bridge_free(ctypes.c_void_p(kernel))

    # Check if output is zeros
    if np.allclose(out, 0):
        return "zeros"
    else:
        return f"ok (mean={out.mean():.3f})"


# Test cases
test_cases = [
    (16, 16, 16),
    (64, 64, 64),
    (256, 256, 64),
    (512, 512, 64),
    (768, 768, 64),
    (512, 1024, 64),
    (512, 512, 256),
    (768, 768, 256),
    (768, 32000, 256),
]

print("=" * 70)
print("Testing ANE evaluation")
print("=" * 70)

for C_in, C_out, S in test_cases:
    result = test_conv(C_in, C_out, S)
    print(f"C_in={C_in:4d}, C_out={C_out:4d}, S={S:3d} | {result}")

print("=" * 70)
