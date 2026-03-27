#!/usr/bin/env python3
"""
Test ANE with exact dimensions from working ANE project.
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


def test_conv(DIM, SEQ, VOCAB):
    """Test classifier-style conv: [VOCAB, DIM] @ [DIM, SEQ]"""
    print(f"\nTesting DIM={DIM}, SEQ={SEQ}, VOCAB={VOCAB}")
    print("-" * 60)

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
        tensor<fp16, [{VOCAB},{DIM},1,1]> We = const()[name=string("We"), val=tensor<fp16, [{VOCAB},{DIM},1,1]>(BLOBFILE(path=string("@model_path/weights/embed.bin"), offset=uint64(64)))];
        tensor<fp16, [1,{VOCAB},1,{SEQ}]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=We,x=x)[name=string("cls")];
    }} -> (out);
}}"""

    mil = mil_text.encode("utf-8")

    # Create weights and input
    embed = np.random.randn(VOCAB, DIM).astype(np.float32) * 0.02
    x = np.random.randn(1, DIM, 1, SEQ).astype(np.float32) * 0.1

    weight_blob = build_weight_blob(embed.ravel())

    in_sz = (ctypes.c_size_t * 1)(DIM * SEQ * 2)
    out_sz = (ctypes.c_size_t * 1)(VOCAB * SEQ * 2)
    wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

    kernel = lib.ane_bridge_compile(
        mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
    )

    if not kernel:
        print("❌ Compilation failed")
        return False

    print("✅ Compilation succeeded")

    # Run
    x_fp16 = x.astype(np.float16)
    lib.ane_bridge_write_input(
        ctypes.c_void_p(kernel), 0, x_fp16.ctypes.data, x_fp16.nbytes
    )

    success = lib.ane_bridge_eval(ctypes.c_void_p(kernel))
    if success:
        print("✅ Evaluation succeeded")

        out_fp16 = np.empty((1, VOCAB, 1, SEQ), dtype=np.float16)
        lib.ane_bridge_read_output(
            ctypes.c_void_p(kernel), 0, out_fp16.ctypes.data, out_fp16.nbytes
        )
        out = out_fp16.astype(np.float32)

        print(f"  Output shape: {out.shape}")
        print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"  Output mean: {out.mean():.4f}")
        print(f"  Output std: {out.std():.4f}")
        print(
            f"  Non-zero: {np.count_nonzero(out)}/{out.size} ({100 * np.count_nonzero(out) / out.size:.1f}%)"
        )

        if np.allclose(out, 0):
            print("  ⚠️  WARNING: Output is all zeros!")
        else:
            print("  ✅ Output has non-zero values")
    else:
        print("❌ Evaluation failed")

    lib.ane_bridge_free(ctypes.c_void_p(kernel))
    return success


# Test with different dimensions
print("=" * 60)
print("Testing ANE with different dimensions")
print("=" * 60)

# Working dimensions from ANE project
test_conv(768, 256, 100)  # DIM=768, SEQ=256 (Stories110M)
test_conv(768, 256, 32000)  # Full classifier
test_conv(512, 256, 100)  # Smaller DIM
test_conv(512, 64, 100)  # SEQ=64 (what we used before)
test_conv(256, 64, 100)  # Even smaller

print("\n" + "=" * 60)
print("Testing complete")
print("=" * 60)
