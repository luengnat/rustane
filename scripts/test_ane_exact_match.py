#!/usr/bin/env python3
"""
Test ANE with exact same approach as working ANE project.
"""

import ctypes
import struct
import numpy as np
from pathlib import Path

# Load bridge
lib = ctypes.CDLL(str(Path("target/debug/libane_bridge.dylib").absolute()))
lib.ane_bridge_init.restype = ctypes.c_int
lib.ane_bridge_compile.restype = ctypes.c_void_p
lib.ane_bridge_compile_multi_weights.restype = ctypes.c_void_p
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
    """Match ANE project exactly"""
    rows, cols = weights_f32.shape
    ws = rows * cols * 2  # fp16 = 2 bytes
    tot = 128 + ws
    buf = bytearray(tot)

    # Headers exactly as in stories_io.h
    buf[0] = 1
    buf[4] = 2
    buf[64] = 0xEF
    buf[65] = 0xBE
    buf[66] = 0xAD
    buf[67] = 0xDE
    buf[68] = 1
    struct.pack_into("<I", buf, 72, ws)
    struct.pack_into("<I", buf, 80, 128)

    # Convert to fp16
    fp16 = np.frombuffer(buf[128:], dtype=np.float16)
    fp16[:] = weights_f32.astype(np.float16).ravel()

    return bytes(buf)


# Test with exact ANE project dimensions: DIM=768, SEQ=256
DIM = 768
SEQ = 256

print(f"\nTesting with DIM={DIM}, SEQ={SEQ} (ANE project values)")
print("=" * 60)

# Build MIL matching the ANE project exactly
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
        tensor<fp16, [{DIM},{DIM},1,1]> Wq = const()[name=string("Wq"), val=tensor<fp16, [{DIM},{DIM},1,1]>(BLOBFILE(path=string("@model_path/weights/wq.bin"), offset=uint64(64)))];
        tensor<fp16, [1,{DIM},1,{SEQ}]> q = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string("q")];
    }} -> (q);
}}"""

mil = mil_text.encode("utf-8")

# Create weight matrix
Wq = np.random.randn(DIM, DIM).astype(np.float32) * 0.02
weight_blob = build_weight_blob(Wq)

print(f"Weight blob size: {len(weight_blob)} bytes ({len(weight_blob) / 1024:.1f} KB)")

# Use multi-weight compile (like ANE project)
weight_name = b"@model_path/weights/wq.bin\x00"
weight_data = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)
weight_len = len(weight_blob)

in_sz = (ctypes.c_size_t * 1)(DIM * SEQ * 2)
out_sz = (ctypes.c_size_t * 1)(DIM * SEQ * 2)

print("Compiling...")
kernel = lib.ane_bridge_compile_multi_weights(
    mil,
    len(mil),
    ctypes.pointer(ctypes.c_char_p(weight_name)),
    ctypes.pointer(ctypes.pointer(weight_data)),
    ctypes.pointer(ctypes.c_size_t(weight_len)),
    1,  # n_weights
    1,
    in_sz,
    1,
    out_sz,
)

if not kernel:
    print("❌ Compilation failed")
else:
    print("✅ Compilation succeeded")

    # Create input
    x = np.random.randn(1, DIM, 1, SEQ).astype(np.float32) * 0.1
    x_fp16 = x.astype(np.float16)

    print(f"Input shape: {x.shape}, bytes: {x_fp16.nbytes}")

    # Write input
    lib.ane_bridge_write_input(
        ctypes.c_void_p(kernel), 0, x_fp16.ctypes.data, x_fp16.nbytes
    )

    # Eval
    print("Running evaluation...")
    success = lib.ane_bridge_eval(ctypes.c_void_p(kernel))

    if success:
        print("✅ Evaluation succeeded")

        # Read output
        out_fp16 = np.empty((1, DIM, 1, SEQ), dtype=np.float16)
        lib.ane_bridge_read_output(
            ctypes.c_void_p(kernel), 0, out_fp16.ctypes.data, out_fp16.nbytes
        )
        out = out_fp16.astype(np.float32)

        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
        print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")

        if np.allclose(out, 0):
            print("⚠️  WARNING: Output is all zeros")
        else:
            print("✅ Output has non-zero values")

            # Compare with CPU reference
            x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, DIM)
            W_reshaped = Wq.T
            cpu_out = (
                (x_reshaped @ W_reshaped).reshape(1, SEQ, DIM, 1).transpose(0, 2, 3, 1)
            )

            diff = np.abs(out - cpu_out)
            print(f"\nCPU reference comparison:")
            print(f"  Max diff: {diff.max():.6f}")
            print(f"  Mean diff: {diff.mean():.6f}")

            if diff.max() < 0.01:
                print("  ✅ ANE matches CPU reference!")
            else:
                print("  ❌ ANE differs from CPU reference")
    else:
        print("❌ Evaluation failed")

    lib.ane_bridge_free(ctypes.c_void_p(kernel))

print("=" * 60)
