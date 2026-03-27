#!/usr/bin/env python3
"""
ANE-accelerated matmul/conv operations for training.
Uses the ANE bridge with proper @model_path format.

Fixed: Weight blob building now uses proper numpy tobytes() method.
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
import os

# Load the ANE bridge library
_bridge_lib = None


def get_bridge():
    """Get or initialize the ANE bridge library."""
    global _bridge_lib
    if _bridge_lib is None:
        lib_path = Path("target/debug/libane_bridge.dylib").absolute()
        if not lib_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(__file__).parent.parent / "target/debug/libane_bridge.dylib",
                Path.cwd() / "target/debug/libane_bridge.dylib",
                Path("/Users/nat/dev/rustane/target/debug/libane_bridge.dylib"),
            ]
            for alt in alt_paths:
                if alt.exists():
                    lib_path = alt
                    break

        _bridge_lib = ctypes.CDLL(str(lib_path))
        _bridge_lib.ane_bridge_init.restype = ctypes.c_int
        _bridge_lib.ane_bridge_compile.restype = ctypes.c_void_p
        _bridge_lib.ane_bridge_free.restype = None
        _bridge_lib.ane_bridge_eval.restype = ctypes.c_bool
        _bridge_lib.ane_bridge_write_input.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        _bridge_lib.ane_bridge_read_output.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]

        ret = _bridge_lib.ane_bridge_init()
        if ret != 0:
            raise RuntimeError(f"ANE bridge initialization failed: {ret}")

    return _bridge_lib


def build_weight_blob(weights_f32):
    """
    Build ANE weight blob with proper 128-byte header.

    CRITICAL: Use numpy.tobytes() for proper float16 conversion.
    Do NOT use frombuffer assignment which can cause issues.

    Format:
    - 64 bytes: global header (version=1, type=2)
    - 64 bytes: chunk header (magic=0xDEADBEEF, chunk_count=1)
    - Remaining: fp16 weight data
    """
    # Flatten and convert to float16
    flat = weights_f32.astype(np.float16).ravel()
    wsize = len(flat) * 2  # fp16 = 2 bytes per element
    total = 128 + wsize

    buf = bytearray(total)

    # Global header
    buf[0] = 0x01  # version
    buf[4] = 0x02  # type

    # Chunk header at offset 64
    buf[64] = 0xEF
    buf[65] = 0xBE
    buf[66] = 0xAD
    buf[67] = 0xDE  # magic
    buf[68] = 0x01  # chunk count
    struct.pack_into("<I", buf, 72, wsize)  # data size
    struct.pack_into("<I", buf, 80, 128)  # data offset

    # Copy float16 bytes directly
    data_bytes = flat.tobytes()
    buf[128 : 128 + len(data_bytes)] = data_bytes

    return bytes(buf)


def compile_conv_kernel(in_channels, out_channels, seq_len, weight_blob):
    """
    Compile a 1x1 conv kernel for ANE.

    Input shape: (1, in_channels, 1, seq_len)
    Weight shape: (out_channels, in_channels, 1, 1)
    Output shape: (1, out_channels, 1, seq_len)
    """
    lib = get_bridge()

    build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]'

    mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, {in_channels}, 1, {seq_len}]> x) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [{out_channels}, {in_channels}, 1, 1]> W = const()[name=string("W"), val=tensor<fp16, [{out_channels}, {in_channels}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        tensor<fp16, [1, {out_channels}, 1, {seq_len}]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("conv")];
    }} -> (y);
}}"""

    mil = mil_text.encode("utf-8")

    in_sz = (ctypes.c_size_t * 1)(in_channels * seq_len * 2)  # fp16
    out_sz = (ctypes.c_size_t * 1)(out_channels * seq_len * 2)  # fp16

    wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

    kernel = lib.ane_bridge_compile(
        mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
    )

    if not kernel:
        return None

    return {
        "kernel": ctypes.c_void_p(kernel),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "seq_len": seq_len,
        "lib": lib,
    }


def conv1x1_forward(kernel_handle, x):
    """
    Run 1x1 conv forward pass on ANE.

    x: numpy array of shape (1, in_channels, 1, seq_len) in float32
    Returns: numpy array of shape (1, out_channels, 1, seq_len) in float32
    """
    if kernel_handle is None:
        raise ValueError("Kernel handle is None")

    lib = kernel_handle["lib"]
    kernel = kernel_handle["kernel"]
    in_channels = kernel_handle["in_channels"]
    out_channels = kernel_handle["out_channels"]
    seq_len = kernel_handle["seq_len"]

    # Convert input to fp16
    x_fp16 = x.astype(np.float16)
    in_bytes = x_fp16.nbytes
    out_bytes = out_channels * seq_len * 2  # fp16

    # Write input
    lib.ane_bridge_write_input(kernel, 0, x_fp16.ctypes.data, in_bytes)

    # Evaluate
    success = lib.ane_bridge_eval(kernel)
    if not success:
        raise RuntimeError("ANE evaluation failed")

    # Read output
    out_fp16 = np.empty((1, out_channels, 1, seq_len), dtype=np.float16)
    lib.ane_bridge_read_output(kernel, 0, out_fp16.ctypes.data, out_bytes)

    return out_fp16.astype(np.float32)


def free_kernel(kernel_handle):
    """Free an ANE kernel."""
    if kernel_handle and kernel_handle["kernel"]:
        kernel_handle["lib"].ane_bridge_free(kernel_handle["kernel"])


class ANEConv1x1:
    """
    1x1 convolution layer using ANE acceleration.
    Compatible with the Conv1x1 interface in train_golf_ane_optimized.py
    """

    def __init__(self, in_channels, out_channels, seq_len=256):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len

        # Initialize weights: (out_channels, in_channels, 1, 1)
        self.weight = (
            np.random.randn(out_channels, in_channels, 1, 1).astype(np.float32) * 0.02
        )
        self._kernel = None
        self._recompile()

    def _recompile(self):
        """Recompile the kernel with current weights."""
        # Free old kernel if exists
        if self._kernel:
            free_kernel(self._kernel)

        # Build weight blob using fixed method
        weight_flat = self.weight.reshape(self.out_channels, self.in_channels).ravel()
        blob = build_weight_blob(weight_flat)

        # Compile kernel
        self._kernel = compile_conv_kernel(
            self.in_channels, self.out_channels, self.seq_len, blob
        )

        if self._kernel is None:
            raise RuntimeError("Failed to compile ANE conv kernel")

    def forward(self, x):
        """
        Forward pass on ANE.

        Input: (B, C_in, 1, S) - currently only supports B=1
        Output: (B, C_out, 1, S)
        """
        B, C_in, H, S = x.shape

        if B != 1:
            # Process batch items sequentially
            outputs = []
            for i in range(B):
                out = self.forward(x[i : i + 1])
                outputs.append(out)
            return np.concatenate(outputs, axis=0)

        if C_in != self.in_channels or S != self.seq_len:
            # Recompile with new dimensions
            self.in_channels = C_in
            self.seq_len = S
            self._recompile()

        return conv1x1_forward(self._kernel, x)

    def update_weights(self, new_weight):
        """Update weights and recompile kernel."""
        self.weight = new_weight.reshape(self.out_channels, self.in_channels, 1, 1)
        self._recompile()

    def __del__(self):
        """Cleanup."""
        if hasattr(self, "_kernel") and self._kernel:
            free_kernel(self._kernel)


# Simple test
if __name__ == "__main__":
    print("Testing ANE convolution...")

    # Create layer with ANE project dimensions
    layer = ANEConv1x1(768, 768, seq_len=256)

    # Create input
    x = np.random.randn(1, 768, 1, 256).astype(np.float32) * 0.1

    # Forward pass
    print("Running forward pass...")
    out = layer.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"Output mean: {out.mean():.4f}, std: {out.std():.4f}")

    if np.allclose(out, 0):
        print("❌ FAILED: Output is all zeros")
    else:
        print("✅ ANE convolution test passed!")
