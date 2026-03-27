#!/usr/bin/env python3
"""
ANE-accelerated backward pass operations.

Implements gradient computation for:
- Linear/Conv1x1 layers: dW, dx
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
from ane_ops import get_bridge, build_weight_blob, free_kernel


def compile_backward_kernel(in_channels, out_channels, seq_len):
    """
    Compile backward kernel for 1x1 conv.

    Forward: y = W @ x
      - x: [1, in_channels, 1, seq_len]
      - W: [out_channels, in_channels, 1, 1]
      - y: [1, out_channels, 1, seq_len]

    Backward for dx: dx = W^T @ dy
      - dy: [1, out_channels, 1, seq_len]
      - W^T: [in_channels, out_channels, 1, 1]
      - dx: [1, in_channels, 1, seq_len]
    """
    lib = get_bridge()

    build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]'

    # For dx = W^T @ dy
    # We bake W^T as weight (shape: [in_channels, out_channels, 1, 1])
    mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, {out_channels}, 1, {seq_len}]> dy) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [{in_channels}, {out_channels}, 1, 1]> Wt = const()[name=string("Wt"), val=tensor<fp16, [{in_channels}, {out_channels}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight_t.bin"), offset=uint64(64)))];
        tensor<fp16, [1, {in_channels}, 1, {seq_len}]> dx = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wt,x=dy)[name=string("dx")];
    }} -> (dx);
}}"""

    mil = mil_text.encode("utf-8")

    # Placeholder - we'll set actual weights later
    dummy_weight = np.zeros((in_channels, out_channels), dtype=np.float32)
    weight_blob = build_weight_blob(dummy_weight)

    in_sz = (ctypes.c_size_t * 1)(out_channels * seq_len * 2)
    out_sz = (ctypes.c_size_t * 1)(in_channels * seq_len * 2)

    wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

    kernel = lib.ane_bridge_compile(
        mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
    )

    if not kernel:
        return None

    return {
        "kernel": ctypes.c_void_p(kernel),
        "in_channels": in_channels,  # Original in_channels (output of backward)
        "out_channels": out_channels,  # Original out_channels (input to backward)
        "seq_len": seq_len,
        "lib": lib,
        "mil": mil,
        "in_sz": in_sz[0],
        "out_sz": out_sz[0],
    }


def update_backward_weights(kernel_handle, weight_T):
    """
    Update backward kernel with transposed weights.

    weight_T: [in_channels, out_channels] transposed from original W
    """
    lib = kernel_handle["lib"]

    # Free old kernel
    lib.ane_bridge_free(kernel_handle["kernel"])

    # Build new weight blob
    weight_blob = build_weight_blob(weight_T.ravel())
    wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

    # Recompile
    mil = kernel_handle["mil"]
    in_sz = (ctypes.c_size_t * 1)(kernel_handle["in_sz"])
    out_sz = (ctypes.c_size_t * 1)(kernel_handle["out_sz"])

    kernel = lib.ane_bridge_compile(
        mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
    )

    if kernel:
        kernel_handle["kernel"] = ctypes.c_void_p(kernel)
        return True
    return False


def backward_dx(kernel_handle, dy):
    """
    Compute dx = W^T @ dy

    dy: [1, out_channels, 1, seq_len]
    Returns: [1, in_channels, 1, seq_len]
    """
    if kernel_handle is None:
        raise ValueError("Kernel handle is None")

    lib = kernel_handle["lib"]
    kernel = kernel_handle["kernel"]
    in_channels = kernel_handle["in_channels"]
    out_channels = kernel_handle["out_channels"]
    seq_len = kernel_handle["seq_len"]

    # Convert input to fp16
    dy_fp16 = dy.astype(np.float16)
    in_bytes = dy_fp16.nbytes
    out_bytes = in_channels * seq_len * 2

    # Write input
    lib.ane_bridge_write_input(kernel, 0, dy_fp16.ctypes.data, in_bytes)

    # Evaluate
    success = lib.ane_bridge_eval(kernel)
    if not success:
        raise RuntimeError("ANE backward evaluation failed")

    # Read output
    dx_fp16 = np.empty((1, in_channels, 1, seq_len), dtype=np.float16)
    lib.ane_bridge_read_output(kernel, 0, dx_fp16.ctypes.data, out_bytes)

    return dx_fp16.astype(np.float32)


class ANELinearBackward:
    """
    Backward pass for ANE linear layer.
    Computes gradients for weight updates and backpropagation.
    """

    def __init__(self, in_channels, out_channels, seq_len=256):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len

        # Kernel for computing dx (gradient w.r.t. input)
        self._dx_kernel = None

    def initialize(self, weight):
        """
        Initialize backward kernels with current weights.

        weight: [out_channels, in_channels, 1, 1]
        """
        # For dx = W^T @ dy, we need W^T
        W_T = weight.reshape(self.out_channels, self.in_channels).T

        # Compile kernel
        dummy_blob = build_weight_blob(W_T.ravel())

        lib = get_bridge()
        build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]'

        mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, {self.out_channels}, 1, {self.seq_len}]> dy) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [{self.in_channels}, {self.out_channels}, 1, 1]> Wt = const()[name=string("Wt"), val=tensor<fp16, [{self.in_channels}, {self.out_channels}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        tensor<fp16, [1, {self.in_channels}, 1, {self.seq_len}]> dx = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wt,x=dy)[name=string("dx")];
    }} -> (dx);
}}"""

        mil = mil_text.encode("utf-8")
        in_sz = (ctypes.c_size_t * 1)(self.out_channels * self.seq_len * 2)
        out_sz = (ctypes.c_size_t * 1)(self.in_channels * self.seq_len * 2)
        wb = (ctypes.c_uint8 * len(dummy_blob))(*dummy_blob)

        kernel = lib.ane_bridge_compile(
            mil, len(mil), wb, len(dummy_blob), 1, in_sz, 1, out_sz
        )

        if not kernel:
            raise RuntimeError("Failed to compile backward kernel")

        self._dx_kernel = {"kernel": ctypes.c_void_p(kernel), "lib": lib}

    def compute_dx(self, dy):
        """
        Compute gradient w.r.t. input.

        dy: [B, out_channels, 1, seq_len]
        Returns: [B, in_channels, 1, seq_len]
        """
        B = dy.shape[0]

        if B != 1:
            # Process batch items sequentially
            outputs = []
            for i in range(B):
                out = self.compute_dx(dy[i : i + 1])
                outputs.append(out)
            return np.concatenate(outputs, axis=0)

        lib = self._dx_kernel["lib"]
        kernel = self._dx_kernel["kernel"]

        # Convert to fp16
        dy_fp16 = dy.astype(np.float16)
        in_bytes = dy_fp16.nbytes
        out_bytes = self.in_channels * self.seq_len * 2

        # Write input
        lib.ane_bridge_write_input(kernel, 0, dy_fp16.ctypes.data, in_bytes)

        # Evaluate
        success = lib.ane_bridge_eval(kernel)
        if not success:
            raise RuntimeError("ANE backward eval failed")

        # Read output
        dx_fp16 = np.empty((1, self.in_channels, 1, self.seq_len), dtype=np.float16)
        lib.ane_bridge_read_output(kernel, 0, dx_fp16.ctypes.data, out_bytes)

        return dx_fp16.astype(np.float32)

    def compute_dW(self, x, dy):
        """
        Compute gradient w.r.t. weights (CPU for now).

        x: [B, in_channels, 1, seq_len]
        dy: [B, out_channels, 1, seq_len]
        Returns: [out_channels, in_channels, 1, 1]
        """
        B = x.shape[0]

        # Reshape for matmul
        # x: [B, C_in, 1, S] -> [B*S, C_in]
        x_reshaped = x.transpose(0, 3, 1, 2).reshape(-1, self.in_channels)
        # dy: [B, C_out, 1, S] -> [B*S, C_out]
        dy_reshaped = dy.transpose(0, 3, 1, 2).reshape(-1, self.out_channels)

        # dW = dy^T @ x / (B*S)
        dW = (dy_reshaped.T @ x_reshaped) / (B * self.seq_len)

        # Reshape to conv format
        dW = dW.reshape(self.out_channels, self.in_channels, 1, 1)

        return dW

    def __del__(self):
        if hasattr(self, "_dx_kernel") and self._dx_kernel:
            self._dx_kernel["lib"].ane_bridge_free(self._dx_kernel["kernel"])


# Test
if __name__ == "__main__":
    print("Testing ANE backward pass...")
    print("=" * 60)

    # Test dimensions
    in_ch, out_ch, seq = 512, 512, 256

    print(f"Dimensions: in={in_ch}, out={out_ch}, seq={seq}")

    # Create weight
    W = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.02

    # Initialize backward
    backward = ANELinearBackward(in_ch, out_ch, seq)
    print("Initializing backward kernel...")
    backward.initialize(W)
    print("✅ Backward kernel compiled")

    # Test inputs
    dy = np.random.randn(1, out_ch, 1, seq).astype(np.float32) * 0.1
    x = np.random.randn(1, in_ch, 1, seq).astype(np.float32) * 0.1

    # Compute dx
    print("\nComputing dx...")
    dx = backward.compute_dx(dy)
    print(f"dx shape: {dx.shape}")
    print(f"dx range: [{dx.min():.4f}, {dx.max():.4f}]")

    # Verify with CPU reference
    W_mat = W.reshape(out_ch, in_ch).T
    dy_mat = dy.transpose(0, 3, 1, 2).reshape(-1, out_ch)
    dx_cpu = (dy_mat @ W_mat.T).reshape(1, seq, in_ch, 1).transpose(0, 2, 3, 1)

    diff = np.abs(dx - dx_cpu)
    print(f"CPU diff: max={diff.max():.6f}, mean={diff.mean():.6f}")

    if diff.max() < 0.01:
        print("✅ ANE backward matches CPU!")
    else:
        print("❌ ANE backward differs from CPU")

    # Compute dW
    print("\nComputing dW (CPU)...")
    dW = backward.compute_dW(x, dy)
    print(f"dW shape: {dW.shape}")
    print(f"dW range: [{dW.min():.6f}, {dW.max():.6f}]")

    print("\n" + "=" * 60)
    print("ANE backward pass test complete!")
