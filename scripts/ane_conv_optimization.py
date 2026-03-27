#!/usr/bin/env python3
"""
ANE Conv Optimization Study

Explores different ways to use conv on ANE for better performance:
1. Fused operations (multiple convs in one kernel)
2. Grouped convolutions
3. Different data layouts
4. Weight packing strategies
"""

import ctypes
import struct
import numpy as np
from pathlib import Path
import time

# Load bridge
_bridge_lib = None


def get_bridge():
    global _bridge_lib
    if _bridge_lib is None:
        lib_path = Path("target/debug/libane_bridge.dylib").absolute()
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
        _bridge_lib.ane_bridge_init()
    return _bridge_lib


def build_weight_blob(weights_f32):
    """Build ANE weight blob."""
    flat = weights_f32.astype(np.float16).ravel()
    wsize = len(flat) * 2
    buf = bytearray(128 + wsize)

    buf[0] = 0x01
    buf[4] = 0x02
    buf[64] = 0xEF
    buf[65] = 0xBE
    buf[66] = 0xAD
    buf[67] = 0xDE
    buf[68] = 0x01
    struct.pack_into("<I", buf, 72, wsize)
    struct.pack_into("<I", buf, 80, 128)

    data_bytes = flat.tobytes()
    buf[128 : 128 + len(data_bytes)] = data_bytes

    return bytes(buf)


class ANEConvLayer:
    """Single conv layer."""

    def __init__(self, in_ch, out_ch, seq_len, name="conv"):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.seq_len = seq_len
        self.name = name

        self.weight = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.02
        self._compile()

    def _compile(self):
        lib = get_bridge()

        build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}})]'

        mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, {self.in_ch}, 1, {self.seq_len}]> x) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        tensor<fp16, [{self.out_ch},{self.in_ch},1,1]> W = const()[name=string("W"), val=tensor<fp16, [{self.out_ch},{self.in_ch},1,1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
        tensor<fp16, [1,{self.out_ch},1,{self.seq_len}]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("{self.name}")];
    }} -> (y);
}}"""

        mil = mil_text.encode("utf-8")
        weight_blob = build_weight_blob(
            self.weight.reshape(self.out_ch, self.in_ch).ravel()
        )

        in_sz = (ctypes.c_size_t * 1)(self.in_ch * self.seq_len * 2)
        out_sz = (ctypes.c_size_t * 1)(self.out_ch * self.seq_len * 2)
        wb = (ctypes.c_uint8 * len(weight_blob))(*weight_blob)

        start = time.time()
        kernel = lib.ane_bridge_compile(
            mil, len(mil), wb, len(weight_blob), 1, in_sz, 1, out_sz
        )
        compile_time = (time.time() - start) * 1000

        if not kernel:
            raise RuntimeError("Compilation failed")

        self._kernel = ctypes.c_void_p(kernel)
        self._lib = lib
        self._compile_time = compile_time

    def forward(self, x):
        x_fp16 = x.astype(np.float16)
        self._lib.ane_bridge_write_input(
            self._kernel, 0, x_fp16.ctypes.data, x_fp16.nbytes
        )

        if not self._lib.ane_bridge_eval(self._kernel):
            raise RuntimeError("Eval failed")

        out_fp16 = np.empty((1, self.out_ch, 1, self.seq_len), dtype=np.float16)
        self._lib.ane_bridge_read_output(
            self._kernel, 0, out_fp16.ctypes.data, out_fp16.nbytes
        )

        return out_fp16.astype(np.float32)


class ANEFusedQKV:
    """
    Fused QKV projection in a single kernel.

    Instead of 3 separate convs, do all 3 in one MIL program.
    This should reduce overhead.
    """

    def __init__(self, dim, num_heads, head_dim, seq_len):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.q_dim = num_heads * head_dim

        # Initialize weights
        self.Wq = np.random.randn(self.q_dim, dim, 1, 1).astype(np.float32) * 0.02
        self.Wk = np.random.randn(self.q_dim, dim, 1, 1).astype(np.float32) * 0.02
        self.Wv = np.random.randn(self.q_dim, dim, 1, 1).astype(np.float32) * 0.02

        self._compile()

    def _compile(self):
        lib = get_bridge()

        build_info = '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}})]'

        # Fused QKV: all 3 projections in one kernel
        mil_text = f"""program(1.3)
{build_info}
{{
    func main<ios18>(tensor<fp16, [1, {self.dim}, 1, {self.seq_len}]> x) {{
        string pt = const()[name=string("pt"), val=string("valid")];
        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
        int32 gr = const()[name=string("gr"), val=int32(1)];
        
        tensor<fp16, [{self.q_dim},{self.dim},1,1]> Wq = const()[name=string("Wq"), val=tensor<fp16, [{self.q_dim},{self.dim},1,1]>(BLOBFILE(path=string("@model_path/weights/wq.bin"), offset=uint64(64)))];
        tensor<fp16, [{self.q_dim},{self.dim},1,1]> Wk = const()[name=string("Wk"), val=tensor<fp16, [{self.q_dim},{self.dim},1,1]>(BLOBFILE(path=string("@model_path/weights/wk.bin"), offset=uint64(64)))];
        tensor<fp16, [{self.q_dim},{self.dim},1,1]> Wv = const()[name=string("Wv"), val=tensor<fp16, [{self.q_dim},{self.dim},1,1]>(BLOBFILE(path=string("@model_path/weights/wv.bin"), offset=uint64(64)))];
        
        tensor<fp16, [1,{self.q_dim},1,{self.seq_len}]> q = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string("cq")];
        tensor<fp16, [1,{self.q_dim},1,{self.seq_len}]> k = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=x)[name=string("ck")];
        tensor<fp16, [1,{self.q_dim},1,{self.seq_len}]> v = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=x)[name=string("cv")];
        
        tensor<fp16, [1,{self.q_dim * 3},1,{self.seq_len}]> qkv = concat(axis=1,x0=q,x1=k,x2=v)[name=string("qkv")];
    }} -> (qkv);
}}"""

        mil = mil_text.encode("utf-8")

        # Build weight blobs for all 3
        wq_blob = build_weight_blob(self.Wq.reshape(self.q_dim, self.dim).ravel())
        wk_blob = build_weight_blob(self.Wk.reshape(self.q_dim, self.dim).ravel())
        wv_blob = build_weight_blob(self.Wv.reshape(self.q_dim, self.dim).ravel())

        # Combine weight blobs
        total_weight_size = len(wq_blob) + len(wk_blob) + len(wv_blob)
        combined_blob = wq_blob + wk_blob + wv_blob

        # For multi-weight, we'd need to use ane_bridge_compile_multi_weights
        # For now, just use single blob with @model_path references
        # This is a placeholder - real implementation needs multi-weight support

        in_sz = (ctypes.c_size_t * 1)(self.dim * self.seq_len * 2)
        out_sz = (ctypes.c_size_t * 1)(self.q_dim * 3 * self.seq_len * 2)
        wb = (ctypes.c_uint8 * len(wq_blob))(*wq_blob)  # Just use Wq for now

        start = time.time()
        kernel = lib.ane_bridge_compile(
            mil, len(mil), wb, len(wq_blob), 1, in_sz, 1, out_sz
        )
        compile_time = (time.time() - start) * 1000

        if not kernel:
            print("Fused QKV compilation failed (expected - need multi-weight support)")
            self._kernel = None
        else:
            self._kernel = ctypes.c_void_p(kernel)
            self._lib = lib

        self._compile_time = compile_time


def benchmark_single_vs_fused():
    """Compare single convs vs fused operations."""
    print("=" * 70)
    print("ANE Conv Optimization Study")
    print("=" * 70)

    dim = 512
    seq_len = 256
    num_runs = 10

    print(f"\nBenchmark: dim={dim}, seq_len={seq_len}")
    print("-" * 70)

    # Test 1: Single conv 512->512
    print("\n1. Single Conv (512->512)")
    conv1 = ANEConvLayer(512, 512, seq_len, "conv1")
    print(f"   Compile time: {conv1._compile_time:.1f}ms")

    # Warmup
    x = np.random.randn(1, 512, 1, seq_len).astype(np.float32) * 0.1
    _ = conv1.forward(x)

    times = []
    for _ in range(num_runs):
        start = time.time()
        out = conv1.forward(x)
        times.append(time.time() - start)

    single_time = np.mean(times) * 1000
    print(f"   Forward time: {single_time:.2f}ms")

    # Test 2: Three separate convs (like QKV)
    print("\n2. Three Separate Convs (Q, K, V)")
    q_conv = ANEConvLayer(512, 512, seq_len, "q")
    k_conv = ANEConvLayer(512, 512, seq_len, "k")
    v_conv = ANEConvLayer(512, 512, seq_len, "v")

    total_compile = q_conv._compile_time + k_conv._compile_time + v_conv._compile_time
    print(f"   Total compile time: {total_compile:.1f}ms")

    # Warmup
    _ = q_conv.forward(x)
    _ = k_conv.forward(x)
    _ = v_conv.forward(x)

    times = []
    for _ in range(num_runs):
        start = time.time()
        q = q_conv.forward(x)
        k = k_conv.forward(x)
        v = v_conv.forward(x)
        times.append(time.time() - start)

    three_time = np.mean(times) * 1000
    print(f"   Total forward time: {three_time:.2f}ms")
    print(f"   Per-conv time: {three_time / 3:.2f}ms")

    # Test 3: Fused QKV (if it works)
    print("\n3. Fused QKV (single kernel)")
    try:
        fused = ANEFusedQKV(dim, 8, 64, seq_len)
        print(f"   Compile time: {fused._compile_time:.1f}ms")

        if fused._kernel:
            # Warmup
            _ = fused.forward(x)

            times = []
            for _ in range(num_runs):
                start = time.time()
                out = fused.forward(x)
                times.append(time.time() - start)

            fused_time = np.mean(times) * 1000
            print(f"   Forward time: {fused_time:.2f}ms")
            print(f"   Speedup vs 3 separate: {three_time / fused_time:.2f}x")
        else:
            print("   (Fused kernel not available - needs multi-weight support)")
    except Exception as e:
        print(f"   Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Single conv:        {single_time:.2f}ms")
    print(f"Three convs:        {three_time:.2f}ms ({three_time / 3:.2f}ms each)")
    print(f"Fused (theoretical): ~{single_time:.2f}ms (same as single conv!)")
    print(f"\nFusing 3 convs into 1 kernel could save:")
    print(f"  - {three_time - single_time:.2f}ms per forward pass")
    print(f"  - {total_compile - conv1._compile_time:.1f}ms in compilation")
    print(f"  - 3x less memory for kernel storage")


if __name__ == "__main__":
    benchmark_single_vs_fused()
