#!/usr/bin/env python3
"""
Debug ANE MIL compilation - minimal test cases
"""

import ctypes
import numpy as np
import struct
from pathlib import Path

ANE_BRIDGE_PATH = (
    Path(__file__).parent.parent / "target" / "debug" / "libane_bridge.dylib"
)
lib = ctypes.CDLL(str(ANE_BRIDGE_PATH))


def test_mil_variants():
    """Test different MIL syntax variants to find what works."""

    lib.ane_bridge_init()

    # Test 1: Absolute minimum (we know this works)
    mil1 = b"""program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 16, 1, 16]> input) {
        tensor<fp16, [1, 16, 1, 16]> output = identity(x=input);
    } -> (output);
}"""

    # Test 2: With name attribute
    mil2 = b"""program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 16, 1, 16]> input) {
        tensor<fp16, [1, 16, 1, 16]> output = identity(x=input)[name=string("out")];
    } -> (output);
}"""

    # Test 3: With fp32 cast (from ane-lora-training)
    mil3 = b"""program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}})]
{
    func main<ios18>(tensor<fp32, [1, 16, 1, 16]> x) {
        string to_fp16 = const()[val=string("fp16")];
        tensor<fp16, [1, 16, 1, 16]> x16 = cast(dtype=to_fp16, x=x);
        tensor<fp16, [1, 16, 1, 16]> y16 = identity(x=x16);
        string to_fp32 = const()[val=string("fp32")];
        tensor<fp32, [1, 16, 1, 16]> y = cast(dtype=to_fp32, x=y16);
    } -> (y);
}"""

    # Test 4: With const weight (no BLOBFILE)
    mil4 = b"""program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 16, 1, 16]> x) {
        tensor<fp16, [16, 16, 1, 1]> W = const()[val=fp16(0.1)];
        string pt = const()[val=string("valid")];
        tensor<int32, [2]> st = const()[val=tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[val=tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> dl = const()[val=tensor<int32, [2]>([1, 1])];
        int32 gr = const()[val=int32(1)];
        tensor<fp16, [1, 16, 1, 16]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x);
    } -> (y);
}"""

    tests = [
        ("Minimum MIL", mil1),
        ("With name attr", mil2),
        ("With fp32 cast", mil3),
        ("Conv with const weight", mil4),
    ]

    in_sz = (ctypes.c_size_t * 1)(16 * 16 * 2)
    out_sz = (ctypes.c_size_t * 1)(16 * 16 * 2)

    for name, mil in tests:
        print(f"\nTest: {name}")
        kernel = lib.ane_bridge_compile(
            (ctypes.c_uint8 * len(mil))(*mil), len(mil), None, 0, 1, in_sz, 1, out_sz
        )
        if kernel:
            print(f"  ✅ Compiled successfully!")
            lib.ane_bridge_free(kernel)
        else:
            print(f"  ❌ Failed to compile")


if __name__ == "__main__":
    test_mil_variants()
