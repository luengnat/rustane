#!/usr/bin/env python3
"""
Simple ANE test - without BLOBFILE, using const weights
"""

import ctypes
import numpy as np
import struct
from pathlib import Path
import time

ANE_BRIDGE_PATH = (
    Path(__file__).parent.parent / "target" / "debug" / "libane_bridge.dylib"
)


def load_ane_bridge():
    """Load the ANE bridge library."""
    lib = ctypes.CDLL(str(ANE_BRIDGE_PATH))

    lib.ane_bridge_init.restype = ctypes.c_int
    lib.ane_bridge_compile.restype = ctypes.c_void_p
    lib.ane_bridge_eval.restype = ctypes.c_int
    lib.ane_bridge_free.restype = None

    return lib


def test_simple_identity():
    """Test simplest possible MIL program - just identity."""
    lib = load_ane_bridge()

    print("Initializing ANE...")
    result = lib.ane_bridge_init()
    if result != 0:
        raise RuntimeError(f"ANE init failed: {result}")
    print("✅ ANE initialized")

    # Simple MIL that we know works
    mil_text = """program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}})]
{
    func main<ios18>(tensor<fp16, [1, 512, 1, 1024]> input) {
        tensor<fp16, [1, 512, 1, 1024]> output = identity(x=input)[name=string("output")];
    } -> (output);
}"""

    print("\nTesting simple identity MIL...")
    mil_bytes = mil_text.encode("utf-8")

    in_sz = (ctypes.c_size_t * 1)(512 * 1024 * 2)  # fp16
    out_sz = (ctypes.c_size_t * 1)(512 * 1024 * 2)

    kernel = lib.ane_bridge_compile(
        (ctypes.c_uint8 * len(mil_bytes))(*mil_bytes),
        len(mil_bytes),
        None,
        0,  # No weights
        1,
        in_sz,
        1,
        out_sz,
    )

    if not kernel:
        print("❌ Simple identity failed too!")
        return False

    print("✅ Simple identity MIL compiled!")

    # Test execution
    input_data = np.zeros((1, 512, 1, 1024), dtype=np.float16)
    input_bytes = input_data.tobytes()

    # Write input (need to define the function)
    # ...

    lib.ane_bridge_free(kernel)
    return True


if __name__ == "__main__":
    test_simple_identity()
