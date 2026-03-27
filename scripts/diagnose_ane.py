#!/usr/bin/env python3
"""
ANE Diagnostic - Check if ANE is actually being used
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


def diagnose_ane():
    """Check if ANE is working properly."""
    print("=" * 70)
    print("ANE DIAGNOSTIC")
    print("=" * 70)

    # 1. Check bridge initialization
    print("\n1. Testing bridge initialization...")
    try:
        ret = get_bridge()
        print(f"   Bridge init returned: {ret}")
        print("   ✅ Bridge initialized")
    except Exception as e:
        print(f"   ❌ Bridge failed: {e}")
        return False

    # 2. Create ANE layer
    print("\n2. Creating ANE layer...")
    try:
        layer = ANEConv1x1(512, 512, 256)
        print(f"   Layer created: {layer}")
        print("   ✅ Layer created")
    except Exception as e:
        print(f"   ❌ Layer creation failed: {e}")
        return False

    # 3. Run forward pass with timing
    print("\n3. Running forward pass (100 iterations)...")
    x = np.random.randn(1, 512, 1, 256).astype(np.float32) * 0.1

    # Warmup
    for _ in range(5):
        _ = layer.forward(x)

    # Time it
    times = []
    for i in range(100):
        start = time.time()
        out = layer.forward(x)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    avg_time = np.mean(times)
    min_time = np.min(times)

    print(f"   Average time: {avg_time:.2f}ms")
    print(f"   Min time: {min_time:.2f}ms")
    print(f"   Output shape: {out.shape}")
    print(f"   Output nonzero: {np.count_nonzero(out)}/{out.size}")

    # 4. Check if ANE is actually being used
    print("\n4. ANE Usage Analysis:")
    print(f"   Expected ANE time: ~1-5ms")
    print(f"   Expected CPU time: ~20-50ms")

    if avg_time < 10:
        print(f"   ✅ ANE appears to be working (fast)")
    elif avg_time < 50:
        print(f"   ⚠️  Might be CPU fallback (moderate speed)")
    else:
        print(f"   ❌ Likely CPU only (slow)")

    # 5. Check Activity Monitor (manual check)
    print("\n5. MANUAL CHECK REQUIRED:")
    print("   Open Activity Monitor and look for:")
    print("   - 'ane' or 'ANE' processes")
    print("   - High CPU usage in Python vs ANE")
    print("   - ANE hardware utilization")

    # 6. Bridge diagnostics
    print("\n6. Bridge diagnostics:")
    print(f"   Bridge library: target/debug/libane_bridge.dylib")
    print(f"   ANE functions: ane_bridge_init, ane_bridge_compile, ane_bridge_eval")

    # Try to detect if compile is working
    print("\n7. Testing recompile timing...")
    new_weight = layer.weight * 0.99

    start = time.time()
    layer.update_weights(new_weight)
    compile_time = (time.time() - start) * 1000

    print(f"   Recompile time: {compile_time:.0f}ms")
    if compile_time > 50:
        print(f"   ✅ Recompile taking time (ANE compiling)")
    else:
        print(f"   ⚠️  Recompile fast (might be cached or CPU)")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    return avg_time < 10  # Return True if ANE seems to be working


def compare_with_cpu():
    """Compare ANE vs CPU directly."""
    print("\n" + "=" * 70)
    print("ANE vs CPU COMPARISON")
    print("=" * 70)

    # CPU matmul
    print("\nCPU (numpy matmul):")
    x = np.random.randn(256, 512).astype(np.float32)
    w = np.random.randn(512, 512).astype(np.float32)

    times = []
    for _ in range(100):
        start = time.time()
        out = x @ w
        times.append((time.time() - start) * 1000)

    cpu_time = np.mean(times)
    print(f"   Average: {cpu_time:.2f}ms")

    # ANE conv (via our layer)
    print("\nANE (our wrapper):")
    layer = ANEConv1x1(512, 512, 256)
    x_ane = np.random.randn(1, 512, 1, 256).astype(np.float32)

    times = []
    for _ in range(100):
        start = time.time()
        out = layer.forward(x_ane)
        times.append((time.time() - start) * 1000)

    ane_time = np.mean(times)
    print(f"   Average: {ane_time:.2f}ms")

    print(f"\nComparison:")
    print(f"   CPU: {cpu_time:.2f}ms")
    print(f"   ANE: {ane_time:.2f}ms")
    if ane_time < cpu_time:
        print(f"   ✅ ANE is {cpu_time / ane_time:.1f}x faster")
    else:
        print(f"   ❌ ANE is {ane_time / cpu_time:.1f}x slower (not working?)")


if __name__ == "__main__":
    working = diagnose_ane()
    compare_with_cpu()

    print("\n" + "=" * 70)
    if working:
        print("✅ ANE appears to be functioning correctly")
    else:
        print("❌ ANE may not be working - check Activity Monitor")
    print("=" * 70)
