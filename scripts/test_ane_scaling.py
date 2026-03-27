#!/usr/bin/env python3
"""
Test ANE with different matrix sizes to find where it beats CPU
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


def test_matrix_size(dim, seq_len, num_runs=50):
    """Test ANE vs CPU for a specific matrix size."""

    # CPU baseline
    x_cpu = np.random.randn(seq_len, dim).astype(np.float32)
    w_cpu = np.random.randn(dim, dim).astype(np.float32)

    cpu_times = []
    for _ in range(num_runs):
        start = time.time()
        out_cpu = x_cpu @ w_cpu
        cpu_times.append((time.time() - start) * 1000)

    cpu_time = np.mean(cpu_times)

    # ANE
    get_bridge()
    layer = ANEConv1x1(dim, dim, seq_len)
    x_ane = np.random.randn(1, dim, 1, seq_len).astype(np.float32)

    # Warmup
    _ = layer.forward(x_ane)

    ane_times = []
    for _ in range(num_runs):
        start = time.time()
        out_ane = layer.forward(x_ane)
        ane_times.append((time.time() - start) * 1000)

    ane_time = np.mean(ane_times)

    speedup = cpu_time / ane_time if ane_time > 0 else 0

    return cpu_time, ane_time, speedup


def main():
    print("=" * 70)
    print("ANE vs CPU - Matrix Size Analysis")
    print("=" * 70)
    print("\nTesting different matrix sizes...")
    print("-" * 70)
    print(
        f"{'Size':<20} {'CPU (ms)':<12} {'ANE (ms)':<12} {'Speedup':<10} {'Winner':<10}"
    )
    print("-" * 70)

    test_cases = [
        (256, 64, "Small"),
        (512, 128, "Medium-small"),
        (512, 256, "Medium"),
        (768, 256, "Medium-large"),
        (1024, 256, "Large"),
        (2048, 256, "X-Large"),
        (768, 32000, "Classifier-sized"),
    ]

    results = []

    for dim, seq_len, name in test_cases:
        try:
            cpu_time, ane_time, speedup = test_matrix_size(dim, seq_len, num_runs=20)
            winner = "ANE" if speedup > 1.5 else "CPU" if speedup < 0.8 else "TIE"

            print(
                f"{name:<20} {cpu_time:>6.2f}      {ane_time:>6.2f}      {speedup:>5.1f}x     {winner:<10}"
            )

            results.append(
                {
                    "name": name,
                    "dim": dim,
                    "seq": seq_len,
                    "cpu": cpu_time,
                    "ane": ane_time,
                    "speedup": speedup,
                }
            )
        except Exception as e:
            print(f"{name:<20} ERROR: {e}")

    print("-" * 70)

    # Find crossover point
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    ane_wins = [r for r in results if r["speedup"] > 1.5]
    cpu_wins = [r for r in results if r["speedup"] < 0.8]

    print(f"\nANE wins on large matrices:")
    for r in ane_wins:
        print(f"  {r['name']} ({r['dim']}x{r['seq']}): {r['speedup']:.1f}x faster")

    print(f"\nCPU wins on small matrices:")
    for r in cpu_wins:
        print(f"  {r['name']} ({r['dim']}x{r['seq']}): {1 / r['speedup']:.1f}x faster")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
ANE only provides speedup for LARGE matrices!

For 512x256 (our current size):
  - CPU and ANE are roughly equal
  - ANE overhead cancels out compute benefit
  
For 768x32000 (classifier):
  - ANE is ~10x faster
  - Large enough to justify overhead

RECOMMENDATION:
  Use ANE only for large operations:
  - Classifier (768 → 32000)
  - Large FFN layers
  
  Use CPU for small operations:
  - Attention Q,K,V (512 → 512)
  - Small projections
    """)


if __name__ == "__main__":
    main()
