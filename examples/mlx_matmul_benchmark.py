#!/usr/bin/env python3
"""MLX Matmul Benchmark - Compare with Rustane ANE performance"""

import mlx.core as mx
import mlx.nn as nn
import time

# Match Rustane benchmark configurations
configs = [
    (1, 256, 512),     # 1×256→512
    (1, 512, 1024),    # 1×512→1024
    (1, 1024, 1024),   # 1×1024×1024
    (4, 256, 512),     # 4×256→512
    (2, 512, 1024),    # 2×512×1024
]

ITERATIONS = 100
WARMUP = 10

def benchmark_mlx_linear(batch, in_feat, out_feat):
    """Benchmark MLX linear layer (matmul)"""
    print(f"\n{'━'*60}")
    print(f"MLX Linear ({batch} × {in_feat} → {out_feat})")
    print(f"{'━'*60}")

    # Create input and weight
    input_data = mx.random.uniform(shape=(batch, in_feat))
    weight = mx.random.uniform(shape=(in_feat, out_feat))

    # Warmup
    for _ in range(WARMUP):
        output = mx.matmul(input_data, weight)
        mx.eval(output)

    # Benchmark
    start = time.time()
    for _ in range(ITERATIONS):
        output = mx.matmul(input_data, weight)
        mx.eval(output)
    end = time.time()

    total_ms = (end - start) * 1000
    avg_ms = total_ms / ITERATIONS
    ops_per_sec = ITERATIONS / (end - start)

    print(f"  Total time ({ITERATIONS} iterations): {total_ms:.2f}ms")
    print(f"  Average time: {avg_ms:.3f}ms")
    print(f"  Throughput: {ops_per_sec:.1f} ops/sec")

    return total_ms, avg_ms, ops_per_sec

def main():
    print("🍎 MLX Matmul Benchmark")
    print("=" * 60)
    print(f"Platform: MLX on Apple Silicon")
    print(f"Iterations: {ITERATIONS}")
    print(f"Warmup: {WARMUP}")
    print()

    results = []

    for batch, in_feat, out_feat in configs:
        print(f"\n{'═'*60}")
        print(f"Configuration: Batch={batch}, Input={in_feat}, Output={out_feat}")
        print(f"Operations: {batch * in_feat * out_feat} mul-add operations per sample")

        total, avg, ops = benchmark_mlx_linear(batch, in_feat, out_feat)
        results.append((batch, in_feat, out_feat, avg, ops))

    # Print summary
    print(f"\n\n{'━'*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'━'*60}")
    print(f"{'Config':<12} {'Avg (ms)':>12} {'Throughput':>15}")
    print("-" * 60)

    for batch, in_feat, out_feat, avg, ops in results:
        config = f"{batch}×{in_feat}→{out_feat}"
        print(f"{config:<12} {avg:>12.3f} {ops:>15.1f}")

    print(f"\n{'━'*60}")
    print("COMPARISON WITH RUSTANE ANE")
    print(f"{'━'*60}")

    # Rustane ANE results from our benchmark
    rustane_results = {
        (1, 256, 512): (0.073, 13741.1),     # 44.83x vs CPU
        (1, 512, 1024): (0.126, 7918.0),    # 103.33x vs CPU
        (1, 1024, 1024): (0.104, 9611.0),   # 255.78x vs CPU
        (4, 256, 512): (0.085, 11728.7),    # 152.75x vs CPU
        (2, 512, 1024): (0.164, 6099.5),    # 159.47x vs CPU
    }

    print(f"\n{'Config':<12} {'MLX (ms)':>10} {'Rustane (ms)':>12} {'Ratio':>10}")
    print("-" * 60)

    for batch, in_feat, out_feat, mlx_avg, mlx_ops in results:
        key = (batch, in_feat, out_feat)
        if key in rustane_results:
            ane_avg, ane_ops = rustane_results[key]
            ratio = mlx_avg / ane_avg
            faster = "MLX" if ratio > 1 else "Rustane"
            print(f"{batch}×{in_feat}→{out_feat:<4} {mlx_avg:>10.3f} {ane_avg:>12.3f} {ratio:>9.2f}x ({faster})")

    print(f"\n✅ MLX benchmark complete!")

if __name__ == "__main__":
    main()
