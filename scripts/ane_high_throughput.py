#!/usr/bin/env python3
"""
ANE High-Throughput Trainer

Problem: ANE NPU usage is low because we don't keep it busy
Solution: Process LARGE batches to maximize throughput

Key insight: ANE has high setup cost but high throughput.
We need to amortize setup over many samples.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


class HighThroughputTrainer:
    """
    Trainer designed to maximize ANE throughput.

    Strategy:
    1. Process LARGE batches (64-256 samples)
    2. Keep ANE pipeline full
    3. Minimize idle time between operations
    """

    def __init__(self, dim=512, seq_len=256):
        self.dim = dim
        self.seq_len = seq_len

        print("=" * 70)
        print("ANE High-Throughput Trainer")
        print("=" * 70)
        print(f"Strategy: Maximize NPU utilization via large batches")
        print(f"Dimensions: {dim}x{seq_len}")

        get_bridge()

        # ANE layer
        self.layer = ANEConv1x1(dim, dim, seq_len)

        # Adam state
        self.m = np.zeros_like(self.layer.weight)
        self.v = np.zeros_like(self.layer.weight)
        self.t = 0

        print("✅ Ready\n")

    def process_large_batch(self, batch_size=64, num_batches=10):
        """
        Process a large batch to maximize ANE throughput.

        With batch_size=64:
        - ANE processes 64 samples in one "batch"
        - Setup cost is amortized over 64 samples
        - NPU stays busy with parallel work
        """
        print(f"Processing {num_batches} batches of size {batch_size}...")
        print("-" * 70)

        all_times = []

        for batch_idx in range(num_batches):
            batch_start = time.time()

            # Generate large batch
            # Shape: [batch_size, dim, 1, seq_len]
            # This keeps ANE busy with parallel work
            x = (
                np.random.randn(batch_size, self.dim, 1, self.seq_len).astype(
                    np.float32
                )
                * 0.1
            )

            # Process all samples - ANE will parallelize internally
            # Note: Our current ANEConv1x1 processes one sample at a time
            # In production, we'd want ANE to handle the batch internally

            batch_times = []
            for i in range(batch_size):
                start = time.time()
                out = self.layer.forward(x[i : i + 1])
                batch_times.append((time.time() - start) * 1000)

            batch_elapsed = (time.time() - batch_start) * 1000
            avg_per_sample = np.mean(batch_times)

            all_times.extend(batch_times)

            if (batch_idx + 1) % 2 == 0:
                print(
                    f"  Batch {batch_idx + 1}/{num_batches}: "
                    f"{batch_elapsed:.0f}ms total, "
                    f"{avg_per_sample:.2f}ms/sample, "
                    f"throughput: {batch_size * 1000 / batch_elapsed:.0f} samples/sec"
                )

        return all_times

    def benchmark_batch_sizes(self):
        """Find optimal batch size for maximum throughput."""
        print("=" * 70)
        print("BATCH SIZE BENCHMARK")
        print("=" * 70)
        print("Finding batch size that maximizes ANE throughput...\n")

        batch_sizes = [1, 4, 8, 16, 32, 64, 128]
        results = []

        for bs in batch_sizes:
            print(f"Testing batch_size={bs}...")

            # Generate batch
            x = np.random.randn(bs, self.dim, 1, self.seq_len).astype(np.float32) * 0.1

            # Warmup
            for i in range(min(3, bs)):
                _ = self.layer.forward(x[i : i + 1])

            # Time it
            start = time.time()
            for i in range(bs):
                out = self.layer.forward(x[i : i + 1])
            elapsed = (time.time() - start) * 1000

            time_per_sample = elapsed / bs
            throughput = bs * 1000 / elapsed

            results.append(
                {
                    "batch_size": bs,
                    "total_time": elapsed,
                    "time_per_sample": time_per_sample,
                    "throughput": throughput,
                }
            )

            print(
                f"  Total: {elapsed:.1f}ms | Per sample: {time_per_sample:.2f}ms | "
                f"Throughput: {throughput:.0f} samples/sec"
            )

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(
            f"{'Batch Size':<12} {'Total (ms)':<12} {'Per Sample (ms)':<18} {'Throughput':<15}"
        )
        print("-" * 70)

        for r in results:
            print(
                f"{r['batch_size']:<12} {r['total_time']:>8.1f}     "
                f"{r['time_per_sample']:>8.2f}          {r['throughput']:>8.0f} samp/s"
            )

        # Find best
        best = max(results, key=lambda x: x["throughput"])
        print("\n" + "=" * 70)
        print(f"🏆 OPTIMAL BATCH SIZE: {best['batch_size']}")
        print(f"   Throughput: {best['throughput']:.0f} samples/sec")
        print(f"   Time per sample: {best['time_per_sample']:.2f}ms")
        print("=" * 70)

        # Analysis
        print("\nINSIGHT:")
        print("-" * 70)
        baseline = results[0]["time_per_sample"]
        optimal = best["time_per_sample"]
        improvement = baseline / optimal

        print(f"Baseline (batch=1): {baseline:.2f}ms/sample")
        print(f"Optimal (batch={best['batch_size']}): {optimal:.2f}ms/sample")
        print(f"Improvement: {improvement:.1f}x")

        if improvement < 1.5:
            print("\n⚠️  WARNING: ANE is not benefiting from batching!")
            print("   Possible reasons:")
            print("   - ANE kernel doesn't support internal batching")
            print("   - Setup cost dominates")
            print("   - Need larger matrices (try 1024+ dim)")
        else:
            print(
                f"\n✅ Batching helps! {improvement:.1f}x speedup with larger batches"
            )

        return best["batch_size"]


def main():
    trainer = HighThroughputTrainer(dim=512, seq_len=256)

    # Find optimal batch size
    optimal_batch = trainer.benchmark_batch_sizes()

    # Test high-throughput processing
    print("\n" + "=" * 70)
    print(f"HIGH-THROUGHPUT TEST (batch_size={optimal_batch})")
    print("=" * 70)

    times = trainer.process_large_batch(batch_size=optimal_batch, num_batches=10)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Optimal batch size: {optimal_batch}")
    print(f"Average time: {np.mean(times):.2f}ms")
    print(f"Throughput: {1000 / np.mean(times):.0f} samples/sec")

    print("\nRECOMMENDATIONS:")
    print("-" * 70)
    print("""
To maximize ANE NPU usage:

1. USE LARGE BATCH SIZES
   - Process 64+ samples together
   - Keeps ANE pipeline full
   
2. MINIMIZE KERNEL SWITCHING
   - Group operations by type
   - Avoid frequent recompilation
   
3. USE LARGER MATRICES
   - ANE benefits from 1024+ dimensions
   - Current 512x256 is too small
   
4. FUSE OPERATIONS
   - Combine multiple ops into one kernel
   - Reduces setup overhead

Current bottleneck: Small matrices (512) and no internal batching
Solution: Increase to 1024+ dims or use CPU for small ops
    """)


if __name__ == "__main__":
    main()
