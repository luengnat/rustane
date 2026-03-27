#!/usr/bin/env python3
"""
ANE Training - Clean Optimized Version

Simple, fast, and reliable. Pushes performance to the limit.
"""

import numpy as np
import struct
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


def load_tokens_simple(filepath):
    """Simple token loader."""
    with open(filepath, "rb") as f:
        data = f.read()
    tokens = struct.unpack(f"{len(data) // 2}H", data)
    return np.array(tokens, dtype=np.int32)


class FastModel:
    """Minimal model for maximum speed."""

    def __init__(self, dim=512, seq_len=256):
        self.dim = dim
        self.seq_len = seq_len

        get_bridge()

        # Single ANE layer
        self.layer = ANEConv1x1(dim, dim, seq_len)

        # Adam state
        self.m = np.zeros_like(self.layer.weight)
        self.v = np.zeros_like(self.layer.weight)
        self.t = 0

    def train_step(self, x, target, lr=0.001):
        """One training step."""
        # Forward
        out = self.layer.forward(x)
        loss = np.mean((out - target) ** 2)

        # Backward (compute dW)
        dy = 2 * (out - target) / np.prod(out.shape)
        x_reshaped = x.transpose(0, 3, 1, 2).reshape(-1, self.dim)
        dy_reshaped = dy.transpose(0, 3, 1, 2).reshape(-1, self.dim)
        dW = (dy_reshaped.T @ x_reshaped).reshape(
            self.dim, self.dim, 1, 1
        ) / self.seq_len

        # Adam update
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * dW
        self.v = 0.999 * self.v + 0.001 * (dW**2)
        m_hat = self.m / (1 - 0.9**self.t)
        v_hat = self.v / (1 - 0.999**self.t)

        new_weight = self.layer.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self.layer.update_weights(new_weight)

        return loss


def benchmark_speeds():
    """Test different configurations to find maximum speed."""
    print("=" * 70)
    print("ANE Speed Benchmark - Finding Maximum Performance")
    print("=" * 70)

    configs = [
        (1, 1, "Baseline"),
        (4, 5, "Small batch"),
        (8, 10, "Medium"),
        (16, 20, "Large batch, high accum"),
        (32, 50, "Extreme"),
    ]

    results = []

    for batch_size, accum_steps, name in configs:
        print(f"\nTesting: {name} (batch={batch_size}, accum={accum_steps})")
        print("-" * 70)

        model = FastModel(dim=512, seq_len=256)

        # Warmup
        for _ in range(3):
            x = np.random.randn(1, 512, 1, 256).astype(np.float32) * 0.1
            target = np.random.randn(1, 512, 1, 256).astype(np.float32) * 0.1
            model.train_step(x, target)

        # Measure
        times = []
        grad_accum = np.zeros_like(model.layer.weight)

        for i in range(accum_steps):
            batch_times = []
            for b in range(batch_size):
                x = np.random.randn(1, 512, 1, 256).astype(np.float32) * 0.1
                target = np.random.randn(1, 512, 1, 256).astype(np.float32) * 0.1

                start = time.time()
                out = model.layer.forward(x)
                loss = np.mean((out - target) ** 2)
                dy = 2 * (out - target) / np.prod(out.shape)

                # Accumulate gradient
                x_r = x.transpose(0, 3, 1, 2).reshape(-1, 512)
                dy_r = dy.transpose(0, 3, 1, 2).reshape(-1, 512)
                dW = (dy_r.T @ x_r).reshape(512, 512, 1, 1) / 256
                grad_accum += dW

                batch_times.append((time.time() - start) * 1000)

            times.extend(batch_times)

        # One update
        update_start = time.time()
        model.t += 1
        grad_mean = grad_accum / accum_steps / batch_size
        model.m = 0.9 * model.m + 0.1 * grad_mean
        model.v = 0.999 * model.v + 0.001 * (grad_mean**2)
        m_hat = model.m / (1 - 0.9**model.t)
        v_hat = model.v / (1 - 0.999**model.t)
        new_w = model.layer.weight - 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)
        model.layer.update_weights(new_w)
        update_time = (time.time() - update_start) * 1000

        # Calculate metrics
        avg_fwd = np.mean(times)
        total_samples = batch_size * accum_steps
        time_per_sample = (np.sum(times) + update_time) / total_samples

        print(f"  Forward: {avg_fwd:.2f}ms/sample")
        print(f"  Update: {update_time:.0f}ms")
        print(f"  Total time: {time_per_sample:.2f}ms/sample")
        print(f"  Throughput: {1000 / time_per_sample:.0f} samples/sec")

        results.append(
            {
                "name": name,
                "batch": batch_size,
                "accum": accum_steps,
                "time_per_sample": time_per_sample,
                "throughput": 1000 / time_per_sample,
            }
        )

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Config':<25} {'Time/ms':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 70)

    for r in results:
        speedup = 4.0 / r["time_per_sample"]  # vs 4ms CPU baseline
        print(
            f"{r['name']:<25} {r['time_per_sample']:>6.2f}      {r['throughput']:>6.0f} samp/s    {speedup:>5.1f}x"
        )

    # Find best
    best = min(results, key=lambda x: x["time_per_sample"])
    print("\n" + "=" * 70)
    print(f"🏆 BEST CONFIGURATION: {best['name']}")
    print(f"   Time: {best['time_per_sample']:.2f}ms/sample")
    print(f"   Throughput: {best['throughput']:.0f} samples/sec")
    print(f"   Speedup: {4.0 / best['time_per_sample']:.1f}x faster than CPU")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_speeds()
