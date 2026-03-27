#!/usr/bin/env python3
"""
CPU vs ANE Comparison - Comprehensive Benchmark

Compares:
1. Forward pass performance and accuracy
2. Backward pass performance and accuracy
3. End-to-end training speed
4. Numerical correctness
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge
from ane_backward import ANELinearBackward


def cpu_conv1x1(x, weight):
    """CPU reference implementation."""
    B, C_in, H, S = x.shape
    C_out = weight.shape[0]

    # Reshape: (B, C_in, 1, S) -> (B*S, C_in)
    x_reshaped = x.transpose(0, 3, 1, 2).reshape(-1, C_in)
    # Weight: (C_out, C_in, 1, 1) -> (C_out, C_in) -> (C_in, C_out)
    w_reshaped = weight.reshape(C_out, C_in).T

    # Matmul
    out = x_reshaped @ w_reshaped

    # Reshape back: (B*S, C_out) -> (B, C_out, 1, S)
    out = out.reshape(B, S, C_out, 1).transpose(0, 2, 3, 1)

    return out


def benchmark_forward(in_ch, out_ch, seq_len, num_runs=10):
    """Benchmark forward pass."""
    print(f"\nForward Pass: {in_ch} -> {out_ch}, seq_len={seq_len}")
    print("-" * 70)

    # Create test data
    x = np.random.randn(1, in_ch, 1, seq_len).astype(np.float32) * 0.1
    W = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.02

    # CPU benchmark
    cpu_times = []
    for _ in range(num_runs):
        start = time.time()
        cpu_out = cpu_conv1x1(x, W)
        cpu_times.append(time.time() - start)
    cpu_time = np.mean(cpu_times) * 1000  # ms

    # ANE benchmark
    ane_layer = ANEConv1x1(in_ch, out_ch, seq_len)
    ane_layer.weight = W.copy()
    ane_layer._recompile()

    # Warmup
    _ = ane_layer.forward(x)

    ane_times = []
    for _ in range(num_runs):
        start = time.time()
        ane_out = ane_layer.forward(x)
        ane_times.append(time.time() - start)
    ane_time = np.mean(ane_times) * 1000  # ms

    # Accuracy check
    diff = np.abs(cpu_out - ane_out)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Results
    speedup = cpu_time / ane_time
    print(f"  CPU:  {cpu_time:6.2f} ms")
    print(f"  ANE:  {ane_time:6.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Accuracy: {'✅ PASS' if max_diff < 0.01 else '❌ FAIL'}")

    return {
        "cpu_time": cpu_time,
        "ane_time": ane_time,
        "speedup": speedup,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }


def benchmark_backward(in_ch, out_ch, seq_len, num_runs=10):
    """Benchmark backward pass."""
    print(f"\nBackward Pass: {out_ch} -> {in_ch}, seq_len={seq_len}")
    print("-" * 70)

    # Create test data
    x = np.random.randn(1, in_ch, 1, seq_len).astype(np.float32) * 0.1
    dy = np.random.randn(1, out_ch, 1, seq_len).astype(np.float32) * 0.1
    W = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.02

    # CPU reference
    def cpu_backward_dx(dy, W):
        W_T = W.reshape(out_ch, in_ch).T
        dy_mat = dy.transpose(0, 3, 1, 2).reshape(-1, out_ch)
        dx = (dy_mat @ W_T.T).reshape(1, seq_len, in_ch, 1).transpose(0, 2, 3, 1)
        return dx

    # CPU benchmark
    cpu_times = []
    for _ in range(num_runs):
        start = time.time()
        cpu_dx = cpu_backward_dx(dy, W)
        cpu_times.append(time.time() - start)
    cpu_time = np.mean(cpu_times) * 1000

    # ANE benchmark
    backward = ANELinearBackward(in_ch, out_ch, seq_len)
    backward.initialize(W)

    # Warmup
    _ = backward.compute_dx(dy)

    ane_times = []
    for _ in range(num_runs):
        start = time.time()
        ane_dx = backward.compute_dx(dy)
        ane_times.append(time.time() - start)
    ane_time = np.mean(ane_times) * 1000

    # Accuracy
    diff = np.abs(cpu_dx - ane_dx)
    max_diff = diff.max()
    mean_diff = diff.mean()

    speedup = cpu_time / ane_time
    print(f"  CPU:  {cpu_time:6.2f} ms")
    print(f"  ANE:  {ane_time:6.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Accuracy: {'✅ PASS' if max_diff < 0.01 else '❌ FAIL'}")

    return {
        "cpu_time": cpu_time,
        "ane_time": ane_time,
        "speedup": speedup,
        "max_diff": max_diff,
    }


def benchmark_training_step(dim, seq_len, num_steps=20):
    """Benchmark full training step."""
    print(f"\nFull Training Step: dim={dim}, seq_len={seq_len}")
    print("-" * 70)

    from ane_ops import ANEConv1x1
    from ane_backward import ANELinearBackward

    # CPU training
    W_cpu = np.random.randn(dim, dim, 1, 1).astype(np.float32) * 0.02
    m_cpu = np.zeros_like(W_cpu)
    v_cpu = np.zeros_like(W_cpu)
    t_cpu = 0

    cpu_losses = []
    cpu_times = []

    for step in range(num_steps):
        x = np.random.randn(1, dim, 1, seq_len).astype(np.float32) * 0.1
        target = np.random.randn(1, dim, 1, seq_len).astype(np.float32) * 0.1

        start = time.time()

        # Forward
        out = cpu_conv1x1(x, W_cpu)

        # Loss
        loss = np.mean((out - target) ** 2)

        # Backward
        dy = 2 * (out - target) / np.prod(out.shape)
        W_T = W_cpu.reshape(dim, dim).T
        dy_mat = dy.transpose(0, 3, 1, 2).reshape(-1, dim)
        dx = (dy_mat @ W_T.T).reshape(1, seq_len, dim, 1).transpose(0, 2, 3, 1)
        dW = (dy_mat.T @ x.transpose(0, 3, 1, 2).reshape(-1, dim)).reshape(
            dim, dim, 1, 1
        ) / seq_len

        # Update (Adam)
        t_cpu += 1
        m_cpu = 0.9 * m_cpu + 0.1 * dW
        v_cpu = 0.999 * v_cpu + 0.001 * (dW**2)
        m_hat = m_cpu / (1 - 0.9**t_cpu)
        v_hat = v_cpu / (1 - 0.999**t_cpu)
        W_cpu -= 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)

        elapsed = time.time() - start
        cpu_times.append(elapsed)
        cpu_losses.append(loss)

    cpu_time = np.mean(cpu_times) * 1000

    # ANE training
    W_ane = W_cpu.copy()
    ane_layer = ANEConv1x1(dim, dim, seq_len)
    ane_layer.weight = W_ane.copy()
    ane_layer._recompile()

    backward = ANELinearBackward(dim, dim, seq_len)
    backward.initialize(W_ane)

    m_ane = np.zeros_like(W_ane)
    v_ane = np.zeros_like(W_ane)
    t_ane = 0

    ane_losses = []
    ane_times = []

    for step in range(num_steps):
        x = np.random.randn(1, dim, 1, seq_len).astype(np.float32) * 0.1
        target = np.random.randn(1, dim, 1, seq_len).astype(np.float32) * 0.1

        start = time.time()

        # Forward (ANE)
        out = ane_layer.forward(x)

        # Loss
        loss = np.mean((out - target) ** 2)

        # Backward (ANE dx + CPU dW)
        dy = 2 * (out - target) / np.prod(out.shape)
        dx = backward.compute_dx(dy)
        dW = backward.compute_dW(x, dy)

        # Update
        t_ane += 1
        m_ane = 0.9 * m_ane + 0.1 * dW
        v_ane = 0.999 * v_ane + 0.001 * (dW**2)
        m_hat = m_ane / (1 - 0.9**t_ane)
        v_hat = v_ane / (1 - 0.999**t_ane)
        W_ane -= 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Update kernels
        ane_layer.update_weights(W_ane)
        backward.initialize(W_ane)

        elapsed = time.time() - start
        ane_times.append(elapsed)
        ane_losses.append(loss)

    ane_time = np.mean(ane_times) * 1000

    # Results
    speedup = cpu_time / ane_time
    print(f"  CPU:  {cpu_time:6.2f} ms/step | Loss: {cpu_losses[-1]:.6f}")
    print(f"  ANE:  {ane_time:6.2f} ms/step | Loss: {ane_losses[-1]:.6f}")
    print(f"  Speedup: {speedup:.1f}x")

    # Check if both are learning
    cpu_learning = cpu_losses[-1] < cpu_losses[0]
    ane_learning = ane_losses[-1] < ane_losses[0]

    print(
        f"  CPU learning: {'✅' if cpu_learning else '❌'} {cpu_losses[0]:.6f} → {cpu_losses[-1]:.6f}"
    )
    print(
        f"  ANE learning: {'✅' if ane_learning else '❌'} {ane_losses[0]:.6f} → {ane_losses[-1]:.6f}"
    )

    return {
        "cpu_time": cpu_time,
        "ane_time": ane_time,
        "speedup": speedup,
        "cpu_learning": cpu_learning,
        "ane_learning": ane_learning,
    }


def main():
    print("=" * 70)
    print("CPU vs ANE Comparison - Comprehensive Benchmark")
    print("=" * 70)

    # Initialize ANE
    print("\nInitializing ANE...")
    get_bridge()
    print("✅ ANE initialized")

    results = {}

    # Test different dimensions
    test_cases = [
        (256, 256, 64),
        (512, 512, 256),
        (768, 768, 256),
        (768, 32000, 256),  # Classifier
    ]

    # Forward pass benchmarks
    print("\n" + "=" * 70)
    print("FORWARD PASS BENCHMARKS")
    print("=" * 70)

    for in_ch, out_ch, seq in test_cases:
        key = f"{in_ch}x{out_ch}_s{seq}"
        results[f"fwd_{key}"] = benchmark_forward(in_ch, out_ch, seq, num_runs=10)

    # Backward pass benchmarks
    print("\n" + "=" * 70)
    print("BACKWARD PASS BENCHMARKS")
    print("=" * 70)

    for in_ch, out_ch, seq in test_cases:
        key = f"{in_ch}x{out_ch}_s{seq}"
        results[f"bwd_{key}"] = benchmark_backward(in_ch, out_ch, seq, num_runs=10)

    # Training step benchmarks
    print("\n" + "=" * 70)
    print("TRAINING STEP BENCHMARKS")
    print("=" * 70)

    for dim, _, seq in [(512, 512, 256)]:
        results["train"] = benchmark_training_step(dim, seq, num_steps=20)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nForward Pass Speedups:")
    for key, val in results.items():
        if key.startswith("fwd_"):
            dims = key.replace("fwd_", "")
            print(f"  {dims:20s}: {val['speedup']:5.1f}x")

    print("\nBackward Pass Speedups:")
    for key, val in results.items():
        if key.startswith("bwd_"):
            dims = key.replace("bwd_", "")
            print(f"  {dims:20s}: {val['speedup']:5.1f}x")

    print("\nTraining Step:")
    print(f"  Speedup: {results['train']['speedup']:.1f}x")
    print(f"  CPU learning: {'✅' if results['train']['cpu_learning'] else '❌'}")
    print(f"  ANE learning: {'✅' if results['train']['ane_learning'] else '❌'}")

    # Overall
    avg_fwd_speedup = np.mean(
        [v["speedup"] for k, v in results.items() if k.startswith("fwd_")]
    )
    avg_bwd_speedup = np.mean(
        [v["speedup"] for k, v in results.items() if k.startswith("bwd_")]
    )

    print("\n" + "=" * 70)
    print(f"Average Forward Speedup: {avg_fwd_speedup:.1f}x")
    print(f"Average Backward Speedup: {avg_bwd_speedup:.1f}x")
    print(f"Training Speedup: {results['train']['speedup']:.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
