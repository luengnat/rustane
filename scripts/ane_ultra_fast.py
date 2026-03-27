#!/usr/bin/env python3
"""
Ultra-Optimized ANE Training

Optimizations:
1. Large batch size (amortize overhead)
2. Aggressive gradient accumulation (minimize recompiles)
3. Process multiple samples efficiently
4. Keep weights in fp16 to avoid conversion
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge
from ane_backward import ANELinearBackward


class UltraFastTrainer:
    """Ultra-optimized ANE trainer."""

    def __init__(self, dim=512, seq_len=256):
        self.dim = dim
        self.seq_len = seq_len

        print("Initializing Ultra-Fast Trainer...")
        get_bridge()

        self.layer = ANEConv1x1(dim, dim, seq_len)
        self.backward = ANELinearBackward(dim, dim, seq_len)
        self.backward.initialize(self.layer.weight)

        # Adam state
        self.m = np.zeros_like(self.layer.weight)
        self.v = np.zeros_like(self.layer.weight)
        self.t = 0

        print(f"  Ready: dim={dim}, seq_len={seq_len}")

    def train_large_batch(self, batch_size=8, accum_steps=20, num_updates=50, lr=0.001):
        """
        Train with large batch and aggressive accumulation.

        With batch_size=8 and accum_steps=20:
        - Effective batch size: 160
        - Recompile only every 160 forward passes
        - Amortizes ANE overhead significantly
        """
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Accumulation steps: {accum_steps}")
        print(f"  Effective batch size: {batch_size * accum_steps}")
        print(f"  Weight updates: {num_updates}")
        print(f"  Forward passes: {num_updates * accum_steps}")
        print()

        losses = []
        forward_times = []
        update_times = []

        grad_accum = np.zeros_like(self.layer.weight)
        step_count = 0

        for update_idx in range(num_updates):
            update_start = time.time()

            # Accumulate gradients over multiple steps
            for accum_idx in range(accum_steps):
                # Generate large batch
                x = (
                    np.random.randn(batch_size, self.dim, 1, self.seq_len).astype(
                        np.float32
                    )
                    * 0.1
                )
                target = (
                    np.random.randn(batch_size, self.dim, 1, self.seq_len).astype(
                        np.float32
                    )
                    * 0.1
                )

                fwd_start = time.time()

                # Process each sample in batch
                batch_loss = 0
                for b in range(batch_size):
                    # Forward
                    out = self.layer.forward(x[b : b + 1])
                    loss = np.mean((out - target[b : b + 1]) ** 2)
                    batch_loss += loss

                    # Backward
                    dy = 2 * (out - target[b : b + 1]) / np.prod(out.shape)
                    dx = self.backward.compute_dx(dy)

                    # Compute dW
                    x_reshaped = (
                        x[b : b + 1].transpose(0, 3, 1, 2).reshape(-1, self.dim)
                    )
                    dy_reshaped = dy.transpose(0, 3, 1, 2).reshape(-1, self.dim)
                    dW = (dy_reshaped.T @ x_reshaped).reshape(
                        self.dim, self.dim, 1, 1
                    ) / self.seq_len
                    grad_accum += dW

                forward_times.append((time.time() - fwd_start) * 1000 / batch_size)
                losses.append(batch_loss / batch_size)
                step_count += 1

            # Weight update (with recompile)
            update_compute_start = time.time()

            self.t += 1
            grad_mean = grad_accum / accum_steps / batch_size
            self.m = 0.9 * self.m + 0.1 * grad_mean
            self.v = 0.999 * self.v + 0.001 * (grad_mean**2)
            m_hat = self.m / (1 - 0.9**self.t)
            v_hat = self.v / (1 - 0.999**self.t)

            new_weight = self.layer.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.layer.update_weights(new_weight)
            self.backward.initialize(new_weight)

            update_times.append((time.time() - update_compute_start) * 1000)
            grad_accum = np.zeros_like(self.layer.weight)

            if (update_idx + 1) % 10 == 0:
                avg_loss = np.mean(losses[-accum_steps * 10 :])
                avg_fwd = np.mean(forward_times[-accum_steps * 10 :])
                print(
                    f"  Update {update_idx + 1}/{num_updates} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Fwd: {avg_fwd:.1f}ms/sample | "
                    f"Update: {update_times[-1]:.0f}ms"
                )

        return losses, forward_times, update_times

    def train_batch_parallel(self, batch_size=4, accum_steps=10, num_updates=30):
        """
        Try to process batch more efficiently.

        If ANE supports larger input sizes, process multiple samples
        in a single kernel call.
        """
        print(f"\nBatch-Parallel Training:")
        print(f"  Batch size: {batch_size}")
        print(f"  Accumulation: {accum_steps}")

        losses = []
        times = []

        grad_accum = np.zeros_like(self.layer.weight)

        for update_idx in range(num_updates):
            for accum_idx in range(accum_steps):
                # Stack samples along batch dimension
                x = (
                    np.random.randn(batch_size, self.dim, 1, self.seq_len).astype(
                        np.float32
                    )
                    * 0.1
                )
                target = (
                    np.random.randn(batch_size, self.dim, 1, self.seq_len).astype(
                        np.float32
                    )
                    * 0.1
                )

                start = time.time()

                # Try processing whole batch at once if ANE supports it
                # Otherwise fall back to sequential
                try:
                    # Reshape to combine batch into channels
                    # [B, C, 1, S] -> [1, B*C, 1, S]
                    x_combined = x.reshape(1, batch_size * self.dim, 1, self.seq_len)
                    out_combined = self.layer.forward(x_combined)

                    # Split back
                    out = out_combined.reshape(batch_size, self.dim, 1, self.seq_len)

                    # Compute loss for all
                    loss = np.mean((out - target) ** 2)

                    # Backward (simplified)
                    dy = 2 * (out - target) / np.prod(out.shape)

                    # Compute dW for all samples
                    for b in range(batch_size):
                        x_reshaped = (
                            x[b : b + 1].transpose(0, 3, 1, 2).reshape(-1, self.dim)
                        )
                        dy_reshaped = (
                            dy[b : b + 1].transpose(0, 3, 1, 2).reshape(-1, self.dim)
                        )
                        dW = (dy_reshaped.T @ x_reshaped).reshape(
                            self.dim, self.dim, 1, 1
                        ) / self.seq_len
                        grad_accum += dW

                    elapsed = (time.time() - start) * 1000 / batch_size

                except:
                    # Fall back to sequential processing
                    for b in range(batch_size):
                        start = time.time()
                        out = self.layer.forward(x[b : b + 1])
                        loss = np.mean((out - target[b : b + 1]) ** 2)
                        elapsed = (time.time() - start) * 1000

                        dy = 2 * (out - target[b : b + 1]) / np.prod(out.shape)
                        dx = self.backward.compute_dx(dy)

                        x_reshaped = (
                            x[b : b + 1].transpose(0, 3, 1, 2).reshape(-1, self.dim)
                        )
                        dy_reshaped = dy.transpose(0, 3, 1, 2).reshape(-1, self.dim)
                        dW = (dy_reshaped.T @ x_reshaped).reshape(
                            self.dim, self.dim, 1, 1
                        ) / self.seq_len
                        grad_accum += dW

                losses.append(loss)
                times.append(elapsed)

            # Update weights
            self.t += 1
            grad_mean = grad_accum / accum_steps / batch_size
            self.m = 0.9 * self.m + 0.1 * grad_mean
            self.v = 0.999 * self.v + 0.001 * (grad_mean**2)
            m_hat = self.m / (1 - 0.9**self.t)
            v_hat = self.v / (1 - 0.999**self.t)

            new_weight = self.layer.weight - 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.layer.update_weights(new_weight)
            self.backward.initialize(new_weight)

            grad_accum = np.zeros_like(self.layer.weight)

            if (update_idx + 1) % 5 == 0:
                avg_loss = np.mean(losses[-accum_steps * 5 :])
                avg_time = np.mean(times[-accum_steps * 5 :])
                print(
                    f"  Update {update_idx + 1}/{num_updates} | Loss: {avg_loss:.4f} | Time: {avg_time:.1f}ms/sample"
                )

        return losses, times


def main():
    print("=" * 70)
    print("Ultra-Optimized ANE Training")
    print("=" * 70)

    # Test 1: Large batch + aggressive accumulation
    print("\n" + "=" * 70)
    print("TEST 1: Large Batch + Aggressive Accumulation")
    print("=" * 70)

    trainer = UltraFastTrainer(dim=512, seq_len=256)
    losses, fwd_times, upd_times = trainer.train_large_batch(
        batch_size=8, accum_steps=20, num_updates=50, lr=0.001
    )

    print("\nResults:")
    print(f"  Average forward: {np.mean(fwd_times):.1f}ms/sample")
    print(f"  Average update: {np.mean(upd_times):.0f}ms")
    print(f"  Final loss: {np.mean(losses[-20:]):.4f}")
    print(f"  Best loss: {min(losses):.4f}")

    # Calculate effective time per sample
    total_samples = 50 * 20 * 8  # updates * accum * batch
    total_fwd_time = np.sum(fwd_times)
    total_update_time = np.sum(upd_times)
    total_time = total_fwd_time + total_update_time

    time_per_sample = total_time / total_samples

    print(f"\n  Total samples: {total_samples:,}")
    print(f"  Total time: {total_time / 1000:.1f}s")
    print(f"  Time per sample: {time_per_sample:.1f}ms")
    print(f"  Throughput: {1000 / time_per_sample:.0f} samples/sec")

    # Test 2: Compare with baseline
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Baseline (accum=1, batch=1):  ~88ms/sample")
    print(f"Optimized (accum=20, batch=8): ~{time_per_sample:.1f}ms/sample")
    print(f"Speedup: {88 / time_per_sample:.1f}x")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print(f"""
By increasing batch size to 8 and accumulation to 20:
- We process {8 * 20}=160 samples per weight update
- Recompile cost is amortized over 160 samples
- Result: {88 / time_per_sample:.1f}x speedup!

To go even faster:
1. Increase batch_size to 16 or 32
2. Increase accum_steps to 50 or 100
3. Implement zero-recompile (would eliminate update cost entirely)
    """)


if __name__ == "__main__":
    main()
