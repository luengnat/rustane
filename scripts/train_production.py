#!/usr/bin/env python3
"""
ANE Production Trainer - Optimized Configuration

Uses optimal settings: batch=32, accum=50
Trains on real FineWeb-10B data at maximum speed.
"""

import numpy as np
import struct
import time
import sys
import os
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


class ProductionTrainer:
    """Production-ready ANE trainer with optimal settings."""

    def __init__(self, dim=512, seq_len=256, batch_size=32, accum_steps=50):
        self.dim = dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.accum_steps = accum_steps

        print("=" * 70)
        print("ANE Production Trainer")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Dimensions: {dim}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Batch size: {batch_size}")
        print(f"  Accumulation steps: {accum_steps}")
        print(f"  Effective batch size: {batch_size * accum_steps}")

        # Initialize ANE
        print("\nInitializing ANE...")
        get_bridge()

        # Model
        self.layer = ANEConv1x1(dim, dim, seq_len)

        # Adam state
        self.m = np.zeros_like(self.layer.weight)
        self.v = np.zeros_like(self.layer.weight)
        self.t = 0

        # Stats
        self.total_tokens = 0
        self.total_time = 0

        print("✅ Ready for training")

    def train_on_data(self, tokens, num_updates=100, lr=0.001, log_interval=10):
        """Train on token data."""
        print(f"\nTraining on {len(tokens):,} tokens...")
        print(f"Target: {num_updates} weight updates")
        print("=" * 70)

        losses = []
        data_idx = 0

        for update_idx in range(num_updates):
            update_start = time.time()

            grad_accum = np.zeros_like(self.layer.weight)
            update_loss = 0

            # Accumulate over multiple steps
            for accum_idx in range(self.accum_steps):
                # Get batch from data
                if data_idx + self.batch_size * self.seq_len + 1 > len(tokens):
                    data_idx = 0

                batch_tokens = tokens[
                    data_idx : data_idx + self.batch_size * self.seq_len + 1
                ]
                data_idx += self.batch_size * self.seq_len

                # Process each sample in batch
                for b in range(self.batch_size):
                    start = b * self.seq_len
                    end = start + self.seq_len

                    x_tokens = batch_tokens[start:end]
                    target_tokens = batch_tokens[start + 1 : end + 1]

                    # Convert to embeddings (simplified - just use tokens as values)
                    x = np.zeros((1, self.dim, 1, self.seq_len), dtype=np.float32)
                    target = np.zeros((1, self.dim, 1, self.seq_len), dtype=np.float32)

                    x[0, : min(len(x_tokens), self.dim), 0, :] = (
                        x_tokens[: self.dim].reshape(-1, 1) * 0.01
                    )
                    target[0, : min(len(target_tokens), self.dim), 0, :] = (
                        target_tokens[: self.dim].reshape(-1, 1) * 0.01
                    )

                    # Forward
                    out = self.layer.forward(x)
                    loss = np.mean((out - target) ** 2)
                    update_loss += loss

                    # Backward
                    dy = 2 * (out - target) / np.prod(out.shape)
                    x_r = x.transpose(0, 3, 1, 2).reshape(-1, self.dim)
                    dy_r = dy.transpose(0, 3, 1, 2).reshape(-1, self.dim)
                    dW = (dy_r.T @ x_r).reshape(self.dim, self.dim, 1, 1) / self.seq_len
                    grad_accum += dW

                    self.total_tokens += self.seq_len

            # Weight update
            self.t += 1
            grad_mean = grad_accum / self.accum_steps / self.batch_size
            self.m = 0.9 * self.m + 0.1 * grad_mean
            self.v = 0.999 * self.v + 0.001 * (grad_mean**2)
            m_hat = self.m / (1 - 0.9**self.t)
            v_hat = self.v / (1 - 0.999**self.t)

            new_w = self.layer.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.layer.update_weights(new_w)

            update_time = time.time() - update_start
            self.total_time += update_time

            avg_loss = update_loss / (self.accum_steps * self.batch_size)
            losses.append(avg_loss)

            if (update_idx + 1) % log_interval == 0:
                tokens_per_sec = (
                    self.batch_size * self.accum_steps * self.seq_len * log_interval
                ) / self.total_time
                print(
                    f"  Update {update_idx + 1}/{num_updates} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {update_time * 1000:.0f}ms | "
                    f"Speed: {tokens_per_sec:.0f} tokens/sec"
                )

        return losses

    def get_stats(self):
        """Get training statistics."""
        return {
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "tokens_per_sec": self.total_tokens / self.total_time
            if self.total_time > 0
            else 0,
        }


def main():
    # Load data
    data_file = "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"

    print("=" * 70)
    print("ANE PRODUCTION TRAINING")
    print("=" * 70)

    if os.path.exists(data_file):
        print(f"Loading data from {data_file}...")
        with open(data_file, "rb") as f:
            data = f.read()
        tokens = struct.unpack(f"{len(data) // 2}H", data)
        tokens = np.array(tokens, dtype=np.int32)
        tokens = np.clip(tokens, 0, 1023)
        print(f"Loaded {len(tokens):,} tokens")
    else:
        print("Data file not found, using synthetic data")
        tokens = np.random.randint(0, 1024, size=10000000, dtype=np.int32)

    # Create trainer with optimal settings
    trainer = ProductionTrainer(
        dim=512,
        seq_len=256,
        batch_size=32,  # Optimal from benchmark
        accum_steps=50,  # Optimal from benchmark
    )

    # Train
    start_time = time.time()
    losses = trainer.train_on_data(
        tokens=tokens, num_updates=100, lr=0.001, log_interval=10
    )
    total_elapsed = time.time() - start_time

    # Summary
    stats = trainer.get_stats()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Best loss: {min(losses):.4f}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Throughput: {stats['tokens_per_sec']:.0f} tokens/sec")
    print(f"Samples/sec: {stats['tokens_per_sec'] / 256:.0f}")
    print(f"Time per sample: {256 * 1000 / stats['tokens_per_sec']:.2f}ms")
    print("=" * 70)

    # Save results
    results = {
        "config": {"batch_size": 32, "accum_steps": 50, "dim": 512, "seq_len": 256},
        "performance": {
            "total_tokens": int(stats["total_tokens"]),
            "total_time_sec": float(total_elapsed),
            "tokens_per_sec": float(stats["tokens_per_sec"]),
            "samples_per_sec": float(stats["tokens_per_sec"] / 256),
            "time_per_sample_ms": float(256 * 1000 / stats["tokens_per_sec"]),
        },
        "training": {
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "best_loss": float(min(losses)),
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open("production_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to: production_results.json")


if __name__ == "__main__":
    main()
