#!/usr/bin/env python3
"""
ANE Long-Running Training on FineWeb-10B

Trains on multiple data files with checkpointing and progress tracking.
"""

import numpy as np
import struct
import time
import sys
import os
import json
import glob
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


class LongRunTrainer:
    """Trainer designed for long runs on real data."""

    def __init__(self, data_dir, vocab_size=1024, dim=512, seq_len=256):
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.data_dir = data_dir

        print("=" * 70)
        print("ANE Long-Run Training on FineWeb-10B")
        print("=" * 70)

        # Find all data files
        pattern = os.path.join(data_dir, "fineweb_train_*.bin")
        self.data_files = sorted(glob.glob(pattern))
        print(f"Found {len(self.data_files)} data files")

        if len(self.data_files) == 0:
            raise ValueError(f"No data files found in {data_dir}")

        # Show first few files
        for i, f in enumerate(self.data_files[:3]):
            size = os.path.getsize(f) / (1024 * 1024)
            print(f"  {i + 1}. {os.path.basename(f)} ({size:.1f} MB)")
        if len(self.data_files) > 3:
            print(f"  ... and {len(self.data_files) - 3} more")

        # Initialize ANE
        print("\nInitializing ANE...")
        get_bridge()

        # Simple model: embedding + 1 ANE layer + head
        print("Initializing model...")
        self.embed = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02
        self.layer = ANEConv1x1(dim, dim, seq_len)

        # Adam state
        self.m = np.zeros_like(self.layer.weight)
        self.v = np.zeros_like(self.layer.weight)
        self.t = 0

        # Stats
        self.total_tokens = 0
        self.total_time = 0
        self.start_time = time.time()

        print(f"✅ Model ready: {vocab_size} vocab, {dim} dim")

    def load_tokens(self, filepath):
        """Load tokens from a single file."""
        with open(filepath, "rb") as f:
            data = f.read()
        tokens = struct.unpack(f"{len(data) // 2}H", data)
        tokens = np.array(tokens, dtype=np.int32)
        tokens = np.clip(tokens, 0, self.vocab_size - 1)
        return tokens

    def train_on_file(
        self, filepath, batch_size=8, accum_steps=20, lr=0.001, max_steps=None
    ):
        """Train on a single file."""
        print(f"\nLoading {os.path.basename(filepath)}...")
        tokens = self.load_tokens(filepath)
        print(f"  Tokens: {len(tokens):,}")

        losses = []
        data_idx = 0
        step = 0

        while data_idx + batch_size * self.seq_len + 1 <= len(tokens):
            if max_steps and step >= max_steps:
                break

            step_start = time.time()

            # Accumulate gradients
            grad_accum = np.zeros_like(self.layer.weight)
            step_loss = 0

            for accum_idx in range(accum_steps):
                if data_idx + batch_size * self.seq_len + 1 > len(tokens):
                    break

                # Get batch
                batch_tokens = tokens[
                    data_idx : data_idx + batch_size * self.seq_len + 1
                ]
                data_idx += batch_size * self.seq_len

                inputs = batch_tokens[:-1].reshape(batch_size, self.seq_len)
                targets = batch_tokens[1:].reshape(batch_size, self.seq_len)

                # Process batch
                for b in range(batch_size):
                    # Forward
                    x = self.embed[inputs[b]]
                    x = x.transpose(1, 0).reshape(1, self.dim, 1, self.seq_len)

                    out = self.layer.forward(x)

                    # Simple loss (MSE against embedded targets)
                    target_embed = self.embed[targets[b]]
                    target_reshaped = target_embed.transpose(1, 0).reshape(
                        1, self.dim, 1, self.seq_len
                    )

                    loss = np.mean((out - target_reshaped) ** 2)
                    step_loss += loss

                    # Backward
                    dy = 2 * (out - target_reshaped) / np.prod(out.shape)

                    x_reshaped = x.transpose(0, 3, 1, 2).reshape(-1, self.dim)
                    dy_reshaped = dy.transpose(0, 3, 1, 2).reshape(-1, self.dim)
                    dW = (dy_reshaped.T @ x_reshaped).reshape(
                        self.dim, self.dim, 1, 1
                    ) / self.seq_len
                    grad_accum += dW

                    self.total_tokens += self.seq_len

            # Update weights
            self.t += 1
            grad_mean = grad_accum / accum_steps / batch_size
            self.m = 0.9 * self.m + 0.1 * grad_mean
            self.v = 0.999 * self.v + 0.001 * (grad_mean**2)
            m_hat = self.m / (1 - 0.9**self.t)
            v_hat = self.v / (1 - 0.999**self.t)

            new_weight = self.layer.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.layer.update_weights(new_weight)

            step_time = time.time() - step_start
            self.total_time += step_time

            avg_loss = step_loss / (accum_steps * batch_size)
            losses.append(avg_loss)
            step += 1

            # Log progress
            if step % 10 == 0:
                elapsed = time.time() - self.start_time
                tokens_per_sec = self.total_tokens / elapsed
                print(
                    f"  Step {step:4d} | Loss: {avg_loss:.4f} | "
                    f"Tokens: {self.total_tokens:,} | "
                    f"Speed: {tokens_per_sec:.0f} tok/s"
                )

        return losses

    def train_multiple_files(
        self, num_files=3, steps_per_file=100, batch_size=8, accum_steps=20, lr=0.001
    ):
        """Train on multiple data files."""
        all_losses = []

        files_to_process = self.data_files[:num_files]

        print(f"\nTraining on {len(files_to_process)} files...")
        print(f"Steps per file: {steps_per_file}")
        print(f"Total steps: {len(files_to_process) * steps_per_file}")
        print("=" * 70)

        for file_idx, filepath in enumerate(files_to_process):
            print(
                f"\n[{file_idx + 1}/{len(files_to_process)}] Processing {os.path.basename(filepath)}..."
            )

            losses = self.train_on_file(
                filepath,
                batch_size=batch_size,
                accum_steps=accum_steps,
                lr=lr,
                max_steps=steps_per_file,
            )

            all_losses.extend(losses)

            # Save checkpoint
            self.save_checkpoint(file_idx + 1)

        return all_losses

    def save_checkpoint(self, file_idx):
        """Save training checkpoint."""
        checkpoint = {
            "embed": self.embed.tolist(),
            "layer_weight": self.layer.weight.tolist(),
            "m": self.m.tolist(),
            "v": self.v.tolist(),
            "t": self.t,
            "total_tokens": self.total_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        filename = f"checkpoint_file{file_idx}.json"
        with open(filename, "w") as f:
            json.dump(checkpoint, f)

        print(f"  💾 Checkpoint saved: {filename}")

    def get_stats(self):
        """Get training statistics."""
        elapsed = time.time() - self.start_time
        return {
            "total_tokens": self.total_tokens,
            "total_time": elapsed,
            "tokens_per_sec": self.total_tokens / elapsed if elapsed > 0 else 0,
            "samples_per_sec": (self.total_tokens / 256) / elapsed
            if elapsed > 0
            else 0,
        }


def main():
    data_dir = "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024"

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Using synthetic data instead...")
        # Create dummy data for testing
        os.makedirs("dummy_data", exist_ok=True)
        for i in range(3):
            tokens = np.random.randint(0, 1024, size=1000000, dtype=np.uint16)
            with open(f"dummy_data/fineweb_train_00000{i}.bin", "wb") as f:
                f.write(tokens.tobytes())
        data_dir = "dummy_data"

    # Create trainer
    trainer = LongRunTrainer(data_dir=data_dir, vocab_size=1024, dim=512, seq_len=256)

    # Train on multiple files
    losses = trainer.train_multiple_files(
        num_files=5,  # Process 5 files
        steps_per_file=100,  # 100 steps per file
        batch_size=8,  # Batch size
        accum_steps=20,  # Accumulation
        lr=0.001,
    )

    # Final summary
    stats = trainer.get_stats()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Statistics:")
    print(f"  Total tokens processed: {stats['total_tokens']:,}")
    print(f"  Total time: {stats['total_time']:.1f}s")
    print(f"  Tokens/sec: {stats['tokens_per_sec']:.0f}")
    print(f"  Samples/sec: {stats['samples_per_sec']:.0f}")
    print(f"\nTraining Performance:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Best loss: {min(losses):.4f}")
    print(f"  Loss reduction: {losses[0] - losses[-1]:.4f}")

    # Save final results
    results = {
        "stats": stats,
        "losses": [float(x) for x in losses],
        "config": {
            "vocab_size": 1024,
            "dim": 512,
            "seq_len": 256,
            "batch_size": 8,
            "accum_steps": 20,
        },
    }

    with open("longrun_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: longrun_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
