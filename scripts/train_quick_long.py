#!/usr/bin/env python3
"""
Quick Long-Run Training Test

Processes substantial data but completes quickly.
"""

import numpy as np
import struct
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


def main():
    print("=" * 70)
    print("Quick Long-Run Training Test")
    print("=" * 70)

    # Load data
    data_file = "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"

    if os.path.exists(data_file):
        print(f"Loading {data_file}...")
        with open(data_file, "rb") as f:
            data = f.read()
        tokens = struct.unpack(f"{len(data) // 2}H", data)
        tokens = np.array(tokens, dtype=np.int32)
        tokens = np.clip(tokens, 0, 1023)
        print(f"Loaded {len(tokens):,} tokens")
    else:
        print("Using synthetic data...")
        tokens = np.random.randint(0, 1024, size=10000000, dtype=np.int32)

    # Initialize
    print("\nInitializing...")
    get_bridge()

    vocab_size = 1024
    dim = 512
    seq_len = 256

    embed = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02
    layer = ANEConv1x1(dim, dim, seq_len)

    m = np.zeros_like(layer.weight)
    v = np.zeros_like(layer.weight)
    t = 0

    # Training params
    batch_size = 8
    accum_steps = 20
    num_updates = 50  # 50 updates = 1000 forward passes
    lr = 0.001

    print(f"\nTraining {num_updates} updates...")
    print(f"  Batch size: {batch_size}")
    print(f"  Accum steps: {accum_steps}")
    print(f"  Effective batch: {batch_size * accum_steps}")
    print("-" * 70)

    losses = []
    data_idx = 0
    total_tokens = 0
    start_time = time.time()

    for update_idx in range(num_updates):
        update_start = time.time()

        grad_accum = np.zeros_like(layer.weight)
        update_loss = 0

        for accum_idx in range(accum_steps):
            if data_idx + batch_size * seq_len + 1 > len(tokens):
                data_idx = 0

            batch_tokens = tokens[data_idx : data_idx + batch_size * seq_len + 1]
            data_idx += batch_size * seq_len

            inputs = batch_tokens[:-1].reshape(batch_size, seq_len)
            targets = batch_tokens[1:].reshape(batch_size, seq_len)

            for b in range(batch_size):
                # Forward
                x = embed[inputs[b]]
                x = x.transpose(1, 0).reshape(1, dim, 1, seq_len)
                out = layer.forward(x)

                # Loss
                target_embed = embed[targets[b]]
                target_reshaped = target_embed.transpose(1, 0).reshape(
                    1, dim, 1, seq_len
                )
                loss = np.mean((out - target_reshaped) ** 2)
                update_loss += loss

                # Backward
                dy = 2 * (out - target_reshaped) / np.prod(out.shape)
                x_r = x.transpose(0, 3, 1, 2).reshape(-1, dim)
                dy_r = dy.transpose(0, 3, 1, 2).reshape(-1, dim)
                dW = (dy_r.T @ x_r).reshape(dim, dim, 1, 1) / seq_len
                grad_accum += dW

                total_tokens += seq_len

        # Update
        t += 1
        grad_mean = grad_accum / accum_steps / batch_size
        m = 0.9 * m + 0.1 * grad_mean
        v = 0.999 * v + 0.001 * (grad_mean**2)
        m_hat = m / (1 - 0.9**t)
        v_hat = v / (1 - 0.999**t)

        new_w = layer.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        layer.update_weights(new_w)

        update_time = time.time() - update_start
        avg_loss = update_loss / (accum_steps * batch_size)
        losses.append(avg_loss)

        if (update_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            print(
                f"  Update {update_idx + 1:2d}/{num_updates} | "
                f"Loss: {avg_loss:.4f} | "
                f"Tokens: {total_tokens:,} | "
                f"Speed: {tokens_per_sec:.0f} tok/s"
            )

    # Summary
    total_elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total updates: {num_updates}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Tokens/sec: {total_tokens / total_elapsed:.0f}")
    print(f"\nLoss:")
    print(f"  Initial: {losses[0]:.4f}")
    print(f"  Final: {losses[-1]:.4f}")
    print(f"  Best: {min(losses):.4f}")
    print(f"  Reduction: {losses[0] - losses[-1]:.4f}")

    if losses[-1] < losses[0]:
        print("\n✅ Model is learning!")
    else:
        print("\n⚠️  Loss stable (may need more steps)")

    print("=" * 70)


if __name__ == "__main__":
    main()
