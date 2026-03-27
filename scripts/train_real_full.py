#!/usr/bin/env python3
"""
Full ANE Training on Real FineWeb-10B Data

Complete training with forward, backward, and weight updates on real tokens.
"""

import numpy as np
import struct
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge
from ane_backward import ANELinearBackward


def load_tokens_bin(filepath, vocab_size=1024):
    """Load tokens from binary file."""
    print(f"Loading {filepath}...")

    with open(filepath, "rb") as f:
        data = f.read()

    # Parse as uint16
    num_tokens = len(data) // 2
    tokens = struct.unpack(f"{num_tokens}H", data[: num_tokens * 2])
    tokens = np.array(tokens, dtype=np.int32)

    print(f"  Loaded {len(tokens):,} tokens ({len(tokens) / 1e6:.1f}M)")
    print(f"  Token range: [{tokens.min()}, {tokens.max()}]")

    # Clamp to vocab_size
    tokens = np.clip(tokens, 0, vocab_size - 1)

    return tokens


class ANEMiniModel:
    """Minimal model for training on real data."""

    def __init__(self, vocab_size=1024, dim=512, seq_len=256):
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len

        print("\nInitializing model...")

        # Embeddings
        self.embed = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02

        # ANE projection layer
        print("  ANE projection layer...")
        self.proj = ANEConv1x1(dim, dim, seq_len)

        # Backward handler
        self.proj_backward = ANELinearBackward(dim, dim, seq_len)
        self.proj_backward.initialize(self.proj.weight)

        # Output head (reuse embeddings)
        self.head = self.embed.T.copy()

        # Adam state
        self.m = np.zeros_like(self.proj.weight)
        self.v = np.zeros_like(self.proj.weight)
        self.t = 0

        print(f"  Parameters: {(self.embed.size + self.proj.weight.size):,}")

    def forward(self, input_ids):
        """Forward pass."""
        batch_size = input_ids.shape[0]

        # Embedding
        x = self.embed[input_ids]
        x = x.transpose(0, 2, 1).reshape(batch_size, self.dim, 1, self.seq_len)

        # Store for backward
        self._input = x.copy()

        # ANE projection
        x = self.proj.forward(x)
        self._proj_out = x.copy()

        # Output projection (CPU)
        x_flat = x.transpose(0, 2, 3, 1).reshape(-1, self.dim)
        logits = x_flat @ self.head
        logits = logits.reshape(batch_size, self.seq_len, self.vocab_size)

        return logits

    def backward_and_update(self, logits, targets, lr=0.001):
        """Backward pass and weight update."""
        B, S, V = logits.shape

        # Softmax
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Gradient w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[np.arange(len(targets_flat)), targets_flat] -= 1
        grad_logits /= B * S

        # Backward through head (CPU)
        grad_x_flat = grad_logits @ self.head.T
        grad_x = grad_x_flat.reshape(B, S, self.dim).transpose(0, 2, 1)
        grad_x = grad_x.reshape(B, self.dim, 1, S)

        # Backward through projection (ANE for dx)
        grad_proj = self.proj_backward.compute_dx(grad_x)

        # Compute dW (CPU)
        x_reshaped = self._input.transpose(0, 3, 1, 2).reshape(-1, self.dim)
        grad_reshaped = grad_x.transpose(0, 3, 1, 2).reshape(-1, self.dim)
        dW = (grad_reshaped.T @ x_reshaped).reshape(self.dim, self.dim, 1, 1) / (B * S)

        # Adam update
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * dW
        self.v = 0.999 * self.v + 0.001 * (dW**2)
        m_hat = self.m / (1 - 0.9**self.t)
        v_hat = self.v / (1 - 0.999**self.t)

        new_weight = self.proj.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Update (this causes recompile)
        self.proj.update_weights(new_weight)
        self.proj_backward.initialize(new_weight)

        return True

    def compute_loss(self, logits, targets):
        """Cross-entropy loss."""
        B, S, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        nll = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        return np.mean(nll)


def train_with_accumulation(
    model, tokens, num_steps=100, batch_size=2, accum_steps=10, lr=0.001, seq_len=256
):
    """Train with gradient accumulation."""

    print("\n" + "=" * 70)
    print("Training with Gradient Accumulation")
    print("=" * 70)
    print(f"Steps: {num_steps}, Batch size: {batch_size}")
    print(f"Accumulation steps: {accum_steps}")
    print(f"Effective batch size: {batch_size * accum_steps}")

    losses = []
    times = []

    grad_accum = np.zeros_like(model.proj.weight)
    step_count = 0

    for step in range(num_steps):
        # Get batch
        start_idx = (step * batch_size * seq_len) % (
            len(tokens) - batch_size * seq_len - 1
        )
        batch_tokens = tokens[start_idx : start_idx + batch_size * seq_len + 1]

        inputs = batch_tokens[:-1].reshape(batch_size, seq_len)
        targets = batch_tokens[1:].reshape(batch_size, seq_len)

        # Training step
        t0 = time.time()

        # Forward
        logits = model.forward(inputs)
        loss = model.compute_loss(logits, targets)

        # Backward (compute gradients)
        B, S, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        grad_logits = probs.copy()
        grad_logits[np.arange(len(targets_flat)), targets_flat] -= 1
        grad_logits /= B * S

        grad_x_flat = grad_logits @ model.head.T
        grad_x = grad_x_flat.reshape(B, S, model.dim).transpose(0, 2, 1)
        grad_x = grad_x.reshape(B, model.dim, 1, S)

        grad_proj = model.proj_backward.compute_dx(grad_x)

        x_reshaped = model._input.transpose(0, 3, 1, 2).reshape(-1, model.dim)
        grad_reshaped = grad_x.transpose(0, 3, 1, 2).reshape(-1, model.dim)
        dW = (grad_reshaped.T @ x_reshaped).reshape(model.dim, model.dim, 1, 1) / (
            B * S
        )

        grad_accum += dW
        step_count += 1

        # Update weights every accum_steps
        if step_count % accum_steps == 0:
            model.t += 1
            model.m = 0.9 * model.m + 0.1 * grad_accum
            model.v = 0.999 * model.v + 0.001 * (grad_accum**2)
            m_hat = model.m / (1 - 0.9**model.t)
            v_hat = model.v / (1 - 0.999**model.t)

            new_weight = model.proj.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            model.proj.update_weights(new_weight)
            model.proj_backward.initialize(new_weight)

            grad_accum = np.zeros_like(model.proj.weight)

        elapsed = time.time() - t0

        losses.append(loss)
        times.append(elapsed)

        if (step + 1) % 20 == 0:
            avg_loss = np.mean(losses[-20:])
            avg_time = np.mean(times[-20:]) * 1000
            print(
                f"  Step {step + 1}/{num_steps} | Loss: {avg_loss:.4f} | Time: {avg_time:.1f}ms"
            )

    return losses, times


if __name__ == "__main__":
    data_file = "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"

    print("=" * 70)
    print("ANE Training on Real FineWeb-10B Data")
    print("=" * 70)

    # Initialize ANE
    print("\nInitializing ANE...")
    get_bridge()
    print("✅ ANE ready")

    # Load tokens
    tokens = load_tokens_bin(data_file, vocab_size=1024)

    # Create model
    model = ANEMiniModel(vocab_size=1024, dim=512, seq_len=256)

    # Train
    losses, times = train_with_accumulation(
        model,
        tokens,
        num_steps=100,
        batch_size=2,
        accum_steps=10,
        lr=0.001,
        seq_len=256,
    )

    # Summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Steps: {len(losses)}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Best loss: {min(losses):.4f}")
    print(f"Avg time: {np.mean(times) * 1000:.1f}ms/step")
    print(f"Tokens: {len(losses) * 2 * 256:,}")

    if losses[-1] < losses[0]:
        print("✅ Model is learning!")
    else:
        print("⚠️  Loss stable (may need more steps or LR tuning)")
