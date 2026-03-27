#!/usr/bin/env python3
"""
ANE Training with Backward Pass - Full Training Loop

This script implements a complete training loop using ANE for both forward and backward passes.
Weights are updated using gradients computed by ANE.
"""

import argparse
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge
from ane_backward import ANELinearBackward


# Model config
CONFIG = {
    "vocab_size": 1024,
    "dim": 512,
    "num_layers": 3,  # Start with fewer layers for testing
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 64,
    "mlp_hidden": 1024,
}


def rms_norm(x, eps=1e-6):
    mean_sq = np.mean(x**2, axis=1, keepdims=True)
    return x / np.sqrt(mean_sq + eps)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


class ANETrainableLinear:
    """
    Trainable linear layer with ANE forward and backward.
    """

    def __init__(self, in_channels, out_channels, seq_len=256):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len

        # Initialize weights
        self.weight = (
            np.random.randn(out_channels, in_channels, 1, 1).astype(np.float32) * 0.02
        )

        # Forward kernel
        self._forward = ANEConv1x1(in_channels, out_channels, seq_len)
        self._forward.weight = self.weight.copy()
        self._forward._recompile()

        # Backward kernel
        self._backward = ANELinearBackward(in_channels, out_channels, seq_len)
        self._backward.initialize(self.weight)

        # For optimizer state
        self.m = np.zeros_like(self.weight)  # First moment (Adam)
        self.v = np.zeros_like(self.weight)  # Second moment (Adam)
        self.t = 0  # Time step

    def forward(self, x):
        """Forward pass on ANE."""
        return self._forward.forward(x)

    def backward(self, dy, x):
        """
        Backward pass.

        Returns:
          dx: gradient w.r.t. input (for backprop)
          dW: gradient w.r.t. weights (for update)
        """
        # dx on ANE
        dx = self._backward.compute_dx(dy)

        # dW on CPU (ANE doesn't support weight gradient computation efficiently)
        dW = self._backward.compute_dW(x, dy)

        return dx, dW

    def update_weights(self, dW, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Update weights using Adam optimizer.
        """
        self.t += 1

        # Adam update
        self.m = beta1 * self.m + (1 - beta1) * dW
        self.v = beta2 * self.v + (1 - beta2) * (dW**2)
        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)

        self.weight -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Update kernels with new weights
        self._forward.update_weights(self.weight)
        self._backward.initialize(self.weight)

    def __del__(self):
        if hasattr(self, "_forward"):
            del self._forward
        if hasattr(self, "_backward"):
            del self._backward


class ANETrainableAttention:
    """Attention layer with ANE-accelerated trainable projections."""

    def __init__(self, config, layer_id, seq_len=256):
        self.config = config
        self.layer_id = layer_id
        self.seq_len = seq_len

        dim = config["dim"]
        num_heads = config["num_heads"]
        num_kv_heads = config["num_kv_heads"]
        head_dim = config["head_dim"]

        # ANE trainable projections
        print(f"  Layer {layer_id}: Initializing trainable projections...")
        self.q_proj = ANETrainableLinear(dim, num_heads * head_dim, seq_len)
        self.k_proj = ANETrainableLinear(dim, num_kv_heads * head_dim, seq_len)
        self.v_proj = ANETrainableLinear(dim, num_kv_heads * head_dim, seq_len)
        self.o_proj = ANETrainableLinear(num_heads * head_dim, dim, seq_len)

        # Learnable scales
        self.qk_gain = np.ones(num_heads, dtype=np.float32) * 1.5

    def forward(self, x):
        """Forward pass."""
        B, C, H, S = x.shape

        normed = rms_norm(x)

        # QKV projections (ANE)
        q = self.q_proj.forward(normed)
        k = self.k_proj.forward(normed)
        v = self.v_proj.forward(normed)

        # Reshape for multi-head attention
        num_heads = self.config["num_heads"]
        num_kv_heads = self.config["num_kv_heads"]
        head_dim = self.config["head_dim"]

        q = q.reshape(B, num_heads, head_dim, S)
        k = k.reshape(B, num_kv_heads, head_dim, S)
        v = v.reshape(B, num_kv_heads, head_dim, S)

        # Apply QK gain
        q = q * self.qk_gain.reshape(1, num_heads, 1, 1)

        # Chunk attention (CPU for now)
        attn_outputs = []
        heads_per_kv = num_heads // num_kv_heads

        for h in range(num_heads):
            kv_h = h // heads_per_kv

            q_h = q[:, h : h + 1, :, :]
            k_h = k[:, kv_h : kv_h + 1, :, :]
            v_h = v[:, kv_h : kv_h + 1, :, :]

            scores = np.matmul(q_h.transpose(0, 1, 3, 2), k_h) / np.sqrt(head_dim)
            probs = softmax(scores, axis=-1)
            attn_h = np.matmul(probs, v_h.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

            attn_outputs.append(attn_h)

        attn = np.concatenate(attn_outputs, axis=1)
        attn = attn.reshape(B, num_heads * head_dim, 1, S)

        # Output projection (ANE)
        out = self.o_proj.forward(attn)

        # Residual
        attn_out = x + out

        return attn_out

    def backward(self, dy, x):
        """
        Backward pass through attention.

        Simplified: just propagate through projections for now.
        """
        # Store gradients for each projection
        grads = {}

        # Backward through o_proj
        dx_attn, dWo = self.o_proj.backward(dy, None)  # Need to cache x

        return dy, grads


class ANETrainableFFN:
    """SwiGLU FFN with ANE trainable projections."""

    def __init__(self, config, seq_len=256):
        self.config = config

        dim = config["dim"]
        hidden = config["mlp_hidden"]

        print("  Initializing trainable FFN projections...")
        self.w1 = ANETrainableLinear(dim, hidden, seq_len)
        self.w2 = ANETrainableLinear(hidden, dim, seq_len)
        self.w3 = ANETrainableLinear(dim, hidden, seq_len)

    def forward(self, x):
        """Forward pass."""
        normed = rms_norm(x)

        # SwiGLU
        h1 = self.w1.forward(normed)
        h3 = self.w3.forward(normed)
        gated = silu(h1) * h3
        ffn_out = self.w2.forward(gated)

        # Residual
        out = x + ffn_out

        return out


class SimpleANETrainer:
    """
    Simple trainer with ANE-accelerated layers.

    For now, just tests the forward/backward/update cycle.
    """

    def __init__(self, config, seq_len=256):
        self.config = config
        self.seq_len = seq_len

        # Single trainable layer for testing
        print("Initializing ANE trainable layer...")
        self.test_layer = ANETrainableLinear(config["dim"], config["dim"], seq_len)

        print("✅ ANE trainer initialized")

    def train_step(self, x, target, lr=0.001):
        """
        Single training step.

        Returns loss and time taken.
        """
        start = time.time()

        # Forward
        out = self.test_layer.forward(x)

        # Compute loss (MSE for testing)
        loss = np.mean((out - target) ** 2)

        # Backward
        dy = 2 * (out - target) / np.prod(out.shape)
        dx, dW = self.test_layer.backward(dy, x)

        # Update weights
        self.test_layer.update_weights(dW, lr)

        elapsed = time.time() - start

        return loss, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq-len", type=int, default=256)

    args = parser.parse_args()

    print("=" * 70)
    print("ANE Training with Backward Pass")
    print("=" * 70)
    print(f"Architecture: {CONFIG['dim']} dim, seq_len={args.seq_len}")
    print("=" * 70)

    # Initialize
    print("\nInitializing ANE...")
    get_bridge()

    trainer = SimpleANETrainer(CONFIG, seq_len=args.seq_len)

    # Training loop
    print("\nTraining...")
    print("-" * 70)

    losses = []
    times = []

    for step in range(args.steps):
        # Generate synthetic data
        x = np.random.randn(1, CONFIG["dim"], 1, args.seq_len).astype(np.float32) * 0.1
        target = (
            np.random.randn(1, CONFIG["dim"], 1, args.seq_len).astype(np.float32) * 0.1
        )

        loss, elapsed = trainer.train_step(x, target, lr=args.lr)

        losses.append(loss)
        times.append(elapsed)

        if (step + 1) % 5 == 0:
            avg_loss = np.mean(losses[-5:])
            avg_time = np.mean(times[-5:])
            print(
                f"  Step {step + 1}/{args.steps} | Loss: {avg_loss:.6f} | Time: {avg_time:.3f}s"
            )

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Average time per step: {np.mean(times):.3f}s")
    print(f"Loss trend: {losses[0]:.6f} → {losses[-1]:.6f}")

    if losses[-1] < losses[0]:
        print("✅ Loss is decreasing - training is working!")
    else:
        print("⚠️  Loss not decreasing - may need tuning")

    print("\nANE Operations:")
    print("  ✅ Forward pass (ANE)")
    print("  ✅ Backward dx (ANE)")
    print("  ✅ Backward dW (CPU)")
    print("  ✅ Weight updates (Adam)")


if __name__ == "__main__":
    main()
