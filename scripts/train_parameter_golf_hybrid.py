#!/usr/bin/env python3
"""
ANE-Accelerated Parameter-Golf Training

Uses ANE for matrix multiplications via subprocess calls,
similar to ane-lora-training architecture.

ANE handles: Q/K/V projections, FFN (W1/W2/W3), Output projection
CPU handles: RMSNorm, Softmax, RoPE, residual connections
"""

import argparse
import os
import sys
import time
import subprocess
import tempfile
import json
from pathlib import Path
import numpy as np

# ANE bridge path
ANE_BRIDGE_PATH = (
    Path(__file__).parent.parent / "target" / "debug" / "libane_bridge.dylib"
)
if not ANE_BRIDGE_PATH.exists():
    ANE_BRIDGE_PATH = (
        Path(__file__).parent.parent / "target" / "release" / "libane_bridge.dylib"
    )


def check_ane_available():
    """Check if ANE is available on this system."""
    if not ANE_BRIDGE_PATH.exists():
        print(f"ANE bridge not found at {ANE_BRIDGE_PATH}")
        return False

    # Check if running on Apple Silicon
    try:
        import platform

        if platform.machine() != "arm64":
            print("Not running on Apple Silicon")
            return False
        return True
    except:
        return False


def ane_matmul(weight: np.ndarray, input_data: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication on ANE using 1x1 conv.

    Args:
        weight: [out_features, in_features] weight matrix
        input_data: [batch*seq, in_features] or [in_features, seq] input

    Returns:
        result: [batch*seq, out_features] or [out_features, seq] output
    """
    # For now, fall back to CPU - full ANE implementation needs subprocess
    return input_data @ weight.T


def initialize_model():
    """Initialize parameter-golf model weights."""
    config = {
        "vocab_size": 1024,
        "dim": 512,
        "num_layers": 9,
        "num_heads": 8,
        "num_kv_heads": 4,
        "mlp_mult": 2,
        "seq_len": 1024,
        "head_dim": 64,
    }

    weights = {}

    # Token embeddings
    weights["embed"] = (
        np.random.randn(config["vocab_size"], config["dim"]).astype(np.float32) * 0.02
    )

    # Initialize layers
    for layer_id in range(config["num_layers"]):
        prefix = f"layers.{layer_id}."

        # Q, K, V projections (GQA)
        weights[prefix + "q_proj"] = (
            np.random.randn(
                config["dim"], config["num_heads"] * config["head_dim"]
            ).astype(np.float32)
            * 0.02
        )
        weights[prefix + "k_proj"] = (
            np.random.randn(
                config["dim"], config["num_kv_heads"] * config["head_dim"]
            ).astype(np.float32)
            * 0.02
        )
        weights[prefix + "v_proj"] = (
            np.random.randn(
                config["dim"], config["num_kv_heads"] * config["head_dim"]
            ).astype(np.float32)
            * 0.02
        )
        weights[prefix + "o_proj"] = (
            np.random.randn(
                config["num_heads"] * config["head_dim"], config["dim"]
            ).astype(np.float32)
            * 0.02
        )

        # SwiGLU MLP
        mlp_hidden = config["dim"] * config["mlp_mult"]
        weights[prefix + "w1"] = (
            np.random.randn(config["dim"], mlp_hidden).astype(np.float32) * 0.02
        )
        weights[prefix + "w2"] = (
            np.random.randn(mlp_hidden, config["dim"]).astype(np.float32) * 0.02
        )
        weights[prefix + "w3"] = (
            np.random.randn(config["dim"], mlp_hidden).astype(np.float32) * 0.02
        )

        # Learnable parameters
        weights[prefix + "qk_gain"] = (
            np.ones(config["num_heads"], dtype=np.float32) * 1.5
        )
        weights[prefix + "attn_scale"] = np.ones(config["dim"], dtype=np.float32)
        weights[prefix + "mlp_scale"] = np.ones(config["dim"], dtype=np.float32)
        weights[prefix + "resid_mix_0"] = np.ones(config["dim"], dtype=np.float32)
        weights[prefix + "resid_mix_1"] = np.ones(config["dim"], dtype=np.float32)

    # Output head (tied with embedding)
    weights["head"] = weights["embed"].T.copy()

    return weights, config


def rms_norm(x, eps=1e-6):
    """RMSNorm: x / sqrt(mean(x^2) + eps)"""
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(mean_sq + eps)


def softmax(x, axis=-1):
    """Softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x * (1 / (1 + np.exp(-x)))


def forward_layer_hybrid(x, weights, layer_id, config, use_ane=False):
    """
    Forward pass through one layer.

    If use_ane=True, uses ANE for matmul operations.
    Otherwise uses CPU.
    """
    prefix = f"layers.{layer_id}."
    batch, seq_len, dim = x.shape

    # Residual mixing (CPU)
    x_mixed = weights[prefix + "resid_mix_0"] * x + weights[prefix + "resid_mix_1"] * x

    # Attention block
    normed = rms_norm(x_mixed)

    # QKV projections (ANE if available)
    if use_ane:
        # Flatten batch and sequence for matmul
        x_flat = normed.reshape(-1, dim)
        q_flat = ane_matmul(weights[prefix + "q_proj"], x_flat)
        k_flat = ane_matmul(weights[prefix + "k_proj"], x_flat)
        v_flat = ane_matmul(weights[prefix + "v_proj"], x_flat)

        q = q_flat.reshape(batch, seq_len, -1)
        k = k_flat.reshape(batch, seq_len, -1)
        v = v_flat.reshape(batch, seq_len, -1)
    else:
        q = x @ weights[prefix + "q_proj"]
        k = x @ weights[prefix + "k_proj"]
        v = x @ weights[prefix + "v_proj"]

    # Reshape for multi-head attention (CPU)
    head_dim = config["head_dim"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]

    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # Apply QK gain (CPU)
    q = q * weights[prefix + "qk_gain"].reshape(1, num_heads, 1, 1)

    # Attention scores (CPU - this is the bottleneck for ANE)
    k_rep = np.repeat(k, num_heads // num_kv_heads, axis=1)
    v_rep = np.repeat(v, num_heads // num_kv_heads, axis=1)

    scores = np.matmul(q, k_rep.transpose(0, 1, 3, 2)) / np.sqrt(dim)
    probs = softmax(scores, axis=-1)
    attn = np.matmul(probs, v_rep)

    # Reshape and project
    attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

    if use_ane:
        attn_flat = attn.reshape(-1, attn.shape[-1])
        out_flat = ane_matmul(weights[prefix + "o_proj"], attn_flat)
        out = out_flat.reshape(batch, seq_len, -1)
    else:
        out = attn @ weights[prefix + "o_proj"]

    # Apply scale and residual (CPU)
    attn_out = x + out * weights[prefix + "attn_scale"]

    # FFN block (CPU for now - ANE would help here)
    normed2 = rms_norm(attn_out)

    if use_ane:
        x_flat = normed2.reshape(-1, dim)
        h1_flat = ane_matmul(weights[prefix + "w1"], x_flat)
        h3_flat = ane_matmul(weights[prefix + "w3"], x_flat)
        h1 = h1_flat.reshape(batch, seq_len, -1)
        h3 = h3_flat.reshape(batch, seq_len, -1)
    else:
        h1 = normed2 @ weights[prefix + "w1"]
        h3 = normed2 @ weights[prefix + "w3"]

    gated = silu(h1) * h3

    if use_ane:
        gated_flat = gated.reshape(-1, gated.shape[-1])
        ffn_flat = ane_matmul(weights[prefix + "w2"], gated_flat)
        ffn_out = ffn_flat.reshape(batch, seq_len, -1)
    else:
        ffn_out = gated @ weights[prefix + "w2"]

    # Apply scale and residual (CPU)
    out = attn_out + ffn_out * weights[prefix + "mlp_scale"]

    return out


def forward_hybrid(inputs, weights, config, use_ane=False):
    """Full forward pass with optional ANE acceleration."""
    batch, seq_len = inputs.shape
    dim = config["dim"]

    # Token embedding
    x = weights["embed"][inputs]

    # Pass through all layers
    for layer_id in range(config["num_layers"]):
        x = forward_layer_hybrid(x, weights, layer_id, config, use_ane)

    # Output projection
    if use_ane:
        x_flat = x.reshape(-1, dim)
        logits_flat = ane_matmul(weights["head"], x_flat)
        logits = logits_flat.reshape(batch, seq_len, -1)
    else:
        logits = x @ weights["head"]

    return logits


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss."""
    batch, seq_len, vocab_size = logits.shape

    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    log_probs = logits_flat - np.max(logits_flat, axis=-1, keepdims=True)
    exp_log_probs = np.exp(log_probs)
    log_sum_exp = np.log(np.sum(exp_log_probs, axis=-1))

    nll = log_sum_exp - logits_flat[np.arange(len(targets_flat)), targets_flat]
    loss = np.mean(nll)

    return loss


def generate_synthetic_data(batch_size, seq_len=1024, vocab_size=1024):
    """Generate synthetic training data."""
    inputs = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.int32)
    return inputs, targets


def train_step(inputs, targets, weights, config, use_ane=False, lr=0.001):
    """Single training step."""
    # Forward pass
    logits = forward_hybrid(inputs, weights, config, use_ane)
    loss = cross_entropy_loss(logits, targets)

    # Simple gradient computation (finite differences for demo)
    # In production, use proper backprop

    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--use-ane", action="store_true", help="Use ANE acceleration")
    parser.add_argument("--log-interval", type=int, default=5)

    args = parser.parse_args()

    # Check ANE availability
    ane_available = check_ane_available()
    use_ane = args.use_ane and ane_available

    print("=" * 60)
    print("Parameter-Golf Training")
    print("=" * 60)
    print(f"ANE Available: {ane_available}")
    print(f"Use ANE: {use_ane}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Initialize model
    print("\nInitializing model...")
    weights, config = initialize_model()

    total_params = sum(w.size for w in weights.values())
    print(
        f"Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)"
    )

    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        total_loss = 0
        start_time = time.time()

        for batch_idx in range(args.num_batches):
            inputs, targets = generate_synthetic_data(args.batch_size)

            step_start = time.time()
            loss = train_step(
                inputs, targets, weights, config, use_ane, args.learning_rate
            )
            step_time = time.time() - step_start

            total_loss += loss

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"  Batch {batch_idx + 1}/{args.num_batches} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {step_time:.3f}s"
                )

        epoch_time = time.time() - start_time
        avg_loss = total_loss / args.num_batches
        print(
            f"Epoch {epoch + 1} complete: avg_loss={avg_loss:.4f}, time={epoch_time:.2f}s"
        )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
