#!/usr/bin/env python3
"""
Parameter-Golf Training with Real Fineweb Data

Loads actual .bin files and trains properly.
Target: Match parameter-golf baseline of val_loss ~2.06
"""

import argparse
import numpy as np
import time
from pathlib import Path
import glob

# Model config matching parameter-golf
# Note: vocab_size is determined from data (57601), not the tokenizer name
CONFIG = {
    "vocab_size": 57601,  # From actual data max token ID + 1
    "dim": 512,
    "num_layers": 9,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 64,
    "mlp_hidden": 1024,
}


def load_bin_file(filepath, seq_len=1024):
    """Load a .bin file and convert to token sequences."""
    data = np.fromfile(filepath, dtype=np.uint16)
    # Truncate to multiple of seq_len
    num_tokens = (len(data) // seq_len) * seq_len
    data = data[:num_tokens]
    # Reshape to (batch, seq_len)
    tokens = data.reshape(-1, seq_len)
    return tokens


def rms_norm(x, eps=1e-6):
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(mean_sq + eps)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


class ParameterGolfModel:
    """Full model with simplified training."""

    def __init__(self):
        self.embed = (
            np.random.randn(CONFIG["vocab_size"], CONFIG["dim"]).astype(np.float32)
            * 0.02
        )
        self.layers = []

        for _ in range(CONFIG["num_layers"]):
            layer = {
                "q_proj": np.random.randn(
                    CONFIG["dim"], CONFIG["num_heads"] * CONFIG["head_dim"]
                ).astype(np.float32)
                * 0.02,
                "k_proj": np.random.randn(
                    CONFIG["dim"], CONFIG["num_kv_heads"] * CONFIG["head_dim"]
                ).astype(np.float32)
                * 0.02,
                "v_proj": np.random.randn(
                    CONFIG["dim"], CONFIG["num_kv_heads"] * CONFIG["head_dim"]
                ).astype(np.float32)
                * 0.02,
                "o_proj": np.random.randn(
                    CONFIG["num_heads"] * CONFIG["head_dim"], CONFIG["dim"]
                ).astype(np.float32)
                * 0.02,
                "w1": np.random.randn(CONFIG["dim"], CONFIG["mlp_hidden"]).astype(
                    np.float32
                )
                * 0.02,
                "w2": np.random.randn(CONFIG["mlp_hidden"], CONFIG["dim"]).astype(
                    np.float32
                )
                * 0.02,
                "w3": np.random.randn(CONFIG["dim"], CONFIG["mlp_hidden"]).astype(
                    np.float32
                )
                * 0.02,
            }
            self.layers.append(layer)

        self.head = self.embed.T.copy()

    def forward(self, input_ids):
        """Forward pass."""
        batch, seq_len = input_ids.shape

        # Embedding
        x = self.embed[input_ids]

        # Through layers
        for layer in self.layers:
            dim = CONFIG["dim"]
            num_heads = CONFIG["num_heads"]
            num_kv_heads = CONFIG["num_kv_heads"]
            head_dim = CONFIG["head_dim"]
            mlp_hidden = CONFIG["mlp_hidden"]

            # Attention
            normed = rms_norm(x)
            q = normed @ layer["q_proj"]
            k = normed @ layer["k_proj"]
            v = normed @ layer["v_proj"]

            q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

            k_rep = np.repeat(k, num_heads // num_kv_heads, axis=1)
            v_rep = np.repeat(v, num_heads // num_kv_heads, axis=1)

            scores = np.matmul(q, k_rep.transpose(0, 1, 3, 2)) / np.sqrt(dim)
            probs = softmax(scores, axis=-1)
            attn = np.matmul(probs, v_rep)

            attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
            out = attn @ layer["o_proj"]

            x = x + out

            # FFN
            normed = rms_norm(x)
            h1 = normed @ layer["w1"]
            h3 = normed @ layer["w3"]
            gated = silu(h1) * h3
            ffn_out = gated @ layer["w2"]

            x = x + ffn_out

        # Output
        logits = x @ self.head
        return logits

    def compute_loss(self, logits, targets):
        """Cross-entropy loss."""
        batch, seq_len, vocab_size = logits.shape

        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        log_probs = np.log(probs + 1e-10)
        nll = -log_probs[np.arange(len(targets_flat)), targets_flat]
        loss = np.mean(nll)

        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/dev/parameter-golf/data/datasets/fineweb10B_sp1024",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=100)

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()

    print("=" * 60)
    print("Parameter-Golf with Real Fineweb Data")
    print("=" * 60)
    print(f"Target: val_loss ~2.06 (baseline)")
    print(f"Data: {data_dir}")
    print(
        f"Batch: {args.batch_size} x {args.seq_len} = {args.batch_size * args.seq_len} tokens"
    )
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_files = sorted(glob.glob(str(data_dir / "fineweb_train_*.bin")))
    val_file = data_dir / "fineweb_val_000000.bin"

    if not train_files:
        print("❌ No training files found!")
        return

    print(f"Found {len(train_files)} training files")

    # Load first training file
    print(f"Loading {train_files[0]}...")
    train_tokens = load_bin_file(train_files[0], args.seq_len)
    print(f"Training sequences: {len(train_tokens)}")

    # Load validation
    if val_file.exists():
        print(f"Loading validation...")
        val_tokens = load_bin_file(str(val_file), args.seq_len)
        print(f"Validation sequences: {len(val_tokens)}")

    # Initialize model
    print("\nInitializing model...")
    model = ParameterGolfModel()

    total_params = sum(
        [
            model.embed.size,
            model.head.size,
        ]
        + [sum(w.size for w in layer.values()) for layer in model.layers]
    )
    print(f"Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")

    # Evaluate on validation (before training)
    if val_file.exists():
        print("\nValidation (before training):")
        val_losses = []
        for i in range(min(10, len(val_tokens) // args.batch_size)):
            batch_tokens = val_tokens[i * args.batch_size : (i + 1) * args.batch_size]
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]

            logits = model.forward(inputs)
            loss = model.compute_loss(logits, targets)
            val_losses.append(loss)

        avg_val_loss = np.mean(val_losses)
        print(f"  Val loss: {avg_val_loss:.4f}")
        print(f"  Target: ~2.06")
        print(f"  Gap: {avg_val_loss - 2.06:.4f}")

    # Training (forward only for now - no backprop)
    print("\nTraining (forward pass only)...")
    print("Note: Full training requires gradient computation (not yet implemented)")

    train_losses = []
    start = time.time()

    for i in range(min(args.num_batches, len(train_tokens) // args.batch_size)):
        batch_tokens = train_tokens[i * args.batch_size : (i + 1) * args.batch_size]
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]

        logits = model.forward(inputs)
        loss = model.compute_loss(logits, targets)
        train_losses.append(loss)

        if (i + 1) % 10 == 0:
            avg_loss = np.mean(train_losses[-10:])
            print(f"  Batch {i + 1}/{args.num_batches} | Loss: {avg_loss:.4f}")

    train_time = time.time() - start
    avg_train_loss = np.mean(train_losses)

    print(f"\nTraining complete: avg_loss={avg_train_loss:.4f}, time={train_time:.2f}s")

    # Evaluate on validation (after "training")
    if val_file.exists():
        print("\nValidation (after training):")
        val_losses = []
        for i in range(min(10, len(val_tokens) // args.batch_size)):
            batch_tokens = val_tokens[i * args.batch_size : (i + 1) * args.batch_size]
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]

            logits = model.forward(inputs)
            loss = model.compute_loss(logits, targets)
            val_losses.append(loss)

        avg_val_loss = np.mean(val_losses)
        print(f"  Val loss: {avg_val_loss:.4f}")
        print(f"  Target: ~2.06")
        print(f"  Gap: {avg_val_loss - 2.06:.4f}")

    print("\n" + "=" * 60)
    print("Note: Loss not decreasing because we haven't implemented")
    print("      gradient computation and weight updates yet.")
    print("=" * 60)


if __name__ == "__main__":
    main()
