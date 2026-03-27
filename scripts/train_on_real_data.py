#!/usr/bin/env python3
"""
Train on Real FineWeb-10B Data

Loads tokenized binary files and trains the ANE-accelerated model.
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
    """
    Load tokens from binary file.

    Binary format: uint16 tokens (2 bytes each)
    Returns: numpy array of token IDs
    """
    print(f"Loading {filepath}...")

    with open(filepath, "rb") as f:
        data = f.read()

    # Parse as uint16 (2 bytes per token)
    num_tokens = len(data) // 2
    tokens = struct.unpack(f"{num_tokens}H", data[: num_tokens * 2])
    tokens = np.array(tokens, dtype=np.int32)

    print(f"  Loaded {len(tokens):,} tokens")
    print(f"  Token range: [{tokens.min()}, {tokens.max()}]")
    print(f"  Unique tokens: {len(np.unique(tokens))}")

    # Clamp to vocab_size if needed
    if tokens.max() >= vocab_size:
        print(f"  Warning: Clamping tokens to vocab_size {vocab_size}")
        tokens = np.clip(tokens, 0, vocab_size - 1)

    return tokens


def create_batches(tokens, batch_size, seq_len):
    """
    Create training batches from tokens.

    Returns: generator of (inputs, targets) batches
    """
    num_tokens = len(tokens)
    batch_len = batch_size * seq_len

    # Calculate number of complete batches
    num_batches = (num_tokens - 1) // batch_len

    print(
        f"Creating {num_batches} batches (batch_size={batch_size}, seq_len={seq_len})"
    )

    for i in range(num_batches):
        start = i * batch_len
        end = start + batch_len + 1  # +1 for target

        batch_tokens = tokens[start:end]

        # Reshape to [batch_size, seq_len]
        inputs = batch_tokens[:-1].reshape(batch_size, seq_len)
        targets = batch_tokens[1:].reshape(batch_size, seq_len)

        yield inputs, targets


class SimpleANEModel:
    """Simple model for testing on real data."""

    def __init__(self, vocab_size=1024, dim=512, seq_len=256):
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len

        # Embeddings
        self.embed = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02

        # Single ANE layer for testing
        print("Initializing ANE layer...")
        self.proj = ANEConv1x1(dim, dim, seq_len)

        # Output head
        self.head = self.embed.T.copy()

        print(f"Model: vocab={vocab_size}, dim={dim}, seq_len={seq_len}")

    def forward(self, input_ids):
        """Forward pass."""
        batch_size = input_ids.shape[0]

        # Embedding
        x = self.embed[input_ids]  # [B, S, D]
        x = x.transpose(0, 2, 1).reshape(batch_size, self.dim, 1, self.seq_len)

        # ANE projection
        x = self.proj.forward(x)

        # Output
        x = x.transpose(0, 2, 3, 1).reshape(-1, self.dim)
        logits = x @ self.head
        logits = logits.reshape(batch_size, self.seq_len, self.vocab_size)

        return logits

    def compute_loss(self, logits, targets):
        """Cross-entropy loss."""
        B, S, V = logits.shape

        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)

        # Softmax
        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # NLL
        nll = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        return np.mean(nll)


def train_on_file(data_file, num_batches=100, batch_size=2):
    """Train on a single data file."""
    print("=" * 70)
    print("ANE Training on Real FineWeb Data")
    print("=" * 70)
    print(f"Data file: {data_file}")

    # Initialize ANE
    print("\nInitializing ANE...")
    get_bridge()

    # Load tokens
    tokens = load_tokens_bin(data_file, vocab_size=1024)

    # Create model
    model = SimpleANEModel(vocab_size=1024, dim=512, seq_len=256)

    # Create batches
    batch_generator = create_batches(tokens, batch_size, seq_len=256)

    # Training loop
    print("\nTraining...")
    print("-" * 70)

    losses = []
    times = []

    for i, (inputs, targets) in enumerate(batch_generator):
        if i >= num_batches:
            break

        start = time.time()

        # Forward
        logits = model.forward(inputs)
        loss = model.compute_loss(logits, targets)

        elapsed = time.time() - start

        losses.append(loss)
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            avg_time = np.mean(times[-10:]) * 1000
            print(
                f"  Batch {i + 1}/{num_batches} | Loss: {avg_loss:.4f} | Time: {avg_time:.1f}ms"
            )

    # Summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Batches processed: {len(losses)}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Best loss: {min(losses):.4f}")
    print(f"Average time: {np.mean(times) * 1000:.1f}ms/batch")
    print(f"Tokens processed: {len(losses) * batch_size * 256:,}")

    return losses, times


if __name__ == "__main__":
    data_file = "/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin"

    losses, times = train_on_file(data_file, num_batches=50, batch_size=2)
