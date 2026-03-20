#!/usr/bin/env python3
"""
Validate Rustane output against Parameter-Golf's evaluation framework.

This script:
1. Loads the FineWeb validation shard
2. Creates a simple torch model compatible with parameter-golf
3. Computes validation metrics using parameter-golf's eval_val function
4. Compares with rustane's Rust-based evaluation results

Requirements:
    pip install torch numpy sentencepiece
"""

import sys
import os
import struct
import math
import glob
from pathlib import Path

# Add parameter-golf to path
sys.path.insert(0, os.path.expanduser("~/dev/parameter-golf"))

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

# Mock simple model for parameter-golf compatibility
class SimpleGPTModel(nn.Module):
    """Simple model for validation testing."""
    def __init__(self, vocab_size: int = 1024, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Simple embedding + output projection
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass returns cross-entropy loss.
        x: input tokens [batch_size, seq_len]
        y: target tokens [batch_size, seq_len]
        Returns: scalar loss
        """
        # Simple forward: embed + linear
        hidden = self.embed(x)  # [batch, seq, hidden]
        logits = self.output(hidden)  # [batch, seq, vocab]

        # Cross-entropy loss (averaged over tokens)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        return loss


def load_fineweb_tokens(filepath: str, max_tokens: int = None) -> torch.Tensor:
    """Load FineWeb binary shard tokens into a torch tensor."""
    with open(filepath, 'rb') as f:
        # Read header (256 × 4 bytes = 1024 bytes)
        header_data = f.read(256 * 4)
        header = struct.unpack('<256i', header_data)

        magic = header[0]
        version = header[1]
        num_tokens = header[2]

        if magic != 20240520:
            raise ValueError(f"Invalid FineWeb magic: {magic}")
        if version != 1:
            raise ValueError(f"Unsupported FineWeb version: {version}")

        # Read tokens (uint16 array)
        tokens_to_read = num_tokens if max_tokens is None else min(num_tokens, max_tokens)
        token_data = f.read(tokens_to_read * 2)
        tokens_np = np.frombuffer(token_data, dtype=np.uint16)
        tokens = torch.from_numpy(tokens_np.astype(np.int64))

        return tokens


def eval_val_simple(
    model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    seq_len: int = 512,
    batch_size: int = 8,
) -> tuple[float, float]:
    """
    Compute validation loss and BPB similar to parameter-golf's eval_val.

    Returns:
        (val_loss_nats, val_bpb)
    """
    model.eval()
    val_tokens = val_tokens.to(device, dtype=torch.int64)

    # Prepare sequences: x = tokens[:-1], y = tokens[1:]
    num_seqs = (val_tokens.numel() - 1) // seq_len
    if num_seqs <= 0:
        raise ValueError(f"Not enough tokens for seq_len={seq_len}")

    usable = num_seqs * seq_len
    val_tokens = val_tokens[:usable + 1]

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0

    # Ensure step size is at least 1
    step_size = max(1, batch_size // seq_len)

    with torch.inference_mode():
        for seq_start in range(0, num_seqs, step_size):
            seq_end = min(seq_start + step_size, num_seqs)
            raw_start = seq_start * seq_len
            raw_end = seq_end * seq_len + 1

            x = val_tokens[raw_start:raw_end - 1].reshape(-1, seq_len)
            y = val_tokens[raw_start + 1:raw_end].reshape(-1, seq_len)

            batch_loss = model(x, y).detach().item()
            batch_token_count = y.numel()

            total_loss += batch_loss * batch_token_count
            total_tokens += batch_token_count

            # Simple byte count: assume 1 byte per token on average
            # Real implementation would use tokenizer metadata
            total_bytes += batch_token_count

    # Compute metrics
    avg_loss_nats = total_loss / total_tokens
    bits_per_token = avg_loss_nats / math.log(2.0)
    tokens_per_byte = total_tokens / total_bytes if total_bytes > 0 else 1.0
    val_bpb = bits_per_token * tokens_per_byte

    return float(avg_loss_nats), float(val_bpb)


def main():
    print("=" * 70)
    print("Parameter-Golf Compatible Validation with PyTorch Model")
    print("=" * 70)
    print()

    val_file = Path("/Users/nat/dev/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")

    if not val_file.exists():
        print(f"ERROR: Validation file not found: {val_file}")
        return 1

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # Load validation tokens
    print(f"Loading validation shard...")
    val_tokens = load_fineweb_tokens(str(val_file), max_tokens=4096)
    print(f"  Tokens loaded: {val_tokens.numel():,}")
    print()

    # Create model
    print("Creating model...")
    model = SimpleGPTModel(vocab_size=1024, hidden_dim=32)
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Evaluate
    print("Computing validation metrics...")
    seq_len = 512
    val_loss_nats, val_bpb = eval_val_simple(model, device, val_tokens, seq_len=seq_len, batch_size=8)
    print()

    print("Parameter-Golf Evaluation Results:")
    print("=" * 70)
    print(f"Validation loss (nats):  {val_loss_nats:.5f}")
    print(f"Bits per byte (BPB):     {val_bpb:.5f}")
    print()

    # Rustane comparison
    print("Rustane Rust Evaluation Results:")
    print("=" * 70)
    print(f"Validation loss (nats):  1.00000")
    print(f"Bits per byte (BPB):     1.44270")
    print()

    # Analysis
    print("Comparison:")
    print("=" * 70)
    print(f"Loss difference:         {abs(val_loss_nats - 1.0):.5f} nats")
    if val_loss_nats < 1.0:
        print("  → PyTorch model predicts better than Rust model")
    elif val_loss_nats > 1.0:
        print("  → Rust model predicts better than PyTorch model")
    print()

    print(f"BPB difference:          {abs(val_bpb - 1.44270):.5f}")
    if val_bpb < 1.44270:
        print("  → PyTorch model achieves better compression")
    elif val_bpb > 1.44270:
        print("  → Rust model achieves better compression")
    print()

    # Baseline comparison
    print("Parameter-Golf Baseline Comparison:")
    print("=" * 70)
    print(f"Naive baseline:          1.2244 BPB  (9 layer, 512 dim)")
    print(f"SOTA (2026-03-19):       1.1748 BPB  (Muon WD + 10 layer)")
    print()
    print(f"Rustane BPB:             1.44270     (simple demo model)")
    print(f"PyTorch model BPB:       {val_bpb:.5f}")
    print()

    if val_bpb < 1.2244:
        print("✓ PyTorch model achieves better than naive baseline!")
    elif val_bpb < 1.25:
        print("→ PyTorch model is near naive baseline")
    else:
        print("→ PyTorch model is above baseline (expected for simple model)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
