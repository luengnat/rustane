#!/usr/bin/env python3
"""
Simple working training - match parameter-golf results
Uses real data and proper gradient updates
"""

import numpy as np
import time
from pathlib import Path
import glob

# Config matching parameter-golf
VOCAB_SIZE = 57601
DIM = 512
NUM_LAYERS = 9
NUM_HEADS = 8
NUM_KV_HEADS = 4
HEAD_DIM = 64
MLP_HIDDEN = 1024


def load_data(filepath, seq_len=1024):
    """Load .bin file."""
    data = np.fromfile(filepath, dtype=np.uint16)
    num_tokens = (len(data) // seq_len) * seq_len
    data = data[:num_tokens]
    return data.reshape(-1, seq_len)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SimpleModel:
    """Simplified model that actually trains."""

    def __init__(self):
        # Embedding
        self.embed = np.random.randn(VOCAB_SIZE, DIM).astype(np.float32) * 0.02

        # One big weight matrix for all layers (simplified)
        self.W = np.random.randn(DIM, VOCAB_SIZE).astype(np.float32) * 0.02

        # Adam state
        self.m_embed = np.zeros_like(self.embed)
        self.v_embed = np.zeros_like(self.embed)
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.t = 0

    def forward(self, inputs):
        """Forward: embed -> project -> softmax"""
        # Embed
        h = self.embed[inputs]  # (batch, seq, dim)

        # Project to vocab
        logits = h @ self.W  # (batch, seq, vocab)

        return logits, h

    def compute_loss(self, logits, targets):
        """Cross-entropy."""
        batch, seq, vocab = logits.shape

        logits_flat = logits.reshape(-1, vocab)
        targets_flat = targets.reshape(-1)

        # Softmax
        probs = softmax(logits_flat)

        # Loss
        nll = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        loss = np.mean(nll)

        # Gradients
        grad_logits = probs.copy()
        grad_logits[np.arange(len(targets_flat)), targets_flat] -= 1
        grad_logits /= batch * seq

        return loss, grad_logits.reshape(batch, seq, vocab)

    def train_step(self, inputs, targets, lr=0.001):
        """Train step with Adam."""
        self.t += 1

        # Forward
        logits, h = self.forward(inputs)
        loss, grad_logits = self.compute_loss(logits, targets)

        # Gradients
        grad_embed = grad_logits @ self.W.T
        grad_W = h.reshape(-1, DIM).T @ grad_logits.reshape(-1, VOCAB_SIZE)

        # Accumulate embed grads
        grad_embed_accum = np.zeros_like(self.embed)
        for b in range(inputs.shape[0]):
            for s in range(inputs.shape[1]):
                grad_embed_accum[inputs[b, s]] += grad_embed[b, s]

        # Adam update for embed
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m_embed = beta1 * self.m_embed + (1 - beta1) * grad_embed_accum
        self.v_embed = beta2 * self.v_embed + (1 - beta2) * (grad_embed_accum**2)
        m_hat = self.m_embed / (1 - beta1**self.t)
        v_hat = self.v_embed / (1 - beta2**self.t)
        self.embed -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Adam update for W
        self.m_W = beta1 * self.m_W + (1 - beta1) * grad_W
        self.v_W = beta2 * self.v_W + (1 - beta2) * (grad_W**2)
        m_hat = self.m_W / (1 - beta1**self.t)
        v_hat = self.v_W / (1 - beta2**self.t)
        self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)

        return loss


def main():
    print("=" * 60)
    print("Simple Training - Test if Loss Decreases")
    print("=" * 60)

    # Load data
    data_dir = Path("~/dev/parameter-golf/data/datasets/fineweb10B_sp1024").expanduser()
    train_files = sorted(glob.glob(str(data_dir / "fineweb_train_*.bin")))

    print(f"\nLoading {train_files[0]}...")
    train_tokens = load_data(train_files[0], 128)
    print(f"Sequences: {len(train_tokens)}")

    # Model
    print("\nInitializing model...")
    model = SimpleModel()
    print(f"Parameters: {model.embed.size + model.W.size:,}")

    # Train
    print("\nTraining...")
    losses = []
    start = time.time()

    for step in range(100):
        idx = step % (len(train_tokens) // 2)
        batch = train_tokens[idx * 2 : (idx + 1) * 2]

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        loss = model.train_step(inputs, targets, lr=0.01)
        losses.append(loss)

        if (step + 1) % 20 == 0:
            recent = np.mean(losses[-20:])
            print(f"  Step {step + 1}: Loss = {recent:.4f}")

    train_time = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {np.mean(losses[-10:]):.4f}")
    print(f"  Time: {train_time:.2f}s")
    print(f"{'=' * 60}")

    if losses[0] > np.mean(losses[-10:]):
        print("✅ Loss is decreasing! Training works.")
    else:
        print("⚠️  Loss not decreasing - need to check gradients")


if __name__ == "__main__":
    main()
