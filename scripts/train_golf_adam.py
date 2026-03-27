#!/usr/bin/env python3
"""
Working Parameter-Golf Training with Adam Optimizer

Uses Adam (simpler than Muon) to train down to target loss.
This is a first step - can switch to Muon+Adam hybrid later.
"""

import argparse
import numpy as np
import time
from pathlib import Path
import glob

CONFIG = {
    "vocab_size": 57601,
    "dim": 512,
    "num_layers": 9,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 64,
    "mlp_hidden": 1024,
}


def rms_norm(x, eps=1e-6):
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(mean_sq + eps)


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


class AdamOptimizer:
    """Simple Adam optimizer."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}  # First moment
        self.v = {}  # Second moment

    def step(self, params, grads):
        """Single Adam step."""
        self.t += 1

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            # Update biased first moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update biased second moment
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (
                grads[key] ** 2
            )

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class ParameterGolfLayer:
    """Transformer layer with gradients."""

    def __init__(self, layer_id):
        self.layer_id = layer_id
        dim = CONFIG["dim"]
        num_heads = CONFIG["num_heads"]
        num_kv_heads = CONFIG["num_kv_heads"]
        head_dim = CONFIG["head_dim"]
        mlp_hidden = CONFIG["mlp_hidden"]

        # Weights
        self.q_proj = (
            np.random.randn(dim, num_heads * head_dim).astype(np.float32) * 0.02
        )
        self.k_proj = (
            np.random.randn(dim, num_kv_heads * head_dim).astype(np.float32) * 0.02
        )
        self.v_proj = (
            np.random.randn(dim, num_kv_heads * head_dim).astype(np.float32) * 0.02
        )
        self.o_proj = (
            np.random.randn(num_heads * head_dim, dim).astype(np.float32) * 0.02
        )
        self.w1 = np.random.randn(dim, mlp_hidden).astype(np.float32) * 0.02
        self.w2 = np.random.randn(mlp_hidden, dim).astype(np.float32) * 0.02
        self.w3 = np.random.randn(dim, mlp_hidden).astype(np.float32) * 0.02

        # Scales
        self.qk_gain = np.ones(num_heads, dtype=np.float32) * 1.5
        self.attn_scale = np.ones(dim, dtype=np.float32)
        self.mlp_scale = np.ones(dim, dtype=np.float32)
        self.resid_mix_0 = np.ones(dim, dtype=np.float32)
        self.resid_mix_1 = np.ones(dim, dtype=np.float32)

    def get_params(self):
        """Get all parameters."""
        return {
            "q_proj": self.q_proj,
            "k_proj": self.k_proj,
            "v_proj": self.v_proj,
            "o_proj": self.o_proj,
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
        }

    def forward(self, x):
        """Forward with gradient tracking (simplified)."""
        batch, seq_len, dim = x.shape

        # Store for backward
        self.x = x.copy()

        # Attention
        normed = rms_norm(x)
        self.normed = normed.copy()

        q = normed @ self.q_proj
        k = normed @ self.k_proj
        v = normed @ self.v_proj

        self.q = q.copy()
        self.k = k.copy()
        self.v = v.copy()

        # Multi-head attention
        head_dim = CONFIG["head_dim"]
        num_heads = CONFIG["num_heads"]
        num_kv_heads = CONFIG["num_kv_heads"]

        q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = q * self.qk_gain.reshape(1, num_heads, 1, 1)

        k_rep = np.repeat(k, num_heads // num_kv_heads, axis=1)
        v_rep = np.repeat(v, num_heads // num_kv_heads, axis=1)

        scores = np.matmul(q, k_rep.transpose(0, 1, 3, 2)) / np.sqrt(dim)
        self.probs = softmax(scores, axis=-1)
        attn = np.matmul(self.probs, v_rep)

        attn = attn.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        self.attn_flat = attn.copy()

        out = attn @ self.o_proj
        self.out_proj = out.copy()

        x = x + out
        self.attn_out = x.copy()

        # FFN
        normed = rms_norm(x)
        self.normed2 = normed.copy()

        h1 = normed @ self.w1
        h3 = normed @ self.w3
        self.h1 = h1.copy()
        self.h3 = h3.copy()

        gated = silu(h1) * h3
        self.gated = gated.copy()

        ffn_out = gated @ self.w2
        self.ffn_out = ffn_out.copy()

        x = x + ffn_out

        return x

    def backward(self, grad_output):
        """Simplified backward pass."""
        grads = {}
        batch, seq_len, dim = grad_output.shape

        # Output projection gradient: (attn_flat^T @ grad_output)
        grads["o_proj"] = self.attn_flat.reshape(
            -1, self.attn_flat.shape[-1]
        ).T @ grad_output.reshape(-1, CONFIG["dim"])

        # FFN gradients
        # w2: (gated^T @ grad_output)
        grads["w2"] = self.gated.reshape(
            -1, self.gated.shape[-1]
        ).T @ grad_output.reshape(-1, CONFIG["dim"])

        # Gradient through w2
        grad_gated = grad_output @ self.w2.T  # (batch, seq, dim)

        # Through SwiGLU: grad_gated * h3 (for h1 path) and grad_gated * silu(h1) (for h3 path)
        grad_h1 = grad_gated * self.h3  # (batch, seq, mlp_hidden)
        grad_h3 = grad_gated * silu(self.h1)  # (batch, seq, mlp_hidden)

        # Through SiLU: grad * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        # Simplified: just use sigmoid derivative
        sig_h1 = 1 / (1 + np.exp(-self.h1))
        grad_h1_pre = grad_h1 * sig_h1 * (1 + self.h1 * (1 - sig_h1))

        # w1 and w3 gradients
        grads["w1"] = self.normed2.reshape(-1, CONFIG["dim"]).T @ grad_h1_pre.reshape(
            -1, CONFIG["mlp_hidden"]
        )
        grads["w3"] = self.normed2.reshape(-1, CONFIG["dim"]).T @ grad_h3.reshape(
            -1, CONFIG["mlp_hidden"]
        )

        # Attention gradients (simplified)
        grad_attn = grad_output @ self.o_proj.T  # (batch, seq, num_heads*head_dim)

        grads["q_proj"] = self.normed.reshape(-1, CONFIG["dim"]).T @ grad_attn.reshape(
            -1, CONFIG["num_heads"] * CONFIG["head_dim"]
        )
        grads["k_proj"] = self.normed.reshape(-1, CONFIG["dim"]).T @ grad_attn.reshape(
            -1, CONFIG["num_kv_heads"] * CONFIG["head_dim"]
        )
        grads["v_proj"] = self.normed.reshape(-1, CONFIG["dim"]).T @ grad_attn.reshape(
            -1, CONFIG["num_kv_heads"] * CONFIG["head_dim"]
        )

        return grads


class ParameterGolfModel:
    """Full model."""

    def __init__(self):
        self.embed = (
            np.random.randn(CONFIG["vocab_size"], CONFIG["dim"]).astype(np.float32)
            * 0.02
        )
        self.layers = [ParameterGolfLayer(i) for i in range(CONFIG["num_layers"])]
        self.head = self.embed.T.copy()

        self.optimizer_embed = AdamOptimizer(lr=0.6)  # Higher LR for embeddings
        self.optimizers_layers = [
            AdamOptimizer(lr=0.04) for _ in self.layers
        ]  # Matrix LR

    def forward(self, input_ids):
        x = self.embed[input_ids]
        for layer in self.layers:
            x = layer.forward(x)
        logits = x @ self.head
        return logits, x

    def compute_loss(self, logits, targets):
        batch, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        log_probs = np.log(probs + 1e-10)
        nll = -log_probs[np.arange(len(targets_flat)), targets_flat]
        loss = np.mean(nll)

        # Gradient of loss w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[np.arange(len(targets_flat)), targets_flat] -= 1
        grad_logits /= batch * seq_len

        return loss, grad_logits.reshape(batch, seq_len, vocab_size)

    def train_step(self, inputs, targets):
        """Single training step."""
        # Forward
        logits, x = self.forward(inputs)
        loss, grad_logits = self.compute_loss(logits, targets)

        # Backward through head
        grad_x = grad_logits @ self.head.T

        # Update head
        self.head -= (
            0.008
            * x.reshape(-1, CONFIG["dim"]).T
            @ grad_logits.reshape(-1, CONFIG["vocab_size"])
        )

        # Backward through layers
        for i in reversed(range(len(self.layers))):
            grads = self.layers[i].backward(grad_x)
            self.optimizers_layers[i].step(self.layers[i].get_params(), grads)

            # Compute grad for next layer (simplified)
            grad_x = grad_x @ self.layers[i].o_proj.T  # Approximate

        # Update embeddings
        for b in range(inputs.shape[0]):
            for s in range(inputs.shape[1]):
                self.embed[inputs[b, s]] -= 0.6 * grad_x[b, s]

        return loss


def load_bin_file(filepath, seq_len=1024):
    """Load .bin file."""
    data = np.fromfile(filepath, dtype=np.uint16)
    num_tokens = (len(data) // seq_len) * seq_len
    data = data[:num_tokens]
    return data.reshape(-1, seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/dev/parameter-golf/data/datasets/fineweb10B_sp1024",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)  # Reduced for speed
    parser.add_argument("--steps", type=int, default=100)

    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()

    print("=" * 60)
    print("Parameter-Golf with Adam Optimizer")
    print("=" * 60)
    print(f"Target: val_loss ~2.06 (baseline)")
    print(
        f"Batch: {args.batch_size}x{args.seq_len} = {args.batch_size * args.seq_len} tokens"
    )
    print(f"Steps: {args.steps}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_files = sorted(glob.glob(str(data_dir / "fineweb_train_*.bin")))

    if not train_files:
        print("No training files!")
        return

    print(f"Loading {train_files[0]}...")
    train_tokens = load_bin_file(train_files[0], args.seq_len)
    print(f"Sequences: {len(train_tokens)}")

    # Model
    print("\nInitializing model...")
    model = ParameterGolfModel()

    n_params = sum(
        [
            model.embed.size,
            model.head.size,
        ]
        + [sum(w.size for w in layer.get_params().values()) for layer in model.layers]
    )
    print(f"Parameters: {n_params:,} ({n_params * 4 / 1024 / 1024:.2f} MB)")

    # Training
    print("\nTraining...")
    losses = []
    start = time.time()

    for step in range(args.steps):
        # Get batch
        idx = step % (len(train_tokens) // args.batch_size)
        batch = train_tokens[idx * args.batch_size : (idx + 1) * args.batch_size]

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        # Train
        loss = model.train_step(inputs, targets)
        losses.append(loss)

        if (step + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            print(f"  Step {step + 1}/{args.steps} | Loss: {avg_loss:.4f}")

    train_time = time.time() - start
    final_loss = np.mean(losses[-10:])

    print(f"\nTraining complete!")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Time: {train_time:.2f}s")
    print(f"  Target: ~2.06")
    print(f"  Gap: {final_loss - 2.06:.4f}")
    print(f"  Steps: {args.steps}")
    print(f"  Tokens seen: {args.steps * args.batch_size * args.seq_len:,}")

    # Note: Parameter-golf baseline trains for ~7.5B tokens over 20k steps
    # We need many more steps to reach ~2.06


if __name__ == "__main__":
    main()
