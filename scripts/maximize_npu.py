#!/usr/bin/env python3
"""
Maximize ANE NPU Utilization

Strategy to fill up NPU:
1. Use larger dimensions (1024 instead of 512)
2. Use ANE only for classifier (512->32000 or 1024->32000)
3. Process large batches to keep NPU busy
4. Minimize kernel switches
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge


class NPUOptimizedModel:
    """Model designed to maximize ANE NPU usage."""

    def __init__(self, vocab_size=32000, dim=1024, seq_len=256):
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len

        print("=" * 70)
        print("NPU-Optimized Model")
        print("=" * 70)
        print(f"Strategy: Maximize ANE NPU utilization")
        print(f"  - Large dim: {dim}")
        print(f"  - Large vocab: {vocab_size}")
        print(f"  - ANE for classifier only")
        print()

        # Initialize ANE
        print("Initializing ANE...")
        get_bridge()

        # Embeddings (CPU)
        print("Creating embeddings...")
        self.embed = np.random.randn(vocab_size, dim).astype(np.float32) * 0.02

        # Simple FFN weights (CPU)
        print("Creating FFN weights...")
        self.W1 = (
            np.random.randn(dim, dim * 4).astype(np.float32) * 0.02
        )  # 1024 -> 4096
        self.W2 = (
            np.random.randn(dim * 4, dim).astype(np.float32) * 0.02
        )  # 4096 -> 1024

        # Classifier on ANE (the big one!)
        print(f"Creating ANE classifier ({dim} -> {vocab_size})...")
        self.ane_classifier = ANEConv1x1(dim, vocab_size, seq_len)
        self.ane_classifier.weight = (
            np.random.randn(vocab_size, dim, 1, 1).astype(np.float32) * 0.02
        )
        self.ane_classifier._recompile()

        print(f"✅ Model ready")
        print(
            f"   Parameters: {(self.embed.size + self.W1.size + self.W2.size + self.ane_classifier.weight.size):,}"
        )

    def forward(self, input_ids):
        """Forward pass with heavy classifier on ANE."""
        batch_size = input_ids.shape[0]

        # Embedding (CPU)
        x = self.embed[input_ids]  # [batch, seq, dim]

        # FFN (CPU - keep it simple)
        h = x @ self.W1  # [batch, seq, 4096]
        h = np.maximum(0, h)  # ReLU
        h = h @ self.W2  # [batch, seq, 1024]
        x = x + h  # Residual

        # Classifier on ANE (THE BIG ONE!)
        x_ane = x.transpose(0, 2, 1).reshape(batch_size, self.dim, 1, self.seq_len)
        logits_ane = self.ane_classifier.forward(x_ane)
        logits = logits_ane.transpose(0, 2, 3, 1).reshape(
            batch_size, self.seq_len, self.vocab_size
        )

        return logits

    def compute_loss(self, logits, targets):
        """Cross-entropy loss."""
        batch_size, seq_len, vocab = logits.shape

        logits_flat = logits.reshape(-1, vocab)
        targets_flat = targets.reshape(-1)

        # Softmax
        max_logits = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Cross-entropy
        nll = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10)
        return np.mean(nll)


def benchmark_npu_utilization():
    """Benchmark designed to show NPU usage."""
    print("\n" + "=" * 70)
    print("BENCHMARK: NPU Utilization Test")
    print("=" * 70)

    # Test 1: Small vocab (1024) - minimal ANE work
    print("\n1. Small vocab (1024) - minimal ANE work:")
    model1 = NPUOptimizedModel(vocab_size=1024, dim=1024)
    input_ids = np.random.randint(0, 1024, (2, 256))
    targets = np.random.randint(0, 1024, (2, 256))

    # Warmup
    for _ in range(3):
        logits = model1.forward(input_ids)
        loss = model1.compute_loss(logits, targets)

    # Time it
    times = []
    for _ in range(10):
        start = time.time()
        logits = model1.forward(input_ids)
        loss = model1.compute_loss(logits, targets)
        times.append((time.time() - start) * 1000)

    avg_time = np.mean(times)
    print(f"   Time: {avg_time:.1f}ms")
    print(f"   Loss: {loss:.4f}")

    # Test 2: Large vocab (32000) - heavy ANE work
    print("\n2. Large vocab (32000) - heavy ANE work:")
    model2 = NPUOptimizedModel(vocab_size=32000, dim=1024)
    input_ids = np.random.randint(0, 32000, (2, 256))
    targets = np.random.randint(0, 32000, (2, 256))

    # Warmup
    print("   Warming up ANE (compiling kernel)...")
    for _ in range(3):
        logits = model2.forward(input_ids)
        loss = model2.compute_loss(logits, targets)

    # Time it
    print("   Benchmarking...")
    times = []
    for _ in range(10):
        start = time.time()
        logits = model2.forward(input_ids)
        loss = model2.compute_loss(logits, targets)
        times.append((time.time() - start) * 1000)

    avg_time2 = np.mean(times)
    print(f"   Time: {avg_time2:.1f}ms")
    print(f"   Loss: {loss:.4f}")

    # Test 3: Continuous load to fill NPU
    print("\n3. Continuous load (fill NPU):")
    print("   Running 100 forward passes to keep NPU busy...")

    start = time.time()
    for i in range(100):
        logits = model2.forward(input_ids)
        loss = model2.compute_loss(logits, targets)
        if (i + 1) % 25 == 0:
            print(f"     Completed {i + 1}/100")

    total_time = (time.time() - start) * 1000
    avg_per_pass = total_time / 100

    print(f"\n   Total time: {total_time:.0f}ms")
    print(f"   Per pass: {avg_per_pass:.1f}ms")
    print(f"   Throughput: {1000 / avg_per_pass:.0f} passes/sec")

    print("\n" + "=" * 70)
    print("INSTRUCTIONS TO CHECK NPU USAGE:")
    print("=" * 70)
    print("""
1. Open Activity Monitor (Cmd+Space, type "Activity Monitor")
2. Go to Window -> GPU History (or Cmd+4)
3. Look for "ANE" or "Neural Engine" in the graph
4. Run this script again
5. You should see ANE usage spike during the classifier phase

If ANE doesn't show up:
  - Try larger batch size (change (2, 256) to (8, 256))
  - Try larger vocab (64000 instead of 32000)
  - ANE may not report usage for small matrices
    """)


def stress_test_npu():
    """Stress test to maximize NPU utilization."""
    print("\n" + "=" * 70)
    print("STRESS TEST: Maximize NPU Utilization")
    print("=" * 70)

    # Create model with maximum load
    model = NPUOptimizedModel(vocab_size=32000, dim=1024)

    # Large batch
    batch_size = 4
    input_ids = np.random.randint(0, 32000, (batch_size, 256))
    targets = np.random.randint(0, 32000, (batch_size, 256))

    print(f"\nRunning stress test with batch_size={batch_size}...")
    print("This should keep ANE busy for several seconds.")
    print("Check Activity Monitor now!\n")

    start = time.time()
    num_iterations = 50

    for i in range(num_iterations):
        logits = model.forward(input_ids)
        loss = model.compute_loss(logits, targets)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            progress = (i + 1) / num_iterations * 100
            print(
                f"Progress: {progress:.0f}% ({i + 1}/{num_iterations}) - "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - start

    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s")
    print(f"Iterations: {num_iterations}")
    print(f"Time per iteration: {total_time / num_iterations * 1000:.0f}ms")
    print(f"Final loss: {loss:.4f}")
    print("\nCheck Activity Monitor - did you see ANE usage?")


if __name__ == "__main__":
    print("=" * 70)
    print("MAXIMIZE ANE NPU UTILIZATION")
    print("=" * 70)

    # Run benchmarks
    benchmark_npu_utilization()

    # Run stress test
    stress_test_npu()

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
If you still don't see NPU usage:

1. Matrix may still be too small
   Try: vocab_size=64000, dim=2048
   
2. ANE may not report usage to Activity Monitor
   Some ANE operations don't show in GPU History
   
3. Verify ANE is working:
   - Check that classifier forward is fast (< 10ms for 32000 vocab)
   - Compare with CPU-only version
   - If faster, ANE is working even if not visible

4. Real NPU usage happens with:
   - Large matrices (1024x1024 or bigger)
   - Continuous workload (no gaps)
   - Single large operation vs many small ones
    """)
