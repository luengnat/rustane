#!/usr/bin/env python3
"""
Profile ANE Training to Find Bottlenecks

Measures time spent in each component to identify optimization opportunities.
"""

import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ane_ops import ANEConv1x1, get_bridge
from ane_backward import ANELinearBackward


class ProfiledANETrainer:
    """Trainer with detailed timing breakdown."""

    def __init__(self, dim=512, seq_len=256):
        self.dim = dim
        self.seq_len = seq_len

        print("Initializing...")
        get_bridge()

        # Single ANE layer
        self.layer = ANEConv1x1(dim, dim, seq_len)
        self.backward = ANELinearBackward(dim, dim, seq_len)
        self.backward.initialize(self.layer.weight)

        # Adam state
        self.m = np.zeros_like(self.layer.weight)
        self.v = np.zeros_like(self.layer.weight)
        self.t = 0

        # Timing stats
        self.stats = {
            "data_prep": [],
            "forward": [],
            "loss": [],
            "backward_dx": [],
            "backward_dW": [],
            "adam_update": [],
            "weight_update": [],
            "total": [],
        }

    def train_step(self, x, target, lr=0.001):
        """Profiled training step."""
        step_start = time.time()

        # 1. Data preparation (fp16 conversion)
        t0 = time.time()
        x_fp16 = x.astype(np.float16)
        data_prep_time = (time.time() - t0) * 1000

        # 2. Forward pass
        t0 = time.time()
        out = self.layer.forward(x)
        forward_time = (time.time() - t0) * 1000

        # 3. Loss computation
        t0 = time.time()
        loss = np.mean((out - target) ** 2)
        loss_time = (time.time() - t0) * 1000

        # 4. Backward - compute dy
        t0 = time.time()
        dy = 2 * (out - target) / np.prod(out.shape)

        # Backward - ANE dx
        t1 = time.time()
        dx = self.backward.compute_dx(dy)
        backward_dx_time = (time.time() - t1) * 1000

        # Backward - CPU dW
        t1 = time.time()
        x_reshaped = x.transpose(0, 3, 1, 2).reshape(-1, self.dim)
        dy_reshaped = dy.transpose(0, 3, 1, 2).reshape(-1, self.dim)
        dW = (dy_reshaped.T @ x_reshaped).reshape(
            self.dim, self.dim, 1, 1
        ) / self.seq_len
        backward_dW_time = (time.time() - t1) * 1000

        backward_time = (time.time() - t0) * 1000

        # 5. Adam update
        t0 = time.time()
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * dW
        self.v = 0.999 * self.v + 0.001 * (dW**2)
        m_hat = self.m / (1 - 0.9**self.t)
        v_hat = self.v / (1 - 0.999**self.t)
        adam_time = (time.time() - t0) * 1000

        # 6. Weight update (includes recompile!)
        t0 = time.time()
        new_weight = self.layer.weight - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        self.layer.update_weights(new_weight)
        self.backward.initialize(new_weight)
        weight_update_time = (time.time() - t0) * 1000

        total_time = (time.time() - step_start) * 1000

        # Store stats
        self.stats["data_prep"].append(data_prep_time)
        self.stats["forward"].append(forward_time)
        self.stats["loss"].append(loss_time)
        self.stats["backward_dx"].append(backward_dx_time)
        self.stats["backward_dW"].append(backward_dW_time)
        self.stats["adam_update"].append(adam_time)
        self.stats["weight_update"].append(weight_update_time)
        self.stats["total"].append(total_time)

        return loss

    def print_stats(self):
        """Print timing statistics."""
        print("\n" + "=" * 70)
        print("TIMING BREAKDOWN (averages in ms)")
        print("=" * 70)

        for key in [
            "data_prep",
            "forward",
            "loss",
            "backward_dx",
            "backward_dW",
            "adam_update",
            "weight_update",
            "total",
        ]:
            times = self.stats[key]
            if times:
                avg = np.mean(times)
                std = np.std(times)
                pct = (avg / np.mean(self.stats["total"])) * 100
                print(f"{key:20s}: {avg:6.2f}ms ± {std:5.2f}ms ({pct:5.1f}%)")

        print("=" * 70)

        # Identify bottlenecks
        total_avg = np.mean(self.stats["total"])
        print("\nBOTTLENECK ANALYSIS:")
        print("-" * 70)

        if np.mean(self.stats["weight_update"]) > total_avg * 0.3:
            print(
                "❌ MAJOR: Weight update (recompilation) is {:.0f}% of time".format(
                    np.mean(self.stats["weight_update"]) / total_avg * 100
                )
            )
            print("   → Solution: Gradient accumulation or zero-recompile")

        if np.mean(self.stats["forward"]) > total_avg * 0.2:
            print(
                "⚠️  MODERATE: Forward pass is {:.0f}% of time".format(
                    np.mean(self.stats["forward"]) / total_avg * 100
                )
            )
            print("   → Solution: Fuse operations, larger batch size")

        if np.mean(self.stats["backward_dx"]) > total_avg * 0.2:
            print(
                "⚠️  MODERATE: Backward dx is {:.0f}% of time".format(
                    np.mean(self.stats["backward_dx"]) / total_avg * 100
                )
            )

        if np.mean(self.stats["data_prep"]) > 1.0:
            print(
                "⚠️  MINOR: Data conversion is {:.1f}ms".format(
                    np.mean(self.stats["data_prep"])
                )
            )
            print("   → Solution: Keep data in fp16")


def main():
    print("=" * 70)
    print("ANE Training Profiler")
    print("=" * 70)

    trainer = ProfiledANETrainer(dim=512, seq_len=256)

    print("\nProfiling 20 training steps...")
    print("-" * 70)

    for i in range(20):
        x = np.random.randn(1, 512, 1, 256).astype(np.float32) * 0.1
        target = np.random.randn(1, 512, 1, 256).astype(np.float32) * 0.1

        loss = trainer.train_step(x, target)

        if (i + 1) % 5 == 0:
            print(
                f"  Step {i + 1}/20: loss={loss:.4f}, time={trainer.stats['total'][-1]:.1f}ms"
            )

    trainer.print_stats()

    print("\n" + "=" * 70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. GRADIENT ACCUMULATION (Immediate - Easy)
   - Accumulate over 10-20 steps
   - Reduces recompilation by 10-20x
   - Expected: 20ms → 7ms

2. BATCH SIZE INCREASE (Immediate - Easy)
   - Process multiple samples per ANE call
   - Amortizes overhead across batch
   - Expected: 10-30% speedup

3. FUSED OPERATIONS (Medium)
   - Combine Q,K,V into single kernel
   - Reduces kernel launch overhead
   - Expected: 10-20% speedup

4. ZERO-RECOMPILE (Hard - High Impact)
   - Input slicing or weight patching
   - Eliminates 100ms recompilation
   - Expected: 7ms → 5ms

5. FP16 THROUGHOUT (Medium)
   - Keep all tensors in fp16
   - Eliminates conversion overhead
   - Expected: 0.5-1ms savings
    """)


if __name__ == "__main__":
    main()
