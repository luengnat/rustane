# Phase-Based Integration Testing

This document describes the phase-based testing strategy adopted from the reference ANE implementation at https://github.com/ncdrone/rustane.

## Philosophy

Instead of flat unit tests, we validate the training pipeline incrementally through phases:
1. **Phase 0** → Foundation (kernel cache creation)
2. **Phase 1** → Single-layer training (loss convergence)
3. **Phase 2** → Gradient accumulation (multi-step)
4. **Phase 3** → Loss scaling stability (FP16 safety)
5. **Phase 4** → Full training pipeline (integrated)
6. **Phase 5** → Hardware alignment validation (ANE constraints)
7. **Benchmarks** → Performance metrics (latency, memory, throughput)

Each phase validates assumptions before moving to the next, catching bugs at the right level.

## Running Phase Tests

### All Phase Tests
```bash
cargo test --test phase_training -- --nocapture
```

### Individual Phases
```bash
cargo test --test phase_training phase0 -- --nocapture
cargo test --test phase_training phase1 -- --nocapture
cargo test --test phase_training phase2 -- --nocapture
cargo test --test phase_training phase3 -- --nocapture
cargo test --test phase_training phase4 -- --nocapture
cargo test --test phase_training hardware_alignment -- --nocapture
```

## Test Output

```
PHASE 0: Kernel Cache Creation
✅ Kernel cache created with default limit (80 kernels)

PHASE 1: Single-Layer Training (Loss Convergence)
Step  1: loss=1.851 | scale=  724.1 | elapsed=254.083µs
Step  2: loss=1.702 | scale=  724.1 | elapsed=255.833µs
...
Step 10: loss=0.509 | scale=  724.1 | elapsed=244.917µs
✅ Loss converged correctly

PHASE 2: Gradient Accumulation (Multi-Step)
Step 1: accumulated 1000 params
Step 2: accumulated 1000 params
Step 3: accumulated 1000 params
Step 4: accumulated 1000 params (complete)
✅ Gradient accumulation validated

PHASE 3: Loss Scaling Stability (FP16 Safety)
Test 1: Valid gradients (no overflow) ✅
Test 2: Overflow detection ✅
Test 3: NaN detection ✅

PHASE 4: Full Training Pipeline
Step  2: loss=1.9000 → UPDATE #1
Step  4: loss=1.8000 → UPDATE #2
...
Step 20: loss=1.0000 → UPDATE #10
✅ Full training pipeline validated

PHASE 5: Hardware Alignment Validation
✓ dim divisible by 128
✓ hidden divisible by 16
✓ IOSurface width multiple of 16
✓ dim ≤ 4096
✅ All hardware alignment requirements satisfied
```

## Benchmarking

### Parameter Sweep Benchmark
```bash
cargo test --test benchmark_harness benchmark_param_sweep -- --ignored --nocapture
```

Output:
```
Config              Params   Layers   Mem (MB)   Avg Step (ms)   Updates
param-golf-baseline  8192      2        0.0        0.426           10
param-golf-scaled-2x 16384     2        0.1        0.858           10
param-golf-scaled-4x 32768     4        0.1        1.160           10

Detailed Latency Statistics:
param-golf-baseline:
  Min:    0.355 ms
  Max:    0.553 ms
  Avg:    0.426 ms
  Median: 0.411 ms
```

### Other Benchmarks (Ignored by Default)
```bash
# Gradient accumulation impact
cargo test --test benchmark_harness benchmark_accumulation_sweep -- --ignored --nocapture

# Model scale impact
cargo test --test benchmark_harness benchmark_model_scale -- --ignored --nocapture

# Detailed single-config analysis
cargo test --test benchmark_harness benchmark_detailed -- --ignored --nocapture
```

## Key Metrics Captured

### Latency
- Min/Max/Avg/Median step time in milliseconds
- Throughput: samples/sec and tokens/sec

### Memory
- Estimated memory footprint (params × 4 bytes)
- Breakdown by component

### Convergence
- Initial and final loss values
- Improvement percentage over training steps

### Training Dynamics
- Loss scaling factor evolution
- Optimizer update frequency
- Gradient accumulation progress

## What These Tests Validate

1. **Phase 0**: Kernel cache is correctly initialized and operational
2. **Phase 1**: Single-layer training loop produces monotonic loss decrease
3. **Phase 2**: Gradient accumulation correctly combines mini-batch gradients
4. **Phase 3**: Loss scaling detects overflow, NaN, and adapts appropriately
5. **Phase 4**: Full 20-step training with 2-step accumulation works end-to-end
6. **Phase 5**: Model dimensions meet ANE hardware constraints
7. **Benchmarks**: Performance is measurable and predictable across scales

## Integration with CI/CD

For continuous integration, run:
```bash
cargo test --lib                      # 139 unit tests (~2s)
cargo test --test phase_training      # 6 phase tests (~10ms)
cargo test --test backend_parity      # ANE-specific tests (requires Apple Silicon)
```

Optional (longer-running):
```bash
cargo test --test benchmark_harness -- --ignored --nocapture  # Benchmarks (~100ms)
```

## Reference Implementation

This testing strategy is modeled on the production ANE training engine at:
- https://github.com/ncdrone/rustane (ncdrone)

Key differences:
- **Reference**: 25 architecture configs, 600M-5B scale, real data
- **parameter-golf**: Proof-of-concept, single config, synthetic data

The reference validates that ANE training scales to production. Our phase-based approach
adopts their incremental validation pattern while keeping parameter-golf focused and testable
on all platforms (including non-Apple Silicon via synthetic data).

## Future Enhancements

1. Add real training data integration (once ANE kernels are activated)
2. Extend benchmarking to measure ANE kernel compilation overhead
3. Add memory profiling (track heap allocations during training)
4. Implement loss trajectory regression testing
5. Add performance regression detection (alert if step latency increases >10%)
