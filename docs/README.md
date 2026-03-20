# Rustane Docs

This page collects the public-facing documentation for Rustane.

## Start Here

- [Main README](../README.md) - overview, installation, and usage

## Core Docs

- [Attention Notes](ATTENTION.md) - background on attention support and layout choices
- [MLX vs Rustane Matmul Comparison](../MLX_MATMUL_COMPARISON.md) - current packed-dynamic benchmark results
- [ANE Backward Limitation](ANE_BACKWARD_LIMITATION.md) - explains ANE single-input MIL constraint
- [ANE Multi-Input Research](ANE_MULTI_INPUT_RESEARCH.md) - investigation of future ANE training support

## Working Examples

- [`examples/simple_inference.rs`](../examples/simple_inference.rs) - minimal inference example
- [`examples/ane_dynamic_matmul_benchmark.rs`](../examples/ane_dynamic_matmul_benchmark.rs) - authoritative packed dynamic matmul benchmark
- [`examples/ane_tiled_rectangular_matmul_benchmark.rs`](../examples/ane_tiled_rectangular_matmul_benchmark.rs) - rectangular tiled benchmark

## Notes

- Rustane is experimental and depends on private Apple ANE APIs.
- Benchmark docs should be treated as point-in-time measurements, not guarantees.
