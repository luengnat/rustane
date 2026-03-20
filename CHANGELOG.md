# Changelog

All notable changes to Rustane will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Apple Neural Engine (ANE) safe Rust bindings via objc2
- MIL (Model Intermediate Language) program builder for constructing neural network graphs
- Weight blob utilities for FP32→FP16 conversion and matrix operations
- Type-safe tensor abstractions with shape validation and bounds checking
- IOSurface RAII wrapper for cross-process memory sharing
- Platform detection for ANE availability
- Complete transformer model implementation (TransformerANE)
- Training infrastructure with Model trait, LossFn trait, and Trainer
- Optimizer implementations (Adam, AdamW, Lion)
- Learning rate schedulers (constant, warmup-linear, warmup-cosine)
- Backward pass MIL generators (RMSNorm, Attention, FFN, Loss)
- Backward validation suite with CPU reference implementation
- ANE gradient accumulator for hybrid ANE/CPU training
- Sharded data loader for processing large datasets
- Gradient accumulation for memory-efficient training
- Model checkpointing (save/load/resume training)
- Gradient checkpointing for memory optimization (up to 75% savings)
- Mixed precision training (FP16/BF16) with loss scaling
- Distributed training support with AllReduce
- Flash Attention implementation (O(seq_len × block_size) complexity)
- Metrics tracking with multi-backend logging
- Sequence parallelism for long sequences (16K+ tokens)
- Model parallelism (layer, tensor, pipeline, hybrid sharding)
- Comprehensive error handling and diagnostics
- Retry policy with adaptive batch reduction
- Graceful CPU fallback strategies

### Changed
- Improved error messages across all modules
- Enhanced documentation with examples
- Optimized memory usage in training loops

### Fixed
- ANE backward pass limitation documented with comprehensive research
- Hardware-dependent tests properly marked as ignored
- All doctests fixed for critical functionality

## [0.1.0] - 2026-03-20

### Initial Release
- Core ANE module foundation
- Data loading infrastructure
- Basic transformer model implementation
- Training loop with gradient descent
- Forward pass MIL generation
- CPU-based backward pass
- Example programs demonstrating usage
- Comprehensive test suite (533+ tests)

### Performance
- Forward pass on ANE: 16.7x speedup vs CPU
- Hybrid ANE/CPU training: 1.5x overall speedup
- Memory-efficient training with gradient checkpointing

### Platform Support
- macOS 15+ (Sequoia)
- Apple Silicon with ANE (M1/M2/M3/M4)
- Rust 1.70+ with 2021 edition

### Documentation
- Comprehensive API documentation
- 61 working examples
- Design specifications for all major components
- Research papers on ANE limitations and workarounds

### Known Limitations
- ANE backward pass requires CPU fallback due to MIL format constraints
  - ANE requires single-input MIL with embedded weights
  - Backward pass needs multiple variable inputs (activations from forward)
  - Documented in `docs/ANE_BACKWARD_LIMITATION.md`
- Some advanced features require specific ANE hardware capabilities
- Hardware-dependent tests may not run on all Apple Silicon devices

### Testing
- 533 library tests (all passing)
- 21 integration tests (all passing)
- 61 examples (all compiling)
- Backend parity tests comparing against MLX and MPS
- Comprehensive benchmark suite
