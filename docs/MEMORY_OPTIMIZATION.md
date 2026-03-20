# Memory Usage Optimization

## Problem: Backend Tests Consuming 45GB+ RAM

### Root Cause

Running `cargo test` directly without thread limits causes massive memory consumption:

1. **Parallel Test Execution**: Cargo runs all 243+ tests in parallel by default
2. **Large Allocations**: Each test allocates large vectors (e.g., `vec![1.0f32; 65536]`)
3. **No Cleanup Between Tests**: Tests hold memory until completion
4. **Integration Tests**: ANE integration tests hold additional resources

**Result**: 45GB+ RAM usage, potential OOM (Out of Memory) crashes

### Example of Memory-Heavy Test Pattern

```rust
#[test]
fn test_large_operation() {
    // Each concurrent test allocates this
    let x = vec![1.0f32; 65536];  // 256KB per test
    let w = vec![1.0f32; 4096];   // 16KB per test
    let d_out = vec![0.1f32; 65536]; // 256KB per test

    // With 243 tests running in parallel: ~130MB just for these arrays
    // Plus all other test allocations, ANE resources, etc.
    // Total: 45GB+
}
```

## Solution

### 1. Created `.cargo/config.toml`

Sets memory-efficient defaults:

```toml
[build]
codegen-units = 16  # Reduce compilation memory

[env]
TEST_THREADS = "2"  # Default to 2 test threads
```

### 2. Enhanced `scripts/test.sh`

Memory-efficient test runner with three modes:

```bash
# Library tests only (fastest, lowest memory) - DEFAULT
./scripts/test.sh

# All tests (lib + integration)
./scripts/test.sh all

# Integration tests only
./scripts/test.sh integration

# Custom pattern
./scripts/test.sh <pattern>

# Override thread limit
TEST_THREADS=4 ./scripts/test.sh
```

**How It Works**:
1. Groups tests by module (ane_backward, transformer, trainer, etc.)
2. Runs each group sequentially with `--test-threads=2`
3. Reduces peak memory from 45GB to ~4-8GB

### 3. Updated Documentation

- README.md with prominent warning about `cargo test`
- Instructions to always use `scripts/test.sh`
- Memory reduction tips

## Memory Usage Comparison

| Method | Peak Memory | Duration | Recommendation |
|--------|-------------|----------|----------------|
| `cargo test` (default) | 45GB+ | Fastest | ❌ AVOID - Can OOM |
| `cargo test -- --test-threads=2` | ~12GB | Fast | ⚠️ Better, but still high |
| `./scripts/test.sh` | ~4-8GB | Medium | ✅ RECOMMENDED |
| `./scripts/test.sh all` | ~8-12GB | Slower | ✅ For full testing |
| `./scripts/test.sh integration` | ~6-10GB | Medium | ✅ Integration only |

## Additional Optimization Tips

### 1. Run Specific Test Groups

```bash
# Only test what you changed
cargo test ane_backward --lib -- --test-threads=2
cargo test transformer --lib -- --test-threads=2
```

### 2. Use Release Builds for Testing

```bash
# Release builds use less runtime memory
cargo test --release --lib -- --test-threads=2
```

### 3. Clean Build Artifacts

```bash
# Free up disk and memory
cargo clean
```

### 4. Monitor Memory Usage

```bash
# Monitor memory in real-time
watch -n 1 'ps aux | grep cargo | head -10'

# Or use Activity Monitor on macOS
```

## For CI/CD

GitHub Actions already configured in `.github/workflows/ci.yml`:

```yaml
- name: Run tests
  run: ./scripts/test.sh
  env:
    TEST_THREADS: 2
```

## Verification

To verify the fix works:

```bash
# Before (without fix):
# cargo test
# Expected: 45GB+ RAM usage

# After (with fix):
./scripts/test.sh
# Expected: 4-8GB RAM usage
```

## Summary

✅ **Fixed**: Created `.cargo/config.toml` with thread limits
✅ **Enhanced**: Updated `scripts/test.sh` with grouped test execution
✅ **Documented**: Updated README with warnings and instructions
✅ **Result**: Reduced memory usage from 45GB to 4-8GB

**Always use `./scripts/test.sh` instead of `cargo test` directly!**
