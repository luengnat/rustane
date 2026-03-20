# Development Guide for Rustane

This guide covers the development workflow, coding standards, and tooling setup for contributing to Rustane.

## Prerequisites

- **macOS 15+** (Sequoia) with Apple Silicon (M1/M2/M3/M4)
- **Rust** 1.70 or later
- **Xcode Command Line Tools**: `xcode-select --install`
- **ANE Bridge**: [maderix/ANE](https://github.com/maderix/ANE) for ANE framework access

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/nlord/rustane.git
cd rustane
```

### 2. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

### 3. Setup ANE Bridge

```bash
# Clone ANE bridge (required for ANE access)
git clone https://github.com/maderix/ANE.git ../ANE
cd ../ANE/bridge
make

# Set environment variables
export ANE_BRIDGE_LIB_PATH="$(pwd)/../ANE/bridge"
export ANE_BRIDGE_INCLUDE_PATH="$(pwd)/../ANE/bridge"

# Return to rustane directory
cd -
```

### 4. Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Development Workflow

### Building the Project

```bash
# Check code compiles (fast)
cargo check

# Full build
cargo build

# Build with optimizations
cargo build --release
```

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_transformer_ane_forward

# Run tests in watch mode (requires cargo-install cargo-watch)
cargo watch -x test
```

### Code Quality Checks

```bash
# Format code
cargo fmt

# Check formatting without modifying
cargo fmt -- --check

# Run linter
cargo clippy

# Fix clippy warnings automatically
cargo clippy --fix --allow-dirty --allow-staged
```

### Documentation

```bash
# Build documentation
cargo doc --open

# Run documentation tests
cargo test --doc
```

### Running Examples

```bash
# List all examples
ls examples/

# Run an example
cargo run --example simple_inference

# Run with logging
RUST_LOG=debug cargo run --example train_transformer_ane
```

## Project Structure

```
rustane/
├── src/                    # Main library source
│   ├── ane/                # ANE integration
│   ├── data/               # Data loading
│   ├── layers/             # Layer implementations
│   ├── training/           # Training infrastructure
│   └── wrapper/           # ANE wrapper bindings
├── examples/               # Example programs
├── tests/                  # Integration tests
├── docs/                   # Additional documentation
└── .github/workflows/     # CI/CD pipelines
```

## Coding Standards

### Rust Style

- Use `rustfmt` for formatting (enforced by pre-commit)
- Follow Rust naming conventions:
  - Types: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `SCREAMING_SNAKE_CASE`
- Maximum line width: 100 characters

### Documentation

- All public items must have rustdoc comments
- Include examples in documentation
- Run `cargo doc --open` to preview documentation

### Testing

- Write tests for all new functionality
- Aim for >80% code coverage
- Unit tests in same module as code (mod tests)
- Integration tests in `tests/` directory

### Error Handling

- Use `Result<T>` for fallible operations
- Provide context with error messages
- Use `anyhow::Error` or custom error types for application errors

## Common Development Tasks

### Adding a New Layer

1. Implement layer in `src/layers/`
2. Add tests in `src/layers/<layer>/tests.rs`
3. Run `cargo test` to verify
4. Add example in `examples/`
5. Update documentation

### Adding ANE Operations

1. Create MIL generator in `src/layers/backward/`
2. Implement `BackwardMILGenerator` trait
3. Add validation tests
4. Integrate with `backward_on_ane()`
5. Benchmark performance

### Adding Examples

1. Create file in `examples/`
2. Add usage documentation
3. Test with `cargo run --example <name>`
4. Update `examples/README.md`

## Debugging Tips

### Enable Logging

```bash
# Set log level
export RUST_LOG=debug

# Specific module logging
export RUST_LOG=rustane::training=debug
```

### ANE-Specific Debugging

```bash
# Enable ANE logging
export ANE_LOGGING=1

# Check ANE availability
./examples/simple_inference
```

### Common Issues

**Problem**: "ANE framework not found"
- **Solution**: Verify ANE bridge is built and paths are set

**Problem**: "Out of memory"
- **Solution**: Reduce batch size or model size

**Problem**: Tests fail on CI but pass locally
- **Solution**: Check macOS version and ANE bridge compatibility

## Release Process

### 1. Update Version

Edit `Cargo.toml` version number:
```toml
[package]
version = "0.x.y"
```

### 2. Update CHANGELOG.md

Add release notes with:
- New features
- Breaking changes
- Bug fixes
- Performance improvements

### 3. Commit and Tag

```bash
git add .
git commit -m "Release v0.x.y"
git tag v0.x.y
git push origin main --tags
```

### 4. GitHub Actions

- CI runs automatically on push
- Release workflow publishes to crates.io
- GitHub release is created automatically

## Getting Help

### Documentation

- [API Documentation](https://docs.rs/rustane)
- [Examples Gallery](./examples/README.md)
- [Roadmap](./ROADMAP_SUMMARY.md)

### Issues

- Report bugs at [GitHub Issues](https://github.com/nlord/rustane/issues)
- Include:
  - Rust version (`rustc --version`)
  - macOS version (`sw_vers`)
  - Error messages
  - Minimal reproduction case

### Discussions

- Use [GitHub Discussions](https://github.com/nlord/rustane/discussions)
- For questions, feature requests, or design discussions
