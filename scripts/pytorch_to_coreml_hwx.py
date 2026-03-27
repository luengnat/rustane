#!/usr/bin/env python3
"""
Convert PyTorch model to CoreML and extract HWX files for ANE training.

This script:
1. Loads a PyTorch model (parameter-golf style transformer)
2. Converts it to CoreML using coremltools
3. Compiles the CoreML model (which generates hwx files internally)
4. Extracts the hwx files from the compiled model bundle

Usage:
    python pytorch_to_coreml_hwx.py --model-path ./checkpoints/model.pt --output-dir ./models/coreml
"""

import argparse
import os
import sys
import shutil
import tempfile
import glob
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: pip install torch coremltools")
    sys.exit(1)


def load_parameter_golf_model(checkpoint_path: str):
    """
    Load a parameter-golf style transformer model.

    Model architecture (from train_gpt.py):
    - vocab_size: 1024
    - dim: 512
    - num_layers: 9
    - num_heads: 8
    - num_kv_heads: 4 (GQA)
    - mlp_mult: 2
    - seq_len: 1024
    """

    # Define the model architecture to match parameter-golf
    class ParameterGolfTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 1024
            self.dim = 512
            self.num_layers = 9
            self.num_heads = 8
            self.num_kv_heads = 4
            self.head_dim = 64  # 512 / 8
            self.mlp_hidden = 1024  # 512 * 2
            self.seq_len = 1024

            # Token embeddings
            self.embed = nn.Embedding(self.vocab_size, self.dim)

            # Transformer layers (simplified - just showing structure)
            self.layers = nn.ModuleList(
                [
                    TransformerLayer(
                        self.dim,
                        self.num_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.mlp_hidden,
                    )
                    for _ in range(self.num_layers)
                ]
            )

            # Output head (tied with embedding)
            self.head = nn.Linear(self.dim, self.vocab_size, bias=False)
            self.head.weight = self.embed.weight  # Tie weights

        def forward(self, x):
            # x: [batch, seq_len]
            x = self.embed(x)  # [batch, seq_len, dim]

            # Pass through transformer layers
            for layer in self.layers:
                x = layer(x)

            # Output projection
            logits = self.head(x)  # [batch, seq_len, vocab_size]
            return logits

    class TransformerLayer(nn.Module):
        def __init__(self, dim, num_heads, num_kv_heads, head_dim, mlp_hidden):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads

            # RMSNorm (simplified - PyTorch doesn't have native RMSNorm)
            self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)

            # Attention projections (GQA)
            self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

            # Learnable scales (from parameter-golf)
            self.qk_gain = nn.Parameter(torch.ones(num_heads))
            self.attn_scale = nn.Parameter(torch.ones(dim))
            self.mlp_scale = nn.Parameter(torch.ones(dim))
            self.resid_mix_0 = nn.Parameter(torch.ones(dim))
            self.resid_mix_1 = nn.Parameter(torch.ones(dim))

            # SwiGLU MLP
            self.w1 = nn.Linear(dim, mlp_hidden, bias=False)
            self.w3 = nn.Linear(dim, mlp_hidden, bias=False)
            self.w2 = nn.Linear(mlp_hidden, dim, bias=False)

        def forward(self, x):
            batch, seq_len, dim = x.shape

            # Residual mixing
            x_mixed = self.resid_mix_0 * x + self.resid_mix_1 * x

            # Attention block
            normed = self.attn_norm(x_mixed)

            # QKV projections
            q = self.q_proj(normed)
            k = self.k_proj(normed)
            v = self.v_proj(normed)

            # Reshape for multi-head attention
            q = q.view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_kv_heads, -1).transpose(1, 2)
            v = v.view(batch, seq_len, self.num_kv_heads, -1).transpose(1, 2)

            # Apply QK gain
            q = q * self.qk_gain.view(1, -1, 1, 1)

            # Attention (simplified - no RoPE for CoreML compatibility)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim**0.5)
            probs = torch.softmax(scores, dim=-1)
            attn = torch.matmul(probs, v)

            # Reshape and project
            attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            out = self.o_proj(attn)

            # Apply scale and residual
            attn_out = x + out * self.attn_scale

            # FFN block with SwiGLU
            normed2 = self.ffn_norm(attn_out)
            h1 = self.w1(normed2)
            h3 = self.w3(normed2)
            gated = torch.silu(h1) * h3
            ffn_out = self.w2(gated)

            # Apply scale and residual
            out = attn_out + ffn_out * self.mlp_scale

            return out

    # Try to load actual checkpoint if available
    model = ParameterGolfTransformer()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"No checkpoint found at {checkpoint_path}, using initialized weights")

    return model


def convert_to_coreml(model, output_dir: str):
    """
    Convert PyTorch model to CoreML format.

    This creates a .mlpackage that can be compiled to ANE hwx files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()

    # Create example input
    example_input = torch.randint(0, 1024, (1, 1024), dtype=torch.long)

    # Trace the model
    print("Tracing PyTorch model...")
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    print("Converting to CoreML...")

    # Define input/output
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_ids", shape=(1, 1024), dtype=int)],
        outputs=[ct.TensorType(name="logits", dtype=float)],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,  # Allow ANE
    )

    # Save the model
    model_path = output_dir / "transformer.mlpackage"
    mlmodel.save(str(model_path))
    print(f"CoreML model saved to: {model_path}")

    return model_path


def compile_coreml_model(model_path: str, output_dir: str):
    """
    Compile CoreML model to generate hwx files.

    This uses the coremlcompiler command-line tool to compile the model
    for ANE execution, which generates the hwx files.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if coremlcompiler is available
    compiler_path = "/Applications/Xcode.app/Contents/Developer/usr/bin/coremlcompiler"
    if not os.path.exists(compiler_path):
        # Try to find it elsewhere
        result = os.system("which coremlcompiler > /dev/null 2>&1")
        if result != 0:
            print("Warning: coremlcompiler not found. Attempting manual compilation...")
            return extract_hwx_manual(model_path, output_dir)

    # Compile the model
    compiled_dir = output_dir / "compiled"
    compiled_dir.mkdir(exist_ok=True)

    cmd = f"{compiler_path} compile {model_path} {compiled_dir}"
    print(f"Running: {cmd}")
    result = os.system(cmd)

    if result != 0:
        print("Compilation failed, trying alternative method...")
        return extract_hwx_manual(model_path, output_dir)

    # Find hwx files in compiled output
    hwx_files = list(compiled_dir.glob("**/*.hwx"))
    print(f"Found {len(hwx_files)} hwx files")

    return hwx_files


def extract_hwx_manual(model_path: Path, output_dir: Path):
    """
    Manually extract hwx files from CoreML model bundle.

    When CoreML compiles a model, it creates intermediate files including hwx.
    This function attempts to extract them from the model bundle.
    """
    print("Attempting manual hwx extraction...")

    # The mlpackage is a directory containing the compiled model
    # Look for ANE-related files

    # Check if there's an internal ANE compilation cache
    ane_cache_paths = [
        Path.home() / "Library/Caches/com.apple.coreml/model_cache",
        Path.home() / "Library/Caches/com.apple.espresso",
        Path.home() / "Library/Caches/ANECompiler",
    ]

    hwx_files = []
    for cache_path in ane_cache_paths:
        if cache_path.exists():
            print(f"Checking cache: {cache_path}")
            # Look for hwx files
            files = list(cache_path.glob("**/*.hwx"))
            for f in files:
                # Copy to output
                dest = output_dir / f.name
                shutil.copy2(f, dest)
                hwx_files.append(dest)
                print(f"  Copied: {f.name}")

    if not hwx_files:
        print("No hwx files found in caches. Creating placeholder instructions...")

        # Create a README with instructions
        readme = output_dir / "EXTRACTION_README.md"
        readme.write_text("""
# HWX Extraction Instructions

The CoreML model has been created but hwx files need to be extracted manually.

## Method 1: Use the CoreML Compiler Directly

```bash
# Find coremlcompiler
COREML_COMPILER=$(xcrun -f coremlcompiler)

# Compile the model
$COREML_COMPILER compile transformer.mlpackage ./compiled

# Look for hwx files in the compiled output
find ./compiled -name "*.hwx"
```

## Method 2: Runtime Extraction

Run the model on a device/simulator with ANE and monitor:

```bash
# Monitor ANE cache during model loading
sudo fs_usage -w | grep -i "hwx\|ane\|espresso"
```

## Method 3: Use Private Framework (for research only)

```python
import ANECompiler
# Use private API to compile and extract hwx
```

## Method 4: tinygrad Approach

Use the tinygrad approach which loads hwx files directly:

```python
# See rustane/src/ane/hwx_loader.rs for implementation
```

## Note

The hwx format is a Mach-O binary with ANE operations at offset 0x4000.
See tinygrad/extra/accel/ane/ for reverse-engineered format details.
""")
        print(f"Created instructions at: {readme}")

    return hwx_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to CoreML and extract HWX files"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/coreml",
        help="Output directory for CoreML model and hwx files",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip CoreML conversion, only extract hwx",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PyTorch to CoreML to HWX Converter")
    print("=" * 60)

    if not args.skip_conversion:
        # Step 1: Load PyTorch model
        print("\n[Step 1/3] Loading PyTorch model...")
        model = load_parameter_golf_model(args.model_path)

        # Step 2: Convert to CoreML
        print("\n[Step 2/3] Converting to CoreML...")
        model_path = convert_to_coreml(model, args.output_dir)
    else:
        model_path = Path(args.output_dir) / "transformer.mlpackage"
        if not model_path.exists():
            print(f"Error: CoreML model not found at {model_path}")
            sys.exit(1)

    # Step 3: Compile and extract hwx
    print("\n[Step 3/3] Extracting HWX files...")
    hwx_files = compile_coreml_model(str(model_path), args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"CoreML model: {model_path}")
    print(f"HWX files found: {len(hwx_files)}")
    for f in hwx_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

    print(f"\nOutput directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Use the hwx files with rustane ANE runtime")
    print("2. Or use CoreML directly for inference")
    print("3. See docs/HWX_INTEGRATION.md for usage")


if __name__ == "__main__":
    main()
