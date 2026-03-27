#!/usr/bin/env python3
"""
Generate ANE-compatible weight blobs for parameter-golf layer 0.

This script creates the binary weight files referenced by the MIL programs:
- layer0_wq.bin (Q projection: [512, 512])
- layer0_wk.bin (K projection: [256, 512])
- layer0_wv.bin (V projection: [256, 512])
- layer0_wo.bin (Output projection: [512, 512])
- layer0_w1.bin (FFN gate: [1024, 512])
- layer0_w3.bin (FFN up: [1024, 512])
- layer0_w2.bin (FFN down: [512, 1024])

Usage:
    python scripts/generate_layer0_weights.py [--checkpoint path/to/model.pt]
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ANE Blob Header Format (128 bytes) - from ane-lora-training
# Layout:
#   [0:4]   version = 1
#   [4:8]   type = 2 (fp16)
#   [8:64]  reserved zeros
#   [64:68] magic = 0xDEADBEEF (LE)
#   [68:72] chunk_count = 1
#   [72:76] data_size
#   [76:80] reserved
#   [80:84] data_offset = 128
#   [84:128] reserved zeros
#   [128:] fp16 data


def create_ane_blob_header(data_size: int) -> bytes:
    """
    Create ANE weight blob header (128 bytes).

    Based on ane-lora-training implementation.
    """
    header = bytearray(128)

    # Global header [0:64]
    header[0:4] = struct.pack("<I", 1)  # version = 1
    header[4:8] = struct.pack("<I", 2)  # type = 2 (fp16)
    # [8:64] reserved zeros

    # Chunk header [64:128]
    header[64:68] = struct.pack("<I", 0xDEADBEEF)  # magic (LE)
    header[68:72] = struct.pack("<I", 1)  # chunk_count = 1
    header[72:76] = struct.pack("<I", data_size)  # data_size
    # [76:80] reserved
    header[80:84] = struct.pack("<I", 128)  # data_offset = 128
    # [84:128] reserved zeros

    return bytes(header)


def save_weight_blob(filepath: str, weights: np.ndarray, data_type: str = "fp16"):
    """
    Save weights as ANE-compatible blob file.

    Args:
        filepath: Output file path
        weights: numpy array of weights
        data_type: 'fp16' or 'fp32'
    """
    # Ensure correct shape
    if weights.ndim == 2:
        # For conv weights, need [out_ch, in_ch, 1, 1]
        weights_4d = weights.reshape(weights.shape[0], weights.shape[1], 1, 1)
    else:
        weights_4d = weights

    # Convert to fp16 and flatten
    if data_type == "fp16":
        weights_flat = weights_4d.astype(np.float16).flatten()
    else:
        weights_flat = weights_4d.astype(np.float32).flatten()

    data_bytes = weights_flat.tobytes()

    # Create 128-byte header
    header = create_ane_blob_header(len(data_bytes))

    # Write file (header + data)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(data_bytes)

    print(f"  Saved: {filepath} ({weights.shape} -> {weights_4d.shape})")


def generate_random_weights(rows: int, cols: int, seed: int = None) -> np.ndarray:
    """Generate random weights with Xavier initialization."""
    if seed is not None:
        np.random.seed(seed)

    # Xavier/Glorot initialization
    limit = np.sqrt(6.0 / (rows + cols))
    weights = np.random.uniform(-limit, limit, (rows, cols))

    return weights.astype(np.float32)


def load_pytorch_checkpoint(checkpoint_path: str) -> dict:
    """Load weights from PyTorch checkpoint."""
    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract layer 0 weights
        weights = {}
        for key, value in checkpoint.items():
            if "layers.0." in key or key.startswith("layers.0."):
                # Map PyTorch keys to our naming
                if "q_proj" in key or "wq" in key.lower():
                    weights["wq"] = value.numpy()
                elif "k_proj" in key or "wk" in key.lower():
                    weights["wk"] = value.numpy()
                elif "v_proj" in key or "wv" in key.lower():
                    weights["wv"] = value.numpy()
                elif "o_proj" in key or "wo" in key.lower():
                    weights["wo"] = value.numpy()
                elif "w1" in key:
                    weights["w1"] = value.numpy()
                elif "w2" in key:
                    weights["w2"] = value.numpy()
                elif "w3" in key:
                    weights["w3"] = value.numpy()

        return weights
    except ImportError:
        print("PyTorch not installed, using random weights")
        return {}
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Generate ANE weight blobs for layer 0"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/layer0/weights",
        help="Output directory for weight blobs",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Random seed for weight initialization"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating ANE Weight Blobs for Layer 0")
    print("=" * 60)

    # Try to load from checkpoint
    checkpoint_weights = {}
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nLoading from checkpoint: {args.checkpoint}")
        checkpoint_weights = load_pytorch_checkpoint(args.checkpoint)

    # Weight specifications for parameter-golf layer 0
    weight_specs = [
        # (name, shape, checkpoint_key)
        ("layer0_wq.bin", (512, 512), "wq"),  # Q projection: [q_dim, dim]
        ("layer0_wk.bin", (256, 512), "wk"),  # K projection: [kv_dim, dim]
        ("layer0_wv.bin", (256, 512), "wv"),  # V projection: [kv_dim, dim]
        ("layer0_wo.bin", (512, 512), "wo"),  # Output: [dim, dim]
        ("layer0_w1.bin", (1024, 512), "w1"),  # FFN gate: [mlp_hidden, dim]
        ("layer0_w3.bin", (1024, 512), "w3"),  # FFN up: [mlp_hidden, dim]
        ("layer0_w2.bin", (512, 1024), "w2"),  # FFN down: [dim, mlp_hidden]
    ]

    print("\nGenerating weight blobs:")
    for filename, shape, ck_key in weight_specs:
        filepath = output_dir / filename

        if ck_key in checkpoint_weights:
            # Use checkpoint weights
            weights = checkpoint_weights[ck_key]
            print(f"  Using checkpoint weights for {filename}")
        else:
            # Generate random weights
            weights = generate_random_weights(shape[0], shape[1], seed=args.seed)
            if ck_key:
                print(f"  Generated random weights for {filename}")

        # Save as ANE blob
        save_weight_blob(str(filepath), weights, data_type="fp16")

    # Also save learnable parameters as separate files
    print("\nGenerating learnable parameters:")

    # q_gain [8]
    q_gain = np.ones(8, dtype=np.float32) * 1.5  # qk_gain_init from train_gpt.py
    q_gain_data = q_gain.astype(np.float16).tobytes()
    q_gain_path = output_dir / "layer0_q_gain.bin"
    with open(q_gain_path, "wb") as f:
        f.write(create_ane_blob_header(len(q_gain_data)))
        f.write(q_gain_data)
    print(f"  Saved: {q_gain_path} (q_gain)")

    # resid_mix_0 [512]
    resid_mix_0 = np.ones(512, dtype=np.float32)
    r0_data = resid_mix_0.astype(np.float16).tobytes()
    r0_path = output_dir / "layer0_resid_mix_0.bin"
    with open(r0_path, "wb") as f:
        f.write(create_ane_blob_header(len(r0_data)))
        f.write(r0_data)
    print(f"  Saved: {r0_path} (resid_mix_0)")

    # resid_mix_1 [512]
    resid_mix_1 = np.ones(512, dtype=np.float32)
    r1_data = resid_mix_1.astype(np.float16).tobytes()
    r1_path = output_dir / "layer0_resid_mix_1.bin"
    with open(r1_path, "wb") as f:
        f.write(create_ane_blob_header(len(r1_data)))
        f.write(r1_data)
    print(f"  Saved: {r1_path} (resid_mix_1)")

    # attn_scale [512]
    attn_scale = np.ones(512, dtype=np.float32)
    as_data = attn_scale.astype(np.float16).tobytes()
    as_path = output_dir / "layer0_attn_scale.bin"
    with open(as_path, "wb") as f:
        f.write(create_ane_blob_header(len(as_data)))
        f.write(as_data)
    print(f"  Saved: {as_path} (attn_scale)")

    # mlp_scale [512]
    mlp_scale = np.ones(512, dtype=np.float32)
    ms_data = mlp_scale.astype(np.float16).tobytes()
    ms_path = output_dir / "layer0_mlp_scale.bin"
    with open(ms_path, "wb") as f:
        f.write(create_ane_blob_header(len(ms_data)))
        f.write(ms_data)
    print(f"  Saved: {ms_path} (mlp_scale)")

    print("\n" + "=" * 60)
    print("Weight generation complete!")
    print(f"Output directory: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.glob("*.bin")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name:25s} ({size_kb:8.2f} KB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
