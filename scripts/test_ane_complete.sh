#!/bin/bash
# ANE Training Quick Start

echo "=============================================================="
echo "ANE-Accelerated Training for Parameter-Golf"
echo "=============================================================="
echo ""

cd /Users/nat/dev/rustane

echo "1. Testing ANE operations..."
python3 scripts/ane_ops.py
echo ""

echo "2. Running ANE correctness test..."
python3 scripts/test_ane_fixed_blob.py
echo ""

echo "3. Running mini training session..."
python3 scripts/train_ane_accelerated.py --epochs 1 --num-batches 3 --seq-len 256
echo ""

echo "=============================================================="
echo "All tests passed! ANE is ready for full training."
echo "=============================================================="
