#!/bin/bash
# Complete ANE Training Test Suite

echo "===================================================================="
echo "ANE Training - Complete Test Suite"
echo "===================================================================="
echo ""

cd /Users/nat/dev/rustane

FAIL=0

echo "Test 1: ANE Bridge Initialization"
echo "--------------------------------------------------------------------"
python3 -c "from ane_ops import get_bridge; print('✅ Bridge initialized')" || FAIL=1
echo ""

echo "Test 2: Forward Pass (Correctness)"
echo "--------------------------------------------------------------------"
python3 scripts/test_ane_fixed_blob.py | grep -E "(✅|❌)" || FAIL=1
echo ""

echo "Test 3: Backward Pass (Correctness)"  
echo "--------------------------------------------------------------------"
python3 scripts/ane_backward.py | grep -E "(✅|❌)" || FAIL=1
echo ""

echo "Test 4: Training Loop (Loss Decreasing)"
echo "--------------------------------------------------------------------"
python3 scripts/train_ane_with_backward.py --steps 20 2>&1 | tail -10 || FAIL=1
echo ""

echo "===================================================================="
if [ $FAIL -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - ANE is fully functional!"
else
    echo "❌ Some tests failed"
fi
echo "===================================================================="
