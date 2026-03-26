#!/bin/bash
# ANE Operator Audit — subprocess-isolated, with timing.
# Usage: bash examples/run_audit.sh [ops...|all]
#
# Tests each (op, size) in its own subprocess to survive ANE crashes.
# Measures compile time, eval time, and compares against CPU baseline.

set -uo pipefail

BINARY="target/debug/examples/test_size_sensitivity"

# Ops to test
ALL_OPS=(cast add mul sub relu tanh sigmoid exp gelu silu softmax reduce_mean layer_norm)
# Sizes: (ch, w) for 4D tensor [1, ch, 1, w]
ALL_SIZES=(
    "64 64"   # 4096 elements — same as working conv1x1
    "64 128"  # 8192 elements
    "64 256"  # 16384 elements
    "128 64"  # 8192 elements (more channels)
    "256 64"  # 16384 elements
    "32 32"   # 1024 elements — smaller
    "16 16"   # 256 elements — tiny
    "4 4"     # 16 elements — very tiny
)

# Parse args
if [ $# -eq 0 ] || [ "$1" = "all" ]; then
    OPS=("${ALL_OPS[@]}")
    SIZES=("${ALL_SIZES[@]}")
else
    # Filter: treat args as ops (default to all sizes)
    OPS=()
    SIZES=("${ALL_SIZES[@]}")
    for arg in "$@"; do
        # If it contains a space, it's a size spec like "64 128"
        if echo "$arg" | grep -q " "; then
            SIZES=("$arg")
        else
            OPS+=("$arg")
        fi
    done
fi

echo "========================================="
echo "  ANE Operator Audit (subprocess-isolated)"
echo "========================================="
echo "  Ops:   ${OPS[*]}"
echo "  Sizes: ${SIZES[*]}"
echo ""

PASS=0
FAIL_COMPILE=0
FAIL_EVAL=0
CRASH=0
TOTAL=0

for size in "${SIZES[@]}"; do
    ch=$(echo "$size" | cut -d' ' -f1)
    w=$(echo "$size" | cut -d' ' -f2)
    elems=$((ch * w))
    
    echo ""
    echo "--- Size: [1, $ch, 1, $w] ($elems elements) ---"
    
    for op in "${OPS[@]}"; do
        TOTAL=$((TOTAL + 1))
        stderr_file="/tmp/ane_audit_${op}_${ch}x${w}.txt"
        
        # Time the whole subprocess (compile + eval)
        start_ns=$(python3 -c "import time; print(int(time.time_ns()))" 2>/dev/null || date +%s%N)
        
        # Run in subprocess — capture stdout for result, stderr for debug
        result=$(timeout 15 "$BINARY" "$op" "$ch" "$w" 2>"$stderr_file")
        exit_code=$?
        
        end_ns=$(python3 -c "import time; print(int(time.time_ns()))" 2>/dev/null || date +%s%N)
        
        # Parse result from stdout
        status=$(echo "$result" | grep -E "^[a-z_]+:" | head -1)
        
        # Check stderr for compile success
        compile_ok=false
        eval_ok=false
        grep -q "COMPILE_OK" "$stderr_file" 2>/dev/null && compile_ok=true
        
        # Extract timing from stderr if available
        elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
        
        if [ "$exit_code" -ge 128 ]; then
            signal=$((exit_code - 128))
            echo "  $op: 💀 CRASH (signal $signal, ${elapsed_ms}ms)"
            CRASH=$((CRASH + 1))
        elif echo "$status" | grep -q ":PASS"; then
            echo "  $op: ✅ ${elapsed_ms}ms"
            PASS=$((PASS + 1))
        elif echo "$status" | grep -q ":WARN"; then
            echo "  $op: ⚠️  zero output (${elapsed_ms}ms)"
            PASS=$((PASS + 1))
        elif echo "$status" | grep -q ":FAIL:eval"; then
            echo "  $op: ❌ eval fail (${elapsed_ms}ms)"
            FAIL_EVAL=$((FAIL_EVAL + 1))
        elif echo "$status" | grep -q ":FAIL:compile"; then
            echo "  $op: ❌ compile fail (${elapsed_ms}ms)"
            FAIL_COMPILE=$((FAIL_COMPILE + 1))
        elif echo "$status" | grep -q ":FAIL"; then
            echo "  $op: ❌ (${elapsed_ms}ms)"
            FAIL_COMPILE=$((FAIL_COMPILE + 1))
        else
            echo "  $op: ❓ unknown (exit=$exit_code, ${elapsed_ms}ms)"
            FAIL_COMPILE=$((FAIL_COMPILE + 1))
        fi
        
        rm -f "$stderr_file" 2>/dev/null
    done
done

echo ""
echo "========================================="
echo "  RESULTS"
echo "========================================="
echo "  ✅ PASS:          $PASS"
echo "  ❌ Compile fail:   $FAIL_COMPILE"
echo "  ❌ Eval fail:      $FAIL_EVAL"
echo "  💀 Crash:          $CRASH"
echo "  ────────────────────────"
echo "  Total tests:     $TOTAL"
echo "========================================="
