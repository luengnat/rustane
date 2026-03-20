#!/bin/bash
# Memory-efficient test runner for Rustane
# Usage: ./scripts/test.sh [test_pattern]

set -e

# Default to 2 test threads to reduce memory pressure
TEST_THREADS=${TEST_THREADS:-2}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Rustane Test Runner ===${NC}"
echo -e "Test threads: ${YELLOW}$TEST_THREADS${NC}"
echo ""

# Check if a specific test pattern was provided
if [ -n "$1" ]; then
    echo -e "Running tests matching: ${YELLOW}$1${NC}"
    cargo test --lib "$1" -- --test-threads="$TEST_THREADS"
else
    echo -e "Running all library tests (grouped by module)"
    echo ""

    # Run tests in groups to reduce memory pressure
    GROUPS=(
        "ane_backward"
        "tensor_sharding"
        "transformer"
        "trainer"
        "scheduler"
        "loss"
        "checkpoint"
        "metrics"
        "wrapper"
        "mil"
        "layers"
        "utils"
    )

    PASSED=0
    FAILED=0

    for group in "${GROUPS[@]}"; do
        echo -e "${YELLOW}Testing: $group${NC}"
        if cargo test --lib "$group" -- --test-threads="$TEST_THREADS" 2>&1 | tail -5; then
            echo -e "${GREEN}✓ $group passed${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ $group failed${NC}"
            ((FAILED++))
        fi
        echo ""
    done

    echo -e "${GREEN}=== Summary ===${NC}"
    echo -e "Groups passed: $PASSED"
    echo -e "Groups failed: $FAILED"

    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
fi
