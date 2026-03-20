#!/bin/bash
# Memory-efficient test runner for Rustane
#
# USAGE:
#   ./scripts/test.sh                    # Run lib tests only (fastest, lowest memory)
#   ./scripts/test.sh integration       # Run integration tests only
#   ./scripts/test.sh all               # Run all tests (lib + integration)
#   ./scripts/test.sh <pattern>         # Run tests matching pattern
#   TEST_THREADS=4 ./scripts/test.sh    # Override thread limit
#
# MEMORY SAVINGS:
#   - Limits test threads to prevent OOM (default: 2)
#   - Groups tests to reduce concurrent memory usage
#   - Skips integration tests by default (use 'all' to include them)

set -e

# Default to 2 test threads to reduce memory pressure
# Override with: TEST_THREADS=4 ./scripts/test.sh
TEST_THREADS=${TEST_THREADS:-2}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Rustane Test Runner ===${NC}"
echo -e "Test threads: ${YELLOW}$TEST_THREADS${NC}"
echo -e "Memory: Limiting concurrent tests to prevent OOM"
echo ""

# Parse arguments
TEST_TYPE="${1:-lib}"

case "$TEST_TYPE" in
    integration)
        echo -e "${BLUE}Running integration tests only${NC}"
        echo -e "${YELLOW}Note: Integration tests may use more memory${NC}"
        echo ""
        cargo test --tests -- --test-threads="$TEST_THREADS"
        ;;

    all)
        echo -e "${BLUE}Running all tests (lib + integration)${NC}"
        echo ""

        # Library tests in groups
        echo -e "${YELLOW}=== Library Tests ===${NC}"
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

        echo -e "${GREEN}=== Integration Tests ===${NC}"
        if cargo test --tests -- --test-threads="$TEST_THREADS"; then
            echo -e "${GREEN}✓ Integration tests passed${NC}"
        else
            echo -e "${RED}✗ Integration tests failed${NC}"
            ((FAILED++))
        fi

        echo ""
        echo -e "${GREEN}=== Summary ===${NC}"
        echo -e "Library groups passed: $PASSED"
        echo -e "Integration tests: $([ $FAILED -eq 0 ] && echo "passed" || echo "failed")"
        echo -e "Total failed: $FAILED"

        if [ $FAILED -gt 0 ]; then
            exit 1
        fi
        ;;

    lib|"")
        # Default: run library tests only (fastest, lowest memory)
        echo -e "${BLUE}Running library tests only${NC}"
        echo -e "Use '${YELLOW}./scripts/test.sh all${NC}' to include integration tests"
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
        ;;

    *)
        # Custom pattern
        echo -e "${BLUE}Running tests matching: ${YELLOW}$1${NC}"
        cargo test "$1" -- --test-threads="$TEST_THREADS"
        ;;
esac

echo ""
echo -e "${GREEN}✓ All tests passed!${NC}"
