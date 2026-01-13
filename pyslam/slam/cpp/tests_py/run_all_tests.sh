#!/usr/bin/env bash

# =============================================================================
# Simple PYSLAM C++ Test Runner
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="${SCRIPT_DIR}"

# Test lists
KDTREE_TESTS=("test_ckdtree_basic.py" "test_ckdtree_validation.py")
SLAM_TESTS=("test_slam_cpp_module.py" "test_slam_cpp_map.py" "test_slam_cpp_feature_matching_equivalence.py" "test_slam_cpp_rotation_histogram.py")
OPTIMIZATION_TESTS=("test_slam_cpp_optimize_pose.py" "test_slam_cpp_optimize_sim3.py" "test_slam_cpp_optimize_essential_graph.py")

# Test tracking
PASSED_TESTS=()
FAILED_TESTS=()
TOTAL_TESTS=0

print_header() {
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

run_test() {
    local test_name=$1
    local test_dir=$2
    
    echo -e "${BLUE}==============================================================================${NC}"
    print_info "Running $test_name in $test_dir"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if python3 $test_name; then
        print_success "Successfully ran $test_name"
        PASSED_TESTS+=("$test_name")
    else
        print_error "Failed to run $test_name"
        FAILED_TESTS+=("$test_name")
        return 1
    fi
}

print_summary() {
    echo
    print_header "TEST SUMMARY"
    echo -e "${BLUE}Total tests run: ${TOTAL_TESTS}${NC}"
    echo -e "${GREEN}Passed: ${#PASSED_TESTS[@]}${NC}"
    echo -e "${RED}Failed: ${#FAILED_TESTS[@]}${NC}"
    
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo
        print_error "Failed tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "${RED}  - $test${NC}"
        done
        echo
        print_error "Some tests failed!"
        exit 1
    else
        echo
        print_success "All tests passed!"
    fi
}

# Main execution
cd $TESTS_DIR

print_header "Starting PYSLAM C++ Tests"

# Run KDTREE tests
if [ ${#KDTREE_TESTS[@]} -gt 0 ]; then
    print_header "Running KDTREE Tests"
    for test in "${KDTREE_TESTS[@]}"; do
        run_test $test $TESTS_DIR || true  # Continue on failure to collect all results
    done
fi

# Run SLAM tests
if [ ${#SLAM_TESTS[@]} -gt 0 ]; then
    print_header "Running SLAM Tests"
    for test in "${SLAM_TESTS[@]}"; do
        run_test $test $TESTS_DIR || true  # Continue on failure to collect all results
    done
fi

# Run OPTIMIZATION tests
if [ ${#OPTIMIZATION_TESTS[@]} -gt 0 ]; then
    print_header "Running OPTIMIZATION Tests"
    for test in "${OPTIMIZATION_TESTS[@]}"; do
        run_test $test $TESTS_DIR || true  # Continue on failure to collect all results
    done
fi

# Print final summary
print_summary