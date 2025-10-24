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
BUILD_DIR="${SCRIPT_DIR}/../build"
TESTS_DIR="${BUILD_DIR}/tests_cpp"

# Test lists
NUMPY_TESTS=("test_numpy_cv_mat" "test_numpy_map_point" "test_numpy_frame" "test_numpy_keyframe" "test_numpy_camera" "test_numpy_map_state")
JSON_TESTS=("test_json_map_point" "test_json_eigen" "test_json_serialization" "test_json_frame" "test_json_keyframe" "test_json_map" "test_json_camera")
KDTREE_TESTS=("test_kdtree_basic" "test_kdtree_validation")
GENERIC_TESTS=("test_image_management")

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
    
    if ./$test_name; then
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

# Run NUMPY tests
if [ ${#NUMPY_TESTS[@]} -gt 0 ]; then
    print_header "Running NUMPY Tests"
    for test in "${NUMPY_TESTS[@]}"; do
        run_test $test $TESTS_DIR || true  # Continue on failure to collect all results
    done
fi

# Run JSON tests
if [ ${#JSON_TESTS[@]} -gt 0 ]; then
    print_header "Running JSON Tests"
    for test in "${JSON_TESTS[@]}"; do
        run_test $test $TESTS_DIR || true  # Continue on failure to collect all results
    done
fi

# Run GENERIC tests
if [ ${#GENERIC_TESTS[@]} -gt 0 ]; then
    print_header "Running GENERIC Tests"
    for test in "${GENERIC_TESTS[@]}"; do
        run_test $test $TESTS_DIR || true  # Continue on failure to collect all results
    done
fi

# Print final summary
print_summary