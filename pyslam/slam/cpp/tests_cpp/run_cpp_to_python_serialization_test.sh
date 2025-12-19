#!/usr/bin/env bash

# =============================================================================
# C++-to-Python Map Serialization Test Runner
# =============================================================================
# This script:
# 1. Generates a test map from C++
# 2. Runs the Python test to verify C++-saved maps can be loaded
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
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
PYTHON_TESTS_DIR="${SCRIPT_DIR}/../tests_py"
BUILD_DIR="${SCRIPT_DIR}/../build"
CPP_TESTS_DIR="${BUILD_DIR}/tests_cpp"
CPP_TEST="test_cpp_to_python_map"
PYTHON_SCRIPT="${PYTHON_TESTS_DIR}/test_cpp_to_python_map_serialization.py"
TEST_DATA_DIR="${PYTHON_TESTS_DIR}/test_data"
TEST_MAP_FILE="${TEST_DATA_DIR}/cpp_saved_map.json"

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

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Step 1: Generate test map from C++
generate_test_map() {
    print_header "Step 1: Generating Test Map from C++"
    
    if [ ! -d "$BUILD_DIR" ]; then
        print_warning "Build directory not found: $BUILD_DIR"
        print_info "You may need to build the C++ module first"
        print_info "Try running: cd $SCRIPT_DIR/.. && ./build.sh"
        exit 1
    fi
    
    if [ ! -f "${CPP_TESTS_DIR}/${CPP_TEST}" ]; then
        print_warning "C++ test executable not found: ${CPP_TESTS_DIR}/${CPP_TEST}"
        print_info "Trying to build it first..."
        build_cpp_test
    fi
    
    # Check if test map already exists
    if [ -f "$TEST_MAP_FILE" ]; then
        print_info "Test map already exists: $TEST_MAP_FILE"
        read -p "Regenerate test map? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Using existing test map"
            return 0
        fi
    fi
    
    print_info "Running: ${CPP_TESTS_DIR}/${CPP_TEST}"
    cd "$CPP_TESTS_DIR"
    
    if ./"$CPP_TEST"; then
        if [ -f "$TEST_MAP_FILE" ]; then
            file_size=$(stat -f%z "$TEST_MAP_FILE" 2>/dev/null || stat -c%s "$TEST_MAP_FILE" 2>/dev/null || echo "unknown")
            print_success "Test map generated successfully"
            print_info "Test map file: $TEST_MAP_FILE (size: $(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo ${file_size} bytes))"
            return 0
        else
            print_error "Test map file was not created: $TEST_MAP_FILE"
            return 1
        fi
    else
        print_error "Failed to generate test map"
        return 1
    fi
}

# Step 2: Build C++ test (if needed)
build_cpp_test() {
    print_header "Step 2: Building C++ Test"
    
    if [ ! -d "$BUILD_DIR" ]; then
        print_warning "Build directory not found: $BUILD_DIR"
        print_info "You may need to build the C++ module first"
        print_info "Try running: cd $SCRIPT_DIR/.. && ./build.sh"
        exit 1
    fi
    
    if [ -f "${CPP_TESTS_DIR}/${CPP_TEST}" ]; then
        print_info "C++ test already built: ${CPP_TESTS_DIR}/${CPP_TEST}"
        return 0
    fi
    
    print_info "C++ test not found. Attempting to build..."
    
    # Try to build using CMake
    if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
        cd "$BUILD_DIR"
        print_info "Running: cmake --build . --target $CPP_TEST"
        if cmake --build . --target "$CPP_TEST" 2>/dev/null; then
            print_success "C++ test built successfully"
        else
            print_warning "CMake build failed. You may need to rebuild the entire project."
            print_info "Try running: cd $SCRIPT_DIR/.. && ./build.sh"
            exit 1
        fi
    else
        print_error "CMake build system not found in $BUILD_DIR"
        print_info "You need to configure and build the project first"
        exit 1
    fi
}

# Step 3: Run Python test
run_python_test() {
    print_header "Step 3: Running Python Test"
    
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python test script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    if [ ! -f "$TEST_MAP_FILE" ]; then
        print_error "Test map file not found: $TEST_MAP_FILE"
        print_info "Trying to generate it first..."
        generate_test_map
    fi
    
    print_info "Running: python3 $PYTHON_SCRIPT"
    cd "$ROOT_DIR"
    
    if python3 "$PYTHON_SCRIPT"; then
        print_success "Python test passed!"
        return 0
    else
        print_error "Python test failed!"
        return 1
    fi
}

# Main execution
main() {
    print_header "C++-to-Python Map Serialization Test"
    
    print_info "This test verifies that maps saved from C++ can be correctly loaded by Python"
    echo
    
    # Step 1: Generate test map
    if ! generate_test_map; then
        print_error "Failed to generate test map"
        exit 1
    fi
    
    echo
    
    # Step 2: Build C++ test (if needed)
    build_cpp_test
    
    echo
    
    # Step 3: Run Python test
    if ! run_python_test; then
        print_error "Python test failed!"
        exit 1
    fi
    
    echo
    print_header "TEST SUMMARY"
    print_success "All tests passed!"
    print_info "C++-to-Python map serialization is working correctly."
}

# Run main function
main

