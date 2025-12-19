#!/usr/bin/env bash

# =============================================================================
# Python-to-C++ Map Serialization Test Runner
# =============================================================================
# This script:
# 1. Generates a test map from Python
# 2. Builds the C++ test (if needed)
# 3. Runs the C++ test to verify Python-saved maps can be loaded
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
PYTHON_SCRIPT="${PYTHON_TESTS_DIR}/test_python_to_cpp_map_serialization.py"
CPP_TEST="test_python_to_cpp_map"
TEST_DATA_DIR="${PYTHON_TESTS_DIR}/test_data"
TEST_MAP_FILE="${TEST_DATA_DIR}/python_saved_map.json"

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

# Step 1: Generate test map from Python
generate_test_map() {
    print_header "Step 1: Generating Test Map from Python"
    
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python test script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    print_info "Running: $PYTHON_SCRIPT"
    
    cd "$PYTHON_TESTS_DIR"
    if python3 "$(basename "$PYTHON_SCRIPT")"; then
        print_success "Test map generated successfully"
        
        if [ -f "$TEST_MAP_FILE" ]; then
            FILE_SIZE=$(du -h "$TEST_MAP_FILE" | cut -f1)
            print_info "Test map file: $TEST_MAP_FILE (size: $FILE_SIZE)"
        else
            print_error "Test map file was not created: $TEST_MAP_FILE"
            exit 1
        fi
    else
        print_error "Failed to generate test map"
        exit 1
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

# Step 3: Run C++ test
run_cpp_test() {
    print_header "Step 3: Running C++ Test"
    
    if [ ! -f "${CPP_TESTS_DIR}/${CPP_TEST}" ]; then
        print_error "C++ test executable not found: ${CPP_TESTS_DIR}/${CPP_TEST}"
        print_info "Trying to build it first..."
        build_cpp_test
    fi
    
    if [ ! -f "$TEST_MAP_FILE" ]; then
        print_error "Test map file not found: $TEST_MAP_FILE"
        print_info "Trying to generate it first..."
        generate_test_map
    fi
    
    print_info "Running: ${CPP_TESTS_DIR}/${CPP_TEST}"
    cd "$CPP_TESTS_DIR"
    
    # Set environment variable with absolute path to test map file
    export PYSLAM_TEST_MAP_FILE="$(cd "$TEST_DATA_DIR" && pwd)/python_saved_map.json"
    
    if ./"$CPP_TEST"; then
        print_success "C++ test passed!"
        return 0
    else
        print_error "C++ test failed!"
        return 1
    fi
}

# Main execution
main() {
    print_header "Python-to-C++ Map Serialization Test"
    echo
    print_info "This test verifies that maps saved from Python can be correctly loaded by C++"
    echo
    
    # Check if test map exists, if not generate it
    if [ ! -f "$TEST_MAP_FILE" ]; then
        print_info "Test map not found, generating it..."
        generate_test_map
        echo
    else
        print_info "Test map already exists: $TEST_MAP_FILE"
        read -p "Regenerate test map? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            generate_test_map
            echo
        fi
    fi
    
    # Build C++ test if needed
    build_cpp_test
    echo
    
    # Run C++ test
    if run_cpp_test; then
        echo
        print_header "TEST SUMMARY"
        print_success "All tests passed!"
        print_info "Python-to-C++ map serialization is working correctly."
        return 0
    else
        echo
        print_header "TEST SUMMARY"
        print_error "Test failed!"
        print_info "Check the output above for details."
        return 1
    fi
}

# Run main function
main "$@"

