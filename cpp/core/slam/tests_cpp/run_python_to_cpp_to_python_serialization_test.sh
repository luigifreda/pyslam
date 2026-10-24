#!/bin/bash
# Script to run Python → C++ → Python round-trip serialization test
# This test verifies that data is correctly preserved when going through both serialization paths

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_TESTS_DIR="${SCRIPT_DIR}"
PYTHON_TESTS_DIR="${SCRIPT_DIR}/../tests_py"
TEST_DATA_DIR="${PYTHON_TESTS_DIR}/test_data"
BUILD_DIR="${SCRIPT_DIR}/../build/tests_cpp"

# Test files
PYTHON_GENERATOR="${PYTHON_TESTS_DIR}/test_python_to_cpp_map_serialization.py"
CPP_TEST="test_python_to_cpp_to_python_map"
PYTHON_VERIFIER="${PYTHON_TESTS_DIR}/test_python_to_cpp_to_python_map_serialization.py"
PYTHON_MAP_FILE="${TEST_DATA_DIR}/python_saved_map.json"
ROUND_TRIP_MAP_FILE="${TEST_DATA_DIR}/python_to_cpp_to_python_map.json"

# Step 1: Generate Python map
generate_python_map() {
    print_header "Step 1: Generating Test Map from Python"
    
    if [ ! -f "$PYTHON_MAP_FILE" ]; then
        print_info "Test map not found, generating it..."
    else
        print_info "Test map already exists: $PYTHON_MAP_FILE"
        read -p "Regenerate test map? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Using existing test map"
            return 0
        fi
    fi
    
    print_info "Running: $PYTHON_GENERATOR"
    cd "$PYTHON_TESTS_DIR"
    
    if python3 "$(basename "$PYTHON_GENERATOR")"; then
        if [ -f "$PYTHON_MAP_FILE" ]; then
            print_success "Test map generated successfully"
            print_info "Test map file: $PYTHON_MAP_FILE (size: $(du -h "$PYTHON_MAP_FILE" | cut -f1))"
            return 0
        else
            print_error "Test map file not found after generation"
            return 1
        fi
    else
        print_error "Failed to generate test map"
        return 1
    fi
}

# Step 2: Build C++ test
build_cpp_test() {
    print_header "Step 2: Building C++ Test"
    
    if [ -f "${BUILD_DIR}/${CPP_TEST}" ]; then
        print_info "C++ test already built: ${BUILD_DIR}/${CPP_TEST}"
        return 0
    fi
    
    print_info "Building C++ test..."
    cd "${SCRIPT_DIR}/../build"
    
    if cmake --build . --target "${CPP_TEST}" 2>&1 | tail -20; then
        if [ -f "${BUILD_DIR}/${CPP_TEST}" ]; then
            print_success "C++ test built successfully"
            return 0
        else
            print_error "C++ test executable not found after build"
            return 1
        fi
    else
        print_error "Failed to build C++ test"
        return 1
    fi
}

# Step 3: Run C++ test (Python → C++ → Python)
run_cpp_test() {
    print_header "Step 3: Running C++ Test (Python → C++ → Python)"
    
    if [ ! -f "${BUILD_DIR}/${CPP_TEST}" ]; then
        print_error "C++ test executable not found: ${BUILD_DIR}/${CPP_TEST}"
        print_info "Trying to build it first..."
        build_cpp_test
    fi
    
    if [ ! -f "$PYTHON_MAP_FILE" ]; then
        print_error "Python map file not found: $PYTHON_MAP_FILE"
        print_info "Trying to generate it first..."
        generate_python_map
    fi
    
    print_info "Running: ${BUILD_DIR}/${CPP_TEST}"
    cd "${BUILD_DIR}"
    
    # Set environment variable with absolute path to test map file
    export PYSLAM_TEST_MAP_FILE="$(cd "$TEST_DATA_DIR" && pwd)/python_saved_map.json"
    
    if ./"$CPP_TEST"; then
        if [ -f "$ROUND_TRIP_MAP_FILE" ]; then
            print_success "C++ test passed!"
            print_info "Round-trip map file: $ROUND_TRIP_MAP_FILE (size: $(du -h "$ROUND_TRIP_MAP_FILE" | cut -f1))"
            return 0
        else
            print_error "Round-trip map file not found after C++ test"
            return 1
        fi
    else
        print_error "C++ test failed!"
        return 1
    fi
}

# Step 4: Run Python verifier
run_python_verifier() {
    print_header "Step 4: Running Python Verifier"
    
    if [ ! -f "$ROUND_TRIP_MAP_FILE" ]; then
        print_error "Round-trip map file not found: $ROUND_TRIP_MAP_FILE"
        print_info "Trying to run C++ test first..."
        run_cpp_test
    fi
    
    print_info "Running: python3 $PYTHON_VERIFIER"
    cd "$PYTHON_TESTS_DIR"
    
    if python3 "$(basename "$PYTHON_VERIFIER")"; then
        print_success "Python verifier passed!"
        return 0
    else
        print_error "Python verifier failed!"
        return 1
    fi
}

# Main execution
main() {
    print_header "Python→C++→Python Round-Trip Map Serialization Test"
    echo
    print_info "This test verifies that maps can be correctly serialized through:"
    print_info "  Python → C++ → Python"
    echo
    
    # Generate Python map
    if ! generate_python_map; then
        echo
        print_header "TEST SUMMARY"
        print_error "Test failed!"
        print_info "Check the output above for details."
        exit 1
    fi
    echo
    
    # Build C++ test
    if ! build_cpp_test; then
        echo
        print_header "TEST SUMMARY"
        print_error "Test failed!"
        print_info "Check the output above for details."
        exit 1
    fi
    echo
    
    # Run C++ test
    if ! run_cpp_test; then
        echo
        print_header "TEST SUMMARY"
        print_error "Test failed!"
        print_info "Check the output above for details."
        exit 1
    fi
    echo
    
    # Run Python verifier
    if ! run_python_verifier; then
        echo
        print_header "TEST SUMMARY"
        print_error "Test failed!"
        print_info "Check the output above for details."
        exit 1
    fi
    
    echo
    print_header "TEST SUMMARY"
    print_success "All tests passed!"
    print_info "Python → C++ → Python round-trip serialization is working correctly."
    return 0
}

# Run main function
main "$@"

