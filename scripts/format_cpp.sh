#!/bin/bash

# C++ Code Formatting Script for PYSLAM
# Usage: ./scripts/format_cpp.sh [--check] [--fix] [files...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CHECK_MODE=false
FIX_MODE=false
FILES=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_MODE=true
            shift
            ;;
        --fix)
            FIX_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--check] [--fix] [files...]"
            echo "  --check: Check formatting without making changes"
            echo "  --fix:   Fix formatting issues"
            echo "  files:   Specific files to format (default: all C++ files)"
            exit 0
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# If no files specified, find all C++ files
if [ ${#FILES[@]} -eq 0 ]; then
    echo "Finding C++ files..."
    FILES=($(find "$PROJECT_ROOT" -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cxx" | \
        grep -v thirdparty | \
        grep -v build | \
        grep -v __pycache__))
fi

echo "Found ${#FILES[@]} C++ files to process"

# Check if clang-format is available
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format not found. Please install it:"
    echo "  Ubuntu/Debian: sudo apt install clang-format"
    echo "  macOS: brew install clang-format"
    echo "  Or use pixi: pixi add clang-format"
    exit 1
fi

# Check if clang-tidy is available
if ! command -v clang-tidy &> /dev/null; then
    echo "Warning: clang-tidy not found. Linting will be skipped."
    echo "Install with: pixi add clang-tidy"
fi

CLANG_FORMAT_CMD="clang-format"
CLANG_TIDY_CMD="clang-tidy"

if [ "$CHECK_MODE" = true ]; then
    echo "Checking C++ formatting..."
    HAS_ISSUES=false
    
    for file in "${FILES[@]}"; do
        if [ -f "$file" ]; then
            if ! $CLANG_FORMAT_CMD -style=file -output-replacements-xml "$file" | grep -q "<replacement "; then
                echo "✓ $file"
            else
                echo "✗ $file (formatting issues found)"
                HAS_ISSUES=true
            fi
        fi
    done
    
    if [ "$HAS_ISSUES" = true ]; then
        echo "Formatting issues found. Run with --fix to fix them."
        exit 1
    else
        echo "All files are properly formatted!"
        exit 0
    fi
    
elif [ "$FIX_MODE" = true ]; then
    echo "Fixing C++ formatting..."
    
    for file in "${FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "Formatting $file"
            $CLANG_FORMAT_CMD -style=file -i "$file"
        fi
    done
    
    echo "Formatting complete!"
    
    # Run clang-tidy if available
    if command -v clang-tidy &> /dev/null; then
        echo "Running clang-tidy..."
        for file in "${FILES[@]}"; do
            if [ -f "$file" ] && [[ "$file" == *.cpp ]]; then
                echo "Linting $file"
                $CLANG_TIDY_CMD "$file" -- -std=c++17 -I"$PROJECT_ROOT/pyslam/slam/cpp" -I"$PROJECT_ROOT/thirdparty/eigen" || true
            fi
        done
    fi
    
else
    echo "Usage: $0 [--check] [--fix] [files...]"
    echo "  --check: Check formatting without making changes"
    echo "  --fix:   Fix formatting issues"
    exit 1
fi
