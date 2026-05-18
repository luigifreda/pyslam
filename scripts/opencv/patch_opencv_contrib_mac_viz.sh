#!/usr/bin/env bash
# Author: Luigi Freda
# This file is part of https://github.com/luigifreda/pyslam
#
# macOS-only patches for OpenCV contrib viz (VTK 9.6+ / recent Xcode).
#
# Newer VTK and libc++ no longer pull in <iostream> transitively, while
# opencv_contrib viz uses std::cout/std::cerr without including it in
# precomp.hpp. This breaks builds after macOS / Homebrew VTK upgrades.
#
# Usage:
#   patch_opencv_contrib_mac_viz.sh <opencv_contrib_dir>
#   OPENCV_CONTRIB_DIR=<path> patch_opencv_contrib_mac_viz.sh
#
# Safe to run multiple times (idempotent).

set -euo pipefail

# Resolve script location (used if callers need a stable path to this script).
SCRIPT_DIR_=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
if command -v readlink &> /dev/null; then
    SCRIPT_DIR_=$(readlink -f "$SCRIPT_DIR_" 2>/dev/null || echo "$SCRIPT_DIR_")
fi

# opencv_contrib root: first CLI arg, else OPENCV_CONTRIB_DIR env var.
OPENCV_CONTRIB_DIR="${1:-${OPENCV_CONTRIB_DIR:-}}"

# Patches target macOS toolchain / Homebrew VTK only; no-op elsewhere.
if [[ "${OSTYPE:-}" != darwin* ]]; then
    echo "[patch_opencv_contrib_mac_viz] skipping (not macOS)"
    exit 0
fi

if [[ -z "$OPENCV_CONTRIB_DIR" ]]; then
    echo "[patch_opencv_contrib_mac_viz] error: missing opencv_contrib directory" >&2
    echo "  usage: $0 <opencv_contrib_dir>" >&2
    exit 1
fi

# Missing tree is non-fatal (e.g. contrib not checked out yet during install).
if [[ ! -d "$OPENCV_CONTRIB_DIR" ]]; then
    echo "[patch_opencv_contrib_mac_viz] warning: directory not found, skipping: $OPENCV_CONTRIB_DIR"
    exit 0
fi

# viz module sources that need patching.
VIZ_SRC="$OPENCV_CONTRIB_DIR/modules/viz/src"
PRECOMP_HPP="$VIZ_SRC/precomp.hpp"
INTERACTOR_CPP="$VIZ_SRC/vtk/vtkVizInteractorStyle.cpp"

patch_count=0

# Patch 1: ensure <iostream> is included in the viz precompiled header.
# viz .cpp files rely on std::cout/std::cerr; older VTK headers used to
# pull <iostream> in transitively, but that no longer happens with VTK 9.6+.
apply_precomp_iostream_patch() {
    if [[ ! -f "$PRECOMP_HPP" ]]; then
        echo "[patch_opencv_contrib_mac_viz] warning: $PRECOMP_HPP not found, skipping iostream patch"
        return 0
    fi

    # Idempotent: skip if a previous run (or upstream) already added it.
    if grep -q '#include <iostream>' "$PRECOMP_HPP"; then
        echo "[patch_opencv_contrib_mac_viz] precomp.hpp already includes <iostream>"
        return 0
    fi

    # Anchor insertion after <iomanip> (stable marker in upstream precomp.hpp).
    if ! grep -q '#include <iomanip>' "$PRECOMP_HPP"; then
        echo "[patch_opencv_contrib_mac_viz] error: expected '#include <iomanip>' in $PRECOMP_HPP" >&2
        exit 1
    fi

    # BSD sed on macOS: -i '' edits in place (empty backup suffix).
    sed -i '' '/#include <iomanip>/a\
#include <iostream>
' "$PRECOMP_HPP"

    echo "[patch_opencv_contrib_mac_viz] added #include <iostream> to precomp.hpp"
    patch_count=$((patch_count + 1))
}

# Patch 2: qualify bare cout/endl with std:: in vtkVizInteractorStyle.cpp.
# Without a using-directive, unqualified cout/endl may not resolve once
# <iostream> is no longer pulled in indirectly via VTK headers.
apply_interactor_cout_patch() {
    if [[ ! -f "$INTERACTOR_CPP" ]]; then
        echo "[patch_opencv_contrib_mac_viz] warning: $INTERACTOR_CPP not found, skipping cout patch"
        return 0
    fi

    local changed=0

    # Only rewrite lines that still use the old unqualified form (idempotent).
    if grep -qE '^[[:space:]]+cout << "Screenshot successfully captured' "$INTERACTOR_CPP"; then
        sed -i '' \
            's|    cout << "Screenshot successfully captured (" << file.c_str() << ")" << endl;|    std::cout << "Screenshot successfully captured (" << file.c_str() << ")" << std::endl;|' \
            "$INTERACTOR_CPP"
        changed=1
    fi

    if grep -qE '^[[:space:]]+cout << "Scene successfully exported' "$INTERACTOR_CPP"; then
        sed -i '' \
            's|    cout << "Scene successfully exported (" << file.c_str() << ")" << endl;|    std::cout << "Scene successfully exported (" << file.c_str() << ")" << std::endl;|' \
            "$INTERACTOR_CPP"
        changed=1
    fi

    if [[ "$changed" -eq 1 ]]; then
        echo "[patch_opencv_contrib_mac_viz] qualified bare cout/endl in vtkVizInteractorStyle.cpp"
        patch_count=$((patch_count + 1))
    else
        echo "[patch_opencv_contrib_mac_viz] vtkVizInteractorStyle.cpp cout patch already applied"
    fi
}

echo "[patch_opencv_contrib_mac_viz] applying macOS viz patches under: $OPENCV_CONTRIB_DIR"

apply_precomp_iostream_patch
apply_interactor_cout_patch

if [[ "$patch_count" -eq 0 ]]; then
    echo "[patch_opencv_contrib_mac_viz] nothing to patch (already up to date)"
else
    echo "[patch_opencv_contrib_mac_viz] applied $patch_count patch(es)"
fi
