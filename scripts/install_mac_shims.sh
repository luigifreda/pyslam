#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
# macOS-only helper to install lightweight compatibility shims for:
#   - pkg_resources  (so that:  from pkg_resources import packaging  works)
#   - clip           (so that:  import clip; clip.load(...)          works)
#
# Motivation:
#   - On some macOS + Python 3.11 setups, the legacy OpenAI CLIP repo
#     and some third-party code (Detic, f3rm, etc.) still rely on:
#         import pkg_resources
#         from pkg_resources import packaging
#     but the runtime environment may not ship a standalone
#     `pkg_resources` package (it is often only an internal part of
#     setuptools), causing ModuleNotFoundError during installation
#     and import.
#   - Similarly, Detic expects a top-level `clip` package that provides:
#         import clip
#         from clip.simple_tokenizer import SimpleTokenizer
#     For pySLAM we can safely reuse the CLIP implementation already
#     vendored via `f3rm`, and expose a thin wrapper under the standard
#     `clip` module name.
#
# What this script does:
#   - Detects the active Python environment (virtualenv / conda / system).
#   - Resolves its `site-packages` directory.
#   - If `import pkg_resources` fails, copies:
#         thirdparty/shims/pkg_resources  ->  <site-packages>/pkg_resources
#   - If `import clip` fails, copies:
#         thirdparty/shims/clip           ->  <site-packages>/clip
#   - If the real modules are already importable, it leaves them untouched.
#
# This keeps the shims:
#   - Optional (Linux users usually don't need them),
#   - Isolated to the Python env rather than the repo root,
#   - Safe to run multiple times.
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

#set -e
STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


print_blue '================================================'
print_blue "Installing mac shims ..."
print_blue '================================================'
echo ""

# ----------------------------------------------------
# Sanity checks
# ----------------------------------------------------

if [[ "$OSTYPE" != "darwin"* ]]; then
    print_yellow "This script is intended for macOS only (detected OSTYPE=$OSTYPE). Skipping."
    cd "$STARTING_DIR"
    exit 0
fi

virtual_env_name=$(get_virtualenv_name)
if [ -z "$virtual_env_name" ]; then
    print_yellow "WARNING: no virtual environment detected (VIRTUAL_ENV/CONDA_DEFAULT_ENV empty)."
    print_yellow "The shims will be installed into the Python found on PATH."
else
    print_blue "Using Python environment: $virtual_env_name"
fi

# Resolve site-packages for the active Python
SITE_PACKAGES=$(python3 - << 'PY'
import sysconfig
print(sysconfig.get_paths().get("purelib", ""))
PY
)

if [ -z "$SITE_PACKAGES" ]; then
    print_red "ERROR: could not determine site-packages directory."
    cd "$STARTING_DIR"
    exit 1
fi

print_blue "Detected site-packages: $SITE_PACKAGES"
echo ""

CLIP_SHIM_SRC="$ROOT_DIR/thirdparty/shims/clip"
PKG_SHIM_SRC="$ROOT_DIR/thirdparty/shims/pkg_resources"

install_shim_module () {
    local module_name="$1"
    local src_dir="$2"

    if [ ! -d "$src_dir" ]; then
        print_red "ERROR: shim source directory not found for $module_name: $src_dir"
        return 1
    fi

    # If the module already imports fine, skip installing the shim
    if python3 -c "import $module_name" >/dev/null 2>&1; then
        print_blue "Python module '$module_name' already available. Skipping shim install."
        return 0
    fi

    local dst_dir="$SITE_PACKAGES/$module_name"

    if [ -d "$dst_dir" ]; then
        print_yellow "WARNING: $dst_dir already exists but 'import $module_name' failed."
        print_yellow "Not overwriting existing directory. Please inspect your environment manually."
        return 1
    fi

    print_blue "Installing shim for '$module_name' into: $dst_dir"
    cp -R "$src_dir" "$dst_dir"
}

install_shim_module "pkg_resources" "$PKG_SHIM_SRC"
install_shim_module "clip" "$CLIP_SHIM_SRC"

echo ""
print_green "Done installing mac shims (where needed)."

cd "$STARTING_DIR"
