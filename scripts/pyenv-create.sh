#!/bin/bash

# Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
# This script creates a Python environment using pyenv and pyenv-virtualenv.

set -e

# --- Input arguments with defaults ---
ENV_NAME="${1:-dlplay}"
PY_VERSION="${2:-3.11.9}"

echo "=== pyenv Python Environment Setup ==="
echo "Virtualenv Name : $ENV_NAME"
echo "Python Version  : $PY_VERSION"
echo

# --- Detect OS ---
OS="$(uname)"
IS_MACOS=false
IS_LINUX=false

if [[ "$OS" == "Darwin" ]]; then
  IS_MACOS=true
elif [[ "$OS" == "Linux" ]]; then
  IS_LINUX=true
else
  echo "❌ Unsupported OS: $OS"
  exit 1
fi

IS_PYENV_INSTALLED=$(command -v pyenv >/dev/null 2>&1)
#IS_PYENV_INSTALLED=false # this is for testing

# --- Install pyenv and pyenv-virtualenv if missing ---
if ! $IS_PYENV_INSTALLED; then
  echo "🔧 Installing pyenv..."

  if $IS_LINUX; then
    sudo apt update
    sudo apt install -y make build-essential libssl-dev zlib1g-dev \
      libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev \
      xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
  elif $IS_MACOS; then
    if ! command -v brew >/dev/null 2>&1; then
      echo "❌ Homebrew is required but not found. Install it from https://brew.sh/"
      exit 1
    fi
    brew update
    brew install pyenv pyenv-virtualenv
  fi

  # Append pyenv init to shell config
  SHELL_CONFIG="$HOME/.bashrc"
  if $IS_MACOS; then
    SHELL_CONFIG="$HOME/.bash_profile"
  fi
  if [[ "$SHELL" == */zsh ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
  fi

  echo "SHELL_CONFIG: $SHELL_CONFIG"

  if ! grep -q 'pyenv init' "$SHELL_CONFIG"; then
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$SHELL_CONFIG"
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> "$SHELL_CONFIG"
    echo 'eval "$(pyenv init --path)"' >> "$SHELL_CONFIG"
    echo 'eval "$(pyenv init -)"' >> "$SHELL_CONFIG"
    echo 'eval "$(pyenv virtualenv-init -)"' >> "$SHELL_CONFIG"
  fi

  echo "✅ pyenv and pyenv-virtualenv installed. Restart your terminal or source your shell config."
  echo "Example: source $SHELL_CONFIG"
fi

# --- Load pyenv in current shell ---
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PATH"

eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# --- Load pyenv-virtualenv ---
if [ -d "$PYENV_ROOT/plugins/pyenv-virtualenv" ] || command -v pyenv-virtualenv-init >/dev/null 2>&1; then
  eval "$(pyenv virtualenv-init -)"
else
  echo "❌ pyenv-virtualenv not found!"
  echo "Installing with:"
  if $IS_MACOS; then
    echo "  brew install pyenv-virtualenv      # macOS"
    brew install pyenv-virtualenv
  else
    echo "  git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv"
    git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
  fi
fi

# --- Install Python version if needed ---
if ! pyenv versions --bare | grep -qx "$PY_VERSION"; then
  echo "📦 Installing Python $PY_VERSION..."
  pyenv install "$PY_VERSION"
else
  echo "✅ Python $PY_VERSION already installed."
fi

# --- Create virtualenv if needed ---
if ! pyenv virtualenvs --bare | grep -qx "$ENV_NAME"; then
  echo "📦 Creating virtualenv '$ENV_NAME'..."
  pyenv virtualenv "$PY_VERSION" "$ENV_NAME"
else
  echo "✅ Virtualenv '$ENV_NAME' already exists."
fi

# --- Set local version in current directory ---
pyenv local "$ENV_NAME"

echo
echo "✅ Environment '$ENV_NAME' using Python $PY_VERSION is ready."
echo "📂 A '.python-version' file has been created in this directory."
echo "💡 To activate the environment manually, run:"
echo "   pyenv activate $ENV_NAME"
