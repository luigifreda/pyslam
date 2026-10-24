#!/usr/bin/env bash
# Quick Linux TBB smoke test (apt libtbb-dev or conda tbb+tbb-devel).
# Run on a Linux host, or: docker run --rm -v "$PWD:/src" -w /src ubuntu:22.04 bash /src/run_linux_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v cmake &>/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq cmake g++ pkg-config libtbb-dev
fi

rm -rf build
mkdir build
cd build
cmake ..
cmake --build .
./tbb_smoke
echo "PASS: TBB headers and link work ($(cmake --version | head -1))"
