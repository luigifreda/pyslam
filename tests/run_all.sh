#!/usr/bin/env bash
# Run the pySLAM checks from the repository root.
# Requires the pySLAM environment, the C++ modules (cpp/build.sh), and the C++ core (cpp/core/slam/build.sh).

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAILED=()

run_step() {
    local name="$1"
    shift
    echo "=============================================================================="
    echo "$name"
    echo "=============================================================================="
    if "$@"; then
        echo "ok: $name"
    else
        echo "failed: $name"
        FAILED+=("$name")
    fi
}

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# pyslam.slam binds to one sparse-SLAM core per process (see tests/conftest.py).
run_step "Python tests (C++ core)" python3 -m pytest tests -q
run_step "Python tests (Python core)" \
    env PYSLAM_USE_CPP=false python3 -m pytest tests/python_core tests/slam -q

run_step "C++ core tests" ./cpp/core/slam/tests_cpp/run_all_tests.sh
run_step "C++ core Python tests" ./cpp/core/slam/tests_py/run_all_tests.sh

# Script-style module checks. pytest ignores these (see pyproject.toml):
# test_volumetric.py and test_sim3solver.py take arguments, and test_pnpsolver.py plots.
for script in \
    tests/modules/test_volumetric.py \
    tests/modules/test_sim3solver.py
do
    run_step "$script" python3 "$script"
done
run_step "tests/modules/test_pnpsolver.py" env MPLBACKEND=Agg python3 tests/modules/test_pnpsolver.py

shopt -s nullglob
casters=(cpp/modules/test_opencv_casters/lib/cvcasters_test*.so)
if (( ${#casters[@]} > 0 )); then
    run_step "OpenCV casters" env RUN_BENCHMARKS=0 ./cpp/modules/test_opencv_casters/run_tests.sh
else
    echo "skip: OpenCV caster binding is not built (cpp/modules/test_opencv_casters)"
fi

echo
if (( ${#FAILED[@]} > 0 )); then
    echo "Failed:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All checks passed."
