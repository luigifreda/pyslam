#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd -P)"
BUILD_DIR="$SCRIPT_DIR/build"
LIB_DIR="$SCRIPT_DIR/lib"
CURRENT_BENCH="$BUILD_DIR/bench_opencv_casters_current"
V1_BENCH="$BUILD_DIR/bench_opencv_casters_v1"
PYTHON_STAMP_FILE="$BUILD_DIR/.python_bin"
TEST_PYTHON="${PYTHON_BIN:-}"
PYTHON_RUNTIME_PATHS=""
RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
BENCH_ITER_SCALE_PCT="${BENCH_ITER_SCALE_PCT:-20}"
BENCH_SAMPLES="${BENCH_SAMPLES:-5}"
ACTIVATE_ENV_SCRIPT="$ROOT_DIR/pyenv-activate.sh"

BUILD_STATUS="not_run"
PYTEST_STATUS="not_run"
BENCH_STATUS="not_run"
current_output=""
v1_output=""

BENCH_CASE_KEYS=(
    "gray_u8_contiguous"
    "color_u8_contiguous"
    "gray_u8_col_step2"
    "gray_u8_transposed"
    "color_u8_reverse"
    "gray_i64_normalized"
)
BENCH_CASE_LABELS=(
    "gray_u8_contiguous"
    "color_u8_contiguous"
    "gray_u8_col_step2"
    "gray_u8_transposed"
    "color_u8_reverse"
    "gray_i64_normalized"
)

print_section() {
    echo "=================================="
    echo "$1"
    echo "=================================="
}

extract_metric() {
    local output="$1"
    local key="$2"
    awk -F= -v key="$key" '$1 == key { print $2 }' <<<"$output"
}

append_candidate() {
    local candidate="$1"
    [[ -z "$candidate" ]] && return 0
    if [[ "$candidate" == */* ]]; then
        [[ -x "$candidate" ]] || return 0
    else
        command -v "$candidate" >/dev/null 2>&1 || return 0
        candidate="$(command -v "$candidate")"
    fi

    local existing
    for existing in "${PYTHON_CANDIDATES[@]:-}"; do
        [[ "$existing" == "$candidate" ]] && return 0
    done
    PYTHON_CANDIDATES+=("$candidate")
}

probe_python() {
    local python_bin="$1"
    "$python_bin" - <<'PY'
import importlib.util
import os
import sys
import sysconfig

def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

include_dirs = [sysconfig.get_path("include"), sysconfig.get_path("platinclude")]
has_include = any(path and os.path.isdir(path) for path in include_dirs)

lib_candidates = []
for base_key, name_key in (
    ("LIBDIR", "LDLIBRARY"),
    ("LIBDIR", "LIBRARY"),
    ("LIBPL", "LDLIBRARY"),
    ("LIBPL", "LIBRARY"),
):
    base = sysconfig.get_config_var(base_key)
    name = sysconfig.get_config_var(name_key)
    if base and name:
        lib_candidates.append(os.path.join(base, name))

has_libpython = any(path and os.path.exists(path) for path in lib_candidates)

fields = [
    sys.executable,
    sys.version.split()[0],
    "1" if has_module("numpy") else "0",
    "1" if has_module("pytest") else "0",
    "1" if has_include else "0",
    "1" if has_libpython else "0",
]
print("\t".join(fields))
PY
}

select_python() {
    local probe_output=""
    local candidate=""

    PYTHON_CANDIDATES=()
    append_candidate "$TEST_PYTHON"
    append_candidate "python"
    append_candidate "python3"
    append_candidate "/usr/bin/python"
    append_candidate "/usr/bin/python3"

    if ((${#PYTHON_CANDIDATES[@]} == 0)); then
        echo "ERROR: no Python interpreter candidates were found" >&2
        exit 1
    fi

    for candidate in "${PYTHON_CANDIDATES[@]}"; do
        probe_output="$(probe_python "$candidate")"
        IFS=$'\t' read -r candidate version has_numpy has_pytest has_include has_libpython <<<"$probe_output"

        if [[ "$has_numpy" == "1" && "$has_pytest" == "1" && "$has_include" == "1" && "$has_libpython" == "1" ]]; then
            TEST_PYTHON="$candidate"
            echo "Using Python interpreter: $TEST_PYTHON (version $version)"
            return 0
        fi

        echo "Skipping Python candidate: $candidate" >&2
        [[ "$has_numpy" == "1" ]] || echo "  - missing module: numpy" >&2
        [[ "$has_pytest" == "1" ]] || echo "  - missing module: pytest" >&2
        [[ "$has_include" == "1" ]] || echo "  - missing Python headers" >&2
        [[ "$has_libpython" == "1" ]] || echo "  - missing libpython for embedding" >&2
    done

    echo "ERROR: could not find a Python interpreter with numpy, pytest, headers, and libpython." >&2
    echo "Set PYTHON_BIN explicitly to a compatible interpreter and re-run." >&2
    exit 1
}

invalidate_build_cache_if_python_changed() {
    local previous_python=""

    if [[ ! -f "$PYTHON_STAMP_FILE" && -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        echo "Existing build cache has no recorded Python interpreter."
        echo "Clearing CMake cache under $BUILD_DIR to avoid reusing stale Python settings."
        rm -f "$BUILD_DIR/CMakeCache.txt"
        rm -rf "$BUILD_DIR/CMakeFiles"
    fi

    if [[ -f "$PYTHON_STAMP_FILE" ]]; then
        previous_python="$(<"$PYTHON_STAMP_FILE")"
    fi

    if [[ -n "$previous_python" && "$previous_python" != "$TEST_PYTHON" ]]; then
        echo "Python interpreter changed since the last configure step."
        echo "Clearing CMake cache under $BUILD_DIR to avoid mixed Python artifacts."
        rm -f "$BUILD_DIR/CMakeCache.txt"
        rm -rf "$BUILD_DIR/CMakeFiles"
    fi

    mkdir -p "$BUILD_DIR"
    printf '%s\n' "$TEST_PYTHON" >"$PYTHON_STAMP_FILE"
}

ensure_shared_module_exists() {
    shopt -s nullglob
    modules=("$LIB_DIR"/cvcasters_test*.so)
    shopt -u nullglob
    if ((${#modules[@]} == 0)); then
        echo "ERROR: cvcasters_test module not found under $LIB_DIR" >&2
        exit 1
    fi
}

ensure_benchmark_exists() {
    local binary="$1"
    [[ -x "$binary" ]] && return 0
    echo "ERROR: benchmark not found: $binary" >&2
    exit 1
}

linked_python_library() {
    local binary="$1"
    ldd "$binary" | awk '/libpython/ { print $3; exit }'
}

selected_python_version() {
    "$TEST_PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'
}

verify_benchmark_python_linkage() {
    local binary="$1"
    local linked_python=""
    local version=""

    linked_python="$(linked_python_library "$binary")"
    version="$(selected_python_version)"

    if [[ -z "$linked_python" ]]; then
        echo "ERROR: could not determine the libpython linked by $binary" >&2
        return 1
    fi

    if [[ "$linked_python" != *"python${version}"* ]]; then
        echo "ERROR: benchmark binary is linked against a different Python runtime." >&2
        echo "  selected interpreter: $TEST_PYTHON (Python $version)" >&2
        echo "  linked libpython:    $linked_python" >&2
        echo "Rebuild after fixing the Python/CMake configuration, or set RUN_BENCHMARKS=0 to skip benchmarks." >&2
        return 1
    fi
}

print_benchmark_comparison() {
    local current="$1"
    local baseline="$2"
    local idx key label current_adj current_alias baseline_adj baseline_alias speedup

    echo
    echo "Benchmark summary (median adjusted ns/op, alias %)"
    printf "%-24s %-18s %-10s %-18s %-10s %-10s\n" "case" "current adj ns/op" "alias %" "v1 adj ns/op" "alias %" "speedup"
    for idx in "${!BENCH_CASE_KEYS[@]}"; do
        key="${BENCH_CASE_KEYS[$idx]}"
        label="${BENCH_CASE_LABELS[$idx]}"
        current_adj="$(extract_metric "$current" "details.${key}.median_adjusted_ns_per_op")"
        current_alias="$(extract_metric "$current" "details.${key}.alias_rate_pct")"
        baseline_adj="$(extract_metric "$baseline" "details.${key}.median_adjusted_ns_per_op")"
        baseline_alias="$(extract_metric "$baseline" "details.${key}.alias_rate_pct")"
        speedup="$(python3 - <<PY
current = "${current_adj}"
baseline = "${baseline_adj}"
try:
    c = float(current)
    b = float(baseline)
    if c > 0:
        print(f"{b / c:.2f}x")
    else:
        print("n/a")
except Exception:
    print("n/a")
PY
)"
        printf "%-24s %-18s %-10s %-18s %-10s %-10s\n" \
            "$label" "${current_adj:-n/a}" "${current_alias:-n/a}" "${baseline_adj:-n/a}" "${baseline_alias:-n/a}" "${speedup:-n/a}"
    done
}

print_summary() {
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        [[ "$BUILD_STATUS" == "running" ]] && BUILD_STATUS="failed"
        [[ "$PYTEST_STATUS" == "running" ]] && PYTEST_STATUS="failed"
        [[ "$BENCH_STATUS" == "running" ]] && BENCH_STATUS="failed"
    fi

    echo
    print_section "Final Summary"
    printf "%-12s %s\n" "build" "$BUILD_STATUS"
    printf "%-12s %s\n" "pytest" "$PYTEST_STATUS"
    printf "%-12s %s\n" "benchmarks" "$BENCH_STATUS"

    if [[ -n "$current_output" && -n "$v1_output" ]]; then
        print_benchmark_comparison "$current_output" "$v1_output"
    fi

    if [[ $exit_code -eq 0 ]]; then
        echo
        echo "All steps completed successfully."
    else
        echo
        echo "Run failed with exit code $exit_code."
    fi
}

trap print_summary EXIT

cd "$SCRIPT_DIR"

if [[ -z "$TEST_PYTHON" && -f "$ACTIVATE_ENV_SCRIPT" ]]; then
    echo "Activating project Python environment via $ACTIVATE_ENV_SCRIPT"
    # shellcheck source=/dev/null
    . "$ACTIVATE_ENV_SCRIPT"
fi

select_python
export PYTHON_BIN="$TEST_PYTHON"
invalidate_build_cache_if_python_changed

print_section "Building OpenCV Type Casters Tests"
BUILD_STATUS="running"
"$SCRIPT_DIR/build.sh" "$@"
BUILD_STATUS="ok"

PYTHON_RUNTIME_PATHS="$("$TEST_PYTHON" -c 'import sys; print(":".join(p for p in sys.path if p))')"
export PYTHONPATH="$LIB_DIR${PYTHON_RUNTIME_PATHS:+:$PYTHON_RUNTIME_PATHS}${PYTHONPATH:+:$PYTHONPATH}"
ensure_shared_module_exists

print_section "Running OpenCV Type Casters Tests"
PYTEST_STATUS="running"
# Disable auto-loading external pytest plugins from the user environment to
# avoid unrelated failures (e.g., torchtyping with incompatible torch).
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$TEST_PYTHON" -m pytest "$SCRIPT_DIR/test_casters.py" -v --tb=short
PYTEST_STATUS="ok"

print_section "Running OpenCV Type Casters Benchmarks"
if [[ "$RUN_BENCHMARKS" == "0" ]]; then
    BENCH_STATUS="skipped"
    echo "Skipping benchmarks because RUN_BENCHMARKS=0"
else
    ensure_benchmark_exists "$CURRENT_BENCH"
    ensure_benchmark_exists "$V1_BENCH"
    verify_benchmark_python_linkage "$CURRENT_BENCH"
    verify_benchmark_python_linkage "$V1_BENCH"

    BENCH_STATUS="running"
    echo "[current]"
    current_output="$(BENCH_ITER_SCALE_PCT="$BENCH_ITER_SCALE_PCT" BENCH_SAMPLES="$BENCH_SAMPLES" "$CURRENT_BENCH")"
    echo "$current_output"

    echo "----------------------------------"
    echo "[v1]"
    v1_output="$(BENCH_ITER_SCALE_PCT="$BENCH_ITER_SCALE_PCT" BENCH_SAMPLES="$BENCH_SAMPLES" "$V1_BENCH")"
    echo "$v1_output"
    BENCH_STATUS="ok"
fi
