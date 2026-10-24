#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifndef OPENCV_CASTER_HEADER_PATH
#error "OPENCV_CASTER_HEADER_PATH must point to the caster header under test"
#endif

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#include STRINGIFY(OPENCV_CASTER_HEADER_PATH)

namespace py = pybind11;

namespace {

constexpr int kDefaultSamples = 5;
volatile std::size_t g_sink = 0;

struct BenchmarkCase {
    std::string name;
    py::array array;
    int iterations;
};

struct CastLoopResult {
    std::size_t checksum = 0;
    int alias_hits = 0;
};

struct Stats {
    double min = 0.0;
    double median = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double stdev = 0.0;
};

struct BenchmarkResult {
    BenchmarkCase bench_case;
    Stats total_ms;
    Stats ns_per_op;
    Stats adjusted_ns_per_op;
    double alias_rate = 0.0;
    std::string dtype;
    std::string shape;
    std::string strides;
};

int env_to_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || *value == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

int scale_iterations(int base_iterations, int scale_pct) {
    const long long scaled =
        (static_cast<long long>(base_iterations) * static_cast<long long>(scale_pct) + 99LL) /
        100LL;
    return std::max(1, static_cast<int>(scaled));
}

std::string format_vector(const std::vector<ssize_t> &values) {
    std::ostringstream oss;
    oss << "(";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << ")";
    return oss.str();
}

bool progress_enabled() {
    return env_to_int("BENCH_PROGRESS", 1) != 0;
}

void print_progress_bar(const std::string &label, int current, int total) {
    if (!progress_enabled()) {
        return;
    }

    const bool interactive = ::isatty(STDERR_FILENO) != 0;
    const int width = 28;
    const int clamped_total = std::max(total, 1);
    const int clamped_current = std::max(0, std::min(current, clamped_total));
    const int filled = (clamped_current * width) / clamped_total;

    std::ostringstream oss;
    oss << '[';
    for (int i = 0; i < width; ++i) {
        oss << (i < filled ? '#' : '-');
    }
    oss << "] " << clamped_current << '/' << clamped_total << "  " << label;

    if (interactive) {
        std::cerr << '\r' << oss.str() << std::flush;
        if (clamped_current == clamped_total) {
            std::cerr << '\n';
        }
    } else {
        std::cerr << oss.str() << '\n';
    }
}

Stats compute_stats(std::vector<double> samples) {
    Stats stats;
    if (samples.empty()) {
        return stats;
    }

    std::sort(samples.begin(), samples.end());
    stats.min = samples.front();
    stats.max = samples.back();
    stats.median = samples[samples.size() / 2];
    stats.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                 static_cast<double>(samples.size());

    double variance = 0.0;
    for (double sample : samples) {
        const double delta = sample - stats.mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(samples.size());
    stats.stdev = std::sqrt(variance);
    return stats;
}

std::size_t run_baseline_loop(const py::array &arr, int iterations) {
    const py::buffer_info info = arr.request();
    const std::size_t rows =
        info.ndim >= 1 ? static_cast<std::size_t>(info.shape[0]) : 0U;
    const std::size_t cols =
        info.ndim >= 2 ? static_cast<std::size_t>(info.shape[1]) : 1U;
    const std::size_t channels =
        info.ndim >= 3 ? static_cast<std::size_t>(info.shape[2]) : 1U;
    const std::size_t dtype_size = static_cast<std::size_t>(info.itemsize);
    const std::size_t pointer_bits =
        reinterpret_cast<std::uintptr_t>(info.ptr) & 0xFFU;

    std::size_t checksum = 0;
    for (int i = 0; i < iterations; ++i) {
        checksum += rows;
        checksum += cols;
        checksum += channels;
        checksum += dtype_size;
        checksum += pointer_bits;
        checksum += static_cast<std::size_t>(i & 0x7);
    }
    return checksum;
}

CastLoopResult run_cast_loop(const py::array &arr, int iterations) {
    const py::buffer_info info = arr.request();
    const auto *original_ptr = static_cast<const std::uint8_t *>(info.ptr);

    CastLoopResult result;
    for (int i = 0; i < iterations; ++i) {
        cv::Mat m = py::cast<cv::Mat>(arr);
        result.checksum += static_cast<std::size_t>(m.rows);
        result.checksum += static_cast<std::size_t>(m.cols);
        result.checksum += static_cast<std::size_t>(m.type());
        result.checksum += reinterpret_cast<std::uintptr_t>(m.data) & 0xFFU;
        result.alias_hits += (m.data == original_ptr) ? 1 : 0;
    }
    return result;
}

BenchmarkResult run_benchmark(const BenchmarkCase &bench_case, int case_index, int total_cases) {
    std::vector<double> total_ms_samples;
    std::vector<double> ns_per_op_samples;
    std::vector<double> adjusted_ns_per_op_samples;
    const int samples = env_to_int("BENCH_SAMPLES", kDefaultSamples);
    total_ms_samples.reserve(samples);
    ns_per_op_samples.reserve(samples);
    adjusted_ns_per_op_samples.reserve(samples);

    const int warmup_iterations = std::max(bench_case.iterations / 20, 1000);
    print_progress_bar("case " + std::to_string(case_index) + "/" + std::to_string(total_cases) +
                           " warmup " + bench_case.name,
                       0, 1);
    g_sink += run_baseline_loop(bench_case.array, warmup_iterations);
    g_sink += run_cast_loop(bench_case.array, warmup_iterations).checksum;
    print_progress_bar("case " + std::to_string(case_index) + "/" + std::to_string(total_cases) +
                           " warmup " + bench_case.name,
                       1, 1);

    int alias_hits = 0;
    int total_casts = 0;
    for (int sample_idx = 0; sample_idx < samples; ++sample_idx) {
        print_progress_bar("case " + std::to_string(case_index) + "/" + std::to_string(total_cases) +
                               " sample " + bench_case.name,
                           sample_idx, samples);
        double baseline_ns = 0.0;
        double cast_ns = 0.0;

        if ((sample_idx % 2) == 0) {
            const auto baseline_start = std::chrono::steady_clock::now();
            const std::size_t baseline_checksum =
                run_baseline_loop(bench_case.array, bench_case.iterations);
            const auto baseline_end = std::chrono::steady_clock::now();
            baseline_ns = std::chrono::duration<double, std::nano>(baseline_end - baseline_start)
                              .count();
            g_sink += baseline_checksum;

            const auto cast_start = std::chrono::steady_clock::now();
            const CastLoopResult cast_result =
                run_cast_loop(bench_case.array, bench_case.iterations);
            const auto cast_end = std::chrono::steady_clock::now();
            cast_ns =
                std::chrono::duration<double, std::nano>(cast_end - cast_start).count();
            g_sink += cast_result.checksum;
            alias_hits += cast_result.alias_hits;
        } else {
            const auto cast_start = std::chrono::steady_clock::now();
            const CastLoopResult cast_result =
                run_cast_loop(bench_case.array, bench_case.iterations);
            const auto cast_end = std::chrono::steady_clock::now();
            cast_ns =
                std::chrono::duration<double, std::nano>(cast_end - cast_start).count();
            g_sink += cast_result.checksum;
            alias_hits += cast_result.alias_hits;

            const auto baseline_start = std::chrono::steady_clock::now();
            const std::size_t baseline_checksum =
                run_baseline_loop(bench_case.array, bench_case.iterations);
            const auto baseline_end = std::chrono::steady_clock::now();
            baseline_ns = std::chrono::duration<double, std::nano>(baseline_end - baseline_start)
                              .count();
            g_sink += baseline_checksum;
        }

        total_casts += bench_case.iterations;
        total_ms_samples.push_back(cast_ns / 1e6);
        ns_per_op_samples.push_back(cast_ns / static_cast<double>(bench_case.iterations));
        adjusted_ns_per_op_samples.push_back(
            std::max(0.0, cast_ns - baseline_ns) / static_cast<double>(bench_case.iterations));
        print_progress_bar("case " + std::to_string(case_index) + "/" + std::to_string(total_cases) +
                               " sample " + bench_case.name,
                           sample_idx + 1, samples);
    }

    const py::buffer_info info = bench_case.array.request();
    BenchmarkResult result{
        bench_case,
        compute_stats(total_ms_samples),
        compute_stats(ns_per_op_samples),
        compute_stats(adjusted_ns_per_op_samples),
        total_casts > 0 ? static_cast<double>(alias_hits) / static_cast<double>(total_casts) : 0.0,
        py::str(bench_case.array.dtype()),
        format_vector(info.shape),
        format_vector(info.strides),
    };
    return result;
}

void print_summary_row(const BenchmarkResult &result) {
    std::cout << std::left << std::setw(24) << result.bench_case.name << std::right
              << std::setw(11) << result.bench_case.iterations << std::setw(15)
              << std::fixed << std::setprecision(2) << result.ns_per_op.median
              << std::setw(16) << result.adjusted_ns_per_op.median << std::setw(14)
              << result.total_ms.median << std::setw(10) << std::setprecision(1)
              << (result.alias_rate * 100.0) << '\n';
}

} // namespace

int main() {
    py::scoped_interpreter guard{};
    py::module_ np = py::module_::import("numpy");
    py::object builtins = py::module_::import("builtins");
    const int iteration_scale_pct = env_to_int("BENCH_ITER_SCALE_PCT", 20);
    const int samples = env_to_int("BENCH_SAMPLES", kDefaultSamples);

    py::object full_slice = builtins.attr("slice")(py::none(), py::none(), py::none());
    py::object step2_slice = builtins.attr("slice")(0, py::none(), 2);
    py::object reverse_slice = builtins.attr("slice")(py::none(), py::none(), -1);

    py::array gray = py::array::ensure(
        np.attr("full")(py::make_tuple(480, 640), 7, py::arg("dtype") = "uint8"));
    py::array color = py::array::ensure(
        np.attr("full")(py::make_tuple(480, 640, 3), 11, py::arg("dtype") = "uint8"));
    py::array sliced = py::array::ensure(
        gray.attr("__getitem__")(py::make_tuple(full_slice, step2_slice)));
    py::array transposed = py::array::ensure(gray.attr("transpose")(py::make_tuple(1, 0)));
    py::array channel_reversed = py::array::ensure(
        color.attr("__getitem__")(py::make_tuple(full_slice, full_slice, reverse_slice)));
    py::array int64_gray = py::array::ensure(
        np.attr("full")(py::make_tuple(480, 640), 42, py::arg("dtype") = "int64"));

    std::vector<BenchmarkCase> cases;
    cases.emplace_back(
        BenchmarkCase{"gray_u8_contiguous", gray, scale_iterations(250000, iteration_scale_pct)});
    cases.emplace_back(
        BenchmarkCase{"color_u8_contiguous", color,
                      scale_iterations(250000, iteration_scale_pct)});
    cases.emplace_back(
        BenchmarkCase{"gray_u8_col_step2", sliced, scale_iterations(25000, iteration_scale_pct)});
    cases.emplace_back(BenchmarkCase{"gray_u8_transposed", transposed,
                                     scale_iterations(12000, iteration_scale_pct)});
    cases.emplace_back(BenchmarkCase{"color_u8_reverse", channel_reversed,
                                     scale_iterations(8000, iteration_scale_pct)});
    cases.emplace_back(BenchmarkCase{"gray_i64_normalized", int64_gray,
                                     scale_iterations(6000, iteration_scale_pct)});

    std::vector<BenchmarkResult> results;
    results.reserve(cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i) {
        results.push_back(run_benchmark(cases[i], static_cast<int>(i + 1),
                                        static_cast<int>(cases.size())));
    }

    std::cout << "samples=" << samples << '\n';
    std::cout << "iteration_scale_pct=" << iteration_scale_pct << '\n';
    std::cout << "metric=median_of_samples\n";
    std::cout << "warmup_iterations=max(iterations/20,1000)\n";
    std::cout << "reported_ns_per_op=raw_cast_time\n";
    std::cout << "adjusted_ns_per_op=cast_time_minus_loop_baseline\n";
    std::cout << "alias_rate_pct=percent_of_casts_reusing_original_numpy_buffer\n";
    std::cout << '\n';

    std::cout << std::left << std::setw(24) << "case" << std::right << std::setw(11) << "iters"
              << std::setw(15) << "ns/op" << std::setw(16) << "adj ns/op" << std::setw(14)
              << "median ms" << std::setw(10) << "alias %" << '\n';
    std::cout << std::string(90, '-') << '\n';
    for (const auto &result : results) {
        print_summary_row(result);
    }

    std::cout << '\n';
    for (const auto &result : results) {
        std::cout << "details." << result.bench_case.name << ".dtype=" << result.dtype << '\n';
        std::cout << "details." << result.bench_case.name << ".shape=" << result.shape << '\n';
        std::cout << "details." << result.bench_case.name << ".strides=" << result.strides
                  << '\n';
        std::cout << "details." << result.bench_case.name
                  << ".median_ns_per_op=" << std::fixed << std::setprecision(2)
                  << result.ns_per_op.median << '\n';
        std::cout << "details." << result.bench_case.name
                  << ".min_ns_per_op=" << result.ns_per_op.min << '\n';
        std::cout << "details." << result.bench_case.name
                  << ".max_ns_per_op=" << result.ns_per_op.max << '\n';
        std::cout << "details." << result.bench_case.name
                  << ".stdev_ns_per_op=" << result.ns_per_op.stdev << '\n';
        std::cout << "details." << result.bench_case.name
                  << ".median_adjusted_ns_per_op=" << result.adjusted_ns_per_op.median << '\n';
        std::cout << "details." << result.bench_case.name
                  << ".alias_rate_pct=" << (result.alias_rate * 100.0) << '\n';
    }

    for (const auto &result : results) {
        if (result.bench_case.name == "gray_u8_contiguous") {
            std::cout << "gray_ms=" << std::fixed << std::setprecision(6)
                      << result.total_ms.median << '\n';
        } else if (result.bench_case.name == "color_u8_contiguous") {
            std::cout << "color_ms=" << std::fixed << std::setprecision(6)
                      << result.total_ms.median << '\n';
        } else if (result.bench_case.name == "gray_u8_col_step2") {
            std::cout << "sliced_ms=" << std::fixed << std::setprecision(6)
                      << result.total_ms.median << '\n';
        }
    }

    std::cout << "global_sink=" << g_sink << '\n';
    return 0;
}
