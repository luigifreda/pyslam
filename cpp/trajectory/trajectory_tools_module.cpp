/**
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PYSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
 */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "trajectory_alignement_incremental.h"
#include "trajectory_alignment.h"

namespace py = pybind11;

// 1: Use zero-memory copy for mapping numpy arrays to Eigen matrices and GIL release
// 0: Use regular management with copy of numpy arrays to Eigen matrices and GIL release
#define USE_ZERO_MEMORY_COPY 1

namespace {

// Mapping helpers requiring contiguous buffers to ensure zero-copy
inline Eigen::Map<const Eigen::VectorXd> map_vector_1d(const py::array &arr) {
    if (!py::isinstance<py::array_t<double>>(arr))
        throw std::invalid_argument("Expected numpy.ndarray of dtype float64 for 1D vector");
    py::buffer_info info = arr.request();
    if (info.ndim != 1)
        throw std::invalid_argument("Expected 1D array");
    if (!(arr.flags() & py::array::c_style))
        throw std::invalid_argument(
            "Vector must be C-contiguous for zero-copy mapping. "
            "For sliced numpy views with arbitrary strides, use np.ascontiguousarray() to create "
            "a contiguous copy first.");
    return Eigen::Map<const Eigen::VectorXd>(static_cast<const double *>(info.ptr),
                                             static_cast<Eigen::Index>(info.shape[0]));
}

using StridesDyn = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
using MapConst3xDyn = Eigen::Map<const Eigen::Matrix<double, 3, Eigen::Dynamic>, 0, StridesDyn>;

inline Eigen::Index stride_in_scalars(py::ssize_t stride_bytes) {
    constexpr py::ssize_t scalar_size = static_cast<py::ssize_t>(sizeof(double));
    if (stride_bytes % scalar_size != 0)
        throw std::invalid_argument("Array strides must be multiples of element size (float64)");
    return static_cast<Eigen::Index>(stride_bytes / scalar_size);
}

inline MapConst3xDyn map_points_array(const py::array &arr, const char *name) {
    if (!py::isinstance<py::array_t<double>>(arr))
        throw std::invalid_argument(std::string(name) +
                                    " must be a numpy.ndarray of dtype float64");
    py::buffer_info info = arr.request();
    if (info.ndim != 2)
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    const bool first_dim_three = info.shape[0] == 3;
    const bool second_dim_three = info.shape[1] == 3;
    if (!first_dim_three && !second_dim_three)
        throw std::invalid_argument(std::string(name) +
                                    " must have shape (3, N) or (N, 3) with float64 entries");

    const bool c_contig = static_cast<bool>(arr.flags() & py::array::c_style);
    const bool f_contig = static_cast<bool>(arr.flags() & py::array::f_style);
    if (!c_contig && !f_contig)
        throw std::invalid_argument(
            std::string(name) +
            " must be contiguous (C or Fortran). "
            "For sliced numpy views with arbitrary strides, use np.ascontiguousarray() or "
            "np.asfortranarray() to create a contiguous copy first.");

    const double *ptr = static_cast<const double *>(info.ptr);
    if (first_dim_three) {
        // Shape is (3, N): numpy row-major layout [x0...x(N-1), y0...y(N-1), z0...z(N-1)]
        // Eigen 3xN column-major: column j = [xj, yj, zj]
        // OuterStride (between columns) = strides[1], InnerStride (within column) = strides[0]
        const Eigen::Index cols = static_cast<Eigen::Index>(info.shape[1]);
        const StridesDyn stride(stride_in_scalars(info.strides[1]),
                                stride_in_scalars(info.strides[0]));
        return MapConst3xDyn(ptr, 3, cols, stride);
    }

    // Shape is (N, 3): numpy strides automatically encode C vs F layout
    // Eigen 3xN column-major: column j = [xj, yj, zj]
    // OuterStride (between columns) = strides[0], InnerStride (within column) = strides[1]
    const Eigen::Index cols = static_cast<Eigen::Index>(info.shape[0]);
    const StridesDyn stride(stride_in_scalars(info.strides[0]), stride_in_scalars(info.strides[1]));
    return MapConst3xDyn(ptr, 3, cols, stride);
}

} // namespace

PYBIND11_MODULE(trajectory_tools, m) {

#if USE_ZERO_MEMORY_COPY
    m.def(
        "find_trajectories_associations",
        [](const py::array &filter_timestamps, const py::array &filter_t_wi_arr,
           const py::array &gt_timestamps, const py::array &gt_t_wi_arr, double max_align_dt,
           bool verbose) {
            // Map numpy buffers without copy (timestamp vectors must be C-contiguous)
            auto ts_filter = map_vector_1d(filter_timestamps);
            auto ts_gt = map_vector_1d(gt_timestamps);
            auto filter_points = map_points_array(filter_t_wi_arr, "filter_t_wi");
            auto gt_points = map_points_array(gt_t_wi_arr, "gt_t_wi");

            // Validate that timestamps and points arrays have matching sizes
            if (ts_filter.size() != filter_points.cols()) {
                throw std::invalid_argument(
                    "filter_timestamps size (" + std::to_string(ts_filter.size()) +
                    ") must match filter_t_wi size (" + std::to_string(filter_points.cols()) + ")");
            }
            if (ts_gt.size() != gt_points.cols()) {
                throw std::invalid_argument("gt_timestamps size (" + std::to_string(ts_gt.size()) +
                                            ") must match gt_t_wi size (" +
                                            std::to_string(gt_points.cols()) + ")");
            }

            // Validate that GT trajectory has at least 2 samples for interpolation
            if (ts_gt.size() < 2) {
                throw std::invalid_argument(
                    "gt_timestamps must have at least 2 samples for interpolation, got " +
                    std::to_string(ts_gt.size()) +
                    ". The function requires at least 2 GT points "
                    "to interpolate between them.");
            }
            trajectory_tools::AssocResult res;
            {
                py::gil_scoped_release release;
                res = trajectory_tools::find_trajectories_associations_eigen(
                    ts_filter, filter_points, ts_gt, gt_points, max_align_dt, verbose);
            }

            return py::make_tuple(res.timestamps, res.filter_points.transpose(),
                                  res.gt_points.transpose());
        },
        py::arg("filter_timestamps"), py::arg("filter_t_wi"), py::arg("gt_timestamps"),
        py::arg("gt_t_wi"), py::arg("max_align_dt") = 1e-1, py::arg("verbose") = true,
        R"pbdoc(
            Find associations between 3D trajectories timestamps in filter and gt.

            Args:
                filter_timestamps: numpy float64 array shaped (Nf,), C-contiguous
                filter_t_wi: numpy float64 array shaped (Nf, 3) C-contiguous or (3, Nf) Fortran-contiguous.
                    For sliced views with arbitrary strides, use np.ascontiguousarray() or np.asfortranarray().
                gt_timestamps: numpy float64 array shaped (Ng,), C-contiguous. Must have Ng >= 2.
                gt_t_wi: numpy float64 array shaped (Ng, 3) C-contiguous or (3, Ng) Fortran-contiguous.
                    For sliced views with arbitrary strides, use np.ascontiguousarray() or np.asfortranarray().
                max_align_dt: Maximum allowed time difference
                verbose: If true, prints alignment info

            Returns:
                Tuple (timestamps_associations [M], filter_associations [Mx3], gt_associations [Mx3])
                
            Note:
                Requires gt_timestamps to have at least 2 samples for interpolation. If Ng < 2,
                the function will raise an exception.
        )pbdoc");

    m.def(
        "align_3d_points_with_svd",
        [](const py::array &gt_points_arr, const py::array &est_points_arr, bool find_scale) {
            Eigen::Matrix4d T_gt_est, T_est_gt;
            bool ok;
            auto gt_points = map_points_array(gt_points_arr, "gt_points");
            auto est_points = map_points_array(est_points_arr, "est_points");

            // Validate that both arrays have the same number of points
            if (gt_points.cols() != est_points.cols()) {
                throw std::invalid_argument("gt_points size (" + std::to_string(gt_points.cols()) +
                                            ") must match est_points size (" +
                                            std::to_string(est_points.cols()) + ")");
            }
            {
                py::gil_scoped_release release;
                std::tie(T_gt_est, T_est_gt, ok) = trajectory_tools::align_3d_points_with_svd_eigen(
                    gt_points, est_points, find_scale);
            }

            return py::make_tuple(T_gt_est, T_est_gt, ok);
        },
        py::arg("gt_points"), py::arg("est_points"), py::arg("find_scale") = true,
        R"pbdoc(
            Align corresponding 3D points using Kabsch-Umeyama SVD.

            Args:
                gt_points: numpy float64 array shaped (N, 3) C-contiguous or (3, N) Fortran-contiguous.
                    For sliced views with arbitrary strides, use np.ascontiguousarray() or np.asfortranarray().
                est_points: numpy float64 array shaped (N, 3) C-contiguous or (3, N) Fortran-contiguous.
                    For sliced views with arbitrary strides, use np.ascontiguousarray() or np.asfortranarray().
                find_scale: Estimate similarity scale (Sim(3))

            Returns:
                Tuple (T_gt_est [4x4], T_est_gt [4x4], success)
        )pbdoc");

#else

    m.def("find_trajectories_associations", &trajectory_tools::find_trajectories_associations,
          py::arg("filter_timestamps"), py::arg("filter_t_wi"), py::arg("gt_timestamps"),
          py::arg("gt_t_wi"), py::arg("max_align_dt") = 1e-1, py::arg("verbose") = true,
          R"pbdoc(
              Find associations between 3D trajectories timestamps in filter and gt.

              Args:
                  filter_timestamps: List of timestamps in seconds
                  filter_t_wi: List of 3D filter poses
                  gt_timestamps: List of timestamps in seconds
                  gt_t_wi: List of 3D ground truth poses
                  max_align_dt: Maximum allowed time difference
                  verbose: If true, prints alignment info

              Returns:
                  Tuple of (timestamps_associations, filter_associations, gt_associations)
          )pbdoc");

    m.def("align_3d_points_with_svd", &trajectory_tools::align_3d_points_with_svd,
          py::arg("gt_points"), py::arg("est_points"), py::arg("find_scale") = true,
          R"pbdoc(
            Align corresponding 3D points using Kabsch–Umeyama SVD.

            Args:
                gt_points: List of 3D ground truth points
                est_points: List of 3D estimated points
                find_scale: Estimate similarity scale (Sim(3))

            Returns:
                Tuple (T_gt_est [4x4], T_est_gt [4x4], success)
        )pbdoc");

#endif

    py::class_<trajectory_tools::AlignmentOptions,
               std::shared_ptr<trajectory_tools::AlignmentOptions>>(m, "AlignmentOptions")
        .def(py::init<>())
        .def_readwrite("max_align_dt", &trajectory_tools::AlignmentOptions::max_align_dt)
        .def_readwrite("find_scale", &trajectory_tools::AlignmentOptions::find_scale)
        .def_readwrite("svd_eps", &trajectory_tools::AlignmentOptions::svd_eps)
        .def_readwrite("verbose", &trajectory_tools::AlignmentOptions::verbose);

    py::class_<trajectory_tools::AlignmentResult,
               std::shared_ptr<trajectory_tools::AlignmentResult>>(m, "AlignmentResult")
        .def_readonly("T_gt_est", &trajectory_tools::AlignmentResult::T_gt_est)
        .def_readonly("T_est_gt", &trajectory_tools::AlignmentResult::T_est_gt)
        .def_readonly("valid", &trajectory_tools::AlignmentResult::valid)
        .def_readonly("n_pairs", &trajectory_tools::AlignmentResult::n_pairs)
        .def_readonly("sigma2_est", &trajectory_tools::AlignmentResult::sigma2_est)
        .def_readonly("mu_est", &trajectory_tools::AlignmentResult::mu_est)
        .def_readonly("mu_gt", &trajectory_tools::AlignmentResult::mu_gt)
        .def_readonly("singvals", &trajectory_tools::AlignmentResult::singvals);

    py::class_<trajectory_tools::IncrementalTrajectoryAlignerNoLBA,
               std::shared_ptr<trajectory_tools::IncrementalTrajectoryAlignerNoLBA>>(
        m, "IncrementalTrajectoryAlignerNoLBA")
        .def(py::init<const std::vector<double> &, const std::vector<Eigen::Vector3d> &>(),
             py::arg("gt_timestamps"), py::arg("gt_t_wi"))
        .def(py::init<const std::vector<double> &, const std::vector<Eigen::Vector3d> &,
                      const trajectory_tools::AlignmentOptions &>(),
             py::arg("gt_timestamps"), py::arg("gt_t_wi"), py::arg("opts"))
        .def("set_options", &trajectory_tools::IncrementalTrajectoryAlignerNoLBA::set_options,
             py::arg("opts"))
        .def("set_gt", &trajectory_tools::IncrementalTrajectoryAlignerNoLBA::set_gt,
             py::arg("gt_timestamps"), py::arg("gt_t_wi"))
        .def("reset", &trajectory_tools::IncrementalTrajectoryAlignerNoLBA::reset)
        .def("add_estimate", &trajectory_tools::IncrementalTrajectoryAlignerNoLBA::add_estimate,
             py::arg("timestamp"), py::arg("est_t_wi"))
        .def("result", &trajectory_tools::IncrementalTrajectoryAlignerNoLBA::result);

    py::class_<trajectory_tools::IncrementalTrajectoryAligner,
               std::shared_ptr<trajectory_tools::IncrementalTrajectoryAligner>>(
        m, "IncrementalTrajectoryAligner")
        .def(py::init<const std::vector<double> &, const std::vector<Eigen::Vector3d> &>(),
             py::arg("gt_timestamps"), py::arg("gt_t_wi"))
        .def(py::init<const std::vector<double> &, const std::vector<Eigen::Vector3d> &,
                      const trajectory_tools::AlignmentOptions &>(),
             py::arg("gt_timestamps"), py::arg("gt_t_wi"), py::arg("opts"))
        .def("set_options", &trajectory_tools::IncrementalTrajectoryAligner::set_options,
             py::arg("opts"))
        .def("set_gt", &trajectory_tools::IncrementalTrajectoryAligner::set_gt,
             py::arg("gt_timestamps"), py::arg("gt_t_wi"))
        .def("reset", &trajectory_tools::IncrementalTrajectoryAligner::reset)
        .def("update_trajectory",
             &trajectory_tools::IncrementalTrajectoryAligner::update_trajectory,
             py::arg("est_timestamps"), py::arg("est_positions"))
        .def("num_associations", &trajectory_tools::IncrementalTrajectoryAligner::num_associations)
        .def("get_estimated_timestamps",
             &trajectory_tools::IncrementalTrajectoryAligner::get_estimated_timestamps)
        .def("get_estimated_positions",
             &trajectory_tools::IncrementalTrajectoryAligner::get_estimated_positions)
        .def("get_gt_interpolated_positions",
             &trajectory_tools::IncrementalTrajectoryAligner::get_gt_interpolated_positions)
        .def("get_associated_pairs",
             &trajectory_tools::IncrementalTrajectoryAligner::get_associated_pairs)
        .def("result", &trajectory_tools::IncrementalTrajectoryAligner::result);
}
