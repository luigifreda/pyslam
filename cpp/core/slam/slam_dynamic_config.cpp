/*
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
#include "slam_dynamic_config.h"

#include "config_parameters.h"
#include "frontend_helpers.h"

#include <algorithm>
#include <iostream>

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace pyslam {

SLAMDynamicConfig::SLAMDynamicConfig(float init_max_descriptor_distance)
    : descriptor_distance_sigma(init_max_descriptor_distance),
      descriptor_distance_min(init_max_descriptor_distance *
                              (1.f - descriptor_distance_max_delta_fraction)),
      descriptor_distance_max(init_max_descriptor_distance *
                              (1.f + descriptor_distance_max_delta_fraction)),
      reproj_err_frame_map_sigma(Parameters::kMaxReprojectionDistanceMap) {}

float SLAMDynamicConfig::update_descriptor_stats(const FramePtr &f_ref, const FramePtr &f_cur,
                                                 const std::vector<int> &idxs_ref,
                                                 const std::vector<int> &idxs_cur) {
    if (!idxs_cur.empty() && f_ref && f_cur && !f_ref->des.empty() && !f_cur->des.empty()) {
        const std::vector<float> distances = descriptor_distances_for_matches(
            f_cur->des, idxs_cur, f_ref->des, idxs_ref);
        const auto [sigma_mad, dists_median] =
            descriptor_sigma_mad(distances, Parameters::kUseDescriptorSigmaMadv2);
        const float delta = Parameters::kUseDescriptorSigmaMadv2
                                ? descriptor_distance_factor * sigma_mad + dists_median
                                : descriptor_distance_factor * sigma_mad;
        descriptor_distance_sigma =
            descriptor_distance_alpha * descriptor_distance_sigma +
            (1.f - descriptor_distance_alpha) * delta;
        descriptor_distance_sigma =
            std::clamp(descriptor_distance_sigma, descriptor_distance_min, descriptor_distance_max);
        Parameters::kMaxDescriptorDistance = descriptor_distance_sigma;
#ifdef USE_PYTHON
        py::gil_scoped_acquire gil;
        py::module_::import("pyslam.config_parameters").attr("Parameters").attr(
            "kMaxDescriptorDistance") = descriptor_distance_sigma;
        py::module_::import("pyslam.slam.feature_tracker_shared")
            .attr("FeatureTrackerShared")
            .attr("update_cpp_module_dynamic_config_parameters")();
#endif
    } else {
        std::cerr << "[SLAMDynamicConfig] No matches to initialize descriptor sigma\n";
    }
    return descriptor_distance_sigma;
}

float SLAMDynamicConfig::update_reproj_err_map_stats(float value) {
    reproj_err_frame_map_sigma = reproj_err_frame_map_alpha * reproj_err_frame_map_sigma +
                                 (1.f - reproj_err_frame_map_alpha) * value;
    reproj_err_frame_map_sigma = std::max(1.f, reproj_err_frame_map_sigma);
    return reproj_err_frame_map_sigma;
}

} // namespace pyslam
