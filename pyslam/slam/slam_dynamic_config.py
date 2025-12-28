"""
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
"""

import math
import numpy as np

from pyslam.utilities.features import descriptor_sigma_mad, descriptor_sigma_mad_v2
from pyslam.utilities.logging import Printer

from .feature_tracker_shared import FeatureTrackerShared

from pyslam.config_parameters import Parameters


# experimental
class SLAMDynamicConfig:
    def __init__(self, init_max_descriptor_distance=Parameters.kMaxDescriptorDistance):

        self.descriptor_distance_sigma = init_max_descriptor_distance
        self.descriptor_distance_alpha = 0.9
        self.descriptor_distance_factor = 3
        self.descriptor_distance_max_delta_fraction = 0.3
        self.descriptor_distance_min = init_max_descriptor_distance * (
            1.0 - self.descriptor_distance_max_delta_fraction
        )
        self.descriptor_distance_max = init_max_descriptor_distance * (
            1.0 + self.descriptor_distance_max_delta_fraction
        )

        self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMap
        self.reproj_err_frame_map_alpha = 0.9
        self.reproj_err_frame_map_factor = 3

    def update_descriptor_stats(self, f_ref, f_cur, idxs_ref, idxs_cur):
        if len(idxs_cur) > 0:
            des_cur = f_cur.des[idxs_cur]
            des_ref = f_ref.des[idxs_ref]
            if Parameters.kUseDescriptorSigmaMadv2:
                sigma_mad, dists_median, _ = descriptor_sigma_mad_v2(
                    des_cur, des_ref, descriptor_distances=FeatureTrackerShared.descriptor_distances
                )
                delta = (
                    self.descriptor_distance_factor * sigma_mad + dists_median
                )  # the final "+ dists_median" is for adding back a bias (the median itself) to the delta threshold since the delta distribution is not centered at zero
            else:
                sigma_mad, dists_median, _ = descriptor_sigma_mad(
                    des_cur, des_ref, descriptor_distances=FeatureTrackerShared.descriptor_distances
                )
                delta = self.descriptor_distance_factor * sigma_mad
            if self.descriptor_distance_sigma is None:
                self.descriptor_distance_sigma = delta
            else:
                self.descriptor_distance_sigma = (
                    self.descriptor_distance_alpha * self.descriptor_distance_sigma
                    + (1.0 - self.descriptor_distance_alpha) * delta
                )
                # clamp the descriptor distance sigma between the min and max values
                self.descriptor_distance_sigma = max(
                    min(self.descriptor_distance_sigma, self.descriptor_distance_max),
                    self.descriptor_distance_min,
                )
            print("descriptor sigma: ", self.descriptor_distance_sigma)
            # dynamicall update the static parameter descriptor distance
            Parameters.kMaxDescriptorDistance = self.descriptor_distance_sigma
            FeatureTrackerShared.update_cpp_module_dynamic_config_parameters()
        else:
            Printer.red(f"[SLAMDynamicConfig] No matches to initialize descriptor sigma")
            self.descriptor_distance_sigma = None
        return self.descriptor_distance_sigma

    def update_reproj_err_map_stats(self, value):
        self.reproj_err_frame_map_sigma = (
            self.reproj_err_frame_map_alpha * self.reproj_err_frame_map_sigma
            + (1.0 - self.reproj_err_frame_map_alpha) * value
        )
        self.reproj_err_frame_map_sigma = max(1.0, self.reproj_err_frame_map_sigma)
        return self.reproj_err_frame_map_sigma
