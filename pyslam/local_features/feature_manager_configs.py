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

from .feature_manager import feature_manager_factory
from .feature_types import FeatureDetectorTypes, FeatureDescriptorTypes

from pyslam.config_parameters import Parameters


kNumFeatures = Parameters.kNumFeatures


"""
Interface for feature manager configurations 
"""


class FeatureManagerConfigs(object):

    # Template configuration used by FeatureManagerConfigs.extract_from() to extract
    # a FeatureManager subconfiguration from a full FeatureTracker input configuration.
    # See for instance the file test_feature_manager.py.
    TEMPLATE = dict(
        num_features=kNumFeatures,
        num_levels=8,
        scale_factor=1.2,
        sigma_level0=Parameters.kSigmaLevel0,
        detector_type=FeatureDetectorTypes.NONE,
        descriptor_type=FeatureDescriptorTypes.NONE,
    )

    # Extract the FeatureManager sub-configuration from a full FeatureTracker input configuration (selected from FeatureTrackerConfigs)
    # See for instance the file test_feature_manager.py.
    @staticmethod
    def extract_from(dict_in):
        dict_out = {
            key: dict_in[key] for key in FeatureManagerConfigs.TEMPLATE.keys() if key in dict_in
        }
        return dict_out

    # NOTE:
    # The following configurations were manually extracted from the corresponding tracker configs in feature_tracker_configs.py.
    # These are just examples.
    # Normally, you should use the FeatureManagerConfigs.extract_from() method instead of manually specifying these configurations.
    # See the comments/notes above and for instance the file test_feature_manager.py.
    ORB2 = dict(
        num_features=kNumFeatures,
        num_levels=8,
        scale_factor=1.2,
        detector_type=FeatureDetectorTypes.ORB2,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        sigma_level0=Parameters.kSigmaLevel0,
        deterministic=False,
    )

    BRISK = dict(
        num_features=kNumFeatures,
        num_levels=4,
        scale_factor=1.2,
        detector_type=FeatureDetectorTypes.BRISK,
        descriptor_type=FeatureDescriptorTypes.BRISK,
        sigma_level0=Parameters.kSigmaLevel0,
    )

    ROOT_SIFT = dict(
        num_features=kNumFeatures,  # independently computes the number of octaves as SIFT
        detector_type=FeatureDetectorTypes.ROOT_SIFT,
        descriptor_type=FeatureDescriptorTypes.ROOT_SIFT,
        sigma_level0=Parameters.kSigmaLevel0,
    )

    SUPERPOINT = dict(
        num_features=kNumFeatures,  # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
        num_levels=1,
        scale_factor=1.2,
        detector_type=FeatureDetectorTypes.SUPERPOINT,
        descriptor_type=FeatureDescriptorTypes.SUPERPOINT,
        sigma_level0=Parameters.kSigmaLevel0,
    )

    XFEAT = dict(
        num_features=kNumFeatures,  # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
        num_levels=1,
        scale_factor=1.2,
        detector_type=FeatureDetectorTypes.XFEAT,
        descriptor_type=FeatureDescriptorTypes.XFEAT,
        sigma_level0=Parameters.kSigmaLevel0,
    )
