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
#include <iostream>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "feature_shared_resources.h" // Add this
#include "frame.h"
#include "keyframe.h"
#include "map_point.h"
#include "semantic_types.h" // Add this
#include "utils/test_utils.h"

using namespace pyslam;
using namespace pyslam::test_utils;

int main() {
    try {
        // Build minimal Frame/KeyFrame graph with IDs
        auto f1 = std::make_shared<Frame>(101);
        auto f2 = std::make_shared<Frame>(102);
        auto kf1 = std::make_shared<KeyFrame>(201);
        auto kf2 = std::make_shared<KeyFrame>(202);

        // MapPoint with fields populated
        Eigen::Vector3d pt(1.0, 2.0, 3.0);
        Eigen::Matrix<unsigned char, 3, 1> color;
        color << 10, 20, 30;

        auto mp = std::make_shared<MapPoint>(pt, color, KeyFramePtr(nullptr), -1, /*id*/ 301);
        mp->normal = Eigen::Vector3d(0.0, 0.0, 1.0);
        mp->_min_distance = 0.5f;
        mp->_max_distance = 25.0f;
        mp->_is_bad = false;
        mp->_num_observations = 2;
        mp->num_times_visible = 7;
        mp->num_times_found = 5;
        mp->last_frame_id_seen = 1234;
        mp->first_kid = 999;
        mp->kf_ref = kf2;

        // Descriptors
        mp->des = cv::Mat(2, 32, CV_8U);
        cv::randu(mp->des, 0, 255);
        mp->semantic_des = cv::Mat(1, 8, CV_32F);
        cv::randu(mp->semantic_des, 0.0f, 1.0f);

        // Set semantic feature type for proper serialization
        FeatureSharedResources::semantic_feature_type = SemanticFeatureType::FEATURE_VECTOR;

        // Observations (by object, will serialize as IDs)
        {
            // add to _observations
            mp->_observations[kf1] = 11;
            mp->_observations[kf2] = 22;

            // add to _frame_views
            mp->_frame_views[f1] = 3;
            mp->_frame_views[f2] = 4;
        }

        // Serialize to JSON and back
        const std::string json_str = mp->to_json();
        auto mp2 = MapPoint::from_json(json_str);

        // Replace IDs with objects using provided pools
        std::vector<MapPointPtr> points{mp2};
        std::vector<FramePtr> frames{f1, f2};
        std::vector<KeyFramePtr> keyframes{kf1, kf2};
        mp2->replace_ids_with_objects(points, frames, keyframes);

        // Basic scalar checks
        if (mp2->id != mp->id)
            throw std::runtime_error("id mismatch");
        if ((mp2->_pt - mp->_pt).norm() > 1e-12)
            throw std::runtime_error("pt mismatch");
        if ((mp2->normal - mp->normal).norm() > 1e-12)
            throw std::runtime_error("normal mismatch");
        if (std::abs(mp2->_min_distance - mp->_min_distance) > 1e-6f)
            throw std::runtime_error("min_distance mismatch");
        if (std::abs(mp2->_max_distance - mp->_max_distance) > 1e-6f)
            throw std::runtime_error("max_distance mismatch");
        if (mp2->_is_bad != mp->_is_bad)
            throw std::runtime_error("_is_bad mismatch");
        if (mp2->_num_observations != mp->_num_observations)
            throw std::runtime_error("_num_observations mismatch");
        if (mp2->num_times_visible != mp->num_times_visible)
            throw std::runtime_error("num_times_visible mismatch");
        if (mp2->num_times_found != mp->num_times_found)
            throw std::runtime_error("num_times_found mismatch");
        if (mp2->last_frame_id_seen != mp->last_frame_id_seen)
            throw std::runtime_error("last_frame_id_seen mismatch");
        if (mp2->first_kid != mp->first_kid)
            throw std::runtime_error("first_kid mismatch");

        // Color check
        for (int i = 0; i < 3; ++i) {
            if (mp2->color[i] != mp->color[i])
                throw std::runtime_error("color mismatch");
        }

        // Descriptor checks
        if (!mats_equal_exact(mp->des, mp2->des))
            throw std::runtime_error("des mismatch");
        if (!mats_equal_exact(mp->semantic_des, mp2->semantic_des))
            throw std::runtime_error("semantic_des mismatch");

        // Reference and relations
        if (!mp2->kf_ref || mp2->kf_ref->id != kf2->id)
            throw std::runtime_error("kf_ref not restored");

        // Observations restored from IDs
        if (mp2->_observations.size() != 2)
            throw std::runtime_error("observations size mismatch");
        if (mp2->_observations.at(kf1) != 11)
            throw std::runtime_error("obs kf1 mismatch");
        if (mp2->_observations.at(kf2) != 22)
            throw std::runtime_error("obs kf2 mismatch");

        // Frame views restored from IDs
        if (mp2->_frame_views.size() != 2)
            throw std::runtime_error("frame_views size mismatch");
        if (mp2->_frame_views.at(f1) != 3)
            throw std::runtime_error("frame_view f1 mismatch");
        if (mp2->_frame_views.at(f2) != 4)
            throw std::runtime_error("frame_view f2 mismatch");

        std::cout << "MapPoint JSON serialization tests passed." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}