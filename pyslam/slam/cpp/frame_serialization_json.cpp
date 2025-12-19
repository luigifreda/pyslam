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

#include "camera.h"
#include "camera_pose.h"
#include "frame.h"
#include "keyframe.h"
#include "map_point.h"

#include "utils/serialization_json.h"

#include <nlohmann/json.hpp>

#include <Eigen/Dense>

namespace pyslam {

std::string Frame::to_json() const {
    nlohmann::json j;

    try {
        // Basic frame information
        j["id"] = id;
        j["timestamp"] = timestamp;
        j["img_id"] = img_id;

        // Pose - convert Eigen::Matrix4d to JSON nested array
        j["pose"] = eigen_matrix_to_json_array(_pose.Tcw());

        // Camera
        if (camera) {
            j["camera"] = nlohmann::json::parse(camera->to_json());
        } else {
            j["camera"] = nullptr;
        }

        // Frame properties
        j["is_keyframe"] = is_keyframe;
        j["median_depth"] = median_depth;

        // FOV centers - convert Eigen::Vector3d to JSON array
        if (fov_center_c != Eigen::Vector3d::Zero()) {
            j["fov_center_c"] = {fov_center_c.x(), fov_center_c.y(), fov_center_c.z()};
        } else {
            j["fov_center_c"] = nullptr;
        }

        if (fov_center_w != Eigen::Vector3d::Zero()) {
            j["fov_center_w"] = {fov_center_w.x(), fov_center_w.y(), fov_center_w.z()};
        } else {
            j["fov_center_w"] = nullptr;
        }

        // Image quality
        j["is_blurry"] = is_blurry;
        j["laplacian_var"] = laplacian_var;

        // Keypoints - convert MatNx2d to JSON array
        if (kps.size() > 0) {
            std::vector<std::vector<float>> kps_list;
            for (int i = 0; i < kps.rows(); ++i) {
                kps_list.push_back({kps(i, 0), kps(i, 1)});
            }
            j["kps"] = kps_list;
        } else {
            j["kps"] = nullptr;
        }

        // Right keypoints
        if (kps_r.size() > 0) {
            std::vector<std::vector<float>> kps_r_list;
            for (int i = 0; i < kps_r.rows(); ++i) {
                kps_r_list.push_back({kps_r(i, 0), kps_r(i, 1)});
            }
            j["kps_r"] = kps_r_list;
        } else {
            j["kps_r"] = nullptr;
        }

        // Undistorted keypoints
        if (kpsu.size() > 0) {
            std::vector<std::vector<float>> kpsu_list;
            for (int i = 0; i < kpsu.rows(); ++i) {
                kpsu_list.push_back({kpsu(i, 0), kpsu(i, 1)});
            }
            j["kpsu"] = kpsu_list;
        } else {
            j["kpsu"] = nullptr;
        }

        // Normalized keypoints
        if (kpsn.size() > 0) {
            std::vector<std::vector<float>> kpsn_list;
            for (int i = 0; i < kpsn.rows(); ++i) {
                kpsn_list.push_back({kpsn(i, 0), kpsn(i, 1)});
            }
            j["kpsn"] = kpsn_list;
        } else {
            j["kpsn"] = nullptr;
        }

        // Semantic keypoints - convert cv::Mat to raw JSON format
        if (kps_sem.rows > 0) {
            j["kps_sem"] = cv_mat_to_json_raw(kps_sem);
        } else {
            j["kps_sem"] = nullptr;
        }

        // Octaves
        if (!octaves.empty()) {
            j["octaves"] = octaves;
        } else {
            j["octaves"] = nullptr;
        }

        // Right octaves
        if (!octaves_r.empty()) {
            j["octaves_r"] = octaves_r;
        } else {
            j["octaves_r"] = nullptr;
        }

        // Sizes
        if (!sizes.empty()) {
            j["sizes"] = sizes;
        } else {
            j["sizes"] = nullptr;
        }

        // Angles
        if (!angles.empty()) {
            j["angles"] = angles;
        } else {
            j["angles"] = nullptr;
        }

        // Descriptors - convert cv::Mat to base64 encoded JSON
        if (!des.empty()) {
            j["des"] = cv_mat_to_json(des);
        } else {
            j["des"] = nullptr;
        }

        // Right descriptors
        if (!des_r.empty()) {
            j["des_r"] = cv_mat_to_json(des_r);
        } else {
            j["des_r"] = nullptr;
        }

        // Depths
        if (!depths.empty()) {
            j["depths"] = depths;
        } else {
            j["depths"] = nullptr;
        }

        // Right u-coordinates
        if (!kps_ur.empty()) {
            j["kps_ur"] = kps_ur;
        } else {
            j["kps_ur"] = nullptr;
        }

        // Map points - store IDs only
        if (!points.empty()) {
            std::vector<int> point_ids;
            for (const auto &point : points) {
                point_ids.push_back(point ? point->id : -1);
            }
            j["points"] = point_ids;
        } else {
            j["points"] = nullptr;
        }

        // Outliers
        if (!outliers.empty()) {
            j["outliers"] = outliers;
        } else {
            j["outliers"] = nullptr;
        }

        // Reference keyframe
        j["kf_ref"] = kf_ref ? kf_ref->id : -1;

        // Images - convert cv::Mat to base64 encoded JSON
        if (!img.empty()) {
            j["img"] = cv_mat_to_json(img);
        } else {
            j["img"] = nullptr;
        }

        if (!depth_img.empty()) {
            j["depth_img"] = cv_mat_to_json(depth_img);
        } else {
            j["depth_img"] = nullptr;
        }

        if (!img_right.empty()) {
            j["img_right"] = cv_mat_to_json(img_right);
        } else {
            j["img_right"] = nullptr;
        }

        if (!semantic_img.empty()) {
            j["semantic_img"] = cv_mat_to_json(semantic_img);
        } else {
            j["semantic_img"] = nullptr;
        }
        if (!semantic_instances_img.empty()) {
            j["semantic_instances_img"] = cv_mat_to_json(semantic_instances_img);
        } else {
            j["semantic_instances_img"] = nullptr;
        }

    } catch (const std::exception &e) {
        throw std::runtime_error("Error in Frame::to_json(): " + std::string(e.what()));
    }

    return j.dump();
}

FramePtr Frame::from_json(const std::string &json_str) {
    nlohmann::json j;

    try {
        j = nlohmann::json::parse(json_str);
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
    }

    // Parse camera with improved error handling
    CameraPtr camera;
    if (j.contains("camera") && !j["camera"].is_null()) {
        try {
            camera = create_camera_from_json(j["camera"]);
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to create camera: " + std::string(e.what()));
        }
    }

    // Parse pose with error handling - handle both raw arrays and JSON-encoded strings
    CameraPose pose;
    if (j.contains("pose") && !j["pose"].is_null()) {
        try {
            Eigen::Matrix4d pose_matrix;
            if (!safe_parse_pose_matrix<double, 4, 4>(j, "pose", pose_matrix)) {
                throw std::runtime_error("Pose is neither 4x4 array nor JSON-encoded array string");
            }
            pose = CameraPose(Eigen::Isometry3d(pose_matrix));
        } catch (const std::exception &e) {
            throw std::runtime_error("Failed to parse pose: " + std::string(e.what()));
        }
    }

    // Parse basic frame info with error handling
    int id = safe_json_get(j, "id", 0);
    double timestamp = safe_json_get(j, "timestamp", 0.0);
    int img_id = safe_json_get(j, "img_id", 0);
    bool is_keyframe = safe_json_get(j, "is_keyframe", false);
    float median_depth = safe_json_get(j, "median_depth", 0.0f);

    // Parse FOV centers with error handling - handle both arrays and JSON-encoded strings
    Eigen::Vector3d fov_center_c = safe_parse_vector3d<double>(j, "fov_center_c");
    Eigen::Vector3d fov_center_w = safe_parse_vector3d<double>(j, "fov_center_w");

    // Parse image quality with defaults
    bool is_blurry = safe_json_get(j, "is_blurry", false);
    float laplacian_var = safe_json_get(j, "laplacian_var", 0.0f);

    // Create the frame
    FramePtr frame =
        FrameNewPtr(camera, cv::Mat(), cv::Mat(), cv::Mat(), pose, id, timestamp, img_id);

    // Set basic properties
    frame->is_keyframe = is_keyframe;
    frame->median_depth = median_depth;
    frame->fov_center_c = fov_center_c;
    frame->fov_center_w = fov_center_w;
    frame->is_blurry = is_blurry;
    frame->laplacian_var = laplacian_var;

    // Parse keypoints with error handling - handles both raw arrays and JSON-encoded strings
    frame->kps = safe_parse_keypoints<float>(j, "kps");
    frame->kps_r = safe_parse_keypoints<float>(j, "kps_r");
    frame->kpsu = safe_parse_keypoints<float>(j, "kpsu");
    frame->kpsn = safe_parse_keypoints<float>(j, "kpsn");

    // Parse semantic keypoints (cv::Mat) - handle raw JSON format
    if (j.contains("kps_sem") && !j["kps_sem"].is_null()) {
        try {
            frame->kps_sem = json_to_cv_mat_raw(j["kps_sem"]);
        } catch (const std::exception &e) {
            // Set empty matrix on error
            frame->kps_sem = cv::Mat();
        }
    }

    // Parse octaves, sizes, angles with error handling
    frame->octaves = safe_json_get_array<int>(j, "octaves");
    frame->octaves_r = safe_json_get_array<int>(j, "octaves_r");
    frame->sizes = safe_json_get_array<float>(j, "sizes");
    frame->angles = safe_json_get_array<float>(j, "angles");

    // Parse descriptors (cv::Mat) with error handling
    // Use flexible parser to handle both structured format and stringified JSON (from Python)
    if (j.contains("des") && !j["des"].is_null()) {
        try {
            frame->des = safe_parse_cv_mat_flexible(j, "des");
        } catch (const std::exception &e) {
            frame->des = cv::Mat();
        }
    }

    if (j.contains("des_r") && !j["des_r"].is_null()) {
        try {
            frame->des_r = safe_parse_cv_mat_flexible(j, "des_r");
        } catch (const std::exception &e) {
            frame->des_r = cv::Mat();
        }
    }

    // Parse depths and kps_ur with error handling
    frame->depths = safe_json_get_array<float>(j, "depths");
    frame->kps_ur = safe_json_get_array<float>(j, "kps_ur");

    // Parse points (map point IDs) with error handling; accept stringified arrays and nulls
    // Use nullable int array parser to handle None/null values in the array
    frame->_points_id_data = safe_json_get_array_nullable_int(j, "points", -1);

    // Parse outliers with error handling
    frame->outliers = safe_json_get_array<bool>(j, "outliers");

    // Parse reference keyframe ID with error handling - handle None/null values
    frame->_kf_ref_id = safe_json_get(j, "kf_ref", -1);

    // Parse images (cv::Mat) with error handling
    // Use flexible parser to handle both structured format and stringified JSON (from Python)
    if (j.contains("img") && !j["img"].is_null()) {
        try {
            frame->img = safe_parse_cv_mat_flexible(j, "img");
        } catch (const std::exception &e) {
            frame->img = cv::Mat();
        }
    }

    if (j.contains("depth_img") && !j["depth_img"].is_null()) {
        try {
            frame->depth_img = safe_parse_cv_mat_flexible(j, "depth_img");
        } catch (const std::exception &e) {
            frame->depth_img = cv::Mat();
        }
    }

    if (j.contains("img_right") && !j["img_right"].is_null()) {
        try {
            frame->img_right = safe_parse_cv_mat_flexible(j, "img_right");
        } catch (const std::exception &e) {
            frame->img_right = cv::Mat();
        }
    }

    if (j.contains("semantic_img") && !j["semantic_img"].is_null()) {
        try {
            frame->semantic_img = safe_parse_cv_mat_flexible(j, "semantic_img");
        } catch (const std::exception &e) {
            frame->semantic_img = cv::Mat();
        }
    }
    if (j.contains("semantic_instances_img") && !j["semantic_instances_img"].is_null()) {
        try {
            frame->semantic_instances_img = safe_parse_cv_mat_flexible(j, "semantic_instances_img");
        } catch (const std::exception &e) {
            frame->semantic_instances_img = cv::Mat();
        }
    }

    // Validate keypoints and points consistency
    if (frame->kps.rows() > 0 && frame->points.size() > 0) {
        if (frame->kps.rows() != static_cast<int>(frame->points.size())) {
            throw std::runtime_error(
                "Keypoints and points size mismatch: " + std::to_string(frame->kps.rows()) +
                " vs " + std::to_string(frame->points.size()));
        }
    }

    return frame;
}

void Frame::replace_ids_with_objects(const std::vector<MapPointPtr> &points,
                                     const std::vector<FramePtr> &frames,
                                     const std::vector<KeyFramePtr> &keyframes) {
    // Pre-build dictionaries for faster lookups
    std::unordered_map<int, MapPointPtr> points_dict;
    std::unordered_map<int, KeyFramePtr> keyframes_dict;

    // Build points lookup dictionary
    for (const auto &point : points) {
        if (point) {
            points_dict[point->id] = point;
        }
    }

    // Build keyframes lookup dictionary
    for (const auto &keyframe : keyframes) {
        if (keyframe) {
            keyframes_dict[keyframe->id] = keyframe;
        }
    }

    // Helper lambda to get object with ID - fixed return type deduction
    auto get_object_with_id = [](int id,
                                 const auto &lookup_dict) -> decltype(lookup_dict.begin()->second) {
        if (id == -1) {
            return nullptr;
        }
        auto it = lookup_dict.find(id);
        return (it != lookup_dict.end()) ? it->second : nullptr;
    };

    // Replace points array with actual MapPoint objects
    if (!_points_id_data.empty()) {
        std::lock_guard<std::mutex> lock(_lock_features);
        this->points.resize(_points_id_data.size(), nullptr);
        int num_restored = 0;
        for (size_t i = 0; i < _points_id_data.size(); ++i) {
            if (_points_id_data[i] != -1) {
                // Extract the ID that was stored during deserialization
                int point_id = _points_id_data[i];
                this->points[i] = get_object_with_id(point_id, points_dict);
                if (this->points[i]) {
                    num_restored++;
                }
            }
        }
        // Debug: Log if no points were restored (might indicate ID mismatch)
        if (num_restored == 0 && _points_id_data.size() > 0) {
            // This is a warning - we had point IDs but couldn't restore any
            // This could mean the points haven't been loaded yet, or IDs don't match
        }
    } else {
        // Debug: _points_id_data is empty - points array won't be populated
        // This could be normal if the frame has no associated map points
    }

    // Replace kf_ref with actual KeyFrame object
    if (_kf_ref_id != -1) {
        // Extract the ID that was stored during deserialization
        this->kf_ref = get_object_with_id(_kf_ref_id, keyframes_dict);
    }
}

} // namespace pyslam