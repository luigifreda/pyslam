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
#include "utils/serialization.h"
#include "utils/serialization_numpy.h"

#include <Eigen/Dense>

namespace pyslam {
std::string Frame::to_json() const {
    nlohmann::json j;

    try {
        // Basic frame information
        j["id"] = id;
        j["timestamp"] = timestamp;
        j["img_id"] = img_id;

        // Pose - convert Eigen::Matrix4d to JSON array
        Eigen::Matrix4d pose_matrix = _pose.Tcw();
        std::vector<std::vector<double>> pose_list(4, std::vector<double>(4));
        for (int i = 0; i < 4; ++i) {
            for (int j_idx = 0; j_idx < 4; ++j_idx) {
                pose_list[i][j_idx] = pose_matrix(i, j_idx);
            }
        }
        j["pose"] = pose_list;

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
        if (kps.rows() > 0) {
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

        // Semantic keypoints - convert cv::Mat to base64 encoded JSON
        if (kps_sem.rows > 0) {
            j["kps_sem"] = cv_mat_to_json(kps_sem);
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

    // Parse pose with error handling
    CameraPose pose;
    if (j.contains("pose") && !j["pose"].is_null()) {
        try {
            auto pose_array = j["pose"].get<std::vector<std::vector<double>>>();
            if (pose_array.size() == 4 && pose_array[0].size() == 4) {
                Eigen::Matrix4d pose_matrix;
                for (int i = 0; i < 4; ++i) {
                    for (int j_idx = 0; j_idx < 4; ++j_idx) {
                        pose_matrix(i, j_idx) = pose_array[i][j_idx];
                    }
                }
                pose = CameraPose(Eigen::Isometry3d(pose_matrix));
            } else {
                throw std::runtime_error("Invalid pose matrix dimensions");
            }
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

    // Parse FOV centers with error handling
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

    // Parse keypoints with error handling
    frame->kps = safe_parse_keypoints<float>(j, "kps");
    frame->kps_r = safe_parse_keypoints<float>(j, "kps_r");
    frame->kpsu = safe_parse_keypoints<float>(j, "kpsu");
    frame->kpsn = safe_parse_keypoints<float>(j, "kpsn");

    // Parse semantic keypoints (cv::Mat)
    if (j.contains("kps_sem") && !j["kps_sem"].is_null()) {
        try {
            frame->kps_sem = json_to_cv_mat(j["kps_sem"]);
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
    if (j.contains("des") && !j["des"].is_null()) {
        try {
            frame->des = json_to_cv_mat(j["des"]);
        } catch (const std::exception &e) {
            frame->des = cv::Mat();
        }
    }

    if (j.contains("des_r") && !j["des_r"].is_null()) {
        try {
            frame->des_r = json_to_cv_mat(j["des_r"]);
        } catch (const std::exception &e) {
            frame->des_r = cv::Mat();
        }
    }

    // Parse depths and kps_ur with error handling
    frame->depths = safe_json_get_array<float>(j, "depths");
    frame->kps_ur = safe_json_get_array<float>(j, "kps_ur");

    // Parse points (map point IDs) with error handling
    if (j.contains("points") && !j["points"].is_null()) {
        try {
            auto point_ids = j["points"].get<std::vector<int>>();
            frame->points.resize(point_ids.size(), nullptr);
            // Store the IDs temporarily using pointer casting
            for (size_t i = 0; i < point_ids.size(); ++i) {
                if (point_ids[i] != -1) {
                    // Store the ID as a pointer value (temporary solution)
                    frame->points[i] = MapPointNewPtr(point_ids[i]);
                }
            }
        } catch (const std::exception &e) {
            // Set empty vector on error
            frame->points.clear();
        }
    }

    // Parse outliers with error handling
    frame->outliers = safe_json_get_array<bool>(j, "outliers");

    // Parse reference keyframe ID with error handling
    int kf_ref_id = safe_json_get(j, "kf_ref", -1);
    if (kf_ref_id != -1) {
        // Store the ID temporarily using pointer casting
        frame->kf_ref = KeyFrameNewPtr(kf_ref_id);
    }

    // Parse images (cv::Mat) with error handling
    if (j.contains("img") && !j["img"].is_null()) {
        try {
            frame->img = json_to_cv_mat(j["img"]);
        } catch (const std::exception &e) {
            frame->img = cv::Mat();
        }
    }

    if (j.contains("depth_img") && !j["depth_img"].is_null()) {
        try {
            frame->depth_img = json_to_cv_mat(j["depth_img"]);
        } catch (const std::exception &e) {
            frame->depth_img = cv::Mat();
        }
    }

    if (j.contains("img_right") && !j["img_right"].is_null()) {
        try {
            frame->img_right = json_to_cv_mat(j["img_right"]);
        } catch (const std::exception &e) {
            frame->img_right = cv::Mat();
        }
    }

    if (j.contains("semantic_img") && !j["semantic_img"].is_null()) {
        try {
            frame->semantic_img = json_to_cv_mat(j["semantic_img"]);
        } catch (const std::exception &e) {
            frame->semantic_img = cv::Mat();
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

    // Helper lambda to get object with ID
    auto get_object_with_id = [](int id, const auto &lookup_dict) -> auto {
        if (id == -1) {
            return static_cast<decltype(lookup_dict.begin()->second)>(nullptr);
        }
        auto it = lookup_dict.find(id);
        return (it != lookup_dict.end()) ? it->second : nullptr;
    };

    // Replace points array with actual MapPoint objects
    if (!this->points.empty()) {
        std::lock_guard<std::mutex> lock(_lock_features);
        for (size_t i = 0; i < this->points.size(); ++i) {
            if (this->points[i]) {
                // Extract the ID that was stored during deserialization
                int point_id = this->points[i]->id;
                this->points[i] = get_object_with_id(point_id, points_dict);
            }
        }
    }

    // Replace kf_ref with actual KeyFrame object
    if (kf_ref) {
        // Extract the ID that was stored during deserialization
        int kf_ref_id = kf_ref->id;
        kf_ref = get_object_with_id(kf_ref_id, keyframes_dict);
    }
}

//=======================================
//         Numpy serialization
//=======================================

// Frame::state_tuple()
py::tuple Frame::state_tuple() const {
    const int version = 1;

    // MapPoints as real objects
    std::vector<std::shared_ptr<MapPoint>> mps;
    mps.reserve(points.size());
    for (auto &p : points) {
        if (p)
            mps.emplace_back(p->shared_from_this());
        else
            mps.emplace_back(std::shared_ptr<MapPoint>()); // keep alignment with indices
    }

    // kf_ref as real object (may be null)
    KeyFramePtr kfref_sp;
    if (kf_ref) {
        // Disambiguate shared_from_this through Frame base, then cast to KeyFrame
        kfref_sp = kf_ref;
    }

    // Build tuple
    return py::make_tuple(
        version,
        // ---- FrameBase core ----
        id, timestamp, img_id,
        // Pose (Tcw); we’ll rebuild via R|t setters to avoid CameraPose ctor needs
        this->Tcw(),
        // stats
        median_depth, fov_center_c, fov_center_w,

        // ---- feature arrays ----
        kps, kps_r, kpsu, kpsn, cvmat_to_numpy(kps_sem), octaves, octaves_r, sizes, angles,
        cvmat_to_numpy(des), cvmat_to_numpy(des_r), depths, kps_ur,

        // ---- map-point associations ----
        mps, outliers,

        // ---- reference KF ----
        kfref_sp,

        // ---- images ----
        cvmat_to_numpy(img), cvmat_to_numpy(img_right), cvmat_to_numpy(depth_img),
        cvmat_to_numpy(semantic_img),

        // ---- misc stats ----
        is_keyframe, is_blurry, laplacian_var);
}

// Frame::restore_from_state()
void Frame::restore_from_state(const py::tuple &t) {
    int idx = 0;
    const int version = t[idx++].cast<int>();
    if (version != 1)
        throw std::runtime_error("Unsupported Frame pickle version");

    // ---- FrameBase core ----
    id = t[idx++].cast<int>();
    timestamp = t[idx++].cast<double>();
    img_id = t[idx++].cast<int>();
    const Eigen::Matrix4d Tcw = t[idx++].cast<Eigen::Matrix4d>();
    {
        // Decompose Tcw and set without requiring a CameraPose constructor
        Eigen::Matrix3d R = Tcw.topLeftCorner<3, 3>();
        Eigen::Vector3d tvec = Tcw.topRightCorner<3, 1>();
        this->update_rotation_and_translation(R, tvec);
    }
    median_depth = t[idx++].cast<float>();
    fov_center_c = t[idx++].cast<Eigen::Vector3d>();
    fov_center_w = t[idx++].cast<Eigen::Vector3d>();

    // ---- feature arrays ----
    kps = t[idx++].cast<MatNx2f>();
    kps_r = t[idx++].cast<MatNx2f>();
    kpsu = t[idx++].cast<MatNx2f>();
    kpsn = t[idx++].cast<MatNx2f>();

    kps_sem = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_32F); // or CV_8U if that’s your dtype

    octaves = t[idx++].cast<std::vector<int>>();
    octaves_r = t[idx++].cast<std::vector<int>>();
    sizes = t[idx++].cast<std::vector<float>>();
    angles = t[idx++].cast<std::vector<float>>();
    des = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U); // adjust if float descriptors
    des_r = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);
    depths = t[idx++].cast<std::vector<float>>();
    kps_ur = t[idx++].cast<std::vector<float>>();

    // ---- map-point associations ----
    {
        auto mps = t[idx++].cast<std::vector<MapPointPtr>>();
        points.resize(mps.size(), nullptr);
        for (size_t i = 0; i < mps.size(); ++i) {
            points[i] = mps[i] ? mps[i] : nullptr;
        }
    }
    outliers = t[idx++].cast<std::vector<bool>>();

    // ---- reference KF ----
    {
        auto kfref_sp = t[idx++].cast<KeyFramePtr>();
        kf_ref = kfref_sp ? kfref_sp : nullptr;
    }

    // ---- images ----
    img = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);
    img_right = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);
    depth_img = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_32F); // if depth is float
    semantic_img = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);

    // ---- misc stats ----
    is_keyframe = t[idx++].cast<bool>();
    is_blurry = t[idx++].cast<bool>();
    laplacian_var = t[idx++].cast<float>();

    // Recreate transient stuff
    _kd.reset(); // kdtree rebuilt lazily elsewhere
    // camera stays as-is (often managed externally); ok to be nullptr for many ops
    // locks are default-constructed
}

} // namespace pyslam