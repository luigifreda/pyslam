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

#include "utils/optional_lock.h"
#include "utils/serialization_numpy.h"

#include <Eigen/Dense>

// Define this before including numpy headers to suppress deprecation warnings
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

namespace pyslam {

//=======================================
//         Numpy serialization
//=======================================

py::tuple Frame::state_tuple(bool need_lock) const {
    const int version = 1;

    CONDITIONAL_LOCK(_lock_pose, need_lock);
    CONDITIONAL_LOCK(_lock_features, need_lock);

    // MapPoints as real objects
    std::vector<int> points_id_data;
    points_id_data.reserve(points.size());
    for (auto &p : points) {
        if (p)
            points_id_data.emplace_back(p->id);
        else
            points_id_data.emplace_back(-1); // keep alignment with indices
    }

    // kf_ref as real object (may be null)
    int kf_ref_id;
    if (kf_ref) {
        // Disambiguate shared_from_this through Frame base, then cast to KeyFrame
        kf_ref_id = kf_ref->id;
    } else {
        kf_ref_id = -1;
    }

    const Eigen::Matrix4d Tcw = _pose.get_matrix();

    // Build tuple
    return py::make_tuple(
        version,
        // ---- FrameBase core ----
        id, timestamp, img_id,
        // Pose (Tcw); we’ll rebuild via R|t setters to avoid CameraPose ctor needs
        Tcw,
        // stats
        median_depth, fov_center_c, fov_center_w,

        // ---- feature arrays ----
        kps, kps_r, kpsu, kpsn, cvmat_to_numpy(kps_sem), octaves, octaves_r, sizes, angles,
        cvmat_to_numpy(des), cvmat_to_numpy(des_r), depths, kps_ur,

        // ---- map-point associations ----
        points_id_data, outliers,

        // ---- reference KF ----
        kf_ref_id,

        // ---- images ----
        cvmat_to_numpy(img), cvmat_to_numpy(img_right), cvmat_to_numpy(depth_img),
        cvmat_to_numpy(semantic_img), cvmat_to_numpy(semantic_instances_img),

        // ---- misc stats ----
        is_keyframe, is_blurry, laplacian_var);
}

void Frame::restore_from_state(const py::tuple &t, bool need_lock) {
    CONDITIONAL_LOCK(_lock_pose, need_lock);
    CONDITIONAL_LOCK(_lock_features, need_lock);

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
        this->update_rotation_and_translation_no_lock_(R, tvec);
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
    // Infer descriptor dtype from numpy array instead of hardcoding CV_8U
    auto des_array = t[idx++].cast<py::array>();
    int des_dtype = des_array.dtype().num() == NPY_FLOAT32 ? CV_32F : CV_8U;
    des = numpy_to_cvmat(des_array, des_dtype);

    auto des_r_array = t[idx++].cast<py::array>();
    int des_r_dtype = des_r_array.dtype().num() == NPY_FLOAT32 ? CV_32F : CV_8U;
    des_r = numpy_to_cvmat(des_r_array, des_r_dtype);
    depths = t[idx++].cast<std::vector<float>>();
    kps_ur = t[idx++].cast<std::vector<float>>();

    // ---- map-point associations ----

    _points_id_data = t[idx++].cast<std::vector<int>>();

    outliers = t[idx++].cast<std::vector<bool>>();

    // ---- reference KF ----
    _kf_ref_id = t[idx++].cast<int>();

    // ---- images ----
    img = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);
    img_right = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_8U);
    depth_img = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_32F); // if depth is float
    // Infer semantic_img dtype from numpy array to support both uint8 (small label sets)
    // and int32 (large label sets like Detic with category IDs > 255)
    auto semantic_img_array = t[idx++].cast<py::array>();
    int semantic_img_dtype = semantic_img_array.dtype().num() == NPY_INT32 ? CV_32S : CV_8U;
    semantic_img = numpy_to_cvmat(semantic_img_array, semantic_img_dtype);
    semantic_instances_img = numpy_to_cvmat(t[idx++].cast<py::array>(), CV_32S);

    // ---- misc stats ----
    is_keyframe = t[idx++].cast<bool>();
    is_blurry = t[idx++].cast<bool>();
    laplacian_var = t[idx++].cast<float>();

    // Recreate transient stuff
    _kd.reset(); // kdtree rebuilt lazily elsewhere
    // camera stays as-is (often managed externally); ok to be nullptr for many ops

    // Note: call replace_ids_with_objects(...) after this to resolve IDs to objects.
}

} // namespace pyslam