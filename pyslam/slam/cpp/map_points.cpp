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
#include "config_parameters.h"
#include "feature_shared_resources.h"
#include "map.h"
#include "utils/cv_ops.h"
#include "utils/messages.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace pyslam {

namespace {

// compute normalized ray in world coordinates
inline Eigen::Vector3d normalized_ray_in_world(const KeyFramePtr &kf, int idx) {
    const double x = kf->kpsn(idx, 0);
    const double y = kf->kpsn(idx, 1);
    const Eigen::Vector3d ray_cam(x, y, 1.0);
    Eigen::Vector3d ray_world = kf->Rwc() * ray_cam;
    const double norm = ray_world.norm();
    if (norm > 1e-12) {
        ray_world /= norm;
    }
    return ray_world;
}

} // namespace

std::tuple<int, std::vector<bool>, std::vector<MapPointPtr>>
Map::add_points(const std::vector<Eigen::Vector3d> &points3d,
                const std::optional<std::vector<bool>> &mask_pts3d_opt, KeyFramePtr &kf1,
                KeyFramePtr &kf2, const std::vector<int> &idxs1, const std::vector<int> &idxs2,
                const cv::Mat &img, bool do_check, double cos_max_parallax,
                std::optional<double> far_points_threshold) {

    const size_t num_points = points3d.size();
    if (num_points == 0) {
        return {0, {}, {}};
    }

    if (!kf1 || !kf2) {
        throw std::invalid_argument("Map::add_points: keyframes must be valid pointers");
    }

    if (idxs1.size() != num_points || idxs2.size() != num_points) {
        throw std::invalid_argument("Map::add_points: indices size mismatch with points");
    }

    std::vector<Eigen::Vector3d> points_world(points3d.begin(), points3d.end());

    std::vector<bool> mask = mask_pts3d_opt ? *mask_pts3d_opt : std::vector<bool>(num_points, true);
    if (mask.size() != num_points) {
        throw std::invalid_argument("Map::add_points: mask size mismatch with points");
    }

    std::vector<bool> out_mask(num_points, false);
    std::vector<MapPointPtr> added_points;
    added_points.reserve(num_points);

    bool init_value =
        do_check ? true : false; // if do_check is true, we initialize the is_bad_point vector with
                                 // true so that the point is validated only at the end of the
                                 // checks if we don't skip the loop iteration for some reason
    std::vector<bool> is_bad_point(num_points, init_value);

    const double far_threshold =
        far_points_threshold.value_or(std::numeric_limits<double>::infinity());

    if (do_check) {
        const bool has_stereo1 = !kf1->kps_ur.empty();
        const bool has_stereo2 = !kf2->kps_ur.empty();
        const double scale_factor = FeatureSharedResources::scale_factor;
        const double ratio_scale_consistency = Parameters::kScaleConsistencyFactor * scale_factor;

        const auto &camera1 = kf1->camera;
        MSG_FORCED_ASSERT(camera1, "Map::add_points: camera1 is nullptr");
        const auto &camera2 = kf2->camera;
        MSG_FORCED_ASSERT(camera2, "Map::add_points: camera2 is nullptr");

        const auto kf1_pose = kf1->Twc();
        const auto kf1_Rwc = kf1_pose.block<3, 3>(0, 0);
        const auto kf1_Ow = kf1_pose.block<3, 1>(0, 3);
        const auto kf2_pose = kf2->Twc();
        const auto kf2_Rwc = kf2_pose.block<3, 3>(0, 0);
        const auto kf2_Ow = kf2_pose.block<3, 1>(0, 3);

        for (size_t i = 0; i < num_points; ++i) {
            const int idx1 = idxs1[i];
            const int idx2 = idxs2[i];

            const auto [uv1, depth1] = kf1->project_point<double>(points_world[i]);
            const auto [uv2, depth2] = kf2->project_point<double>(points_world[i]);

            const bool is_bad_depths1 = depth1 <= 0.0 || depth1 > far_threshold;
            const bool is_bad_depths2 = depth2 <= 0.0 || depth2 > far_threshold;
            if (is_bad_depths1 || is_bad_depths2) {
                continue;
            }

            const bool is_stereo1 = has_stereo1 && idx1 >= 0 &&
                                    idx1 < static_cast<int>(kf1->kps_ur.size()) &&
                                    kf1->kps_ur[idx1] >= 0.0f;
            const bool is_stereo2 = has_stereo2 && idx2 >= 0 &&
                                    idx2 < static_cast<int>(kf2->kps_ur.size()) &&
                                    kf2->kps_ur[idx2] >= 0.0f;

            // compute back-projected rays (unit vectors)
            const Eigen::Vector3d ray1 = normalized_ray_in_world(kf1, idx1);
            const Eigen::Vector3d ray2 = normalized_ray_in_world(kf2, idx2);

            // compute dot product of rays
            const double cos_parallax = ray1.dot(ray2);

            // check if we can use depths in case of bad parallax
            // NOTE: 2.0 is certainly higher than any cos_parallax value
            double cos_parallax_stereo1 = 2.0;
            double cos_parallax_stereo2 = 2.0;

            double stereo_depth1 = 0.0;
            double stereo_depth2 = 0.0;

            if (is_stereo1) {
                stereo_depth1 = static_cast<double>(kf1->depths[idx1]);
                if (std::isfinite(stereo_depth1) && stereo_depth1 > 0.0) {
                    const double angle = 2.0 * std::atan2(camera1->b / 2.0, stereo_depth1);
                    cos_parallax_stereo1 = std::cos(angle);
                }
            }

            if (is_stereo2) {
                stereo_depth2 = static_cast<double>(kf2->depths[idx2]);
                if (std::isfinite(stereo_depth2) && stereo_depth2 > 0.0) {
                    const double angle = 2.0 * std::atan2(camera2->b / 2.0, stereo_depth2);
                    cos_parallax_stereo2 = std::cos(angle);
                }
            }

            const double cos_parallax_stereo = std::min(cos_parallax_stereo1, cos_parallax_stereo2);

            // check if we can recover bad-parallx points from stereo/rgbd data
            const bool try_recover3d_from_stereo = (cos_parallax < 0.0) ||
                                                   (cos_parallax > cos_parallax_stereo) ||
                                                   (cos_parallax > cos_max_parallax);

            bool is_recovered3d_from_stereo = false;
            if (try_recover3d_from_stereo && is_stereo1 &&
                cos_parallax_stereo1 < cos_parallax_stereo2) {
                const bool ok1 =
                    stereo_depth1 > Parameters::kMinDepth; // && stereo_depth1 < far_threshold;
                if (ok1) {
                    const Vec3<double> pt_c(stereo_depth1 * kf1->kpsn(idx1, 0),
                                            stereo_depth1 * kf1->kpsn(idx1, 1), stereo_depth1);
                    const Vec3<double> pt_w = kf1_Rwc * pt_c + kf1_Ow;
                    points_world[i] = pt_w;
                    is_recovered3d_from_stereo = true;
                }
            } else if (try_recover3d_from_stereo && is_stereo2 &&
                       cos_parallax_stereo2 < cos_parallax_stereo1) {
                const bool ok2 =
                    stereo_depth2 > Parameters::kMinDepth; // && stereo_depth2 < far_threshold;
                if (ok2) {
                    const Vec3<double> pt_c(stereo_depth2 * kf2->kpsn(idx2, 0),
                                            stereo_depth2 * kf2->kpsn(idx2, 1), stereo_depth2);
                    const Vec3<double> pt_w = kf2_Rwc * pt_c + kf2_Ow;
                    points_world[i] = pt_w;
                    is_recovered3d_from_stereo = true;
                }
            }

            bool is_bad_cos_parallax =
                ((cos_parallax < 0.0) || (cos_parallax > cos_max_parallax)) &&
                !is_recovered3d_from_stereo;

            if (is_bad_cos_parallax) {
                continue;
            }

            const Eigen::Vector2d diff1 = uv1 - kf1->keypoint_undistorted(idx1);
            const double inv_sigma2_1 =
                FeatureSharedResources::inv_level_sigmas2[kf1->octaves[idx1]];

            bool is_bad_chis2_1 = false;
            if (is_stereo1) {
                const double inv_depth =
                    std::isfinite(stereo_depth1) && stereo_depth1 > 0.0 ? 1.0 / stereo_depth1 : 0.0;
                const double u_r_pred = uv1.x() - camera1->bf * inv_depth;
                const double err_r = u_r_pred - static_cast<double>(kf1->kps_ur[idx1]);
                const double chi2_stereo = (diff1.squaredNorm() + err_r * err_r) * inv_sigma2_1;
                is_bad_chis2_1 = chi2_stereo > Parameters::kChi2Stereo;
            } else {
                const double chi2_mono1 = diff1.squaredNorm() * inv_sigma2_1;
                is_bad_chis2_1 = chi2_mono1 > Parameters::kChi2Mono;
            }

            if (is_bad_chis2_1) {
                continue;
            }

            const Eigen::Vector2d diff2 = uv2 - kf2->keypoint_undistorted(idx2);
            const double inv_sigma2_2 =
                FeatureSharedResources::inv_level_sigmas2[kf2->octaves[idx2]];

            bool is_bad_chis2_2 = false;
            if (is_stereo2) {
                const double inv_depth =
                    std::isfinite(stereo_depth2) && stereo_depth2 > 0.0 ? 1.0 / stereo_depth2 : 0.0;
                const double u_r_pred = uv2.x() - camera2->bf * inv_depth;
                const double err_r = u_r_pred - static_cast<double>(kf2->kps_ur[idx2]);
                const double chi2_stereo = (diff2.squaredNorm() + err_r * err_r) * inv_sigma2_2;
                is_bad_chis2_2 = chi2_stereo > Parameters::kChi2Stereo;
            } else {
                const double chi2_mono2 = diff2.squaredNorm() * inv_sigma2_2;
                is_bad_chis2_2 = chi2_mono2 > Parameters::kChi2Mono;
            }

            if (is_bad_chis2_2) {
                continue;
            }

            const double scale_depth1 =
                FeatureSharedResources::scale_factors[kf1->octaves[idx1]] * depth1;
            const double scale_depth2 =
                FeatureSharedResources::scale_factors[kf2->octaves[idx2]] * depth2;

            const double scale_depth1_ratio = scale_depth1 * ratio_scale_consistency;
            const double scale_depth2_ratio = scale_depth2 * ratio_scale_consistency;
            const bool is_bad_scale_consistency =
                (scale_depth1 > scale_depth2_ratio) || (scale_depth2 > scale_depth1_ratio);

            is_bad_point[i] = is_bad_cos_parallax || is_bad_depths1 || is_bad_depths2 ||
                              is_bad_chis2_1 || is_bad_chis2_2 || is_bad_scale_consistency;
        } // end for

    } // end do_check

    const cv::Vec3f default_color(255.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < num_points; ++i) {
        if (!mask[i] || is_bad_point[i]) {
            continue;
        }
        const int idx1 = idxs1[i];
        const int idx2 = idxs2[i];

        const int u = static_cast<int>(std::floor(kf1->kps(idx1, 0)));
        const int v = static_cast<int>(std::floor(kf1->kps(idx1, 1)));
        const cv::Vec3f mean_color =
            compute_patch_mean(img, u, v, Parameters::kSparseImageColorPatchDelta, default_color);
        const Eigen::Matrix<unsigned char, 3, 1> color = to_color_vector(mean_color);

        auto mp = MapPointNewPtr(points_world[i], color, kf2, idx2);
        this->add_point(mp);

        mp->add_observation(kf1, idx1);
        mp->add_observation(kf2, idx2);
        mp->update_info();

        out_mask[i] = true;
        added_points.push_back(mp);
    }
    return {static_cast<int>(added_points.size()), out_mask, added_points};
}

int Map::add_stereo_points(const std::vector<Eigen::Vector3d> &points3d,
                           const std::optional<std::vector<bool>> &mask_pts3d_opt, FramePtr &f,
                           KeyFramePtr &kf, const std::vector<int> &idxs, const cv::Mat &img) {

    if (points3d.empty()) {
        return 0;
    }

    if (!kf || !f) {
        throw std::invalid_argument("Map::add_stereo_points: invalid frame or keyframe");
    }

    const size_t num_points = points3d.size();
    if (idxs.size() != num_points) {
        throw std::invalid_argument("Map::add_stereo_points: indices size mismatch with points");
    }

    std::vector<bool> mask = mask_pts3d_opt ? *mask_pts3d_opt : std::vector<bool>(num_points, true);
    if (mask.size() != num_points) {
        throw std::invalid_argument("Map::add_stereo_points: mask size mismatch with points");
    }

    const cv::Vec3f default_color(255.0f, 0.0f, 0.0f);

    int num_added_points = 0;
    for (size_t i = 0; i < num_points; ++i) {
        if (!mask[i]) {
            continue;
        }

        const int idx = idxs[i];
        const int u = static_cast<int>(std::floor(kf->kps(idx, 0)));
        const int v = static_cast<int>(std::floor(kf->kps(idx, 1)));
        const cv::Vec3f mean_color =
            compute_patch_mean(img, u, v, Parameters::kSparseImageColorPatchDelta, default_color);
        const auto color = to_color_vector(mean_color);

        auto mp = MapPointNewPtr(points3d[i], color, kf, idx);
        this->add_point(mp);

        if (idx >= 0 && idx < static_cast<int>(f->points.size())) {
            f->points[idx] = mp;
        }

        mp->add_observation(kf, idx);
        mp->update_info();
        ++num_added_points;
    }

    return num_added_points;
}

void Map::remove_points_with_big_reproj_err(const std::vector<MapPointPtr> &points_to_check) {
    const auto &inv_level_sigmas2 = FeatureSharedResources::inv_level_sigmas2;

    std::vector<MapPointPtr> candidates;
    if (points_to_check.empty()) {
        std::lock_guard<MapMutex> lock_map(_lock);
        candidates.assign(points.begin(), points.end());
    } else {
        candidates = points_to_check;
    }

    // std::lock_guard<MapMutex> lock_map(_lock);
    std::scoped_lock lock(_update_lock);

    for (auto &p : candidates) {
        if (!p || p->is_bad()) {
            continue;
        }

        const auto observations = p->observations();
        if (observations.empty()) {
            continue;
        }

        // compute average reprojection error as chi2 mean
        double chi2_sum = 0.0;
        int count = 0;

        for (const auto &[frame, idx] : observations) {
            if (!frame || idx < 0 || idx >= static_cast<int>(frame->kpsu.rows())) {
                continue;
            }

            const auto [proj, _] = frame->project_map_point<double>(p);
            const Eigen::Vector2d uv = frame->keypoint_undistorted(idx);
            const Eigen::Vector2d err = proj - uv;
            const double inv_sigma2 = inv_level_sigmas2[frame->octaves[idx]];
            chi2_sum += err.squaredNorm() * inv_sigma2;
            ++count;
        }

        if (count == 0) {
            continue;
        }

        const double chi2_mean = chi2_sum / static_cast<double>(count);
        if (chi2_mean > Parameters::kChi2Mono) {
            this->remove_point(p);
        }
    }
}

float Map::compute_mean_reproj_error(const std::vector<MapPointPtr> &points_subset) const {
    std::scoped_lock lock(_update_lock);

    const bool use_subset = !points_subset.empty();
    double chi2 = 0.0;
    size_t num_obs = 0;

    const auto accumulate_point = [&](const MapPointPtr &p) {
        if (!p || p->is_bad()) {
            return;
        }

        const auto observations = p->observations();
        for (const auto &[frame, idx] : observations) {
            if (!frame || idx < 0 || idx >= static_cast<int>(frame->kpsu.rows())) {
                continue;
            }
            const auto [proj, _] = frame->project_map_point<double>(p);
            const Eigen::Vector2d uv = frame->keypoint_undistorted(idx);
            const Eigen::Vector2d err = proj - uv;
            const double inv_sigma2 =
                FeatureSharedResources::inv_level_sigmas2[frame->octaves[idx]];
            chi2 += err.squaredNorm() * inv_sigma2;
            ++num_obs;
        }
    };

    if (use_subset) {
        for (auto &p : points_subset) {
            accumulate_point(p);
        }
    } else {
        for (auto &p : points) {
            accumulate_point(p);
        }
    }

    if (num_obs == 0) {
        return 0.0f;
    }
    return static_cast<float>(chi2 / static_cast<double>(num_obs));
}

} // namespace pyslam
