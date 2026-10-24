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
#include "initializer.h"

#include "config_parameters.h"
#include "feature_shared_resources.h"
#include "frontend_helpers.h"
#include "map.h"
#include "map_point.h"
#include "rotation_histogram.h"

#include <algorithm>
#include <iostream>
#include <opencv2/calib3d.hpp>

namespace pyslam {

Initializer::Initializer(SensorType sensor_type_) : sensor_type(sensor_type_) {
    is_monocular = sensor_type == SensorType::MONOCULAR;
    num_min_features = is_monocular ? Parameters::kInitializerNumMinFeatures
                                    : Parameters::kInitializerNumMinFeaturesStereo;
    num_min_triangulated_points = is_monocular ? Parameters::kInitializerNumMinTriangulatedPoints
                                               : Parameters::kInitializerNumMinTriangulatedPointsStereo;
    check_min_frame_distance = Parameters::kInitializerUseMinFrameDistanceCheck;
    check_cell_coverage = Parameters::kInitializerUseCellCoverageCheck;
}

void Initializer::reset() {
    frames.clear();
    f_ref.reset();
}

void Initializer::push_frame(const FramePtr &frame) {
    frames.push_back(frame);
    while (static_cast<int>(frames.size()) > kMaxLenFrameDeque) {
        frames.pop_front();
    }
}

void Initializer::init(const FramePtr &f_cur, const cv::Mat &img_cur, const cv::Mat &img_right_cur,
                       const cv::Mat &depth_img_cur) {
    if (!img_cur.empty()) {
        f_cur->img = img_cur;
    }
    if (!img_right_cur.empty()) {
        f_cur->set_img_right(img_right_cur);
    }
    if (!depth_img_cur.empty()) {
        f_cur->set_depth_img(depth_img_cur);
    }
    push_frame(f_cur);
    f_ref = f_cur;
}

std::pair<InitializerOutput, bool> Initializer::initialize(const FramePtr &f_cur,
                                                           const cv::Mat &img_cur,
                                                           const cv::Mat &img_right_cur,
                                                           const cv::Mat &depth_img_cur) {
    if (!img_cur.empty()) {
        f_cur->img = img_cur;
    }
    if (!img_right_cur.empty()) {
        f_cur->set_img_right(img_right_cur);
    }
    if (!depth_img_cur.empty()) {
        f_cur->set_depth_img(depth_img_cur);
    }
    return initialize_simple(f_cur, img_cur);
}

void Initializer::check_num_failures() {
    if (num_failures > kNumOfFailuresAfterWichNumMinTriangulatedPointsIsReduced) {
        const double factor = 2.0 / 3.0;
        num_min_triangulated_points =
            factor * (is_monocular ? Parameters::kInitializerNumMinTriangulatedPoints
                                   : Parameters::kInitializerNumMinTriangulatedPointsStereo);
        check_cell_coverage = false;
    }
}

std::optional<Eigen::Matrix4d> Initializer::estimate_pose(const MatNx2f &kpn_ref,
                                                          const MatNx2f &kpn_cur) {
    const int n = std::min(kpn_ref.rows(), kpn_cur.rows());
    if (n < 5) {
        return std::nullopt;
    }
    std::vector<cv::Point2f> pts_cur;
    std::vector<cv::Point2f> pts_ref;
    pts_cur.reserve(static_cast<size_t>(n));
    pts_ref.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        pts_cur.emplace_back(kpn_cur(i, 0), kpn_cur(i, 1));
        pts_ref.emplace_back(kpn_ref(i, 0), kpn_ref(i, 1));
    }
    const int method = cv::USAC_MAGSAC;
    cv::Mat E = cv::findEssentialMat(pts_cur, pts_ref, 1.0, cv::Point2d(0.0, 0.0), method,
                                     Parameters::kRansacProb, Parameters::kRansacThresholdNormalized,
                                     mask_match);
    if (E.empty() || E.total() < 9) {
        return std::nullopt;
    }
    cv::Mat R;
    cv::Mat t;
    cv::recoverPose(E, pts_cur, pts_ref, R, t, 1.0, cv::Point2d(0.0, 0.0));
    if (R.empty() || t.empty()) {
        return std::nullopt;
    }
    R.convertTo(R, CV_64F);
    t.convertTo(t, CV_64F);
    return pose_from_Rt(R, t);
}

std::pair<InitializerOutput, bool> Initializer::initialize_simple(const FramePtr &f_cur,
                                                                  const cv::Mat &img_cur) {
    check_num_failures();
    if (f_ref && f_cur->id - f_ref->id > kMaxIdDistBetweenIntializingFrames && !frames.empty()) {
        f_ref = frames.back();
    }
    FramePtr reference = f_ref;
    push_frame(f_cur);
    return process_frames(f_cur, img_cur, reference);
}

std::pair<InitializerOutput, bool> Initializer::process_frames(const FramePtr &f_cur,
                                                               const cv::Mat &img_cur,
                                                               const FramePtr &f_ref_in) {
    InitializerOutput out;
    bool is_ok = false;
    if (!f_cur || !f_ref_in) {
        ++num_failures;
        return {out, false};
    }
    if (check_min_frame_distance && is_monocular &&
        f_cur->id - f_ref_in->id < kMinIdDistBetweenIntializingFrames) {
        ++num_failures;
        return {out, false};
    }
    const int num_kps_ref = f_ref_in->kps.rows();
    const int num_kps_cur = f_cur->kps.rows();
    if (num_kps_ref < num_min_features || num_kps_cur < num_min_features) {
        ++num_failures;
        return {out, false};
    }

    const auto matches = match_descriptors(f_cur->img, f_ref_in->img, f_cur->des, f_ref_in->des,
                                           f_cur->kps, f_ref_in->kps,
                                           Parameters::kInitializerFeatureMatchRatioTest);
    std::vector<int> idxs_cur = matches.first;
    std::vector<int> idxs_ref = matches.second;
    if (idxs_cur.empty() || idxs_ref.empty()) {
        return {out, false};
    }
    if (FeatureSharedResources::oriented_features) {
        const std::vector<int> valid = RotationHistogram::filter_matches_with_histogram_orientation(
            idxs_cur, idxs_ref, angles_as_double(f_cur->angles), angles_as_double(f_ref_in->angles));
        if (!valid.empty()) {
            idxs_cur = apply_match_indices(idxs_cur, valid);
            idxs_ref = apply_match_indices(idxs_ref, valid);
        }
    }
    if (static_cast<int>(idxs_cur.size()) < 5) {
        ++num_failures;
        return {out, false};
    }

    const MatNx2f kpsn_ref = keypoint_rows(f_ref_in->kpsn, idxs_ref);
    const MatNx2f kpsn_cur = keypoint_rows(f_cur->kpsn, idxs_cur);
    const std::optional<Eigen::Matrix4d> Trc = estimate_pose(kpsn_ref, kpsn_cur);
    if (!Trc) {
        ++num_failures;
        return {out, false};
    }
    const Eigen::Matrix4d Tcr = Trc->inverse();
    f_ref_in->update_pose(Eigen::Matrix4d::Identity());
    f_cur->update_pose(Tcr);

    std::vector<int> idxs_ref_inliers;
    std::vector<int> idxs_cur_inliers;
    if (!mask_match.empty()) {
        const cv::Mat mask = mask_match.reshape(1, static_cast<int>(mask_match.total()));
        const int n = std::min({mask.rows, static_cast<int>(idxs_cur.size()),
                                static_cast<int>(idxs_ref.size())});
        for (int i = 0; i < n; ++i) {
            if (mask.at<uchar>(i) != 0) {
                idxs_cur_inliers.push_back(idxs_cur[static_cast<size_t>(i)]);
                idxs_ref_inliers.push_back(idxs_ref[static_cast<size_t>(i)]);
            }
        }
    }
    num_inliers = static_cast<int>(idxs_cur_inliers.size());

    auto temp_map = std::make_shared<Map>();
    f_ref_in->reset_points();
    f_cur->reset_points();
    KeyFramePtr kf_ref = std::make_shared<KeyFrame>(f_ref_in, f_ref_in->img);
    KeyFramePtr kf_cur = std::make_shared<KeyFrame>(f_cur, img_cur);
    temp_map->add_keyframe(kf_ref);
    temp_map->add_keyframe(kf_cur);

    std::vector<Eigen::Vector3d> pts3d;
    std::vector<bool> mask_pts3d;
    bool do_check = true;
    if (sensor_type == SensorType::RGBD || sensor_type == SensorType::STEREO) {
        auto unprojected = kf_ref->unproject_points_3d<double>(idxs_ref_inliers, false);
        pts3d = std::move(unprojected.first);
        mask_pts3d = std::move(unprojected.second);
        const int num_valid = static_cast<int>(
            std::count(mask_pts3d.begin(), mask_pts3d.end(), true));
        if (num_valid < Parameters::kInitializerNumMinNumPointsForPnPWithDepth) {
            return {out, false};
        }
        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2d> image_points;
        for (int i = 0; i < static_cast<int>(mask_pts3d.size()); ++i) {
            if (!mask_pts3d[static_cast<size_t>(i)]) {
                continue;
            }
            object_points.emplace_back(pts3d[static_cast<size_t>(i)].x(),
                                       pts3d[static_cast<size_t>(i)].y(),
                                       pts3d[static_cast<size_t>(i)].z());
            const int kp_idx = idxs_cur_inliers[static_cast<size_t>(i)];
            image_points.emplace_back(kf_cur->kps(kp_idx, 0), kf_cur->kps(kp_idx, 1));
        }
        cv::Mat rvec;
        cv::Mat tvec;
        const cv::Mat &K = kf_cur->camera ? kf_cur->camera->K_cv : cv::Mat();
        const cv::Mat &D = kf_cur->camera ? kf_cur->camera->D_cv : cv::Mat();
        const bool success =
            cv::solvePnP(object_points, image_points, K, D, rvec, tvec, false, cv::SOLVEPNP_EPNP);
        Eigen::Matrix4d Trc_pnp = Eigen::Matrix4d::Identity();
        if (success) {
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            R.convertTo(R, CV_64F);
            tvec.convertTo(tvec, CV_64F);
            Trc_pnp = pose_from_Rt(R, tvec);
        }
        kf_cur->update_pose(Trc_pnp);
        f_cur->update_pose(Trc_pnp);
        do_check = false;
    } else {
        const MatNx2f kpn_cur = keypoint_rows(kf_cur->kpsn, idxs_cur_inliers);
        const MatNx2f kpn_ref = keypoint_rows(kf_ref->kpsn, idxs_ref_inliers);
        auto triangulated =
            triangulate_normalized_points(kf_cur->Tcw(), kf_ref->Tcw(), kpn_cur, kpn_ref);
        pts3d = std::move(triangulated.first);
        mask_pts3d = std::move(triangulated.second);
    }

    auto [new_pts_count, mask_points, added_map_points] = temp_map->add_points(
        pts3d, mask_pts3d, kf_cur, kf_ref, idxs_cur_inliers, idxs_ref_inliers, img_cur, do_check,
        Parameters::kCosMaxParallaxInitializer, std::nullopt);

    double median_depth = -1.0;
    if (new_pts_count > num_min_triangulated_points) {
        const float reproj_error_before = temp_map->compute_mean_reproj_error();
        const double reproj_error_after =
            temp_map->optimize(Parameters::kLargeBAWindowSize, false, 20, true, false, nullptr);
        const int num_map_points = temp_map->num_points();
        is_ok = reproj_error_after < reproj_error_before &&
                num_map_points > num_min_triangulated_points;
        if (is_ok) {
            out.pts.reserve(added_map_points.size());
            for (const auto &point : added_map_points) {
                if (point) {
                    out.pts.push_back(point->pt());
                }
            }
            out.kf_cur = kf_cur;
            out.kf_ref = kf_ref;
            out.idxs_cur = select_indices(idxs_cur_inliers, mask_points);
            out.idxs_ref = select_indices(idxs_ref_inliers, mask_points);
        }
        if (is_ok && is_monocular) {
            MatNx3d points(static_cast<int>(out.pts.size()), 3);
            for (int i = 0; i < points.rows(); ++i) {
                points.row(i) = out.pts[static_cast<size_t>(i)].transpose();
            }
            median_depth = kf_cur->compute_points_median_depth<double>(points, 0.5);
            const double baseline = (kf_cur->Ow() - kf_ref->Ow()).norm();
            is_ok = median_depth > 0.0 && baseline > 0.0;
            if (is_ok) {
                const double ratio = median_depth / baseline;
                if (ratio > Parameters::kInitializerMinRatioDepthBaseline) {
                    is_ok = false;
                }
            }
            if (is_ok) {
                const double desired = Parameters::kInitializerDesiredMedianDepth;
                const double depth_scale = desired / median_depth;
                for (auto &pt : out.pts) {
                    pt *= depth_scale;
                }
                kf_cur->update_translation(kf_cur->tcw() * depth_scale);
            }
            if (is_ok && check_cell_coverage && kf_cur->camera) {
                ImageGrid grid(kf_cur->camera->width, kf_cur->camera->height, 3, 3);
                grid.add_points(keypoint_rows(kf_cur->kps, out.idxs_cur));
                is_ok = grid.is_each_cell_covered(1);
            }
        }
    }

    temp_map->delete_map();
    if (kf_cur) {
        kf_cur->map = nullptr;
    }
    if (kf_ref) {
        kf_ref->map = nullptr;
    }
    temp_map->keyframes.clear();
    temp_map->keyframes_map.clear();
    temp_map->points.clear();

    if (!is_ok) {
        ++num_failures;
        out = InitializerOutput();
    }
    return {out, is_ok};
}

} // namespace pyslam
