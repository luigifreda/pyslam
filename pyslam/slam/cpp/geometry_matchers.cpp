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

#include "geometry_matchers.h"
#include "config_parameters.h"
#include "feature_shared_resources.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "rotation_histogram.h"
#include "utils/geom_2views.h"
#include "utils/messages.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace pyslam {

// search by projection matches between {map points of f_ref} and {keypoints of f_cur},  (access
// frames from tracking thread, no need to lock)
std::tuple<std::vector<int>, std::vector<int>, int> ProjectionMatcher::search_frame_by_projection(
    const FramePtr &f_ref, FramePtr &f_cur,
    float max_reproj_distance /*= Parameters::kDefaultMaxReprojDistanceFrame*/,
    float max_descriptor_distance /*= -1.0f*/,
    float ratio_test /*= Parameters::kDefaultMatchRatioTestMap*/, bool is_monocular /*= true*/,
    const std::vector<int> &already_matched_ref_idxs /*= {}*/) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance = Parameters::kMaxDescriptorDistance; // take the updated value
    }

    int found_count_out = 0;
    std::vector<int> idxs_ref_out, idxs_cur_out;
    idxs_ref_out.reserve(f_ref->points.size());
    idxs_cur_out.reserve(f_cur->points.size());

    std::vector<bool> already_matched_ref_idxs_flags;
    const bool check_already_matched_ref_idxs = !already_matched_ref_idxs.empty();
    if (check_already_matched_ref_idxs) {
        already_matched_ref_idxs_flags.resize(f_ref->points.size(), false);
        for (const int &idx : already_matched_ref_idxs) {
            already_matched_ref_idxs_flags[idx] = true;
        }
    }

    RotationHistogram rot_histo;
    const bool check_orientation =
        Parameters::kCheckFeaturesOrientation && FeatureSharedResources::oriented_features;

    const auto &scale_factors = FeatureSharedResources::scale_factors;

    const bool is_stereo = f_cur->camera->is_stereo();
    const bool do_check_stereo_reproj_err = !f_cur->kps_ur.empty();
    if (is_stereo && !do_check_stereo_reproj_err) {
        MSG_RED_WARN("search_frame_by_projection: stereo but no stereo reproj error check");
    }

    // KD for f_cur
    const auto &kd_cur = f_cur->kd();
    if (!kd_cur) {
        MSG_ERROR("search_frame_by_projection: f_cur->kd() is not initialized");
        return std::make_tuple(std::vector<int>(), std::vector<int>(), 0);
    }

    bool check_forward_backward = !is_monocular;
    bool forward = false;
    bool backward = false;
    if (check_forward_backward) {
        const auto &Tcw = f_cur->pose();
        const Eigen::Matrix3d Rcw = Tcw.block<3, 3>(0, 0);
        const Eigen::Vector3d tcw = Tcw.block<3, 1>(0, 3);
        const Eigen::Vector3d twc = -Rcw.transpose() * tcw;

        const auto &Trw = f_ref->pose();
        const Eigen::Matrix3d Rrw = Trw.block<3, 3>(0, 0);
        const Eigen::Vector3d trw = Trw.block<3, 1>(0, 3);
        const Eigen::Vector3d trc = Rrw.transpose() * twc + trw;

        forward = trc[2] > f_cur->camera->b;
        backward = trc[2] < -f_cur->camera->b;
        check_forward_backward = forward || backward;
    }

    const auto Ow = f_cur->Ow();

    const int points_size = static_cast<int>(f_ref->points.size());
    for (int i = 0; i < points_size; ++i) {
        const MapPointPtr &p_ref = f_ref->points[i];
        // Consider only non-outlier
        if (!p_ref || f_ref->outliers[i])
            continue;

        // Discard already matched points
        if (check_already_matched_ref_idxs && already_matched_ref_idxs_flags[i])
            continue;

        // Project point on f_cur
        const auto [proj, depth] = f_cur->project_map_point<float>(p_ref, is_stereo);
        const auto &proj_uv = proj.head<2>();

        // Check if point projection is in image
        if (!f_cur->is_in_image<float>(proj_uv, depth))
            continue;
#if 0
        // Check if point is visible  (disabled in contiguous-frame matching)
        const auto PO = p_ref->pt() - Ow;
        const float dist3D = PO.norm();
        if (dist3D < p_ref->min_distance() || dist3D > p_ref->max_distance()) {
            continue;
        }
        const auto normal = p_ref->get_normal();
        const float cos_view = PO.dot(normal) / dist3D;
        if (cos_view < Parameters::kViewingCosLimitForPoint) {
            continue;
        }
#endif

        float best_dist = std::numeric_limits<float>::infinity();
        int best_k_idx = -1;
        int best_ref_idx = -1;

        const int kp_ref_octave = f_ref->octaves[i];

        const float kp_ref_scale_factor =
            (kp_ref_octave >= 0 && kp_ref_octave < (int)scale_factors.size())
                ? scale_factors[kp_ref_octave]
                : 1.0f;
        const float proj_radius = max_reproj_distance * kp_ref_scale_factor;
        const auto f_cur_idxs = kd_cur->query_ball_point(proj_uv, proj_radius);
        if (f_cur_idxs.empty())
            continue;

        // check each kp neighbor in f_cur
        for (const int f_cur_idx : f_cur_idxs) {
            const MapPointPtr &p_cur = f_cur->points[f_cur_idx];
            // skip if already matched p_cur
            if (p_cur && p_cur->num_observations() > 0) // we already matched p_cur => discard it
                continue;

            const int &kp_cur_octave = f_cur->octaves[f_cur_idx];

            if (do_check_stereo_reproj_err) {
                const float &kp_ur = f_cur->kps_ur[f_cur_idx];
                if (kp_ur >= 0) {
                    // check right u coordinate consistency
                    const float err_ur = std::fabs(proj[2] - kp_ur);
                    const float kp_cur_scale_factor = scale_factors[kp_cur_octave];
                    if (err_ur > max_reproj_distance * kp_cur_scale_factor)
                        continue;
                }
            }

            // check if point is in the same octave as the reference point
            if (check_forward_backward) {
                if (backward && kp_cur_octave > kp_ref_octave) {
                    continue;
                } else if (forward && kp_cur_octave < kp_ref_octave) {
                    continue;
                } else if (kp_cur_octave < (kp_ref_octave - 1) ||
                           kp_cur_octave > (kp_ref_octave + 1)) {
                    continue;
                }
            } else {
                if (kp_cur_octave < (kp_ref_octave - 1) || kp_cur_octave > (kp_ref_octave + 1))
                    continue;
            }

            // descriptor distance
            const float d = p_ref->min_des_distance(f_cur->des.row(f_cur_idx));
            if (d < best_dist) {
                best_dist = d;
                best_k_idx = f_cur_idx;
                best_ref_idx = i;
            }
        }

        if (best_k_idx >= 0 && best_dist < max_descriptor_distance) {
            if (p_ref->add_frame_view(f_cur, best_k_idx)) {
                ++found_count_out;
                idxs_ref_out.push_back(best_ref_idx);
                idxs_cur_out.push_back(best_k_idx);

                if (check_orientation) {
                    const int index_match = idxs_cur_out.size() - 1;
                    const float rot = f_ref->angles[best_ref_idx] - f_cur->angles[best_k_idx];
                    rot_histo.push(rot, index_match);
                }
            }
        }
    }

    if (check_orientation) {
        const std::vector<int> valid_match_idxs = rot_histo.get_valid_idxs();
        const int num_valid_matches = valid_match_idxs.size();
        const int num_total_matches = idxs_cur_out.size();
        // filter out invalid matches
        std::vector<int> filtered_idxs_ref_out;
        std::vector<int> filtered_idxs_cur_out;
        filtered_idxs_ref_out.reserve(num_valid_matches);
        filtered_idxs_cur_out.reserve(num_valid_matches);
        for (int idx : valid_match_idxs) {
            filtered_idxs_ref_out.push_back(idxs_ref_out[idx]);
            filtered_idxs_cur_out.push_back(idxs_cur_out[idx]);
        }
        idxs_ref_out = std::move(filtered_idxs_ref_out);
        idxs_cur_out = std::move(filtered_idxs_cur_out);

        std::cout << "checking orientation consistency - valid matches % :"
                  << static_cast<float>(num_valid_matches) / std::max(1, num_total_matches) * 100
                  << "% of " << num_total_matches << "matches" << std::endl;
    }
    found_count_out = idxs_cur_out.size();

    return std::make_tuple(std::move(idxs_ref_out), std::move(idxs_cur_out), found_count_out);
}

// Project keyframe map points onto f_cur; nearest unmatched f_cur keypoints by descriptor.
std::tuple<std::vector<int>, std::vector<int>, int>
ProjectionMatcher::search_keyframe_by_projection(
    const KeyFramePtr &kf_ref, FramePtr &f_cur, float max_reproj_distance,
    float max_descriptor_distance /*= -1.0f*/,
    float ratio_test /*= Parameters::kDefaultMatchRatioTestMap*/,
    const std::vector<int> &already_matched_ref_idxs /*= {}*/) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance = Parameters::kMaxDescriptorDistance; // take the updated value
    }

    if (!kf_ref->is_keyframe) {
        MSG_ERROR("search_keyframe_by_projection: kf_ref must be a KeyFrame");
        return std::make_tuple(std::vector<int>(), std::vector<int>(), 0);
    }

    int found_count_out = 0;
    std::vector<int> idxs_ref_out, idxs_cur_out;

    RotationHistogram rot_histo;
    const bool check_orientation =
        Parameters::kCheckFeaturesOrientation && FeatureSharedResources::oriented_features;

    const Eigen::Matrix4d Tcw = f_cur->pose();
    const Eigen::Matrix3d Rcw = Tcw.block<3, 3>(0, 0);
    const Eigen::Vector3d tcw = Tcw.block<3, 1>(0, 3);
    const Eigen::Vector3d Ow = -Rcw.transpose() * tcw;

    // Keyframe matched points (by index alignment to keypoints)
    const auto ref_mps = kf_ref->get_matched_points();
    if (ref_mps.empty())
        return std::make_tuple(std::vector<int>(), std::vector<int>(), 0);

    std::vector<bool> already_matched_ref_idxs_flags;
    const bool check_already_matched_ref_idxs = !already_matched_ref_idxs.empty();
    if (check_already_matched_ref_idxs) {
        already_matched_ref_idxs_flags.resize(ref_mps.size(), false);
        for (const int &idx : already_matched_ref_idxs) {
            already_matched_ref_idxs_flags[idx] = true;
        }
    }

    const auto &scale_factors = FeatureSharedResources::scale_factors;

    // KD for f_cur
    const auto &kd_cur = f_cur->kd();
    if (!kd_cur) {
        MSG_ERROR("search_frame_by_projection: f_cur->kd() is not initialized");
        return std::make_tuple(std::vector<int>(), std::vector<int>(), 0);
    }

    for (int i = 0; i < (int)ref_mps.size(); ++i) {
        const MapPointPtr &p_ref = ref_mps[i];
        if (!p_ref || p_ref->is_bad())
            continue;

        if (check_already_matched_ref_idxs && already_matched_ref_idxs_flags[i])
            continue;

        const auto [proj_uv, depth] = f_cur->project_map_point<float>(p_ref);
        if (!f_cur->is_in_image<float>(proj_uv, depth))
            continue;

        const auto PO = p_ref->pt() - Ow;
        const float dist3D = PO.norm();
        if (dist3D < p_ref->min_distance() || dist3D > p_ref->max_distance()) {
            continue;
        }
        const auto normal = p_ref->get_normal();
        const float cos_view = PO.dot(normal) / dist3D;
        if (cos_view < Parameters::kViewingCosLimitForPoint) {
            continue;
        }

        const int predicted_level = p_ref->predict_detection_level(dist3D);
        const float kp_ref_scale_factor =
            (predicted_level >= 0 && predicted_level < (int)scale_factors.size())
                ? scale_factors[predicted_level]
                : 1.0f;
        const float proj_radius = max_reproj_distance * kp_ref_scale_factor;

        const auto &kd_cur_idxs = kd_cur->query_ball_point(proj_uv, proj_radius);
        if (kd_cur_idxs.empty())
            continue;

        float best_dist = std::numeric_limits<float>::infinity();
        float best_dist2 = std::numeric_limits<float>::infinity();
        int best_level = -1;
        int best_level2 = -1;
        int best_k_idx = -1;

        for (const int f_cur_idx : kd_cur_idxs) {
            const MapPointPtr &p_cur = f_cur->points[f_cur_idx];
            if (p_cur)
                continue; // skip if already matched

            const int &kp_cur_level = f_cur->octaves[f_cur_idx];

            if (kp_cur_level < (predicted_level - 1) || kp_cur_level > (predicted_level + 1))
                continue;

            const float d = p_ref->min_des_distance(f_cur->des.row(f_cur_idx));
            if (d < best_dist) {
                best_dist2 = best_dist;
                best_level2 = best_level;
                best_dist = d;
                best_level = kp_cur_level;
                best_k_idx = f_cur_idx;
            } else if (d < best_dist2) {
                best_dist2 = d;
                best_level2 = kp_cur_level;
            }
        }

        if (best_k_idx >= 0 && best_dist < max_descriptor_distance) {
            if (best_level == best_level2 && best_dist > best_dist2 * ratio_test)
                continue;

            if (p_ref->add_frame_view(f_cur, best_k_idx)) {
                ++found_count_out;
                idxs_ref_out.push_back(i);
                idxs_cur_out.push_back(best_k_idx);

                if (check_orientation) {
                    const int index_match = idxs_cur_out.size() - 1;
                    const float rot = kf_ref->angles[i] - f_cur->angles[best_k_idx];
                    rot_histo.push(rot, index_match);
                }
            }
        }
    }

    if (check_orientation) {
        const std::vector<int> valid_match_idxs = rot_histo.get_valid_idxs();
        const int num_valid_matches = valid_match_idxs.size();
        const int num_total_matches = idxs_cur_out.size();
        // filter out invalid matches
        std::vector<int> filtered_idxs_ref_out;
        std::vector<int> filtered_idxs_cur_out;
        filtered_idxs_ref_out.reserve(num_valid_matches);
        filtered_idxs_cur_out.reserve(num_valid_matches);
        for (int idx : valid_match_idxs) {
            filtered_idxs_ref_out.push_back(idxs_ref_out[idx]);
            filtered_idxs_cur_out.push_back(idxs_cur_out[idx]);
        }
        idxs_ref_out = std::move(filtered_idxs_ref_out);
        idxs_cur_out = std::move(filtered_idxs_cur_out);

        std::cout << "checking orientation consistency - valid matches % :"
                  << static_cast<float>(num_valid_matches) / std::max(1, num_total_matches) * 100
                  << "% of " << num_total_matches << "matches" << std::endl;
    }
    found_count_out = idxs_cur_out.size();

    return std::make_tuple(std::move(idxs_ref_out), std::move(idxs_cur_out), found_count_out);
}

// Search matches between input map points and unmatched keypoints of f_cur.
std::pair<int, std::vector<int>> ProjectionMatcher::search_map_by_projection(
    const std::vector<MapPointPtr> &points, FramePtr &f_cur,
    float max_reproj_distance /*= Parameters::kMaxReprojectionDistanceMap*/,
    float max_descriptor_distance /*= -1.0f*/,
    float ratio_test /*= Parameters::kMatchRatioTestMap*/,
    float far_points_threshold /*= std::numeric_limits<float>::infinity()*/) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance = Parameters::kMaxDescriptorDistance; // take the updated value
    }

    if (points.empty())
        return std::pair(0, std::vector<int>());

    int found_count_out = 0;
    std::vector<int> idxs_cur_out;

    // KD neighbors
    const auto &kd_cur = f_cur->kd();
    if (!kd_cur) {
        MSG_ERROR("search_map_by_projection: f_cur->kd() is not initialized");
        return std::pair(0, std::vector<int>());
    }

    const auto &scale_factors = FeatureSharedResources::scale_factors;

    const auto f_cur_Ow = f_cur->Ow();

    const int points_size = static_cast<int>(points.size());
    for (int i = 0; i < points_size; ++i) {
        const MapPointPtr &p = points[i];
        if (!p || p->is_bad())
            continue;

        if (p->last_frame_id_seen == f_cur->id)
            continue;

        const auto [proj_uv, depth] = f_cur->project_map_point<float>(p);
        if (!f_cur->is_in_image<float>(proj_uv, depth))
            continue;
        if (depth > far_points_threshold)
            continue;

        const auto PO = p->pt() - f_cur_Ow;
        const float dist3D = PO.norm();
        if (dist3D < p->min_distance() || dist3D > p->max_distance()) {
            continue;
        }
        const auto normal = p->get_normal();
        const float cos_view = PO.dot(normal) / dist3D;
        if (cos_view < Parameters::kViewingCosLimitForPoint) {
            continue;
        }

        p->increase_visible();

        const int predicted_level = p->predict_detection_level(dist3D);
        const float kp_ref_scale_factor =
            (predicted_level >= 0 && predicted_level < (int)scale_factors.size())
                ? scale_factors[predicted_level]
                : 1.0f;
        const float proj_radius = max_reproj_distance * kp_ref_scale_factor;
        const auto &kd_cur_idxs = kd_cur->query_ball_point(proj_uv, proj_radius);
        if (kd_cur_idxs.empty())
            continue;

        float best_dist = std::numeric_limits<float>::infinity();
        float best_dist2 = std::numeric_limits<float>::infinity();
        int best_level = -1;
        int best_level2 = -1;
        int best_k_idx = -1;

        for (const int f_cur_idx : kd_cur_idxs) {
            const MapPointPtr &p_cur = f_cur->points[f_cur_idx];
            if (p_cur && p_cur->num_observations() > 0)
                continue; // already matched p_cur => discard it

            const int &kp_cur_level = f_cur->octaves[f_cur_idx];

            if (kp_cur_level < (predicted_level - 1) || kp_cur_level > predicted_level)
                continue;

            const float d = p->min_des_distance(f_cur->des.row(f_cur_idx));
            if (d < best_dist) {
                best_dist2 = best_dist;
                best_level2 = best_level;
                best_dist = d;
                best_level = kp_cur_level;
                best_k_idx = f_cur_idx;
            } else if (d < best_dist2) {
                best_dist2 = d;
                best_level2 = kp_cur_level;
            }
        }

        if (best_k_idx >= 0 && best_dist < max_descriptor_distance) {
            if (best_level == best_level2 && best_dist > best_dist2 * ratio_test)
                continue;

            if (p->add_frame_view(f_cur, best_k_idx)) {
                ++found_count_out;
                idxs_cur_out.push_back(best_k_idx);
            }
        }
    }

    return std::pair(found_count_out, std::move(idxs_cur_out));
}

// search by projection matches between {map points of last frames} and {unmatched keypoints of
// f_cur}, (access frame from tracking thread, no need to lock)
std::pair<int, std::vector<int>> ProjectionMatcher::search_local_frames_by_projection(
    Map *map, FramePtr &f_cur, int local_window_size, float max_descriptor_distance /*= -1.0f*/) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance = Parameters::kMaxDescriptorDistance; // take the updated value
    }

    if (!map)
        return std::pair(0, std::vector<int>());

    // Collect frames
    auto frames = map->get_last_keyframes(local_window_size);
    std::unordered_set<MapPointPtr> unique_pts;
    for (const auto &frame : frames) {
        auto pts = frame->get_points();
        for (auto &p : pts)
            if (p)
                unique_pts.insert(p);
    }
    std::vector<MapPointPtr> pts(unique_pts.begin(), unique_pts.end());
    return search_map_by_projection(pts, f_cur, Parameters::kMaxReprojectionDistanceMap,
                                    max_descriptor_distance);
}

// search by projection matches between {all map points} and {unmatched keypoints of f_cur}
std::pair<int, std::vector<int>>
ProjectionMatcher::search_all_map_by_projection(Map *map, FramePtr &f_cur,
                                                float max_descriptor_distance /*= -1.0f*/) {
    if (max_descriptor_distance < 0) {
        max_descriptor_distance = Parameters::kMaxDescriptorDistance; // take the updated value
    }

    if (!map)
        return std::pair(0, std::vector<int>());
    auto pts_vec = map->get_points_vector();
    return ProjectionMatcher::search_map_by_projection(
        pts_vec, f_cur, Parameters::kMaxReprojectionDistanceMap, max_descriptor_distance);
}

// search by projection more matches between {input map points} and {unmatched keypoints of frame
// f_cur}
//  in:
//    points: input map points
//    f_cur: current frame
//    Scw: suggested se3 or sim3 transformation
//    f_cur_matched_points: matched points in current frame  (f_cur_matched_points[i] is the i-th
//    map point matched on f_cur or None)
// NOTE1: f_cur_matched_points is modified in place and passed by reference in output
// NOTE2: The suggested transformation Scw (in se3 or sim3) is used in the search (instead of using
//  the current frame pose)
std::pair<int, std::vector<MapPointPtr> &> ProjectionMatcher::search_more_map_points_by_projection(
    const std::vector<MapPointPtr> &points, FramePtr &f_cur, const Sim3Pose &Scw,
    std::vector<MapPointPtr> &f_cur_matched_points,
    const std::vector<int> &f_cur_matched_points_idxs /*= {}*/,
    float max_reproj_distance /*= Parameters::kMaxReprojectionDistanceMap*/,
    float max_descriptor_distance /*= -1.0f*/) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance =
            0.5 * Parameters::kMaxDescriptorDistance; // take the updated value
    }

    if (points.empty())
        return std::pair<int, std::vector<MapPointPtr> &>(0, f_cur_matched_points);

    if (f_cur_matched_points.size() != f_cur->points.size()) {
        MSG_ERROR("search_more_map_points_by_projection: f_cur_matched_points.size() != "
                  "f_cur->points.size()");
        return std::pair<int, std::vector<MapPointPtr> &>(0, f_cur_matched_points);
    }

    auto kd_cur = f_cur->kd();
    if (!kd_cur) {
        MSG_ERROR("search_more_map_points_by_projection: f_cur->kd() is not initialized");
        return std::pair<int, std::vector<MapPointPtr> &>(0, f_cur_matched_points);
    }

    // extract from sim3 Scw=[s*Rcw, tcw; 0, 1] the corresponding se3 transformation Tcw=[Rcw,
    // tcw/s]
    const double scw = Scw.s();
    const Eigen::Matrix3d Rcw = Scw.R();
    const Eigen::Vector3d tcw = Scw.t() / scw;
    const Eigen::Vector3d Ow = -Rcw.transpose() * tcw;

    std::unordered_set<int> f_cur_matched_points_idxs_set;
    if (!f_cur_matched_points_idxs.empty()) {
        f_cur_matched_points_idxs_set = std::unordered_set<int>(f_cur_matched_points_idxs.begin(),
                                                                f_cur_matched_points_idxs.end());
    }

    int found_count_out = 0;

    const auto &camera = f_cur->get_camera();
    if (!camera) {
        MSG_ERROR("search_more_map_points_by_projection: f_cur->get_camera() is not initialized");
        return std::pair<int, std::vector<MapPointPtr> &>(0, f_cur_matched_points);
    }

    const auto &scale_factors = FeatureSharedResources::scale_factors;

    for (const auto &p : points) {
        if (!p || p->is_bad())
            continue;

        if (f_cur_matched_points_idxs_set.contains(p->id))
            continue;

        const Eigen::Vector3d pt_w = p->pt();
        const Eigen::Vector3f pt_c = (Rcw * pt_w + tcw).cast<float>();
        const auto [proj_uv, depth] = camera->project_point(pt_c);

        // check if visible in image
        if (!camera->is_in_image<float>(proj_uv, depth))
            continue;

        const Eigen::Vector3d PO = pt_w - Ow;
        const float dist3D = PO.norm();
        if (dist3D < p->min_distance() || dist3D > p->max_distance()) {
            continue;
        }
        const auto normal = p->get_normal();
        const float cos_view = normal.dot(PO) / dist3D;
        if (cos_view < Parameters::kViewingCosLimitForPoint) {
            continue;
        }

        const int predicted_level = p->predict_detection_level(dist3D);
        const float kp_ref_scale_factor =
            (predicted_level >= 0 && predicted_level < (int)scale_factors.size())
                ? scale_factors[predicted_level]
                : 1.0f;
        const float proj_radius = max_reproj_distance * kp_ref_scale_factor;
        const auto &kd_cur_idxs = kd_cur->query_ball_point(proj_uv, proj_radius);
        if (kd_cur_idxs.empty())
            continue;

        float best_dist = std::numeric_limits<float>::infinity();
        int best_k_idx = -1;

        for (const int f_cur_idx : kd_cur_idxs) {
            const MapPointPtr &p_cur = f_cur->points[f_cur_idx];
            if (p_cur)
                continue; // already matched p_cur => discard it

            const int &kp_cur_level = f_cur->octaves[f_cur_idx];

            if (kp_cur_level < (predicted_level - 1) || kp_cur_level > predicted_level)
                continue;

            const float d = p->min_des_distance(f_cur->des.row(f_cur_idx));
            if (d < best_dist) {
                best_dist = d;
                best_k_idx = f_cur_idx;
            }
        }

        if (best_k_idx >= 0 && best_dist < max_descriptor_distance) {
            f_cur_matched_points[best_k_idx] = p;
            ++found_count_out;
        }
    }

    return std::pair<int, std::vector<MapPointPtr> &>(found_count_out, f_cur_matched_points);
}

// search by projection matches between {input map points} and {keyframe points} and fuse them if
// they are close enough
int ProjectionMatcher::search_and_fuse(const std::vector<MapPointPtr> &points,
                                       KeyFramePtr &keyframe, float max_reproj_distance,
                                       float max_descriptor_distance, float ratio_test) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance =
            0.5f * Parameters::kMaxDescriptorDistance; // more conservative check
    }

    int fused_pts_count = 0;
    if (points.empty()) {
        MSG_RED_WARN("search_and_fuse - no points");
        return fused_pts_count;
    }

    const auto &scale_factors = FeatureSharedResources::scale_factors;
    const auto &inv_level_sigmas2 = FeatureSharedResources::inv_level_sigmas2;

    const bool is_stereo = keyframe->camera->is_stereo();
    const auto &camera = keyframe->camera;
    const auto &kf_Ow = keyframe->Ow();

    const bool do_stereo_check = !keyframe->kps_ur.empty();
    if (is_stereo && !do_stereo_check) {
        MSG_RED_WARN("search_and_fuse: stereo but no stereo reproj error check");
    }

    // KD tree for keyframe
    const auto &kd = keyframe->kd();
    if (!kd) {
        MSG_ERROR("search_and_fuse: keyframe->kd() is not initialized");
        return fused_pts_count;
    }

    for (MapPointPtr p : points) {
        if (!p || p->is_bad_or_is_in_keyframe(keyframe)) {
            continue;
        }

        const auto [proj, depth] = keyframe->project_point<double>(p->pt(), is_stereo);
        const auto proj_uv = proj.head<2>();
        if (!camera->is_in_image<double>(proj_uv, depth)) {
            continue;
        }

        const auto PO = p->pt() - kf_Ow;
        const float dist3D = PO.norm();
        if (dist3D < p->min_distance() || dist3D > p->max_distance()) {
            continue;
        }
        const auto normal = p->get_normal();
        const float cos_view = PO.dot(normal) / dist3D;
        if (cos_view < Parameters::kViewingCosLimitForPoint) {
            continue;
        }

        const int predicted_level = p->predict_detection_level(dist3D);
        const float kp_scale_factor = scale_factors[predicted_level];
        const float radius = max_reproj_distance * kp_scale_factor;
        const auto kd_idxs = kd->query_ball_point(proj_uv, radius);
        if (kd_idxs.empty()) {
            continue;
        }

        float best_dist = std::numeric_limits<float>::infinity();
        int best_kd_idx = -1;

        for (const int kd_idx : kd_idxs) {
            // Check detection level
            const int kp_level = keyframe->octaves[kd_idx];
            if (kp_level < (predicted_level - 1) || kp_level > predicted_level) {
                continue;
            }

            // Check reprojection error
            const float invSigma2 = inv_level_sigmas2[kp_level];
            const auto kpsu = keyframe->kpsu.row(kd_idx);
            const Eigen::Vector2f err = proj_uv.cast<float>() - kpsu.transpose();
            float chi2 = err.squaredNorm() * invSigma2;
            const float kp_ur = keyframe->kps_ur[kd_idx];
            if (do_stereo_check && kp_ur >= 0) {
                const float proj_ur = proj[2];
                chi2 += (kp_ur - proj_ur) * (kp_ur - proj_ur) * invSigma2;
                if (chi2 > Parameters::kChi2Stereo) {
                    continue;
                }
            } else {
                if (chi2 > Parameters::kChi2Mono) {
                    continue;
                }
            }

            // Descriptor distance
            const float descriptor_dist = p->min_des_distance(keyframe->des.row(kd_idx));
            if (descriptor_dist < best_dist) {
                best_dist = descriptor_dist;
                best_kd_idx = kd_idx;
            }
        }

        if (best_dist < max_descriptor_distance) {
            auto p_keyframe = keyframe->get_point_match(best_kd_idx);
            if (p_keyframe) {
                const auto [p_keyframe_is_bad, p_keyframe_is_good_with_better_num_obs] =
                    p_keyframe->is_bad_and_is_good_with_min_obs(p->num_observations());
                if (!p_keyframe_is_bad) {
                    if (p_keyframe_is_good_with_better_num_obs) {
                        p->replace_with(p_keyframe);
                    } else {
                        p_keyframe->replace_with(p);
                    }
                }
            } else {
                p->add_observation(keyframe, best_kd_idx);
            }
            fused_pts_count++;
        }
    }

    return fused_pts_count;
}

// search by projection matches between {input map points} and {keyframe points} and fuse them if
// they are close enough use suggested Scw to project
std::vector<MapPointPtr> &ProjectionMatcher::search_and_fuse_for_loop_correction(
    const KeyFramePtr &keyframe, const Sim3Pose &Scw, const std::vector<MapPointPtr> &points,
    std::vector<MapPointPtr> &replace_points, float max_reproj_distance,
    float max_descriptor_distance) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance =
            0.5f * Parameters::kMaxDescriptorDistance; // more conservative check
    }

    if (points.size() != replace_points.size()) {
        MSG_RED_WARN("search_and_fuse_for_loop_correction: points.size() != replace_points.size()");
        return replace_points;
    }

    int fused_pts_count = 0;
    if (points.empty()) {
        MSG_RED_WARN("search_and_fuse_for_loop_correction - no points");
        return replace_points;
    }

    // Extract transformation from Sim3
    const double scw = Scw.s();
    const Eigen::Matrix3d Rcw = Scw.R();
    const Eigen::Vector3d tcw = Scw.t() / scw;
    const Eigen::Vector3d Ow = -Rcw.transpose() * tcw;

    const auto &scale_factors = FeatureSharedResources::scale_factors;
    const auto &camera = keyframe->camera;

    // KD tree for keyframe
    const auto &kd = keyframe->kd();
    if (!kd) {
        MSG_ERROR("search_and_fuse_for_loop_correction: keyframe->kd() is not initialized");
        return replace_points;
    }

    const int points_size = static_cast<int>(points.size());
    for (int i = 0; i < points_size; ++i) {
        MapPointPtr p = points[i];
        if (!p || p->is_bad() || p->is_in_keyframe(keyframe)) {
            continue;
        }

        const Eigen::Vector3d pt_w = p->pt();
        const Eigen::Vector3f pt_c = (Rcw * pt_w + tcw).cast<float>();
        const auto [proj_uv, depth] = camera->project_point(pt_c);
        if (!camera->is_in_image<float>(proj_uv, depth)) {
            continue;
        }

        const auto PO = pt_w - Ow;
        const float dist3D = PO.norm();
        if (dist3D < p->min_distance() || dist3D > p->max_distance()) {
            continue;
        }
        const auto normal = p->get_normal();
        const float cos_view = PO.dot(normal) / dist3D;
        if (cos_view < Parameters::kViewingCosLimitForPoint) {
            continue;
        }

        const int predicted_level = p->predict_detection_level(dist3D);
        const float kp_scale_factor = scale_factors[predicted_level];
        const float radius = max_reproj_distance * kp_scale_factor;
        const auto kd_idxs = kd->query_ball_point(proj_uv, radius);
        if (kd_idxs.empty()) {
            continue;
        }

        float best_dist = std::numeric_limits<float>::infinity();
        int best_kd_idx = -1;

        for (const int kd_idx : kd_idxs) {
            const int kp_level = keyframe->octaves[kd_idx];
            if (kp_level < (predicted_level - 1) || kp_level > predicted_level) {
                continue;
            }

            const float descriptor_dist = p->min_des_distance(keyframe->des.row(kd_idx));
            if (descriptor_dist < best_dist) {
                best_dist = descriptor_dist;
                best_kd_idx = kd_idx;
            }
        }
        if (best_dist < max_descriptor_distance) {
            auto p_keyframe = keyframe->get_point_match(best_kd_idx);
            // if there is already a map point replace it
            if (p_keyframe) {
                if (!p_keyframe->is_bad()) {
                    replace_points[i] = p_keyframe;
                }
            } else {
                p->add_observation(keyframe, best_kd_idx);
            }
            fused_pts_count++;
        }
    }

    return replace_points;
}

// search new matches between unmatched map points of kf1 and kf2 by using a know sim3
// transformation (guided matching) in:
//   kf1, kf2: keyframes
//   idxs1, idxs2:  kf1.points(idxs1[i]) is matched with kf2.points(idxs2[i])
//   s12, R12, t12: sim3 transformation that guides the matching
// out:
//   - new_matches12: where kf2.points(new_matches12[i]) is matched to i-th map point in kf1
//   (includes the input matches) if new_matches12[i]>0
//   - new_matches21: where kf1.points(new_matches21[i]) is matched to i-th map point in kf2
//   (includes the input matches) if new_matches21[i]>0
std::tuple<int, std::vector<int>, std::vector<int>>
ProjectionMatcher::search_by_sim3(const KeyFramePtr &kf1, const KeyFramePtr &kf2,
                                  const std::vector<int> &idxs1, const std::vector<int> &idxs2,
                                  float s12, const Eigen::Matrix3d &R12, const Eigen::Vector3d &t12,
                                  float max_reproj_distance, float max_descriptor_distance) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance = Parameters::kMaxDescriptorDistance;
    }

    if (idxs1.size() != idxs2.size()) {
        MSG_ERROR("search_by_sim3: idxs1.size() != idxs2.size()");
        return std::make_tuple(0, std::vector<int>(), std::vector<int>());
    }

    // Sim3 transformations between cameras
    const Eigen::Matrix3d sR12 = s12 * R12;
    const Eigen::Matrix3d sR21 = (1.0 / s12) * R12.transpose();
    const Eigen::Vector3d t21 = -sR21 * t12;

    const auto map_points1 = kf1->get_points();
    const int n1 = map_points1.size();
    // kf2.points(new_matches12[i]) is matched to i-th map point in kf1
    std::vector<int> new_matches12(n1, -1);

    const auto map_points2 = kf2->get_points();
    const int n2 = map_points2.size();
    // kf1.points(new_matches21[i]) is matched to i-th map point in kf2
    std::vector<int> new_matches21(n2, -1);

    // Filter and integrate existing matches
    std::unordered_set<int> already_matched_idxs1;
    std::unordered_set<int> already_matched_idxs2;
    for (size_t i = 0; i < idxs1.size(); ++i) {
        const int idx1 = idxs1[i];
        const int idx2 = idxs2[i];
        const MapPointPtr &mp1 = map_points1[idx1];
        const MapPointPtr &mp2 = map_points2[idx2];
        if (!mp1 || !mp2 || mp1->is_bad() || mp2->is_bad()) {
            continue;
        }
        new_matches12[idx1] = idx2;
        new_matches21[idx2] = idx1;
        already_matched_idxs1.insert(idx1);
        already_matched_idxs2.insert(idx2);
    }

    const auto &scale_factors = FeatureSharedResources::scale_factors;

    // KD tree for kf1
    const auto &kd1 = kf1->kd();
    if (!kd1) {
        MSG_ERROR("search_by_sim3: kf1->kd() is not initialized");
        return std::make_tuple(0, std::vector<int>(), std::vector<int>());
    }

    // KD tree for kf2
    const auto &kd2 = kf2->kd();
    if (!kd2) {
        MSG_ERROR("search_by_sim3: kf2->kd() is not initialized");
        return std::make_tuple(0, std::vector<int>(), std::vector<int>());
    }

    const auto &camera1 = kf1->camera;
    const auto Rc1w = kf1->Rcw();
    const auto tc1w = kf1->tcw();

    const auto &camera2 = kf2->camera;
    const auto Rc2w = kf2->Rcw();
    const auto tc2w = kf2->tcw();

    // Find unmatched map points of kf1
    for (size_t idx1 = 0; idx1 < n1; ++idx1) {
        const MapPointPtr &mp1 = map_points1[idx1];
        if (!mp1 || mp1->is_bad()) {
            continue;
        }
        if (already_matched_idxs1.contains(idx1)) {
            continue;
        }

        const Eigen::Vector3d pt1_w = mp1->pt();
        const Eigen::Vector3d pt1_c1 = (Rc1w * pt1_w + tc1w);               // world to camera 1
        const Eigen::Vector3f pt1_c2 = (sR21 * pt1_c1 + t21).cast<float>(); // camera 1 to camera 2
        const auto [proj_uv2, depth2] = camera2->project_point(pt1_c2);
        if (!camera2->is_in_image<float>(proj_uv2, depth2)) {
            continue;
        }

        const float dist3D2 = pt1_c2.norm();
        if (dist3D2 < mp1->min_distance() || dist3D2 > mp1->max_distance()) {
            continue;
        }
        const auto normal2 = mp1->get_normal().cast<float>();
#if 0        
        const float cos_view2 = pt1_c2.dot(normal2) / dist3D2;
        if (cos_view2 < Parameters::kViewingCosLimitForPoint) {
            continue;
        }
#endif

        const int predicted_level2 = mp1->predict_detection_level(dist3D2);
        const float kp_scale_factor2 = scale_factors[predicted_level2];
        const float radius2 = max_reproj_distance * kp_scale_factor2;
        const auto kd2_idxs = kd2->query_ball_point(proj_uv2, radius2);
        if (kd2_idxs.empty()) {
            continue;
        }

        float best_dist2 = std::numeric_limits<float>::infinity();
        int best_kd_idx2 = -1;

        for (const int kd2_idx : kd2_idxs) {
            const int kp_level2 = kf2->octaves[kd2_idx];
            if (kp_level2 < (predicted_level2 - 1) || kp_level2 > predicted_level2) {
                continue;
            }

            const float descriptor_dist2 = mp1->min_des_distance(kf2->des.row(kd2_idx));
            if (descriptor_dist2 < best_dist2) {
                best_dist2 = descriptor_dist2;
                best_kd_idx2 = kd2_idx;
            }
        }
        if (best_dist2 < max_descriptor_distance) {
            if (new_matches21[best_kd_idx2] == -1) {
                new_matches12[idx1] = best_kd_idx2;
            }
        }
    }

    // Find unmatched map points of kf2
    for (size_t idx2 = 0; idx2 < n2; ++idx2) {
        const MapPointPtr &mp2 = map_points2[idx2];
        if (!mp2 || mp2->is_bad()) {
            continue;
        }
        if (already_matched_idxs2.contains(idx2)) {
            continue;
        }

        const Eigen::Vector3d pt2_w = mp2->pt();
        const Eigen::Vector3d pt2_c2 = (Rc2w * pt2_w + tc2w);               // world to camera 2
        const Eigen::Vector3f pt2_c1 = (sR12 * pt2_c2 + t12).cast<float>(); // camera 2 to camera 1
        const auto [proj_uv1, depth1] = camera1->project_point(pt2_c1);
        if (!camera1->is_in_image<float>(proj_uv1, depth1)) {
            continue;
        }

        const float dist3D1 = pt2_c1.norm();
        if (dist3D1 < mp2->min_distance() || dist3D1 > mp2->max_distance()) {
            continue;
        }
        const auto normal1 = mp2->get_normal().cast<float>();
#if 0        
        const float cos_view1 = pt2_c1.dot(normal1) / dist3D1;
        if (cos_view1 < Parameters::kViewingCosLimitForPoint) {
            continue;
        }
#endif

        const int predicted_level1 = mp2->predict_detection_level(dist3D1);
        const float kp_scale_factor1 = scale_factors[predicted_level1];
        const float radius1 = max_reproj_distance * kp_scale_factor1;
        const auto kd1_idxs = kd1->query_ball_point(proj_uv1, radius1);
        if (kd1_idxs.empty()) {
            continue;
        }

        float best_dist1 = std::numeric_limits<float>::infinity();
        int best_kd_idx1 = -1;

        for (const int kd1_idx : kd1_idxs) {
            const int kp_level1 = kf1->octaves[kd1_idx];
            if (kp_level1 < (predicted_level1 - 1) || kp_level1 > predicted_level1) {
                continue;
            }

            const float descriptor_dist1 = mp2->min_des_distance(kf1->des.row(kd1_idx));
            if (descriptor_dist1 < best_dist1) {
                best_dist1 = descriptor_dist1;
                best_kd_idx1 = kd1_idx;
            }
        }
        if (best_dist1 < max_descriptor_distance) {
            if (new_matches12[best_kd_idx1] == -1) {
                new_matches21[idx2] = best_kd_idx1;
            }
        }
    }

    // Check agreement
    int num_matches_found = 0;
    for (int i1 = 0; i1 < n1; ++i1) {
        const int idx2 = new_matches12[i1]; // index of kf2 point that matches with i1-th kf1 point
        if (idx2 >= 0) {
            const int idx1 =
                new_matches21[idx2]; // index of kf1 point that matches with idx2-th kf2 point
            if (idx1 != i1) {        // reset if mismatch
                new_matches12[i1] = -1;
                new_matches21[idx2] = -1;
            } else {
                num_matches_found++;
            }
        }
    }

    return std::make_tuple(num_matches_found, std::move(new_matches12), std::move(new_matches21));
}

// search keypoint matches (for triangulations) between f1 and f2
// search for matches between unmatched keypoints (without a corresponding map point)
// in input we have already some pose estimates for f1 and f2
std::tuple<std::vector<int>, std::vector<int>, int> EpipolarMatcher::search_frame_for_triangulation(
    const KeyFramePtr &kf1, const KeyFramePtr &kf2, const std::vector<int> &idxs1_in /*= {}*/,
    const std::vector<int> &idxs2_in /*= {}*/, float max_descriptor_distance /*= -1.0f*/,
    bool is_monocular /*= true*/) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance =
            0.5 * Parameters::kMaxDescriptorDistance; // take the updated value
    }

    if (!kf1 || !kf2)
        return std::tuple(std::vector<int>(), std::vector<int>(), 0);

    const Eigen::Vector3d O1w = kf1->Ow();
    const Eigen::Vector3d O2w = kf2->Ow();
    // compute epipoles
    const auto [e1_d, _1] = kf1->project_point<double>(O2w); // in first frame
    const auto [e2_d, _2] = kf2->project_point<double>(O1w); // in second frame
    const Eigen::Vector2f e1 = e1_d.cast<float>();
    const Eigen::Vector2f e2 = e2_d.cast<float>();

    const double baseline = (O1w - O2w).norm();

    if (!is_monocular) { // we assume the Inializer has been used for building the first map
        if (baseline < kf2->camera->b)
            return std::tuple(std::vector<int>(), std::vector<int>(), 0);
    } else {
        double median_depth = kf2->compute_points_median_depth<double>();
        if (median_depth == -1) {
            MSG_WARN("search_frame_for_triangulation: f2 with negative median depth");
            median_depth = kf1->compute_points_median_depth<double>();
        }
        const double ratio_baseline_depth = baseline / median_depth;
        if (ratio_baseline_depth < Parameters::kMinRatioBaselineDepth) {
            MSG_WARN("search_frame_for_triangulation: impossible with too low ratioBaselineDepth!");
            return std::tuple(std::vector<int>(), std::vector<int>(), 0);
        }
    }

    // compute the fundamental matrix between the two frames by using their estimated poses
    const auto [F12, H21] = geom_2views::computeF12(kf1, kf2);

    std::vector<int> idxs1 = idxs1_in;
    std::vector<int> idxs2 = idxs2_in;

    if (idxs1.empty() || idxs2.empty()) {
        const float ratio_test = Parameters::kFeatureMatchDefaultRatioTest;
        const bool kRowMatching = false;
        const float max_disparity = std::numeric_limits<float>::infinity();
        const auto [idxs1_, idxs2_] = FeatureSharedResources::feature_matching_callback(
            kf1->img, kf2->img, kf1->des, kf2->des, kf1->kps, kf2->kps, ratio_test, kRowMatching,
            max_disparity);
        idxs1 = std::move(idxs1_);
        idxs2 = std::move(idxs2_);
    }

    if (idxs1.empty() || idxs2.empty()) {
        return std::tuple(std::vector<int>(), std::vector<int>(), 0);
    }

    if (idxs1.size() != idxs2.size()) {
        MSG_ERROR("search_frame_for_triangulation: idxs1.size() != idxs2.size()");
        return std::tuple(std::vector<int>(), std::vector<int>(), 0);
    }

    const bool check_orientation = FeatureSharedResources::oriented_features;
    RotationHistogram rot_histo;

    const auto &level_sigmas2 = FeatureSharedResources::level_sigmas2;
    const auto &scale_factors = FeatureSharedResources::scale_factors;
    const float kMinDistanceFromEpipole2 =
        Parameters::kMinDistanceFromEpipole * Parameters::kMinDistanceFromEpipole;

    std::vector<int> final_idxs1;
    std::vector<int> final_idxs2;
    int num_found_matches = 0;
    final_idxs1.reserve(idxs1.size());
    final_idxs2.reserve(idxs2.size());

    for (size_t i = 0; i < idxs1.size(); ++i) {
        const int idx1 = idxs1[i];
        const int idx2 = idxs2[i];

        const MapPointPtr &p1 = kf1->get_point_match(idx1);
        const MapPointPtr &p2 = kf2->get_point_match(idx2);

        if (p1 || p2) {
            continue; // we triangulate only unmatched points
        }

        const auto des1 = kf1->des.row(idx1);
        const auto des2 = kf2->des.row(idx2);
        const auto d = descriptor_distance(des1, des2, FeatureSharedResources::norm_type);
        if (d > max_descriptor_distance) {
            continue;
        }

        const auto kp1 = kf1->kpsu.row(idx1).transpose();
        const auto kp2 = kf2->kpsu.row(idx2).transpose();
        const int octaves2 = kf2->octaves[idx2];

        const auto kp2_scale_factor = scale_factors[octaves2];
        const auto min_epipole_distance_sq = kMinDistanceFromEpipole2 * kp2_scale_factor;
        const auto epipole_distance_sq = (kp2 - e2).squaredNorm();
        if (epipole_distance_sq < min_epipole_distance_sq) {
            continue;
        }

        const auto sigma2_kps2 = level_sigmas2[octaves2];
        const auto epipolar_line = F12.transpose() * kp1.homogeneous().cast<double>();
        const double numerator =
            kp2(0) * epipolar_line[0] + kp2(1) * epipolar_line[1] + epipolar_line[2];
        const double denominator =
            (epipolar_line[0] * epipolar_line[0]) + (epipolar_line[1] * epipolar_line[1]);
        if (denominator < 1e-20)
            continue;
        const double dist_sq = (numerator * numerator) / denominator;
        const double chi2_threshold = 3.84 * sigma2_kps2;
        if (dist_sq > chi2_threshold) {
            continue;
        }

        final_idxs1.push_back(idx1);
        final_idxs2.push_back(idx2);

        if (check_orientation) {
            const int index_match = final_idxs1.size() - 1;
            rot_histo.push(kf1->angles[idx1] - kf2->angles[idx2], index_match);
        }
    }

    if (check_orientation) {
        const auto valid_match_idxs = rot_histo.get_valid_idxs();
        std::vector<int> final_idxs1_valid;
        std::vector<int> final_idxs2_valid;
        final_idxs1_valid.reserve(valid_match_idxs.size());
        final_idxs2_valid.reserve(valid_match_idxs.size());
        for (const int idx : valid_match_idxs) {
            final_idxs1_valid.push_back(final_idxs1[idx]);
            final_idxs2_valid.push_back(final_idxs2[idx]);
        }
        if (!valid_match_idxs.empty()) {
            final_idxs1 = std::move(final_idxs1_valid);
            final_idxs2 = std::move(final_idxs2_valid);
        }
    }

    num_found_matches = final_idxs1.size();
    return std::tuple(std::move(final_idxs1), std::move(final_idxs2), num_found_matches);
}

} // namespace pyslam
