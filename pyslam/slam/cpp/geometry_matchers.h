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

#pragma once

#include "config_parameters.h"
#include "sim3_pose.h"
#include "smart_pointers.h"

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tuple>

namespace pyslam {

class ProjectionMatcher {
  public:
    // propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need
    // to lock)
    static std::tuple<std::vector<int>, std::vector<int>, int> search_frame_by_projection(
        const FramePtr &f_ref, FramePtr &f_cur,
        float max_reproj_distance = Parameters::kMaxReprojectionDistanceFrame,
        float max_descriptor_distance = -1.0f, float ratio_test = Parameters::kMatchRatioTestMap,
        bool is_monocular = true, const std::vector<int> &already_matched_ref_idxs = {});

    // search by projection matches between {map points of f_ref} and {keypoints of f_cur},  (access
    // frames from tracking thread, no need to lock)
    static std::tuple<std::vector<int>, std::vector<int>, int>
    search_keyframe_by_projection(const KeyFramePtr &kf_ref, FramePtr &f_cur,
                                  float max_reproj_distance, float max_descriptor_distance = -1.0f,
                                  float ratio_test = Parameters::kMatchRatioTestMap,
                                  const std::vector<int> &already_matched_ref_idxs = {});

    // search by projection matches between {input map points} and {unmatched keypoints of frame
    // f_cur}, (access frame from tracking thread, no need to lock)
    static std::pair<int, std::vector<int>>
    search_map_by_projection(const std::vector<MapPointPtr> &points, FramePtr &f_cur,
                             float max_reproj_distance = Parameters::kMaxReprojectionDistanceMap,
                             float max_descriptor_distance = -1.0f,
                             float ratio_test = Parameters::kMatchRatioTestMap,
                             float far_points_threshold = std::numeric_limits<float>::infinity());

    // search by projection matches between {map points of last frames} and {unmatched keypoints of
    // f_cur}, (access frame from tracking thread, no need to lock)
    static std::pair<int, std::vector<int>>
    search_local_frames_by_projection(Map *map, FramePtr &f_cur,
                                      int local_window_size = Parameters::kLocalBAWindowSize,
                                      float max_descriptor_distance = -1.0f);

    // search by projection matches between {all map points} and {unmatched keypoints of f_cur}
    static std::pair<int, std::vector<int>>
    search_all_map_by_projection(Map *map, FramePtr &f_cur, float max_descriptor_distance = -1.0f);

    // search by projection more matches between {input map points} and {unmatched keypoints of
    // frame f_cur}
    //  in:
    //    points: input map points
    //    f_cur: current frame
    //    Scw: suggested se3 or sim3 transformation
    //    f_cur_matched_points: matched points in current frame  (f_cur_matched_points[i] is the
    //    i-th map point matched on f_cur or None)
    // NOTE1: f_cur_matched_points is modified in place and passed by reference in output
    // NOTE2: The suggested transformation Scw (in se3 or sim3) is used in the search (instead of
    // using
    //  the current frame pose)
    static std::pair<int, std::vector<MapPointPtr> &> search_more_map_points_by_projection(
        const std::vector<MapPointPtr> &points, FramePtr &f_cur, const Sim3Pose &Scw,
        std::vector<MapPointPtr> &f_cur_matched_points,
        const std::vector<int> &f_cur_matched_points_idxs = {},
        float max_reproj_distance = Parameters::kMaxReprojectionDistanceMap,
        float max_descriptor_distance = -1.0f);

    // For each input map point visible in keyframe, find closest keyframe keypoint by descriptor.
    // If that keyframe keypoint already has a different point, prefer replacement by observation
    // count; otherwise add observation to keyframe.
    static int search_and_fuse(const std::vector<MapPointPtr> &points, KeyFramePtr &keyframe,
                               float max_reproj_distance = Parameters::kMaxReprojectionDistanceFuse,
                               float max_descriptor_distance = -1.0f,
                               float ratio_test = Parameters::kMatchRatioTestMap);

    // search by projection matches between {input map points} and {keyframe points} and fuse them
    // if they are close enough use suggested Scw to project
    static std::vector<MapPointPtr> &search_and_fuse_for_loop_correction(
        const KeyFramePtr &keyframe, const Sim3Pose &Scw, const std::vector<MapPointPtr> &points,
        std::vector<MapPointPtr> &replace_points,
        float max_reproj_distance = Parameters::kLoopClosingMaxReprojectionDistanceFuse,
        float max_descriptor_distance = -1.0f);

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
    static std::tuple<int, std::vector<int>, std::vector<int>>
    search_by_sim3(const KeyFramePtr &kf1, const KeyFramePtr &kf2, const std::vector<int> &idxs1,
                   const std::vector<int> &idxs2, float s12, const Eigen::Matrix3d &R12,
                   const Eigen::Vector3d &t12,
                   float max_reproj_distance = Parameters::kMaxReprojectionDistanceSim3,
                   float max_descriptor_distance = -1.0f);
};

class EpipolarMatcher {

  public:
    // search keypoint matches (for triangulations) between f1 and f2
    // search for matches between unmatched keypoints (without a corresponding map point)
    // in input we have already some pose estimates for f1 and f2
    static std::tuple<std::vector<int>, std::vector<int>, int>
    search_frame_for_triangulation(const KeyFramePtr &kf1, const KeyFramePtr &kf2,
                                   const std::vector<int> &idxs1 = {},
                                   const std::vector<int> &idxs2 = {},
                                   float max_descriptor_distance = -1.0f, bool is_monocular = true);
};

} // namespace pyslam