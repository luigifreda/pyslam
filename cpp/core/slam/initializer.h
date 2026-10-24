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

#include "frame.h"
#include "keyframe.h"

#include <deque>
#include <optional>
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

namespace pyslam {

struct InitializerOutput {
    std::vector<Eigen::Vector3d> pts;
    KeyFramePtr kf_cur;
    KeyFramePtr kf_ref;
    std::vector<int> idxs_cur;
    std::vector<int> idxs_ref;
};

class Initializer {
  public:
    explicit Initializer(SensorType sensor_type = SensorType::MONOCULAR);

    void reset();
    void init(const FramePtr &f_cur, const cv::Mat &img_cur, const cv::Mat &img_right_cur = cv::Mat(),
              const cv::Mat &depth_img_cur = cv::Mat());
    std::pair<InitializerOutput, bool> initialize(const FramePtr &f_cur, const cv::Mat &img_cur,
                                                  const cv::Mat &img_right_cur = cv::Mat(),
                                                  const cv::Mat &depth_img_cur = cv::Mat());

    SensorType sensor_type;
    bool is_monocular = true;
    cv::Mat mask_match;
    std::deque<FramePtr> frames;
    FramePtr f_ref;
    int num_min_features = 100;
    double num_min_triangulated_points = 150;
    int num_failures = 0;
    int num_inliers = 0;
    bool check_min_frame_distance = true;
    bool check_cell_coverage = true;

  private:
    static constexpr int kMinIdDistBetweenIntializingFrames = 2;
    static constexpr int kMaxIdDistBetweenIntializingFrames = 5;
    static constexpr int kMaxLenFrameDeque = kMaxIdDistBetweenIntializingFrames + 1;
    static constexpr int kNumOfFailuresAfterWichNumMinTriangulatedPointsIsReduced = 10;

    void push_frame(const FramePtr &frame);
    void check_num_failures();
    std::optional<Eigen::Matrix4d> estimate_pose(const MatNx2f &kpn_ref, const MatNx2f &kpn_cur);
    std::pair<InitializerOutput, bool> initialize_simple(const FramePtr &f_cur, const cv::Mat &img_cur);
    std::pair<InitializerOutput, bool> process_frames(const FramePtr &f_cur, const cv::Mat &img_cur,
                                                      const FramePtr &f_ref_in);
};

} // namespace pyslam
