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

#include "camera/camera.h"
#include "slam/direct/core/direct_params.h"

#include <opencv2/core.hpp>

#include <vector>

namespace pyslam {

// Pyramidal Lucas-Kanade (KLT) point tracking. It produces 2D correspondences and does not
// optimize a photometric pose error. Used by temporal tracking in klt/hybrid, by left/right
// stereo matching in every mode, by semi_direct seed tracking and by the monocular initializer.

// Output arrays keep input-point indexing, including failed tracks (status == 0).
struct LkTrack {
    std::vector<Vec2d> points_cur;
    std::vector<unsigned char> status;
};

// Seed forward LK with predicted pixels, then reject inconsistent backward tracks.
LkTrack track_pyramidal_lk(const cv::Mat &img_ref, const cv::Mat &img_cur,
                           const std::vector<Vec2d> &points_ref,
                           const std::vector<Vec2d> &points_cur_init, const DirectSlamParams &params);

} // namespace pyslam
