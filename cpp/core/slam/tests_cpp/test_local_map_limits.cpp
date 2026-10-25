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
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include <iostream>

int main() {
    using namespace pyslam;
    auto camera = std::make_shared<PinholeCamera>();
    auto make_keyframe = [&]() {
        auto frame = std::make_shared<Frame>(camera);
        return std::make_shared<KeyFrame>(frame);
    };
    auto current = make_keyframe();
    std::vector<KeyFramePtr> neighbors;
    const int limit = Parameters::kMaxNumOfKeyframesInLocalMap;
    for (int i = 0; i < limit + 20; ++i) {
        auto neighbor = make_keyframe();
        neighbors.push_back(neighbor);
        current->ordered_keyframes_weights.emplace_back(neighbor, limit + 20 - i);
    }
    neighbors.front()->_is_bad = true;
    auto weak = neighbors.back();
    auto point = std::make_shared<MapPoint>();
    point->_observations[current] = 0;
    point->_observations[weak] = 0;
    current->points.push_back(point);
    LocalCovisibilityMap local_map(nullptr);
    const auto [local, points, fixed] = local_map.update(current);
    const bool ok = local.size() == static_cast<size_t>(limit) && local.count(current) &&
                    local.count(neighbors[1]) && !local.count(neighbors[0]) &&
                    !local.count(weak) && points.count(point) && fixed.count(weak);
    current->points.clear();
    point->_observations.clear();
    if (!ok) {
        std::cerr << "Local BA did not bound optimized poses and preserve fixed observers\n";
        return 1;
    }
    std::cout << "Local BA: strongest neighbors bounded; other observers remain fixed\n";
}
