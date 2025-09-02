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

#include "map.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace pyslam {

// Serialization
std::string Map::to_json(const std::string &out_json) const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"num_keyframes\": " << num_keyframes() << ", ";
    oss << "\"num_points\": " << num_points() << ", ";
    oss << "\"max_point_id\": " << max_point_id << ", ";
    oss << "\"max_frame_id\": " << max_frame_id << ", ";
    oss << "\"max_keyframe_id\": " << max_keyframe_id;
    oss << "}";
    return oss.str();
}

std::string Map::serialize() const { return to_json(); }

void Map::from_json(const std::string &loaded_json) {
    // TODO: implement this
    // This is a simplified implementation
    // The actual implementation would parse JSON and restore map state
}

void Map::deserialize(const std::string &s) { from_json(s); }

void Map::save(const std::string &filename) const {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << to_json();
        file.close();
    }
}

void Map::load(const std::string &filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        from_json(content);
        file.close();
    }
}

} // namespace pyslam