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

#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "utils/serialization_json.h"

#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

namespace pyslam {

// Serialization
std::string Map::to_json(const std::string &out_json) const {
    std::lock_guard<MapMutex> lock(_lock);
    std::lock_guard<MapMutex> update_lock(_update_lock);

    nlohmann::json json_obj;

    try {
        // Static stuff - these are global counters in Python
        // In C++, we access them via static methods to match Python behavior
        json_obj["FrameBase._id"] = FrameBase::next_id();
        json_obj["MapPointBase._id"] = MapPointBase::next_id();

        // Non-static stuff
        // Serialize frames
        nlohmann::json frames_array = nlohmann::json::array();
        for (const auto &frame : frames) {
            if (frame) {
                frames_array.push_back(nlohmann::json::parse(frame->to_json()));
            }
        }
        json_obj["frames"] = frames_array;

        // Serialize keyframes (only good ones)
        nlohmann::json keyframes_array = nlohmann::json::array();
        for (const auto &keyframe : keyframes) {
            if (keyframe && !keyframe->is_bad()) {
                keyframes_array.push_back(nlohmann::json::parse(keyframe->to_json()));
            }
        }
        json_obj["keyframes"] = keyframes_array;

        // Serialize points (only good ones)
        nlohmann::json points_array = nlohmann::json::array();
        for (const auto &point : points) {
            if (point && !point->is_bad()) {
                points_array.push_back(nlohmann::json::parse(point->to_json()));
            }
        }
        json_obj["points"] = points_array;

        // Serialize keyframe origins
        nlohmann::json keyframe_origins_array = nlohmann::json::array();
        for (const auto &keyframe : keyframe_origins) {
            if (keyframe) {
                keyframe_origins_array.push_back(nlohmann::json::parse(keyframe->to_json()));
            }
        }
        json_obj["keyframe_origins"] = keyframe_origins_array;

        // Map metadata
        json_obj["max_frame_id"] = max_frame_id;
        json_obj["max_point_id"] = max_point_id;
        json_obj["max_keyframe_id"] = max_keyframe_id;

        // Viewer scale
        json_obj["viewer_scale"] = viewer_scale;

    } catch (const std::exception &e) {
        throw std::runtime_error("Error in Map::to_json(): " + std::string(e.what()));
    }

    return json_obj.dump();
}

std::string Map::serialize() const { return to_json(); }

void Map::from_json(const std::string &loaded_json) {
    nlohmann::json json_obj;

    try {
        json_obj = nlohmann::json::parse(loaded_json);
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to parse JSON: " + std::string(e.what()));
    }

    std::lock_guard<MapMutex> lock(_lock);
    std::lock_guard<MapMutex> update_lock(_update_lock);

    try {
        // Static stuff - restore global counters (match Python behavior)
        int frame_base_id = safe_json_get(json_obj, "FrameBase._id", 0);
        int mappoint_base_id = safe_json_get(json_obj, "MapPointBase._id", 0);
        FrameBase::set_id(frame_base_id);
        MapPointBase::set_id(mappoint_base_id);
        // Also update instance variables for consistency
        max_frame_id = frame_base_id;
        max_point_id = mappoint_base_id;

        // Clear existing data
        frames.clear();
        keyframes.clear();
        points.clear();
        keyframe_origins.clear();
        keyframes_map.clear();

        // Load frames
        if (json_obj.contains("frames") && !json_obj["frames"].is_null()) {
            for (const auto &frame_json : json_obj["frames"]) {
                try {
                    // Determine if it's a keyframe or regular frame
                    bool is_keyframe = safe_json_get(frame_json, "is_keyframe", false);
                    if (is_keyframe) {
                        auto keyframe = KeyFrame::from_json(frame_json.dump());
                        if (keyframe) {
                            frames.push_back(keyframe);
                        }
                    } else {
                        auto frame = Frame::from_json(frame_json.dump());
                        if (frame) {
                            frames.push_back(frame);
                        }
                    }
                } catch (const std::exception &e) {
                    // Skip invalid frames
                    continue;
                }
            }
        }

        // Load keyframes
        if (json_obj.contains("keyframes") && !json_obj["keyframes"].is_null()) {
            for (const auto &keyframe_json : json_obj["keyframes"]) {
                try {
                    auto keyframe = KeyFrame::from_json(keyframe_json.dump());
                    if (keyframe) {
                        keyframes.insert(keyframe);
                        keyframes_map[keyframe->id] = keyframe;
                    }
                } catch (const std::exception &e) {
                    // Skip invalid keyframes
                    continue;
                }
            }
        }

        // Load points
        if (json_obj.contains("points") && !json_obj["points"].is_null()) {
            for (const auto &point_json : json_obj["points"]) {
                try {
                    auto point = MapPoint::from_json(point_json.dump());
                    if (point) {
                        points.insert(point);
                    }
                } catch (const std::exception &e) {
                    // Skip invalid points
                    continue;
                }
            }
        }

        // Load keyframe origins
        if (json_obj.contains("keyframe_origins") && !json_obj["keyframe_origins"].is_null()) {
            for (const auto &origin_json : json_obj["keyframe_origins"]) {
                try {
                    int origin_id = safe_json_get(origin_json, "id", -1);
                    if (origin_id >= 0) {
                        auto it = keyframes_map.find(origin_id);
                        if (it != keyframes_map.end()) {
                            keyframe_origins.insert(it->second);
                        }
                    }
                } catch (const std::exception &e) {
                    // Skip invalid origins
                    continue;
                }
            }
        }

        // Load map metadata
        max_frame_id = safe_json_get(json_obj, "max_frame_id", max_frame_id);
        max_point_id = safe_json_get(json_obj, "max_point_id", max_point_id);
        max_keyframe_id = safe_json_get(json_obj, "max_keyframe_id", max_keyframe_id);

        // Load viewer scale
        viewer_scale = safe_json_get(json_obj, "viewer_scale", -1.0f);

        // Now replace IDs with actual objects in all map assets
        // This is crucial for restoring object relationships

        // Replace IDs with objects in frames
        for (auto &frame : frames) {
            if (frame) {
                frame->replace_ids_with_objects(
                    std::vector<MapPointPtr>(points.begin(), points.end()),
                    std::vector<FramePtr>(frames.begin(), frames.end()),
                    std::vector<KeyFramePtr>(keyframes.begin(), keyframes.end()));
            }
        }

        // Replace IDs with objects in keyframes
        for (auto &keyframe : keyframes) {
            if (keyframe) {
                keyframe->replace_ids_with_objects(
                    std::vector<MapPointPtr>(points.begin(), points.end()),
                    std::vector<FramePtr>(frames.begin(), frames.end()),
                    std::vector<KeyFramePtr>(keyframes.begin(), keyframes.end()));
                keyframe->map = this; // Set the map reference
            }
        }

        // Replace IDs with objects in points
        for (auto &point : points) {
            if (point) {
                point->replace_ids_with_objects(
                    std::vector<MapPointPtr>(points.begin(), points.end()),
                    std::vector<FramePtr>(frames.begin(), frames.end()),
                    std::vector<KeyFramePtr>(keyframes.begin(), keyframes.end()));
                point->map = this; // Set the map reference
            }
        }

        // Set up reloaded session map info
        reloaded_session_map_info = std::make_unique<ReloadedSessionMapInfo>(
            static_cast<int>(keyframes.size()), static_cast<int>(points.size()), max_point_id,
            max_frame_id, max_keyframe_id);

        std::cout << "Map: from_json - FrameBase._id: " << FrameBase::next_id() << std::endl;
        std::cout << "Map: from_json - MapPointBase._id: " << MapPointBase::next_id() << std::endl;

        // Sync static ID counters with max IDs so new objects continue from the correct ID
        FrameBase::set_id(max_frame_id);
        MapPointBase::set_id(max_point_id);

    } catch (const std::exception &e) {
        throw std::runtime_error("Error in Map::from_json(): " + std::string(e.what()));
    }
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