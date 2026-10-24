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
#include "local_mapping_core.h"
#include "config_parameters.h"
#include "eigen_aliases.h"
#include "geometry_matchers.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"

#include <execution>
#include <iostream>
#include <tbb/global_control.h>
#include <tbb/parallel_for_each.h>

namespace pyslam {

// ==========================================
// Constructor
// ==========================================

LocalMappingCore::LocalMappingCore(Map *map, const SensorType sensor_type)
    : map_(map), sensor_type_(sensor_type) {}

// ==========================================
// State Management
// ==========================================
void LocalMappingCore::reset() { recently_added_points_.clear(); }

void LocalMappingCore::set_kf_cur(KeyFramePtr kf) { kf_cur_ = std::move(kf); }

KeyFramePtr &LocalMappingCore::get_kf_cur() { return kf_cur_; }

void LocalMappingCore::set_kid_last_BA(int kid) { kid_last_BA_ = kid; }

int &LocalMappingCore::get_kid_last_BA() { return kid_last_BA_; }

void LocalMappingCore::set_opt_abort_flag(bool value) { opt_abort_flag_ = value; }

bool &LocalMappingCore::get_opt_abort_flag() { return opt_abort_flag_; }

void LocalMappingCore::add_points(const std::vector<MapPointPtr> &points) {
    recently_added_points_.insert(points.begin(), points.end());
}

void LocalMappingCore::remove_points(const std::vector<MapPointPtr> &points) {
    for (const auto &p : points) {
        recently_added_points_.erase(p);
    }
}

void LocalMappingCore::clear_recent_points() { recently_added_points_.clear(); }

size_t LocalMappingCore::num_recent_points() const { return recently_added_points_.size(); }

std::vector<MapPointPtr> LocalMappingCore::get_recently_added_points() const {
    return std::vector<MapPointPtr>(recently_added_points_.begin(), recently_added_points_.end());
}

// ==========================================
// Computation Methods
// ==========================================

void LocalMappingCore::process_new_keyframe() {
    // associate map points to keyframe observations (only good points)
    // and update normal and descriptor
    // Use get_matched_good_points_and_idxs() to match Python implementation exactly
    auto good_points_and_idxs = kf_cur_->get_matched_good_points_and_idxs();
    for (const auto &[p, idx] : good_points_and_idxs) {
        // Try to add observation
        const bool added = p->add_observation(kf_cur_, idx);
        if (added) {
            p->update_info();
        } else {
            // this happens for new stereo points inserted by Tracking
            recently_added_points_.insert(p);
        }
    }
    // Update connections
    kf_cur_->update_connections();
}

int LocalMappingCore::cull_map_points() {
    const int th_num_observations = (sensor_type_ == SensorType::MONOCULAR) ? 2 : 3;
    const float min_found_ratio = 0.25f;

    const int current_kid = kf_cur_->kid;
    int num_culled_points = 0;
    auto it = recently_added_points_.begin();
    while (it != recently_added_points_.end()) {
        auto p = *it;
        if (p->is_bad()) {
            num_culled_points++;
            it = recently_added_points_.erase(it);
        } else if (p->get_found_ratio() < min_found_ratio) {
            p->set_bad();
            map_->remove_point(p);
            num_culled_points++;
            it = recently_added_points_.erase(it);
        } else if ((current_kid - p->first_kid) >= 2 &&
                   p->num_observations() <= th_num_observations) {
            p->set_bad();
            map_->remove_point(p);
            num_culled_points++;
            it = recently_added_points_.erase(it);
        } else if ((current_kid - p->first_kid) >= 3) {
            // after three keyframes we do not consider the point a recent one
            num_culled_points++;
            it = recently_added_points_.erase(it);
        } else {
            it++;
        }
    }

    return num_culled_points;
}

// Check if once we remove "kf_to_remove" from covisible_kfs we still have that the max distance
// among fov centers is less than distance "dist"
bool LocalMappingCore::check_remaining_fov_centers_max_distance(
    const std::vector<KeyFramePtr> &covisible_kfs, const KeyFramePtr &kf_to_remove, float dist) {
    // fov centers that remain if we remove kf_to_remove
    std::vector<Eigen::Vector3d> remaining_fov_centers;
    remaining_fov_centers.reserve(covisible_kfs.size());

    for (const auto &kf : covisible_kfs) {
        if (kf != kf_to_remove) {
            remaining_fov_centers.push_back(kf->fov_center_w);
        }
    }

    if (remaining_fov_centers.empty()) {
        return false;
    }

    // Convert to Eigen matrix for KD-tree
    MatNx3d data(remaining_fov_centers.size(), 3);
    for (size_t i = 0; i < remaining_fov_centers.size(); ++i) {
        data.row(i) = remaining_fov_centers[i];
    }

    // Build KD-tree
    CKDTreeEigen<double, 3> tree(data, 10);

    // Check the distance to the nearest neighbor for each remaining point
    for (size_t i = 0; i < remaining_fov_centers.size(); ++i) {
        const Eigen::Vector3d &point = remaining_fov_centers[i];
        // k=2 because the closest point is itself
        auto [distances, indices] = tree.query(point.data(), 2);

        // distances[1] is the distance to the nearest neighbor (not itself)
        if (distances.size() < 2 || distances[1] >= dist) {
            return false;
        }
    }
    return true;
}

int LocalMappingCore::cull_keyframes(bool use_fov_centers_based_kf_generation,
                                     float max_fov_centers_distance) {
    // check redundant keyframes in local keyframes: a keyframe is considered redundant if the 90%
    // of the MapPoints it sees, are seen in at least other 3 keyframes (in the same or finer scale)
    int num_culled_keyframes = 0;
    const int th_num_observations = 3;

    auto covisible_kfs = kf_cur_->get_covisible_keyframes();

    for (const auto &kf : covisible_kfs) {
        if (kf->kid == 0) {
            continue;
        }

        int kf_num_points = 0;                 // num good points for kf
        int kf_num_redundant_observations = 0; // num redundant observations for kf

        auto points = kf->get_points();

        for (size_t i = 0; i < points.size(); ++i) {
            auto p = points[i];
            if (!p || p->is_bad()) {
                continue;
            }

            // Check depth if available
            if (sensor_type_ != SensorType::MONOCULAR && kf->depths.size() > i) {
                const float &depth = kf->depths[i];
                if (depth > kf->camera->depth_threshold || depth < 0.0f) {
                    continue;
                }
            }

            kf_num_points++;

            if (p->num_observations() > th_num_observations) {
                const int scale_level = kf->octaves[i]; // scale level of observation in kf
                int p_num_observations = 0;

                auto observations = p->observations();
                for (const auto &[kf_j, idx_j] : observations) {
                    if (kf_j == kf) {
                        continue;
                    }

                    if (kf_j->is_bad()) {
                        continue;
                    }

                    const int scale_level_i =
                        kf_j->octaves[idx_j]; // scale level of observation in kfi
                    if (scale_level_i <= scale_level + 1) {
                        // N.B.1 <- more aggressive culling (expecially when scale_factor=2)
                        // if scale_level_i <= scale_level:     // N.B.2 <- only same scale or finer
                        p_num_observations++;
                        if (p_num_observations >= th_num_observations) {
                            break;
                        }
                    }
                }

                if (p_num_observations >= th_num_observations) {
                    kf_num_redundant_observations++;
                }
            }
        }

        bool remove_kf = ((kf_num_redundant_observations >
                           Parameters::kKeyframeCullingRedundantObsRatio * kf_num_points) &&
                          (kf_num_points > Parameters::kKeyframeCullingMinNumPoints));
        if (remove_kf) {
            // check if the keyframe is too close in time to its parent
            const double delta_time_parent = std::abs(kf->timestamp - kf->parent->timestamp);
            if (delta_time_parent < Parameters::kKeyframeMaxTimeDistanceInSecForCulling) {
                remove_kf = false;
            }
            // check if the keyframe is too far from the FOV centers of the covisible keyframes
            if (use_fov_centers_based_kf_generation) {
                if (!check_remaining_fov_centers_max_distance(covisible_kfs, kf,
                                                              max_fov_centers_distance)) {
                    remove_kf = false;
                }
            }
        }

        if (remove_kf) {
            kf->set_bad();
            num_culled_keyframes++;
            // TODO: move this to a logger to local mapping file
            std::cout << "culling keyframe " << kf->id << " (set it bad) - redundant observations: "
                      << static_cast<float>(kf_num_redundant_observations) /
                             std::max(kf_num_points, 1)
                      << "%" << std::endl;
        }
    }

    return num_culled_keyframes;
}

std::pair<float, int> LocalMappingCore::local_BA() {
    // local optimization
    float err = static_cast<float>(
        map_->locally_optimize(kf_cur_, false, 10, &opt_abort_flag_ // verbose, rounds, abort_flag
                               ));

    int num_kf_ref_tracked_points =
        kf_cur_->num_tracked_points(Parameters::kNumMinObsForKeyFrameTrackedPoints);

    return {err, num_kf_ref_tracked_points};
}

float LocalMappingCore::large_window_BA() {
    // large window optimization of the map
    kid_last_BA_ = kf_cur_->kid;
    float err =
        static_cast<float>(map_->optimize(Parameters::kLargeBAWindowSize, // local_window_size
                                          false,                          // verbose
                                          10,                             // rounds
                                          false,                          // use_robust_kernel
                                          false,                          // do_cull_points
                                          &opt_abort_flag_                // abort_flag
                                          ));

    return err;
}

int LocalMappingCore::fuse_map_points(float descriptor_distance_sigma) {
    int total_fused_pts = 0;

    int num_neighbors = Parameters::kLocalMappingNumNeighborKeyFramesStereo;
    if (sensor_type_ == SensorType::MONOCULAR) {
        num_neighbors = Parameters::kLocalMappingNumNeighborKeyFramesMonocular;
    }

    // 1. Get direct neighbors
    auto local_keyframes = map_->local_map->get_best_neighbors(kf_cur_, num_neighbors);
    std::set<KeyFramePtr> target_kfs;
    for (const auto &kf : local_keyframes) {
        if (kf == kf_cur_ || kf->is_bad()) {
            continue;
        }
        target_kfs.insert(kf);
        // 2. Add second neighbors
        const auto second_neighbors = map_->local_map->get_best_neighbors(kf, 5);
        for (const auto &kf2 : second_neighbors) {
            if (kf2 == kf_cur_ || kf2->is_bad()) {
                continue;
            }
            target_kfs.insert(kf2);
        }
    }

    // 3. Fuse current keyframe's points into all target keyframes
    for (KeyFramePtr kf : target_kfs) {
        const auto kf_cur_points = kf_cur_->get_points();
        int num_fused_pts = ProjectionMatcher::search_and_fuse(
            kf_cur_points, kf, Parameters::kMaxReprojectionDistanceFuse,
            0.5f * descriptor_distance_sigma);
        total_fused_pts += num_fused_pts;
    }

    // 4. Fuse all target keyframes' points into current keyframe
    std::set<MapPointPtr> fuse_candidates;
    for (const auto &kf : target_kfs) {
        auto points = kf->get_points();
        for (const auto &p : points) {
            if (p && !p->is_bad()) {
                fuse_candidates.insert(p);
            }
        }
    }

    std::vector<MapPointPtr> candidates_list(fuse_candidates.begin(), fuse_candidates.end());
    int num_fused_pts = ProjectionMatcher::search_and_fuse(candidates_list, kf_cur_,
                                                           Parameters::kMaxReprojectionDistanceFuse,
                                                           0.5f * descriptor_distance_sigma);
    total_fused_pts += num_fused_pts;

    // 5. Update all map points in current keyframe
    auto points_to_update = kf_cur_->get_matched_good_points();
#if 0
    for (const auto &p : points_to_update) {
        if (p) {
            p->update_info();
        }
    }
#else
    // parallel update of the map points
    tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
                              Parameters::kLocalMappingParallelFusePointsNumWorkers);

    tbb::parallel_for_each(points_to_update.begin(), points_to_update.end(),
                           [](const MapPointPtr &p) {
                               if (p) {
                                   p->update_info();
                               }
                           });
#endif

    // 6. Update connections in covisibility graph
    kf_cur_->update_connections();

    return total_fused_pts;
}

} // namespace pyslam