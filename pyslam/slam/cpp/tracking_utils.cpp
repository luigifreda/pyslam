#include "tracking_utils.h"

// Utility functions implementation
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

#include "Eigen/Dense"
#include "config_parameters.h"
#include "frame.h"
#include "map_point.h"
#include "smart_pointers.h"
#include "utils/cv_ops.h"
#include "utils/image_processing.h"
#include "utils/messages.h"

namespace pyslam {

// propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need to
// lock)
std::tuple<int, std::vector<int>, std::vector<int>> TrackingUtils::propagate_map_point_matches(
    const FramePtr &f_ref, FramePtr &f_cur, const std::vector<int> &idxs_ref,
    const std::vector<int> &idxs_cur, float max_descriptor_distance /* = -1.0f*/) {

    if (max_descriptor_distance < 0) {
        max_descriptor_distance = Parameters::kMaxDescriptorDistance; // take the updated value
    }

    int num_matched_map_pts = 0;
    std::vector<int> idx_ref_out, idx_cur_out;
    idx_ref_out.reserve(idxs_ref.size());
    idx_cur_out.reserve(idxs_cur.size());

    // populate f_cur with map points by propagating map point matches of f_ref;
    // to this aim, we use map points observed in f_ref and keypoint matches between f_ref and f_cur
    const auto &pts_ref = f_ref->points;
    const auto &outliers_ref = f_ref->outliers;
    const auto &pts_cur = f_cur->points;
    const auto &des_cur = f_cur->des;

    const int idxs_ref_size = static_cast<int>(idxs_ref.size());
    const int idxs_cur_size = static_cast<int>(idxs_cur.size());
    const int pts_ref_size = static_cast<int>(f_ref->points.size());
    const int pts_cur_size = static_cast<int>(f_cur->points.size());

    for (int i = 0; i < idxs_ref_size; ++i) {
        const int idx_ref = idxs_ref[i];
        if (idx_ref < 0 || idx_ref >= pts_ref_size)
            continue;
        const MapPointPtr &p_ref = pts_ref[idx_ref];
        if (!p_ref || outliers_ref[idx_ref] || p_ref->is_bad())
            continue;

        const int idx_cur = idxs_cur[i];
        if (idx_cur < 0 || idx_cur >= pts_cur_size)
            continue;
        const MapPointPtr &p_cur = pts_cur[idx_cur];
        if (p_cur)
            continue; // already has a match

        // distance to descriptor
        float des_dist = p_ref->min_des_distance(des_cur.row(idx_cur));
        if (des_dist > max_descriptor_distance)
            continue;

        if (p_ref->add_frame_view(f_cur, idx_cur)) {
            // => P is matched to the i-th matched keypoint in f_cur
            ++num_matched_map_pts;
            idx_ref_out.push_back(idx_ref);
            idx_cur_out.push_back(idx_cur);
        }
    }
    return std::make_tuple(num_matched_map_pts, std::move(idx_ref_out), std::move(idx_cur_out));
}

std::vector<MapPointPtr> TrackingUtils::create_vo_points(FramePtr &frame, int max_num_points,
                                                         const Vec3b &color) {
    std::lock_guard<std::mutex> lock(frame->_lock_features);

    const auto &depths = frame->depths;
    const auto &camera = frame->camera;
    auto &points = frame->points;

    // Check depths are available
    if (depths.empty()) {
        MSG_WARN("TrackingUtils::create_vo_points() - depths are empty");
        return {};
    }

    // Use camera depth threshold if not provided
    const float depth_threshold = camera->depth_threshold;

    // Collect all points with valid depths
    std::vector<std::pair<int, float>> valid_idxs_depths;
    valid_idxs_depths.reserve(depths.size());
    for (int i = 0; i < static_cast<int>(depths.size()); ++i) {
        if (depths[i] > Parameters::kMinDepth) {
            valid_idxs_depths.push_back(std::make_pair(i, depths[i]));
        }
    }
    if (valid_idxs_depths.empty()) {
        return {};
    }

    // Sort by depth (increasing order)
    std::sort(valid_idxs_depths.begin(), valid_idxs_depths.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                  return a.second < b.second;
              });

    int num_vo_points = 0;
    std::vector<MapPointPtr> vo_points;
    vo_points.reserve(max_num_points);

    // Insert all points that are within the depth threshold or at least the closest max_num_points
    // points
    for (const auto &[idx, depth] : valid_idxs_depths) {
        const auto &p = points[idx];
        // If the point is not already observed, create a new map point
        if (p && p->num_observations() < 1) {
            auto [pt3d, is_valid] =
                frame->unproject_point_3d<double>(idx, /*transform_in_world*/ true);
            if (is_valid) {
                auto mp = std::make_shared<MapPoint>(pt3d, color, frame, idx);
                vo_points.push_back(mp);
                points[idx] = mp;
                num_vo_points++;
            }
        }

        // Stop when we have selected at least a maximum number of points and the depth is greater
        // than the threshold
        if (num_vo_points >= max_num_points && depth > depth_threshold) {
            break;
        }
    }
    return vo_points;
}

int TrackingUtils::create_and_add_stereo_map_points_on_new_kf(FramePtr &frame, KeyFramePtr &kf,
                                                              MapPtr &map, const cv::Mat &img) {

    const auto &depths = kf->depths;
    const auto &camera = kf->camera;
    auto &frame_points = frame->points;
    auto &kf_points = kf->points;

    // Check depths are available
    if (depths.empty()) {
        MSG_WARN("TrackingUtils::create_and_add_stereo_map_points_on_new_kf() - depths are empty");
        return {};
    }

    // Use camera depth threshold if not provided
    const float depth_threshold = camera->depth_threshold;

    // Collect all points with valid depths
    std::vector<std::pair<int, float>> valid_idxs_depths;
    valid_idxs_depths.reserve(depths.size());
    for (int i = 0; i < static_cast<int>(depths.size()); ++i) {
        if (depths[i] > Parameters::kMinDepth) {
            valid_idxs_depths.push_back(std::make_pair(i, depths[i]));
        }
    }
    if (valid_idxs_depths.empty()) {
        return 0;
    }

    // Sort by depth (increasing order)
    std::sort(valid_idxs_depths.begin(), valid_idxs_depths.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b) {
                  return a.second < b.second;
              });

    const int max_num_stereo_points = Parameters::kMaxNumStereoPointsOnNewKeyframe;
    int num_stereo_points = 0;

    utils::ImageColorExtractor img_color_extractor(img);
    const int num_channels = img.channels();
    if (num_channels != 1 && num_channels != 3) {
        MSG_ERROR("TrackingUtils::create_and_add_stereo_map_points_on_new_kf() - image channels "
                  "are not 1 or 3");
        return 0;
    }
    const bool is_color = num_channels == 3;

    // Insert all points that are within the depth threshold or at least the closest max_num_points
    // points
    for (const auto &[idx, depth] : valid_idxs_depths) {
        const auto &p = kf_points[idx];
        // If the point is not already observed, create a new map point
        if (p && p->num_observations() < 1) {
            auto [pt3d, is_valid] =
                frame->unproject_point_3d<double>(idx, /*transform_in_world*/ true);
            if (is_valid) {
                const auto kp = frame->kps.row(idx);
                const auto color_cv = img_color_extractor.extract_mean_color(kp, is_color);
                // Convert cv::Vec<float,3> to Eigen::Matrix<unsigned char,3,1>
                Vec3b color_eigen(
                    static_cast<unsigned char>(std::clamp(color_cv[0], 0.0f, 255.0f)),
                    static_cast<unsigned char>(std::clamp(color_cv[1], 0.0f, 255.0f)),
                    static_cast<unsigned char>(std::clamp(color_cv[2], 0.0f, 255.0f)));
                auto mp = std::make_shared<MapPoint>(pt3d, color_eigen, kf, idx);

                frame_points[idx] = mp; // add point to the frame
                map->add_point(mp);     // add point to the map
                mp->add_observation(kf, idx);
                mp->update_info();
                num_stereo_points++;
            }
        }

        // Stop when we have selected at least a maximum number of points and the depth is greater
        // than the threshold
        if (num_stereo_points >= max_num_stereo_points && depth > depth_threshold) {
            break;
        }
    }
    return num_stereo_points;
}

//===============================================
// WIP
//===============================================

// Essential matrix pose estimation
std::pair<Eigen::Matrix4d, cv::Mat> TrackingUtils::estimate_pose_ess_mat(const MatNx2f &kps_ref,
                                                                         const MatNx2f &kps_cur,
                                                                         int method, double prob,
                                                                         double threshold) {

    // Convert to OpenCV format
    std::vector<cv::Point2f> points_ref, points_cur;
    for (int i = 0; i < kps_ref.rows(); ++i) {
        points_ref.emplace_back(kps_ref(i, 0), kps_ref(i, 1));
        points_cur.emplace_back(kps_cur(i, 0), kps_cur(i, 1));
    }

    cv::Mat E, mask;
    E = cv::findEssentialMat(points_ref, points_cur, 1.0, cv::Point2d(0, 0), method, prob,
                             threshold, mask);

    // Recover pose from essential matrix
    cv::Mat R, t;
    cv::recoverPose(E, points_ref, points_cur, R, t, 1.0, cv::Point2d(0, 0), mask);

    // Convert to Eigen format
    Eigen::Matrix4d Mrc = Eigen::Matrix4d::Identity();
    Mrc.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3d>(R.ptr<double>()).transpose();
    Mrc.block<3, 1>(0, 3) = Eigen::Map<Eigen::Vector3d>(t.ptr<double>());

    return {Mrc, mask};
}

// Homography estimation with RANSAC
std::pair<cv::Mat, cv::Mat> TrackingUtils::find_homography_with_ransac(const MatNx2f &kps_cur,
                                                                       const MatNx2f &kps_ref,
                                                                       int method,
                                                                       double reproj_threshold) {

    // Convert to OpenCV format
    std::vector<cv::Point2f> points_cur, points_ref;
    for (int i = 0; i < kps_cur.rows(); ++i) {
        points_cur.emplace_back(kps_cur(i, 0), kps_cur(i, 1));
        points_ref.emplace_back(kps_ref(i, 0), kps_ref(i, 1));
    }

    cv::Mat H, mask;
    H = cv::findHomography(points_cur, points_ref, method, reproj_threshold);

    return {H, mask};
}

} // namespace pyslam