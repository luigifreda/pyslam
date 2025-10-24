#include "tracking_core.h"

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
#include "utils/geom_2views.h"
#include "utils/image_processing.h"
#include "utils/messages.h"

// Add this helper function in tracking_core.cpp (or as an inline function in a header)
namespace {
// Get the best available RANSAC method
inline int get_ransac_method() {
#if defined(OPENCV_VERSION_MAJOR) && OPENCV_VERSION_MAJOR >= 4
#if defined(USAC_MAGSAC)
    return cv::USAC_MAGSAC;
#else
    return cv::RANSAC;
#endif
#else
    return cv::RANSAC;
#endif
}
} // namespace

namespace pyslam {

// Essential matrix pose estimation
std::tuple<std::vector<int>, std::vector<int>, int>
TrackingCore::estimate_pose_by_fitting_ess_mat(const FramePtr &f_ref, FramePtr &f_cur,
                                               const std::vector<int> &idxs_ref,
                                               const std::vector<int> &idxs_cur) {
    int num_inliers = 0;

    if (idxs_ref.empty() || idxs_cur.empty() || idxs_cur.size() != idxs_ref.size()) {
        MSG_RED_WARN("idxs_ref or idxs_cur is empty or len(idxs_cur) != len(idxs_ref)");
        return std::make_tuple(idxs_ref, idxs_cur, num_inliers);
    }

    try {
        const int ransac_method = get_ransac_method();

        // Extract keypoints manually
        MatNx2d kpsn_ref(idxs_ref.size(), 2);
        MatNx2d kpsn_cur(idxs_cur.size(), 2);
        for (size_t i = 0; i < idxs_ref.size(); ++i) {
            kpsn_ref(i, 0) = f_ref->kpsn(idxs_ref[i], 0);
            kpsn_ref(i, 1) = f_ref->kpsn(idxs_ref[i], 1);

            kpsn_cur(i, 0) = f_cur->kpsn(idxs_cur[i], 0);
            kpsn_cur(i, 1) = f_cur->kpsn(idxs_cur[i], 1);
        }

        // Estimate inter frame camera motion by using found keypoint matches
        // output of the following function is: Trc = [Rrc, trc] with ||trc||=1 where c=cur, r=ref
        // and pr = Trc * pc
        const auto [Mrc, mask_match] = geom_2views::estimate_pose_ess_mat(
            kpsn_ref, kpsn_cur, ransac_method, Parameters::kRansacProb,
            Parameters::kRansacThresholdNormalized);

        // Mcr = inv_T(Mrc)
        const auto Mcr = inv_T(Mrc);

        // estimated_Tcw = Mcr * f_ref.pose()
        const auto f_ref_pose = f_ref->pose();
        const auto estimated_Tcw = Mcr * f_ref_pose;

        // Remove outliers from keypoint matches by using the mask
        std::vector<int> idxs_ref_out, idxs_cur_out;
        const size_t num_matches = mask_match.size();

        for (size_t i = 0; i < num_matches; ++i) {
            if (mask_match[i] == 1) {
                idxs_ref_out.push_back(idxs_ref[i]);
                idxs_cur_out.push_back(idxs_cur[i]);
            }
        }

        num_inliers = static_cast<int>(idxs_ref_out.size());
        MSG_INFO_STREAM("# inliers: " << num_inliers);

        // If there are not enough inliers do not use the estimated pose
        if (num_inliers < Parameters::kNumMinInliersEssentialMat) {
            MSG_RED_WARN("Essential mat: not enough inliers!");
        } else {
            // Use the estimated pose as an initial guess for the subsequent pose optimization
            // Set only the estimated rotation (essential mat computation does not provide a scale
            // for the translation)
            const Mat3d Rcw = estimated_Tcw.block<3, 3>(0, 0); // copy only the rotation
            const Vec3d tcw =
                f_ref_pose.block<3, 1>(0, 3); // override translation with ref frame translation
            f_cur->update_rotation_and_translation(Rcw, tcw);
        }

        return std::make_tuple(std::move(idxs_ref_out), std::move(idxs_cur_out), num_inliers);

    } catch (const std::exception &e) {
        MSG_RED_WARN_STREAM("Error in estimate_pose_ess_mat: " << e.what());
        return std::make_tuple(idxs_ref, idxs_cur, num_inliers);
    }
}

// Homography estimation with RANSAC
std::tuple<bool, std::vector<int>, std::vector<int>, int, int>
TrackingCore::find_homography_with_ransac(const FramePtr &f_cur, const FramePtr &f_ref,
                                          const std::vector<int> &idxs_cur,
                                          const std::vector<int> &idxs_ref,
                                          const double reproj_threshold /* = 5.0 */,
                                          const int min_num_inliers /* = 15 */) {
    int num_matched_kps = 0;
    int num_outliers = 0;

    if (idxs_cur.empty() || idxs_cur.size() != idxs_ref.size()) {
        MSG_RED_WARN(
            "find_homography_with_ransac: idxs_cur is empty or len(idxs_cur) != len(idxs_ref)");
        return std::make_tuple(false, std::vector<int>(), std::vector<int>(), 0, 0);
    }

    const int ransac_method = get_ransac_method();

    // Get keypoints
    std::vector<cv::Point2f> kps_cur, kps_ref;
    kps_cur.reserve(idxs_cur.size());
    kps_ref.reserve(idxs_ref.size());
    for (size_t i = 0; i < idxs_cur.size(); ++i) {
        const float x_cur = f_cur->kps(idxs_cur[i], 0);
        const float y_cur = f_cur->kps(idxs_cur[i], 1);
        const float x_ref = f_ref->kps(idxs_ref[i], 0);
        const float y_ref = f_ref->kps(idxs_ref[i], 1);
        kps_cur.emplace_back(x_cur, y_cur);
        kps_ref.emplace_back(x_ref, y_ref);
    }

    cv::Mat H, mask;
    H = cv::findHomography(kps_cur, kps_ref, ransac_method, reproj_threshold, mask);

    // Check if homography was found (mask is not None)
    if (mask.empty()) {
        return std::make_tuple(false, std::vector<int>(), std::vector<int>(), 0, 0);
    }

    // Convert mask to boolean and count inliers
    int num_inliers = 0;
    std::vector<int> idxs_cur_out, idxs_ref_out;
    idxs_cur_out.reserve(idxs_cur.size());
    idxs_ref_out.reserve(idxs_ref.size());

    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i) == 1) {
            num_inliers++;
            idxs_cur_out.push_back(idxs_cur[i]);
            idxs_ref_out.push_back(idxs_ref[i]);
        }
    }

    num_outliers = static_cast<int>(idxs_cur.size()) - num_inliers;

    if (num_inliers < min_num_inliers) {
        return std::make_tuple(false, std::vector<int>(), std::vector<int>(), 0, 0);
    }

    num_matched_kps = static_cast<int>(idxs_cur_out.size());
    return std::make_tuple(true, std::move(idxs_cur_out), std::move(idxs_ref_out), num_matched_kps,
                           num_outliers);
}

// propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need to
// lock)
std::tuple<int, std::vector<int>, std::vector<int>> TrackingCore::propagate_map_point_matches(
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

std::vector<MapPointPtr> TrackingCore::create_vo_points(FramePtr &frame, int max_num_points,
                                                        const Vec3b &color) {
    std::lock_guard<std::mutex> lock(frame->_lock_features);

    const auto &depths = frame->depths;
    const auto &camera = frame->camera;
    auto &points = frame->points;

    // Check depths are available
    if (depths.empty()) {
        MSG_WARN("TrackingCore::create_vo_points() - depths are empty");
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
        // If there is not a point or the point is not already observed,
        // then create a new map point
        if (!p || (p && p->num_observations() < 1)) {
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

int TrackingCore::create_and_add_stereo_map_points_on_new_kf(FramePtr &frame, KeyFramePtr &kf,
                                                             MapPtr &map, const cv::Mat &img) {

    const auto &depths = kf->depths;
    const auto &camera = kf->camera;
    auto &frame_points = frame->points;
    auto &kf_points = kf->points;

    // Check depths are available
    if (depths.empty()) {
        MSG_WARN("TrackingCore::create_and_add_stereo_map_points_on_new_kf() - depths are empty");
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
        MSG_ERROR("TrackingCore::create_and_add_stereo_map_points_on_new_kf() - image channels "
                  "are not 1 or 3");
        return 0;
    }
    const bool is_color = num_channels == 3;

    // Insert all points that are within the depth threshold or at least the closest max_num_points
    // points
    for (const auto &[idx, depth] : valid_idxs_depths) {
        const auto &p = kf_points[idx];
        // If there is not a point or the point is not already observed,
        // then create a new map point
        if (!p || (p && p->num_observations() < 1)) {
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

std::tuple<int, int, std::vector<bool>>
TrackingCore::count_tracked_and_non_tracked_close_points(const FramePtr &f_cur,
                                                         const SensorType sensor_type) {
    // Count how many "close" points are being tracked and how many "close" points could be
    // potentially created. For convenience, we also compute and return a mask of tracked points.

    int num_non_tracked_close = 0;
    int num_tracked_close = 0;
    std::vector<bool> tracked_mask;

    const auto &points = f_cur->points;
    const auto &outliers = f_cur->outliers;
    const auto &depths = f_cur->depths;
    const auto &camera = f_cur->camera;
    const float depth_threshold = camera->depth_threshold;
    const size_t num_points = points.size();

    if (sensor_type != SensorType::MONOCULAR) {
        tracked_mask.resize(num_points, false);
        for (size_t i = 0; i < num_points; ++i) {
            const bool is_close =
                (depths[i] > Parameters::kMinDepth && depths[i] < depth_threshold);
            const bool is_tracked = points[i] && !outliers[i];
            if (is_tracked) { // point is tracked
                tracked_mask[i] = true;
                if (is_close) { // point is close and tracked
                    num_tracked_close++;
                }
            } else if (is_close) { // point is close but not tracked
                num_non_tracked_close++;
            }
        }
    }
    return std::make_tuple(num_tracked_close, num_non_tracked_close, std::move(tracked_mask));
}

} // namespace pyslam