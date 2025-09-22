"""
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
"""

import numpy as np
import cv2

from pyslam.slam.camera import Camera
from pyslam.local_features.feature_tracker import (
    FeatureTrackerTypes,
    FeatureTrackingResult,
    FeatureTracker,
)
from pyslam.utilities.utils_geom import poseRt, is_rotation_matrix, closest_rotation_matrix
from pyslam.utilities.timer import TimerFps
from pyslam.io.ground_truth import GroundTruth
from pyslam.slam.visual_odometry_base import VoState, VisualOdometryBase


kVerbose = True

kMinNumFeature = 2000
kRansacThresholdNormalized = (
    0.0004  # metric threshold used for normalized image coordinates (originally 0.0003)
)
kRansacThresholdPixels = 0.1  # pixel threshold used for image coordinates
kUseEssentialMatrixEstimation = True  # using the essential matrix fitting algorithm is more robust RANSAC given five-point algorithm solver
kRansacProb = 0.999  # (originally 0.999)
kMinAveragePixelShiftForMotionEstimation = 1.5  # if the average pixel shift is below this threshold, motion is considered to be small enough to be ignored

kUseGroundTruthScale = True
kAbsoluteScaleThresholdKitti = 0.1  # absolute translation scale; it is also the minimum translation norm for an accepted motion
kAbsoluteScaleThresholdIndoor = 0.015  # absolute translation scale; it is also the minimum translation norm for an accepted motion


# This "educational" class is a first start to understand the basics of inter frame feature tracking and camera pose estimation.
# It combines the simplest VO ingredients without performing any image point triangulation or
# windowed bundle adjustment. At each step $k$, it estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$.
# The inter frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $||t_{k-1,k}||=1$.
# With this very basic approach, you need to use a ground truth in order to recover a reasonable inter-frame scale $s$ and estimate a
# valid trajectory by composing $C_k = C_{k-1} * [R_{k-1,k}, s t_{k-1,k}]$.
class VisualOdometryEducational(VisualOdometryBase):
    def __init__(self, cam: Camera, groundtruth: GroundTruth, feature_tracker: FeatureTracker):
        super().__init__(cam=cam, groundtruth=groundtruth)

        self.kps_ref = None  # reference keypoints
        self.des_ref = None  # refeference descriptors
        self.kps_cur = None  # current keypoints
        self.des_cur = None  # current descriptors

        self.feature_tracker: FeatureTracker = feature_tracker

        self.pose_estimation_inliers = None

        self.absolute_scale_threshold = 0.0
        if self.groundtruth.type == "kitti":
            self.absolute_scale_threshold = kAbsoluteScaleThresholdKitti
        else:
            self.absolute_scale_threshold = kAbsoluteScaleThresholdIndoor

        self.timer_pose_est = TimerFps("PoseEst", is_verbose=self.timer_verbose)
        self.timer_feat = TimerFps("Feature", is_verbose=self.timer_verbose)

    def computeFundamentalMatrix(self, kps_ref, kps_cur):
        F, mask = cv2.findFundamentalMat(
            kps_ref, kps_cur, cv2.FM_RANSAC, kRansacThresholdPixels, kRansacProb
        )
        if F is None or F.shape == (1, 1):
            # no fundamental matrix found
            raise Exception("No fundamental matrix found")
        elif F.shape[0] > 3:
            # more than one matrix found, just pick the first
            F = F[0:3, 0:3]
        return np.matrix(F), mask

    def removeOutliersByMask(self, mask):
        if mask is not None:
            n = self.kpn_cur.shape[0]
            mask_index = [i for i, v in enumerate(mask) if v > 0]
            self.kpn_cur = self.kpn_cur[mask_index]
            self.kpn_ref = self.kpn_ref[mask_index]
            if self.des_cur is not None:
                self.des_cur = self.des_cur[mask_index]
            if self.des_ref is not None:
                self.des_ref = self.des_ref[mask_index]
            if kVerbose:
                print("removed ", n - self.kpn_cur.shape[0], " outliers")

    # Fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
    # out: [Rrc, trc]   (with respect to 'ref' frame)
    # N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
    # N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie on a ruled quadric
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
    # N.B.3: The five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
    # N.B.4: As it is reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return the rotation
    def estimatePose(self, kps_ref, kps_cur):
        kp_ref_u = self.cam.undistort_points(kps_ref)
        kp_cur_u = self.cam.undistort_points(kps_cur)
        self.kpn_ref = self.cam.unproject_points(kp_ref_u)
        self.kpn_cur = self.cam.unproject_points(kp_cur_u)
        if kUseEssentialMatrixEstimation:
            ransac_method = None
            try:
                # with VO RANSAC seems to return better results than USAC_MAGSAC
                # probably, better tuning is needed for USAC_MAGSAC
                ransac_method = cv2.RANSAC  # cv2.USAC_MAGSAC
            except:
                ransac_method = cv2.RANSAC
            # the essential matrix algorithm is more robust since it uses the five-point algorithm solver by D. Nister (see the notes and paper above )
            E, self.mask_match = cv2.findEssentialMat(
                self.kpn_cur,
                self.kpn_ref,
                focal=1,
                pp=(0.0, 0.0),
                method=ransac_method,
                prob=kRansacProb,
                threshold=kRansacThresholdNormalized,
            )
        else:
            # just for the hell of testing fundamental matrix fitting ;-)
            F, self.mask_match = self.computeFundamentalMatrix(kp_cur_u, kp_ref_u)
            E = self.cam.K.T @ F @ self.cam.K  # E = K.T * F * K
        # self.removeOutliersFromMask(self.mask)  # do not remove outliers, the last unmatched/outlier features can be matched and recognized as inliers in subsequent frames
        self.pose_estimation_inliers, R, t, mask = cv2.recoverPose(
            E, self.kpn_cur, self.kpn_ref, focal=1, pp=(0.0, 0.0)
        )
        print(f"num inliers in pose estimation: {self.pose_estimation_inliers}")
        return R, t  # Rrc, trc (with respect to 'ref' frame)

    def process_first_frame(self, frame_id) -> None:
        # convert image to gray if needed
        if self.cur_image.ndim > 2:
            self.cur_image = cv2.cvtColor(self.cur_image, cv2.COLOR_RGB2GRAY)
        # only detect on the current image
        self.kps_ref, self.des_ref = self.feature_tracker.detectAndCompute(self.cur_image)
        # convert from list of keypoints to an array of points
        self.kps_ref = (
            np.array([x.pt for x in self.kps_ref], dtype=np.float32)
            if self.kps_ref is not None
            else None
        )
        self.draw_img = self.drawFeatureTracks(self.cur_image)

    def process_frame(self, frame_id) -> None:
        # convert image to gray if needed
        if self.cur_image.ndim > 2:
            self.cur_image = cv2.cvtColor(self.cur_image, cv2.COLOR_RGB2GRAY)
        # track features
        self.timer_feat.start()
        self.track_result = self.feature_tracker.track(
            self.prev_image, self.cur_image, self.kps_ref, self.des_ref
        )
        self.timer_feat.refresh()
        # estimate pose
        self.timer_pose_est.start()
        R, t = self.estimatePose(
            self.track_result.kps_ref_matched, self.track_result.kps_cur_matched
        )
        self.timer_pose_est.refresh()
        # update keypoints history
        self.kps_ref = self.track_result.kps_ref
        self.kps_cur = self.track_result.kps_cur
        self.des_cur = self.track_result.des_cur
        self.num_matched_kps = self.kpn_ref.shape[0]
        self.num_inliers = np.sum(self.mask_match)
        # compute average delta pixel shift
        self.average_pixel_shift = np.mean(
            np.abs(self.track_result.kps_ref_matched - self.track_result.kps_cur_matched)
        )
        print(f"average pixel shift: {self.average_pixel_shift}")
        if kVerbose:
            matcher_type = (
                self.feature_tracker.matcher.matcher_type.name
                if self.feature_tracker.matcher is not None
                else self.feature_tracker.matcher_type
            )
            print(
                "# matched points: ",
                self.num_matched_kps,
                ", # inliers: ",
                self.num_inliers,
                ", matcher type: ",
                matcher_type,
                ", tracker type: ",
                self.feature_tracker.tracker_type.name,
            )
        # t is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with the previous estimated ones)
        self.update_gt_data(frame_id)
        absolute_scale = self.gt_scale if kUseGroundTruthScale else 1.0
        print("absolute scale: ", absolute_scale)
        # NOTE: This simplistic estimation approach provide reasonable results with Kitti where a good ground velocity provides a decent interframe parallax. It does not work with indoor datasets.
        if (
            absolute_scale > self.absolute_scale_threshold
            and self.average_pixel_shift > kMinAveragePixelShiftForMotionEstimation
            and self.pose_estimation_inliers > 5
        ):
            # compose absolute motion [Rwa,twa] with estimated relative motion [Rab,s*tab] (s is the scale extracted from the ground truth)
            # [Rwb,twb] = [Rwa,twa]*[Rab,tab] = [Rwa*Rab|twa + Rwa*tab]
            print("estimated t with norm |t|: ", np.linalg.norm(t), " (just for sake of clarity)")
            self.cur_R = self.cur_R.dot(R)
            if not is_rotation_matrix(self.cur_R):
                print(f"Correcting rotation matrix: {self.cur_R}")
                self.cur_R = closest_rotation_matrix(self.cur_R)
            self.cur_t = self.cur_t + absolute_scale * self.cur_R @ t
        # draw image
        self.draw_img = self.drawFeatureTracks(self.cur_image)
        # check if we have enough features to track otherwise detect new ones and start tracking from them (used for LK tracker)
        if (self.feature_tracker.tracker_type == FeatureTrackerTypes.LK) and (
            self.kps_ref.shape[0] < self.feature_tracker.num_features
        ):
            self.kps_cur, self.des_cur = self.feature_tracker.detectAndCompute(self.cur_image)
            self.kps_cur = np.array(
                [x.pt for x in self.kps_cur], dtype=np.float32
            )  # convert from list of keypoints to an array of points
            if kVerbose:
                print("# new detected points: ", self.kps_cur.shape[0])
        self.kps_ref = self.kps_cur
        self.des_ref = self.des_cur

    def drawFeatureTracks(self, img, reinit=False):
        draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        num_outliers = 0
        if self.state == VoState.GOT_FIRST_IMAGE:
            if reinit:
                for p1 in self.kps_cur:
                    a, b = p1.ravel()
                    cv2.circle(draw_img, (a, b), 1, (0, 255, 0), -1)
            else:
                print(
                    f"drawing feature tracks, num features matched: {len(self.track_result.kps_ref_matched)}"
                )
                for i, pts in enumerate(
                    zip(self.track_result.kps_ref_matched, self.track_result.kps_cur_matched)
                ):
                    drawAll = False  # set this to true if you want to draw outliers
                    if self.mask_match[i] or drawAll:
                        p1, p2 = pts
                        a, b = p1.astype(int).ravel()
                        c, d = p2.astype(int).ravel()
                        cv2.line(draw_img, (a, b), (c, d), (0, 255, 0), 1)
                        cv2.circle(draw_img, (a, b), 1, (0, 0, 255), -1)
                    else:
                        num_outliers += 1
            if kVerbose:
                print("# outliers: ", num_outliers)
        return draw_img
