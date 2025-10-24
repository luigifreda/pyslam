import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


from pyslam.config import Config


os.environ["PYSLAM_USE_CPP"] = "true"  # set environment variable to use C++ module

from pyslam.slam import (
    Frame,
    Camera,
    PinholeCamera,
    MapPoint,
    KeyFrame,
    Sim3Pose,
    Map,
    optimizer_g2o,
)

from pyslam.viz.mplot_figure import MPlotFigure

from pyslam.utilities.utils_geom import add_ones, poseRt, skew
from pyslam.utilities.utils_draw import draw_points2, draw_feature_matches
from pyslam.slam.search_points_test import search_frame_for_triangulation_test
from pyslam.slam.slam import Slam
from pyslam.slam.initializer import Initializer
from pyslam.utilities.timer import TimerFps

from pyslam.utilities.utils_sys import Printer

from pyslam.local_features.feature_tracker import FeatureTrackerTypes

from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.utilities.timer import Timer

from pyslam.config_parameters import Parameters
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.slam.frame import match_frames


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../../../.."
kDataFolder = kRootFolder + "/test/data"


if __name__ == "__main__":

    config = Config()
    # forced camera settings to be kept coherent with the input file below
    config.config[config.dataset_type]["settings"] = "settings/KITTI04-12.yaml"
    config.sensor_type = "mono"
    config.get_general_system_settings()  # parse again the settings file

    # dataset = dataset_factory(config)
    # groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)
    print(f"camera: {cam.to_json()}")

    # ============================================
    # Init Feature Tracker
    # ============================================

    # select your tracker configuration (see the file feature_tracker_configs.py)
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config["num_features"] = 1000
    tracker_config["deterministic"] = True
    feature_tracker = feature_tracker_factory(**tracker_config)
    FeatureTrackerShared.set_feature_tracker(feature_tracker, force=True)

    # ============================================

    # N.B.: keep this coherent with the above forced camera settings
    img_ref = cv2.imread(f"{kDataFolder}/kitti06-12.png", cv2.IMREAD_COLOR)
    # img_cur = cv2.imread('../data/kitti06-12-01.png',cv2.IMREAD_COLOR)
    img_cur = cv2.imread(f"{kDataFolder}/kitti06-13.png", cv2.IMREAD_COLOR)

    print(f"camera: {cam.width}x{cam.height}")

    f_ref = Frame(cam, img_ref, img_id=0)
    f_cur = Frame(cam, img_cur, img_id=1)

    print(f"f_ref: {f_ref.pose()}")
    print(f"f_cur: {f_cur.pose()}")

    timer = Timer()
    print("matching frames...")

    print(f"max descriptor distance: {Parameters.kMaxDescriptorDistance}")

    timer.start()

    matching_result = match_frames(f_ref, f_cur, ratio_test=0.8)
    idxs_ref, idxs_cur, num_found_matches = (
        matching_result.idxs1,
        matching_result.idxs2,
        len(matching_result.idxs1),
    )

    elapsed = timer.elapsed()
    print("time:", elapsed)
    print("# found matches:", num_found_matches)

    N = len(idxs_ref)

    pts_ref = f_ref.kpsu[idxs_ref[:N]]
    pts_cur = f_cur.kpsu[idxs_cur[:N]]

    img_ref, img_cur = draw_points2(img_ref, img_cur, pts_ref, pts_cur)
    img_matches = draw_feature_matches(img_ref, img_cur, pts_ref, pts_cur, horizontal=False)

    fig_ref = MPlotFigure(img_ref, title="image ref")
    fig_cur = MPlotFigure(img_cur, title="image cur")
    fig_matches = MPlotFigure(img_matches, title="image matches")

    MPlotFigure.show()
