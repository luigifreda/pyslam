import os
import sys


from pyslam.config import Config

config = Config()

from pyslam.utilities.file_management import gdrive_download_lambda
from pyslam.utilities.system import getchar, Printer
from pyslam.utilities.img_management import (
    float_to_color,
    convert_float_to_colored_uint8_image,
    LoopCandidateImgs,
)
from pyslam.utilities.features import transform_float_to_binary_descriptor

import math
import cv2
import numpy as np

from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType
from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.local_features.feature_types import FeatureInfo

config.set_lib("pyobindex2")
import pyobindex2 as obindex2


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"
kOrbVocabFile = kDataFolder + "/ORBvoc.txt"


kMinDeltaFrameForMeaningfulLoopClosure = 10
kMaxResultsForLoopClosure = 5


# online loop closure detection by using DBoW3
if __name__ == "__main__":

    dataset = dataset_factory(config)

    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config["num_features"] = 2000

    print("tracker_config: ", tracker_config)
    feature_tracker = feature_tracker_factory(**tracker_config)

    # Creating a new index of images
    index = obindex2.ImageIndex(16, 150, 4, obindex2.MERGE_POLICY_AND, True)

    # to nicely visualize current loop candidates in a single image
    loop_closure_imgs = LoopCandidateImgs()

    # init the similarity matrix
    S_float = np.empty([dataset.num_frames, dataset.num_frames], "float32")
    S_color = np.empty([dataset.num_frames, dataset.num_frames, 3], "uint8")
    # S_color = np.full([dataset.num_frames, dataset.num_frames, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear

    cv2.namedWindow("S", cv2.WINDOW_NORMAL)

    entry_id = 0
    img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed
    while dataset.is_ok:

        timestamp = dataset.getTimestamp()  # get current timestamp
        img = dataset.getImageColor(img_id)

        if img is not None:
            print("----------------------------------------")
            print(f"processing img {img_id}")

            loop_closure_imgs.reset()

            # Find the keypoints and descriptors in img1
            kps, des = feature_tracker.detectAndCompute(
                img
            )  # with DL matchers this a null operation
            kps_ = [
                (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps
            ]  # tuple_x_y_size_angle_response_octave

            des_ = des
            if FeatureInfo.norm_type[feature_tracker.descriptor_type] != cv2.NORM_HAMMING:
                des_ = transform_float_to_binary_descriptor(des)

            if entry_id == 0:
                index.addImage(img_id, kps_, des_)
            elif entry_id >= 1:
                matches_feats = index.searchDescriptors(des_, 2, 64)

                # Filter matches according to the ratio test
                matches = []
                for (
                    m
                ) in (
                    matches_feats
                ):  # vector of pairs of tuples (queryIdx, trainIdx, imgIdx, distance)
                    if m[0][3] < m[1][3] * 0.8:
                        matches.append(m[0])

                if len(matches) > 0:
                    image_matches = index.searchImages(des_, matches, True)
                    for i, m in enumerate(image_matches):
                        float_value = m.score
                        color_value = float_to_color(m.score)
                        S_float[img_id, m.image_id] = float_value
                        S_float[m.image_id, img_id] = float_value
                        S_color[img_id, m.image_id] = color_value
                        S_color[m.image_id, img_id] = color_value

                        # visualize non-trivial loop closures: we check the query results are not too close to the current image
                        if abs(m.image_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure:
                            print(f"result - best id: {m.image_id}, score: {m.score}")
                            loop_img = dataset.getImageColor(m.image_id)
                            loop_closure_imgs.add(loop_img, m.image_id, m.score)

                        if i >= (kMaxResultsForLoopClosure - 1):
                            break

                # Addthe image to the index.
                index.addImage(img_id, kps_, des_, matches)

            # Reindex features every 500 images
            if entry_id % 250 == 0 and entry_id > 0:
                print("------ Rebuilding indices ------")
                index.rebuild()

            font_pos = (50, 50)
            cv2.putText(
                img,
                f"id: {img_id}",
                font_pos,
                LoopCandidateImgs.kFont,
                LoopCandidateImgs.kFontScale,
                LoopCandidateImgs.kFontColor,
                LoopCandidateImgs.kFontThickness,
                cv2.LINE_AA,
            )
            cv2.imshow("img", img)

            cv2.imshow("S", S_color)
            # cv2.imshow('S', convert_float_to_colored_uint8_image(S_float))

            if loop_closure_imgs.candidates is not None:
                cv2.imshow("loop_closure_imgs", loop_closure_imgs.candidates)

            cv2.waitKey(1)
        else:
            getchar()

        img_id += 1
        entry_id += 1
