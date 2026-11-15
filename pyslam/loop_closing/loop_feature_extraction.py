"""Raw keyframe preparation owned exclusively by the loop-detection process."""

from collections import OrderedDict

import numpy as np


class LoopKeyframeFeatureExtractor:
    # Keep the geometry descriptors separate from any descriptors recomputed by
    # the retrieval backend (which may use a different local feature manager).
    fields = ("kps", "kps_r", "kpsu", "kpsn", "octaves", "octaves_r",
              "sizes", "angles", "des", "des_r", "depths", "kps_ur")
    # Only recently queued neighbors are looked up (direct keyframes link to
    # the previous kRecentCovisibility ones), so a short history is enough and
    # memory stays bounded on long runs with large global descriptors.
    max_global_descriptors = 64

    def __init__(self):
        self.config = None
        self.has_right_tracker = False
        self.global_descriptors = OrderedDict()

    def reset(self):
        self.global_descriptors.clear()

    def extract(self, task):
        request = getattr(task, "feature_extraction", None)
        if request is None:
            return None

        from pyslam.local_features.feature_tracker import feature_tracker_factory
        from pyslam.slam import Frame
        from pyslam.slam.feature_tracker_shared import FeatureTrackerShared

        config = request["config"]
        if self.config != config:
            FeatureTrackerShared.set_feature_tracker(feature_tracker_factory(**config), force=True)
            self.config = dict(config)
            self.has_right_tracker = False
        if request["image_right"] is not None and not self.has_right_tracker:
            FeatureTrackerShared.set_feature_tracker_right(
                feature_tracker_factory(**config), force=True)
            self.has_right_tracker = True

        # Reuse the standard Frame path, including calibration, depth-factor
        # conversion, stereo matching and the configured detector/descriptor.
        frame = Frame(request["camera"], task.keyframe_data.img,
                      img_right=request["image_right"], depth=request["depth"],
                      id=task.keyframe_data.id)
        features = {}
        for name in self.fields:
            value = getattr(frame, name)
            features[name] = None if value is None else np.array(value, copy=True)
        for name in ("kps", "octaves", "sizes", "angles", "des"):
            setattr(task.keyframe_data, name, features[name])

        # Neighbor tasks may have been queued before their results reached the
        # parent. Their global descriptors are already available in this worker.
        for neighbor in task.covisible_keyframes_data:
            if neighbor.g_des is None:
                neighbor.g_des = self.global_descriptors.get(neighbor.id)
        return features

    def remember(self, task, output):
        if getattr(task, "feature_extraction", None) is not None:
            descriptor = task.keyframe_data.g_des
            if descriptor is None:
                descriptor = output.g_des_vec
            self.global_descriptors[task.keyframe_data.id] = descriptor
            self.global_descriptors.move_to_end(task.keyframe_data.id)
            while len(self.global_descriptors) > self.max_global_descriptors:
                self.global_descriptors.popitem(last=False)
