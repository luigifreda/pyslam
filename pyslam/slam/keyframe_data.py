from pyslam.slam import KeyFrame

# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .keyframe import KeyFrame


# A wrapper with essential data for making a KeyFrame picklable
class KeyFrameData:
    def __init__(self, keyframe: KeyFrame):
        # replicate the main fields of KeyFrame without locks
        self.id = keyframe.id
        self.kid = keyframe.kid
        # self.is_keyframe = keyframe.is_keyframe

        self.img = keyframe.img
        # self.img_d = keyframe.img_d
        self.depth_img = keyframe.depth_img
        self.semantic_img = keyframe.semantic_img
        self.camera = keyframe.camera

        # self.pose = keyframe.pose
        kf_Tcw = keyframe.Tcw()
        self.Rcw = kf_Tcw[:3, :3]
        self.tcw = kf_Tcw[:3, 3]
        self.Tcw = kf_Tcw
        self.timestamp = keyframe.timestamp

        self.is_bad = keyframe.is_bad()

        self.kps = keyframe.kps
        self.kps_r = keyframe.kps_r
        self.kpsu = keyframe.kpsu
        self.kpsn = keyframe.kpsn
        self.kps_sem = keyframe.kps_sem
        self.octaves = keyframe.octaves
        self.octaves_r = keyframe.octaves_r
        self.sizes = keyframe.sizes
        self.angles = keyframe.angles
        self.des = keyframe.des
        self.des_r = keyframe.des_r
        self.depths = keyframe.depths
        self.kps_ur = keyframe.kps_ur

        self.g_des = keyframe.g_des

        self.points = keyframe.points
        self.outliers = keyframe.outliers
