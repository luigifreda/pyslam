import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

sys.path.append("./lib/")
from orbslam2_features import ORBextractor, ORBextractorDeterministic

if __name__ == "__main__":
    # read image
    img = cv2.imread("./kitti06-436.png", cv2.IMREAD_COLOR)

    # main settings
    num_features = 2000
    num_levels = 8
    scale_factor = 2

    ORBExtractorClass = ORBextractorDeterministic
    # ORBExtractorClass = ORBextractor

    # declare ORB extractor
    orb_extractor = ORBExtractorClass(num_features, scale_factor, num_levels)
    kps, des = orb_extractor.detectAndCompute(img)

    N = 10
    for i in range(N):
        orb_extractor = ORBExtractorClass(num_features, scale_factor, num_levels)
        kpsi, desi = orb_extractor.detectAndCompute(img)

        # compute the difference between the two sets of keypoints
        diffs_kps_i = []
        for kp1, kp2 in zip(kps, kpsi):
            # kp is a tuple (x, y, size, angle, response, octave)
            kp1_pt = np.array([kp1[0], kp1[1]])
            kp2_pt = np.array([kp2[0], kp2[1]])
            diffs_kps_i.append(np.linalg.norm(kp1_pt - kp2_pt))

        diffs_des_i = np.linalg.norm(des - desi)

        mean_diff_kps = np.mean(diffs_kps_i)
        std_diff_kps = np.std(diffs_kps_i)
        mean_diff_des = np.mean(diffs_des_i)
        std_diff_des = np.std(diffs_des_i)
        print(
            f"Diffs {i}: kps mean {i}: {mean_diff_kps}, kps std {i}: {std_diff_kps}, des mean {i}: {mean_diff_des}, des std {i}: {std_diff_des}"
        )

    # convert keypoint tuples in cv2.KeyPoints
    kps_cv = [cv2.KeyPoint(*kp) for kp in kps]

    imgDraw = cv2.drawKeypoints(
        img, kps_cv, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Show image
    plt.imshow(imgDraw)
    plt.show()
