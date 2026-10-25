import cv2
import numpy as np

if __name__ == "__main__":
    # Test FAST directly on the same image

    img = cv2.imread("../data/kitti06-12-color.png", cv2.IMREAD_COLOR)
    fast = cv2.FastFeatureDetector_create(threshold=20)

    kps1 = fast.detect(img)

    N = 10
    for i in range(N):
        kpsi = fast.detect(img)

        # compute the difference between the two sets of keypoints
        diffs_i = []
        for kp1, kp2 in zip(kps1, kpsi):
            kp1_pt = np.array([kp1.pt[0], kp1.pt[1]])
            kp2_pt = np.array([kp2.pt[0], kp2.pt[1]])
            diffs_i.append(np.linalg.norm(kp1_pt - kp2_pt))

        mean_diff = np.mean(diffs_i)
        std_diff = np.std(diffs_i)
        print(f"Mean difference {i}: {mean_diff}, standard deviation {i}: {std_diff}")
