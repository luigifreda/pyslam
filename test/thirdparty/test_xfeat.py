import torch
import cv2
import time
import platform 
import sys 
import torch
import numpy as np

sys.path.append("../../")
from config import Config
import config
config.cfg.set_lib('xfeat') 

from pyslam.io.dataset import dataset_factory
from accelerated_features.modules.xfeat import XFeat


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    return img_matches


if __name__ == "__main__":

    config = Config()

    xfeat = XFeat()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

    dataset = dataset_factory(config)

    img_id=0
    video_path = '../../videos/kitti00/video.mp4'
    cap = cv2.VideoCapture(video_path)
    ret, previous_frame = cap.read()
    previous_frame = cv2.resize(previous_frame, (1226,370))
    while dataset.isOk():

        current_frame = dataset.getImage(img_id)
        time0=time.time()
        # ret, current_frame = cap.read()
        # if not ret:
        #     break
        mkpts_0, mkpts_1 = xfeat.match_xfeat(current_frame, previous_frame, top_k = 4096)
        
        #kp=mkpts_1.cpu().numpy()
        for k in mkpts_1:
            #print(k)
            try:
                cv2.circle(current_frame,(int(k[0]),int(k[1])),3,(255),-1)
            except:
                pass
        #print(kp)
        
        # Process matches here (e.g., for tracking, stabilization, etc.)

        img_matches = warp_corners_and_draw_matches(mkpts_1, mkpts_0, previous_frame, current_frame)
        
        cv2.imshow("current_frame",current_frame)
        cv2.imshow("img_matches",img_matches)
        cv2.waitKey(2)        
        
        #viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        # Update previous_frame for the next iteration
        img_id +=1
        previous_frame = current_frame
        print(1/(time.time()-time0))
    cap.release()
