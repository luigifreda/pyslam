# open video file and play it

import cv2
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(this_dir, "..", "..")

if __name__ == "__main__":

    video_path = os.path.join(root_dir, "data", "videos", "kitti06", "video_color.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()