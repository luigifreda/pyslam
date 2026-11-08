"""
KITTI 06 image pairs with ground-truth poses.   Run from the repo root:
    python -m geomatch.demo [--show]

A match counts as correct if the point in image 2 lies within ERR_PX of the epipolar line
given by the ground-truth camera motion. For each weight w of the geometric descriptor it
prints: #matches, #correct, #wrong, precision.
"""

import sys

import cv2
import numpy as np

from geomatch.core import concat, depth_descriptor, detect, match
from geomatch.depth import predict_depth

VIDEO = "data/videos/kitti06/video_color.mp4"
POSES = "data/videos/kitti06/06.txt"  # KITTI format: 3x4 camera-to-world per frame
K = np.array([[707.0912, 0, 601.8873], [0, 707.0912, 183.1104], [0, 0, 1]])  # settings/KITTI04-12.yaml
STARTS = range(50, 1050, 100)  # first frame of each pair (frames 0-5: GT rotation ~0.5 deg off)
GAP = 10                      # frames between the two images (bigger = harder)
ERR_PX = 4.0                  # max distance to the GT epipolar line for a correct match


def read_frames(idxs):
    """Read selected frames (RGB) by decoding the video sequentially (seeking is unreliable)."""
    cap, out, i = cv2.VideoCapture(VIDEO), {}, 0
    while len(out) < len(idxs):
        ok, frame = cap.read()
        if not ok:
            break
        if i in idxs:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    return out


def epipolar_error(p1, p2, T1, T2):
    """Distance (px) of each p2 to the epipolar line of p1, using the GT relative pose."""
    T21 = np.linalg.inv(T2) @ T1                       # camera 1 -> camera 2
    R, t = T21[:3, :3], T21[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ tx @ R @ Kinv                         # fundamental matrix
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    lines = x1 @ F.T                                   # epipolar lines in image 2
    return np.abs(np.sum(x2 * lines, 1)) / np.hypot(lines[:, 0], lines[:, 1])


poses = np.tile(np.eye(4), (1101, 1, 1))
poses[:, :3] = np.loadtxt(POSES).reshape(-1, 3, 4)
frames = read_frames({i for s in STARTS for i in (s, s + GAP)})

WEIGHTS = [0.0, 0.1, 0.3, 0.5]
totals = {w: np.zeros(2) for w in WEIGHTS}  # matches, correct

for s in STARTS:
    img1, img2 = frames[s], frames[s + GAP]
    depth1, depth2 = predict_depth(img1), predict_depth(img2)
    pts1, sizes1, des1 = detect(img1)
    pts2, sizes2, des2 = detect(img2)
    g1 = depth_descriptor(depth1, pts1, sizes1)
    g2 = depth_descriptor(depth2, pts2, sizes2)

    for w in WEIGHTS:
        i1, i2 = match(concat(des1, g1, w), concat(des2, g2, w))
        p1, p2 = pts1[i1], pts2[i2]
        correct = epipolar_error(p1, p2, poses[s], poses[s + GAP]) < ERR_PX
        totals[w] += [len(p1), correct.sum()]

print(f"\n{len(STARTS)} pairs, gap {GAP} frames, correct = within {ERR_PX}px of GT epipolar line")
print(f"{'w':>4} | {'matches':>7} {'correct':>7} {'wrong':>6} {'precision':>9}")
for w, (n, c) in totals.items():
    print(f"{w:4.1f} | {n:7.0f} {c:7.0f} {n-c:6.0f} {c/n:9.1%}")

SHOW_PAIR = 0     # which pair to plot (index into STARTS)
SHOW_W = 0.0      # geometry weight for the plotted matches (0 = SIFT only)

if "--show" in sys.argv:  # each plot in its own window
    import matplotlib.pyplot as plt

    s = STARTS[SHOW_PAIR]
    img1, img2 = frames[s], frames[s + GAP]
    depth1, depth2 = predict_depth(img1), predict_depth(img2)
    pts1, sizes1, des1 = detect(img1)
    pts2, sizes2, des2 = detect(img2)
    g1 = depth_descriptor(depth1, pts1, sizes1)
    g2 = depth_descriptor(depth2, pts2, sizes2)
    i1, i2 = match(concat(des1, g1, SHOW_W), concat(des2, g2, SHOW_W))
    correct = epipolar_error(pts1[i1], pts2[i2], poses[s], poses[s + GAP]) < ERR_PX

    def window(title, img, **kw):
        plt.figure(title, figsize=(12, 4 if img.shape[0] < 500 else 7))
        plt.imshow(img, **kw)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()

    window(f"frame {s}", img1)
    window(f"DA3 depth, frame {s}", depth1, cmap="magma_r")
    window(f"matches {s}->{s + GAP}, w={SHOW_W}: {correct.sum()} correct (green), "
           f"{(~correct).sum()} wrong (red)", np.vstack([img1, img2]))
    h = img1.shape[0]
    for a, b, c in zip(pts1[i1], pts2[i2], correct):
        plt.plot([a[0], b[0]], [a[1], b[1] + h], "g-" if c else "r-", lw=0.6)
    plt.show()
