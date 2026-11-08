"""
The basics, as plain functions on numpy arrays.

  pts   Nx2 pixel coordinates (x, y)
  des   NxD float32 descriptors (L2)
  depth HxW float32
  K     3x3 intrinsics
"""

import cv2
import numpy as np


# ---------- appearance baseline ----------

def detect(img_rgb, n=2000):
    """RootSIFT. Returns pts (Nx2), sizes (N), des (Nx128)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kps, des = cv2.SIFT_create(nfeatures=n).detectAndCompute(gray, None)
    des = np.sqrt(des / (des.sum(axis=1, keepdims=True) + 1e-7))  # RootSIFT
    pts = np.array([k.pt for k in kps], np.float32)
    sizes = np.array([k.size for k in kps], np.float32)
    return pts, sizes, des.astype(np.float32)


def match(des1, des2, ratio=0.8):
    """Nearest neighbour + Lowe ratio test. Returns (i1, i2) index arrays."""
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    return np.array([m.queryIdx for m in good]), np.array([m.trainIdx for m in good])


# ---------- depth helpers ----------

def depth_at(depth, pts):
    """Depth at pixel positions of keypoints (nearest pixel)."""
    h, w = depth.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return depth[y, x] #z = depth[y, x], z[0] = depth[y[0], x[0]]; z[1] = depth[y[1], y[1]] ...


# ---------- WP2 : geometric descriptor, concatenated ----------

def depth_descriptor(depth, pts, sizes, grid=5, scale=3.0):
    """
    Relative depth around each keypoint: log(depth / center depth) on a grid x grid
    lattice spanning `scale * keypoint size`. Using log ratios makes it independent of the
    global depth scale. Returns N x grid^2, each row unit length.
    """
    r = np.linspace(-1, 1, grid)
    dx, dy = np.meshgrid(r, r)
    offsets = np.stack([dx.ravel(), dy.ravel()], 1)                      # G x 2
    samples = pts[:, None, :] + (scale * sizes)[:, None, None] * offsets  # N x G x 2

    z = depth_at(depth, samples.reshape(-1, 2)).reshape(len(pts), -1)
    zc = depth_at(depth, pts)[:, None]
    g = np.log(np.maximum(z, 1e-3) / np.maximum(zc, 1e-3))
    return (g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def concat(des, g, w):
    """[appearance | w * geometry]. w = 0 is exactly the appearance baseline."""
    return np.hstack([des, w * g]).astype(np.float32)

