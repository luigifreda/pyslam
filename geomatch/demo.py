"""
KITTI 06 image pairs with ground-truth poses.   Run from the repo root:
    python -m geomatch.demo [--show]

For each match, errors are computed against the ground-truth camera motion and grouped:
  depth-independent (GT pose only): epipolar, symmetric epipolar, Sampson distance,
      triangulation reprojection (+ cheirality)
  depth-dependent (adds DA3 depth): symmetric reprojection error
A match counts as correct under a metric if its error is below ERR_PX. For each weight w
of the geometric descriptor the table prints #matches, #wrong and per-metric precision,
plus median errors. Then, for the same putative matches, all filter configurations in
FILTER_CONFIGS (see geomatch/filtering.py) are compared: kept matches and precision after
filtering.   Run from the repo root:  python -m geomatch.demo [--show]
"""

import sys

import cv2
import numpy as np

from geomatch.core import concat, depth_at, depth_descriptor, detect, match
from geomatch.depth import predict_depth
from geomatch.filtering import MatchContext, filter_matches

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


def fundamental_matrix(T1, T2):
    """Fundamental matrix from the GT relative camera motion (camera 1 -> camera 2)."""
    T21 = np.linalg.inv(T2) @ T1                       # camera 1 -> camera 2
    R, t = T21[:3, :3], T21[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(K)
    return Kinv.T @ tx @ R @ Kinv


def epipolar_error(p1, p2, T1, T2):
    """Distance (px) of each p2 to the epipolar line of p1, using the GT relative pose."""
    F = fundamental_matrix(T1, T2)
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    lines = x1 @ F.T                                   # epipolar lines in image 2
    return np.abs(np.sum(x2 * lines, 1)) / np.hypot(lines[:, 0], lines[:, 1])


def symmetric_epipolar_error(p1, p2, T1, T2):
    """Mean of the point-to-epipolar-line distances in both images (px), GT relative pose."""
    F = fundamental_matrix(T1, T2)
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    l2 = x1 @ F.T                                      # epipolar lines in image 2
    l1 = x2 @ F                                        # epipolar lines in image 1
    d2 = np.abs(np.sum(x2 * l2, 1)) / np.hypot(l2[:, 0], l2[:, 1])
    d1 = np.abs(np.sum(x1 * l1, 1)) / np.hypot(l1[:, 0], l1[:, 1])
    return 0.5 * (d1 + d2)


def sampson_error(p1, p2, T1, T2):
    """Sampson distance (px): first-order approximation of the geometric error w.r.t.
    the fundamental matrix from the GT relative pose."""
    F = fundamental_matrix(T1, T2)
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    Fx1 = x1 @ F.T                                     # F x1
    Ftx2 = x2 @ F                                      # F^T x2
    num = np.sum(x2 * Fx1, 1) ** 2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(denom, 1e-12))


def triangulation_error(p1, p2, T1, T2):
    """Triangulate each match with the GT relative pose and return the mean reprojection
    error (px) of the triangulated 3D point in both cameras. Cheirality is enforced:
    matches whose 3D point lies behind either camera get infinite error.
    Depth-independent: uses only the GT pose, never the predicted depth."""
    T21 = np.linalg.inv(T2) @ T1                       # camera 1 -> camera 2
    R, t = T21[:3, :3], T21[:3, 3]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # projection in camera 1 frame
    P2 = K @ T21[:3]                                   # projection of camera 2 in camera 1 frame
    Xh = cv2.triangulatePoints(P1, P2, p1.T, p2.T)     # 4xN homogeneous, camera 1 frame
    X = (Xh[:3] / Xh[3]).T                             # Nx3 in camera 1
    X2 = X @ R.T + t                                   # in camera 2

    def project(Xc):
        proj = Xc @ K.T
        return proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)

    err = 0.5 * (np.linalg.norm(p1 - project(X), axis=1) + np.linalg.norm(p2 - project(X2), axis=1))
    err[(X[:, 2] <= 1e-6) | (X2[:, 2] <= 1e-6)] = np.inf  # cheirality violation
    return err


def _reproject_one_way(p_src, p_dst, z_src, T_src, T_dst):
    """Pixel distance between p_dst and the projection into the destination camera of
    p_src's 3D point, obtained by back-projecting p_src with depth z_src and applying
    the GT relative pose. Points landing behind the destination camera get infinite error."""
    T_sd = np.linalg.inv(T_dst) @ T_src                # source -> destination camera
    R, t = T_sd[:3, :3], T_sd[:3, 3]
    Kinv = np.linalg.inv(K)
    xs = np.c_[p_src, np.ones(len(p_src))]
    Xs = (xs @ Kinv.T) * z_src[:, None]                # Nx3 points in source camera
    Xd = Xs @ R.T + t                                  # in destination camera
    proj = Xd @ K.T
    uv = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-9)
    err = np.linalg.norm(p_dst - uv, axis=1)
    err[Xd[:, 2] <= 1e-6] = np.inf                     # behind destination camera
    return err


def reprojection_error(p1, p2, z1, z2, T1, T2):
    """Symmetric reprojection error: mean of the 1->2 and 2->1 pixel reprojection errors,
    each computed by back-projecting with the point's own depth and applying the GT relative
    pose. Each direction absorbs only its own depth error, so the average is more robust than
    the one-sided version. Infinite if the 3D point is behind either camera."""
    return 0.5 * (_reproject_one_way(p1, p2, z1, T1, T2) + _reproject_one_way(p2, p1, z2, T2, T1))


poses = np.tile(np.eye(4), (1101, 1, 1))
poses[:, :3] = np.loadtxt(POSES).reshape(-1, 3, 4)
frames = read_frames({i for s in STARTS for i in (s, s + GAP)})

WEIGHTS = [0.0, 0.1, 0.3, 0.5]
# filter configurations compared on the same putative matches: label -> filter_matches kwargs
FILTER_CONFIGS = [
    ("none",                        dict(method="none", ransac="none")),
    ("ransac",                      dict(method="none", ransac="after")),
    ("depth_valid",                 dict(method="depth_valid", ransac="none")),
    ("desc_thresh",                 dict(method="desc_threshold", ransac="none", max_dist=1.0)),
    ("rel_depth_thresh_0.1",        dict(method="relative_depth_threshold", ransac="none", rel_depth_thresh=0.1)),
    ("rel_depth_thresh_0.1+ransac", dict(method="relative_depth_threshold", ransac="after", rel_depth_thresh=0.1)),
    ("ransac+rel_depth_thresh_0.1", dict(method="relative_depth_threshold", ransac="before", rel_depth_thresh=0.1)),
    ("rel_depth_thresh_0.2",        dict(method="relative_depth_threshold", ransac="none", rel_depth_thresh=0.2)),
    ("rel_depth_thresh_0.2+ransac", dict(method="relative_depth_threshold", ransac="after", rel_depth_thresh=0.2)),
    ("ransac+rel_depth_thresh_0.2", dict(method="relative_depth_threshold", ransac="before", rel_depth_thresh=0.2)),
    ("rel_depth_thresh_0.4",        dict(method="relative_depth_threshold", ransac="none", rel_depth_thresh=0.4)),
    ("rel_depth_thresh_0.4+ransac", dict(method="relative_depth_threshold", ransac="after", rel_depth_thresh=0.4)),
    ("ransac+rel_depth_thresh_0.4", dict(method="relative_depth_threshold", ransac="before", rel_depth_thresh=0.4)),
    ("rel_depth_thresh_0.5",        dict(method="relative_depth_threshold", ransac="none", rel_depth_thresh=0.5)),
    ("rel_depth_thresh_0.5+ransac", dict(method="relative_depth_threshold", ransac="after", rel_depth_thresh=0.5)),
    ("ransac+rel_depth_thresh_0.5", dict(method="relative_depth_threshold", ransac="before", rel_depth_thresh=0.5)),
    ("rel_depth_thresh_0.9",        dict(method="relative_depth_threshold", ransac="none", rel_depth_thresh=0.9)),
    ("gt_epipolar (oracle)",        dict(method="gt_epipolar", ransac="none")),  # upper bound, uses GT
]
# per w: matches, and #correct under each metric (epi, sym-epi, sampson, triang, reproj)
totals = {w: np.zeros(6) for w in WEIGHTS}
epi_errors = {w: [] for w in WEIGHTS}   # per-match epipolar errors (depth-independent)
tri_errors = {w: [] for w in WEIGHTS}   # per-match triangulation errors (depth-independent)
rep_errors = {w: [] for w in WEIGHTS}   # per-match reprojection errors (depth-dependent)
fstats = {}                             # (w, label) -> [kept, correct(epi), correct(reproj)]
fstats_epi = {}                         # (w, label) -> list of per-match epipolar errors of kept matches

for s in STARTS:
    img1, img2 = frames[s], frames[s + GAP]
    depth1, depth2 = predict_depth(img1), predict_depth(img2)
    pts1, sizes1, des1 = detect(img1)
    pts2, sizes2, des2 = detect(img2)
    g1 = depth_descriptor(depth1, pts1, sizes1)
    g2 = depth_descriptor(depth2, pts2, sizes2)
    z1 = depth_at(depth1, pts1)
    z2 = depth_at(depth2, pts2)
    T1, T2 = poses[s], poses[s + GAP]

    for w in WEIGHTS:
        i1, i2 = match(concat(des1, g1, w), concat(des2, g2, w))
        p1, p2 = pts1[i1], pts2[i2]
        epi = epipolar_error(p1, p2, T1, T2)
        sym = symmetric_epipolar_error(p1, p2, T1, T2)
        sam = sampson_error(p1, p2, T1, T2)
        tri = triangulation_error(p1, p2, T1, T2)
        rep = reprojection_error(p1, p2, z1[i1], z2[i2], T1, T2)
        totals[w] += [len(p1), (epi < ERR_PX).sum(), (sym < ERR_PX).sum(),
                      (sam < ERR_PX).sum(), (tri < ERR_PX).sum(), (rep < ERR_PX).sum()]
        epi_errors[w].append(epi)
        tri_errors[w].append(tri)
        rep_errors[w].append(rep)

        # compare all filter configurations on the same putative matches
        ctx = MatchContext(i1=i1, i2=i2, pts1=pts1, pts2=pts2,
                           des1=concat(des1, g1, w), des2=concat(des2, g2, w),
                           depth1=depth1, depth2=depth2, z1=z1, z2=z2, K=K, T1=T1, T2=T2)
        for label, cfg in FILTER_CONFIGS:
            keep = filter_matches(ctx, **cfg)
            k1, k2 = i1[keep], i2[keep]
            if len(k1):
                f_epi_err = epipolar_error(pts1[k1], pts2[k2], T1, T2)
                f_epi = (f_epi_err < ERR_PX).sum()
                f_rep = (reprojection_error(pts1[k1], pts2[k2], z1[k1], z2[k2], T1, T2) < ERR_PX).sum()
            else:
                f_epi_err = np.empty(0)
                f_epi = f_rep = 0.0
            fstats[(w, label)] = fstats.get((w, label), np.zeros(3)) + [len(k1), f_epi, f_rep]
            fstats_epi.setdefault((w, label), []).append(f_epi_err)

print(f"\n{len(STARTS)} pairs, gap {GAP} frames; a match is 'correct' under a metric if its error < {ERR_PX} px")
print("'wrong' = matches - correct(epipolar); all errors in pixels; prec = share of matches correct\n")

print("DEPTH-INDEPENDENT metrics (ground-truth pose only)")
print(f"{'w':>4} | {'matches':>7} {'wrong':>6} | {'epi prec':>8} {'sym-epi':>8} {'sampson':>8} {'triang':>8} {'med tri err':>11}")
for w, (n, c_epi, c_sym, c_sam, c_tri, c_rep) in totals.items():
    med_tri = np.median(np.concatenate(tri_errors[w]))
    print(f"{w:4.1f} | {n:7.0f} {n-c_epi:6.0f} | {c_epi/n:8.1%} {c_sym/n:8.1%} {c_sam/n:8.1%} {c_tri/n:8.1%} {med_tri:11.2f}")

print("\nDEPTH-DEPENDENT metrics (add DA3 predicted depth)")
print(f"{'w':>4} | {'reproj prec':>11} {'med rep err':>11}")
for w, (n, c_epi, c_sym, c_sam, c_tri, c_rep) in totals.items():
    med_rep = np.median(np.concatenate(rep_errors[w]))
    print(f"{w:4.1f} | {c_rep/n:11.1%} {med_rep:11.2f}")

EPILINE_THRESHOLDS = list(range(1, 11))  # px
print("\nEPIPOLAR DISTANCE sweep (depth-independent, GT pose): precision at each threshold + avg/median")
print(f"{'w':>4} |" + "".join(f"{f'epi@{t}px':>8}" for t in EPILINE_THRESHOLDS) + f" | {'avg epi':>8} {'med epi':>8}")
for w in WEIGHTS:
    epi_all = np.concatenate(epi_errors[w])
    print(f"{w:4.1f} |" + "".join(f"{(epi_all < t).mean():8.1%}" for t in EPILINE_THRESHOLDS)
          + f" | {epi_all.mean():8.2f} {np.median(epi_all):8.2f}")

print("\nlegend: epi = epipolar distance to GT line | sym-epi = symmetric epipolar distance | "
      "sampson = Sampson distance | triang = GT-pose triangulation reprojection (cheirality enforced) | "
      "reproj = symmetric reprojection using predicted depth")

label_w = max(len("filter"), max(len(label) for label, _ in FILTER_CONFIGS))
for w in WEIGHTS:
    print(f"\nFILTER COMPARISON for w={w} (same putative matches, filter applied on top)")
    print(f"{'filter':>{label_w}} | {'kept':>6} {'of':>6} | {'epi prec':>8} {'reproj prec':>11}")
    n_all = totals[w][0]
    for label, _ in FILTER_CONFIGS:
        kept, c_epi, c_rep = fstats[(w, label)]
        prec_epi = c_epi / kept if kept else 0.0
        prec_rep = c_rep / kept if kept else 0.0
        print(f"{label:>{label_w}} | {kept:6.0f} {n_all:6.0f} | {prec_epi:8.1%} {prec_rep:11.1%}")

    print(f"\nFILTER COMPARISON epipolar sweep for w={w} (precision of KEPT matches at each threshold)")
    print(f"{'filter':>{label_w}} |" + "".join(f"{f'@{t}px':>7}" for t in EPILINE_THRESHOLDS)
          + f" | {'avg':>7} {'med':>7}")
    for label, _ in FILTER_CONFIGS:
        parts = fstats_epi.get((w, label), [])
        e = np.concatenate(parts) if parts else np.empty(0)
        if len(e):
            row = "".join(f"{(e < t).mean():7.1%}" for t in EPILINE_THRESHOLDS) + f" | {e.mean():7.2f} {np.median(e):7.2f}"
        else:
            row = "".join(f"{0.0:7.1%}" for t in EPILINE_THRESHOLDS) + f" | {'-':>7} {'-':>7}"
        print(f"{label:>{label_w}} |{row}")

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
