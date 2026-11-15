"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.

Tests for pyslam.scene_from_views.

They use synthetic scenes (known cameras + known depth maps) and fake models, so no network
weights are needed. Tests that need an optional model package are skipped when it is missing.
"""

import importlib
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trimesh")
pytest.importorskip("roma")

import trimesh  # noqa: E402

from pyslam.scene_from_views.scene_from_views_base import (  # noqa: E402
    SceneFromViewsBase,
    SceneFromViewsResult,
    depth_from_world_points,
)


# ---------------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------------


def _rotation(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _pose(rx, ry, rz, t):
    T = np.eye(4)
    T[:3, :3] = _rotation(rx, ry, rz)
    T[:3, 3] = t
    return T


def _depth_map(h, w, base=2.0, phase=0.0):
    yy, xx = np.mgrid[0:h, 0:w]
    return (base + 0.3 * np.sin(0.3 * xx + phase) + 0.2 * np.cos(0.25 * yy + phase)).astype(
        np.float64
    )


def _pointmap(depth, f, cx, cy, cam2world=None):
    """Back-project a depth map (pinhole) and optionally move it to the world frame."""
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([(xx - cx) * depth / f, (yy - cy) * depth / f, depth], axis=-1)
    if cam2world is not None:
        pts = pts @ cam2world[:3, :3].T + cam2world[:3, 3]
    return pts


def _import_or_skip(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as e:  # missing optional dependency
        pytest.skip(f"cannot import {module_name}: {e}")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def test_depth_from_world_points_recovers_camera_depth():
    h, w, f = 12, 16, 20.0
    depth = _depth_map(h, w)
    cam2world = _pose(0.2, -0.3, 0.1, [1.0, -0.5, 2.0])
    pts_world = _pointmap(depth, f, w / 2, h / 2, cam2world)

    # world Z is NOT the depth (this was the bug)
    assert not np.allclose(pts_world[..., 2], depth, atol=1e-3)

    out = depth_from_world_points(pts_world, cam2world)
    np.testing.assert_allclose(out, depth, atol=1e-9)

    # flat input + mask: depth is zeroed outside the mask
    mask = np.zeros((h, w), dtype=bool)
    mask[2:5, 3:9] = True
    out = depth_from_world_points(pts_world.reshape(-1, 3), cam2world, mask)
    np.testing.assert_allclose(out[mask], depth[mask], atol=1e-9)
    assert np.all(out[~mask] == 0)


class _DummyReconstructor(SceneFromViewsBase):
    def __init__(self, fail=False):
        super().__init__(device="cpu")
        self.fail = fail
        self.created_dirs = []

    def preprocess_images(self, images, **kwargs):
        temp_dir = self._make_temp_dir(suffix="_test")
        with open(os.path.join(temp_dir, "x.bin"), "wb") as f:
            f.write(b"0")
        self.created_dirs.append(temp_dir)
        return images

    def infer(self, processed_images, **kwargs):
        if self.fail:
            raise RuntimeError("inference failed")
        return processed_images

    def postprocess_results(self, raw_output, images, processed_images, **kwargs):
        # temp dirs must still exist during postprocessing
        assert all(os.path.isdir(d) for d in self.created_dirs)
        return SceneFromViewsResult(processed_images=list(processed_images))


@pytest.mark.parametrize("fail", [False, True])
def test_reconstruct_removes_temp_dirs(fail):
    rec = _DummyReconstructor(fail=fail)
    images = [np.zeros((4, 4, 3), np.uint8)]
    if fail:
        with pytest.raises(RuntimeError):
            rec.reconstruct(images)
    else:
        result = rec.reconstruct(images)
        assert len(result.processed_images) == 1
    assert rec.created_dirs
    assert not any(os.path.exists(d) for d in rec.created_dirs)


def test_builtin_optimizers_do_not_claim_result_postprocessing():
    from pyslam.scene_from_views.optimizers import (
        DenseSceneOptimizer,
        SceneOptimizerBase,
        SparseSceneOptimizer,
    )

    result = SceneFromViewsResult()
    assert not DenseSceneOptimizer(device="cpu").can_optimize_from_result(result)
    assert not SparseSceneOptimizer(device="cpu").can_optimize_from_result(result)

    class _PostProcessingOptimizer(DenseSceneOptimizer):
        def optimize_from_result(self, result, processed_images=None, verbose=True, **kw):
            return {"scene": None}

    assert isinstance(_PostProcessingOptimizer(device="cpu"), SceneOptimizerBase)
    assert _PostProcessingOptimizer(device="cpu").can_optimize_from_result(result)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "type_name, module_name, class_name, default_type",
    [
        ("MAST3R", "scene_from_views_mast3r", "SceneFromViewsMast3r", "sparse_scene_optimizer"),
        ("DUST3R", "scene_from_views_dust3r", "SceneFromViewsDust3r", "dense_scene_optimizer"),
    ],
)
def test_factory_accepts_optimizer_config(
    monkeypatch, type_name, module_name, class_name, default_type
):
    from pyslam.scene_from_views import scene_from_views_factory, SceneFromViewsType

    module = _import_or_skip(f"pyslam.scene_from_views.{module_name}")
    captured = {}

    class _Fake:
        def __init__(self, device=None, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(module, class_name, _Fake)
    stype = SceneFromViewsType[type_name]

    scene_from_views_factory(stype, device="cpu")
    assert captured["optimizer_config"]["type"] == default_type

    captured.clear()
    custom = {"type": "dense_scene_optimizer", "niter": 7}
    # used to raise "got multiple values for keyword argument 'optimizer_config'"
    scene_from_views_factory(stype, device="cpu", optimizer_config=custom)
    assert captured["optimizer_config"] is custom


# ---------------------------------------------------------------------------
# Sparse optimizer: kinematic chain initialization and root gauge
# ---------------------------------------------------------------------------


def _anchors_for(h, w, subsample):
    """Anchor pixels at the center of each subsample block (as condense_data would produce)."""
    ys, xs = np.mgrid[subsample // 2 : h : subsample, subsample // 2 : w : subsample]
    n = ys.size
    uv1 = torch.tensor(np.stack([xs.ravel(), ys.ravel(), np.ones(n)], axis=-1), dtype=torch.float32)
    idxs = torch.arange(n)
    offsets = torch.ones(n)
    return uv1, idxs, offsets


def test_sparse_optimizer_initial_poses_match_input_poses(monkeypatch):
    """With niter=0, the optimizer output must reproduce the initial absolute poses.

    Before the fix the root camera was forced to the origin while the other cameras were chained
    from the root *anchor* frame, which displaced cameras by ~ the median scene depth.
    """
    from pyslam.scene_from_views.helpers import sparse_ga
    from pyslam.scene_from_views.helpers.sparse_ga import PairOfSlices

    n, W, H, subsample = 3, 32, 24, 8
    root = 1
    abs_poses = np.stack(
        [
            _pose(0.05, -0.2, 0.03, [-0.6, 0.1, 0.2]),
            _pose(0.10, 0.15, -0.05, [0.3, -0.2, 0.4]),
            _pose(-0.1, 0.25, 0.08, [0.9, 0.05, -0.1]),
        ]
    )

    def _fake_extract(*args, **kwargs):
        return None, None, torch.tensor(abs_poses, dtype=torch.float32)

    monkeypatch.setattr(sparse_ga, "_extract_initial_poses_from_pairwise", _fake_extract)

    imgs = [f"img{i}" for i in range(n)]
    imsizes = torch.tensor([[W, H]] * n)
    pps = torch.tensor([[14.0, 13.0], [16.5, 11.0], [15.0, 12.5]])  # off-center on purpose
    base_focals = torch.tensor([30.0, 28.0, 32.0])
    H2, W2 = H // subsample, W // subsample
    core_depth = [
        torch.tensor(_depth_map(H2, W2, base=1.5 + 0.5 * i), dtype=torch.float32)
        for i in range(n)
    ]
    anchors = {i: _anchors_for(H, W, subsample) for i in range(n)}
    # an empty correspondence set (no matches) must not crash matching_check()
    empty_slice = PairOfSlices(
        0, slice(0, 0), torch.zeros(0, 2), torch.zeros(0, dtype=torch.long),
        1, slice(0, 0), torch.zeros(0, 2), torch.zeros(0, dtype=torch.long),
        torch.zeros(0), 0.0,
    )  # fmt: skip
    corres = (torch.zeros(0), 0.0, [empty_slice])
    corres2d = [(i, torch.zeros(0, 2), torch.zeros(0), 0.0, []) for i in range(n)]
    mst = (root, [(1, 0), (1, 2)])

    _, res_coarse, res_fine = sparse_ga.sparse_scene_optimizer(
        imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d,
        preds_21={}, canonical_paths=[], mst=mst, cache_path=None,
        niter1=0, niter2=0, device="cpu", dtype=torch.float32,
        pairs_output={"unused": None}, verbose=False,
    )  # fmt: skip
    assert res_fine is None
    cam2w = res_coarse["cam2w"].double().numpy()

    # root camera has identity rotation (gauge), and all poses relative to the root are preserved
    np.testing.assert_allclose(cam2w[root, :3, :3], np.eye(3), atol=1e-5)
    inv_root_in = np.linalg.inv(abs_poses[root])
    inv_root_out = np.linalg.inv(cam2w[root])
    for i in range(n):
        np.testing.assert_allclose(
            inv_root_out @ cam2w[i], inv_root_in @ abs_poses[i], atol=1e-4, err_msg=f"camera {i}"
        )


def test_initialize_kinematic_chain_root_is_identity():
    from pyslam.scene_from_views.helpers.sparse_ga import initialize_kinematic_chain_from_abs_poses

    n = 4
    abs_poses = torch.tensor(
        np.stack([_pose(0.1 * i, -0.05 * i, 0.02 * i, [0.3 * i, 0.1, -0.2 * i]) for i in range(n)]),
        dtype=torch.float64,
    )
    quats = [torch.nn.Parameter(torch.zeros(4, dtype=torch.float64)) for _ in range(n)]
    trans = [torch.nn.Parameter(torch.ones(3, dtype=torch.float64)) for _ in range(n)]
    mst = (2, [(2, 1), (1, 0), (2, 3)])
    dists = initialize_kinematic_chain_from_abs_poses(
        abs_poses,
        mst,
        median_depths=torch.full((n,), 2.0, dtype=torch.float64),
        imsizes=torch.tensor([[32, 24]] * n),
        base_focals=torch.full((n,), 30.0, dtype=torch.float64),
        pps=torch.full((n, 2), 0.5, dtype=torch.float64),
        quats=quats,
        trans=trans,
    )
    np.testing.assert_allclose(quats[2].detach().numpy(), [0, 0, 0, 1])
    np.testing.assert_allclose(trans[2].detach().numpy(), [0, 0, 0])
    centers = abs_poses[:, :3, 3]
    np.testing.assert_allclose(dists.numpy(), (centers - centers[2]).norm(dim=-1).numpy())


# ---------------------------------------------------------------------------
# Sparse optimizer: densification uses the optimization subsample
# ---------------------------------------------------------------------------


def test_sparse_ga_dense_points_use_optimization_subsample(monkeypatch, tmp_path):
    from pyslam.scene_from_views.helpers import sparse_ga
    from pyslam.scene_from_views.helpers.canonical_views import CanonicalViewUtils
    from pyslam.scene_from_views.helpers.geometry import ProjectionUtils

    H, W = 8, 8
    path = str(tmp_path / "canon.pth")
    canon = torch.ones(H, W, 3)
    torch.save(((canon, torch.ones(H, W), torch.ones(H, W)), torch.tensor(10.0)), path)

    scene = object.__new__(sparse_ga.SparseGA)
    scene.canonical_paths = [path]
    scene.subsample = 4
    scene.cam2w = torch.eye(4)[None]
    scene.intrinsics = torch.eye(3)[None]
    scene.depthmaps = [torch.ones(2, 2)]

    used = []

    def _fake_offsets(canon_depth, pixels, subsample=8):
        used.append(subsample)
        return {0: torch.zeros(1, dtype=torch.long)}, {0: torch.ones(1)}

    def _fake_make_pts3d(*args, **kwargs):
        return [torch.zeros(1, 3)], [torch.ones(1)]

    monkeypatch.setattr(CanonicalViewUtils, "anchor_depth_offsets", staticmethod(_fake_offsets))
    monkeypatch.setattr(ProjectionUtils, "make_pts3d", staticmethod(_fake_make_pts3d))

    scene.get_dense_pts3d(clean_depth=False)
    assert used == [4]  # used to be the hard-coded default 8
    scene.get_dense_pts3d(clean_depth=False, subsample=2)
    assert used == [4, 2]


# ---------------------------------------------------------------------------
# Canonical views
# ---------------------------------------------------------------------------


def _save_pair(cache_dir, name, X1, C1, X2, C2):
    path = os.path.join(cache_dir, name)
    torch.save((X1, C1, X2, C2), path)
    return path


def _save_corres(cache_dir, name, n, H, W, seed):
    g = torch.Generator().manual_seed(seed)
    xy1 = torch.stack([torch.randint(0, W, (n,), generator=g), torch.randint(0, H, (n,), generator=g)], 1)
    xy2 = torch.stack([torch.randint(0, W, (n,), generator=g), torch.randint(0, H, (n,), generator=g)], 1)
    confs = torch.rand(n, generator=g) + 1.5
    path = os.path.join(cache_dir, name)
    torch.save(((1.0, float(confs.sum()), n), (xy1.float(), xy2.float(), confs)), path)
    return path


def test_prepare_canonical_data_ignores_wrong_frame_and_verbose(tmp_path):
    """A pair without reverse prediction must not inject points in the wrong frame."""
    from pyslam.scene_from_views.helpers.canonical_views import CanonicalViewUtils

    H = W = 16
    f, subsample = 20.0, 8
    poses = {
        "a": np.eye(4),
        "b": _pose(0.0, 0.2, 0.0, [0.5, 0.0, 0.0]),
        "c": _pose(0.1, -0.2, 0.0, [-0.4, 0.1, 0.1]),
    }
    depths = {k: _depth_map(H, W, base=2.0 + i, phase=i) for i, k in enumerate("abc")}

    def in_frame(src, ref):
        """pointmap of image src expressed in the camera frame of image ref"""
        rel = np.linalg.inv(poses[ref]) @ poses[src]
        return torch.tensor(_pointmap(depths[src], f, W / 2, H / 2, rel), dtype=torch.float32)

    conf = torch.full((H, W), 2.0)
    cache = str(tmp_path)
    pairs = {}
    for i, j, has_reverse in [("a", "b", True), ("b", "c", True), ("a", "c", False)]:
        p1 = _save_pair(cache, f"{i}{j}.pth", in_frame(i, i), conf, in_frame(j, i), conf)
        p2 = (
            _save_pair(cache, f"{j}{i}.pth", in_frame(j, j), conf, in_frame(i, j), conf)
            if has_reverse
            else None
        )
        pc = _save_corres(cache, f"corres_{i}{j}.pth", 6, H, W, seed=len(pairs))
        pairs[(i, j)] = ((p1, p2), pc)

    imgs = ["a", "b", "c"]
    tmp_pairs, scores, canonical_views, canonical_paths, preds_21 = (
        CanonicalViewUtils.prepare_canonical_data(
            imgs, pairs, subsample, cache_path=cache, mode="avg-angle", device="cpu", verbose=True
        )
    )
    # verbose must not be part of the cache key (it prevented cache reuse)
    assert all("verbose" not in os.path.basename(p) for p in canonical_paths)

    # image c has a single prediction in its own frame: (b, c) reversed. The (a, c) prediction
    # lives in a's frame and must be ignored, so the canonical view of c equals its own pointmap.
    (canon_c, _, _), _ = torch.load(canonical_paths[2])
    np.testing.assert_allclose(canon_c.numpy(), in_frame("c", "c").numpy(), atol=1e-5)

    # downstream condensing must work with the missing reverse pair
    CanonicalViewUtils.condense_data(imgs, tmp_pairs, canonical_views, preds_21)


# ---------------------------------------------------------------------------
# DUSt3R
# ---------------------------------------------------------------------------


def _dust3r_module():
    return _import_or_skip("pyslam.scene_from_views.scene_from_views_dust3r")


def test_dust3r_no_fabricated_correspondences(monkeypatch, tmp_path):
    """When descriptor and 3D-NN matching both fail, no identity matches must be produced."""
    mod = _dust3r_module()
    from pyslam.scene_from_views.helpers.canonical_views import CanonicalViewUtils

    # make every matcher unavailable
    monkeypatch.setitem(sys.modules, "mast3r.fast_nn", types.ModuleType("mast3r.fast_nn"))

    H = W = 16
    f = 20.0
    poses = [np.eye(4), _pose(0.0, 0.15, 0.0, [0.4, 0.0, 0.0])]
    depths = [_depth_map(H, W, 2.0), _depth_map(H, W, 2.5, 1.0)]

    def in_frame(src, ref):
        rel = np.linalg.inv(poses[ref]) @ poses[src]
        return torch.tensor(_pointmap(depths[src], f, W / 2, H / 2, rel), dtype=torch.float32)

    conf = torch.full((H, W), 3.0)
    img = torch.zeros(1, 3, H, W)
    raw_output = {
        "view1": {"idx": [0, 1], "img": [img, img]},
        "view2": {"idx": [1, 0], "img": [img, img]},
        "pred1": {"pts3d": [in_frame(0, 0), in_frame(1, 1)], "conf": [conf, conf]},
        "pred2": {"pts3d_in_other_view": [in_frame(1, 0), in_frame(0, 1)], "conf": [conf, conf]},
    }
    rec = object.__new__(mod.SceneFromViewsDust3r)
    rec.device = torch.device("cpu")
    opt_input = rec.create_optimizer_input(
        raw_output,
        pairs=[],
        processed_images=[{"img": img}, {"img": img}],
        filelist=["image_0", "image_1"],
        cache_dir=str(tmp_path),
        subsample=8,
    )
    assert opt_input.pairs_output
    for (_, _), corres_path in opt_input.pairs_output.values():
        _, (xy1, xy2, confs) = torch.load(corres_path)
        assert len(xy1) == len(xy2) == len(confs) == 0

    # the canonical pipeline must accept pairs without correspondences
    tmp_pairs, _, canonical_views, _, preds_21 = CanonicalViewUtils.prepare_canonical_data(
        opt_input.filelist, opt_input.pairs_output, 8, cache_path=str(tmp_path), device="cpu"
    )
    CanonicalViewUtils.condense_data(opt_input.filelist, tmp_pairs, canonical_views, preds_21)


class _FakeDenseScene:
    def __init__(self, pts_world, imgs, confs, poses, focals):
        self._pts = [torch.tensor(p, dtype=torch.float32) for p in pts_world]
        self.imgs = imgs
        self.im_conf = [torch.tensor(c, dtype=torch.float32) for c in confs]
        self._poses = torch.tensor(np.stack(poses), dtype=torch.float32)
        self._focals = torch.tensor(focals, dtype=torch.float32)

    def get_pts3d(self):
        return self._pts

    def get_im_poses(self):
        return self._poses

    def get_focals(self):
        return self._focals


@pytest.mark.parametrize("min_conf_thr", [0.0, 100.0])
def test_dust3r_depth_in_camera_frame(min_conf_thr):
    mod = _dust3r_module()
    H, W, f = 12, 16, 20.0
    poses = [np.eye(4), _pose(0.1, 0.3, -0.1, [0.5, -0.2, 0.3])]
    depths = [_depth_map(H, W, 2.0), _depth_map(H, W, 3.0, 0.5)]
    pts = [_pointmap(d, f, W / 2, H / 2, T) for d, T in zip(depths, poses)]
    rng = np.random.default_rng(0)
    confs = [1.0 + rng.random((H, W)) for _ in poses]
    imgs = [np.full((H, W, 3), 0.5, np.float32) for _ in poses]
    scene = _FakeDenseScene(pts, imgs, confs, poses, [f, f])

    rec = object.__new__(mod.SceneFromViewsDust3r)
    result = rec.postprocess_results(
        {"scene": scene, "min_conf_thr": min_conf_thr, "verbose": False},
        images=[None, None],
        processed_images=[],
    )
    # with the 100th percentile only the max-confidence point survives (used to index out of range)
    assert sum(int((d > 0).sum()) for d in result.depth_predictions) >= 1
    for i, depth in enumerate(result.depth_predictions):
        valid = depth > 0
        if min_conf_thr == 0.0:
            assert valid.all()
        np.testing.assert_allclose(depth[valid], depths[i][valid], atol=1e-4)


# ---------------------------------------------------------------------------
# MV-DUSt3R
# ---------------------------------------------------------------------------


def test_mvdust3r_view_order_roundtrip():
    mod = _import_or_skip("pyslam.scene_from_views.scene_from_views_mvdust3r")
    for n in range(1, 40):
        order = mod.compute_mvdust3r_view_order(n)
        assert sorted(order) == list(range(n))
        assert order[0] == 0  # the first view stays the reference
        permuted = [f"v{k}" for k in order]
        assert mod.restore_view_order(permuted, order) == [f"v{k}" for k in range(n)]
    assert mod.compute_mvdust3r_view_order(5)[:4] == [0, 3, 2, 1]


def test_mvdust3r_outputs_follow_input_order(monkeypatch):
    """End-to-end infer/postprocess with a fake model: outputs must be in the input order."""
    mod = _import_or_skip("pyslam.scene_from_views.scene_from_views_mvdust3r")

    n, h, w, f = 5, 32, 32, 30.0
    poses = [np.eye(4)] + [
        _pose(0.05 * i, -0.08 * i, 0.03 * i, [0.2 * i, -0.05 * i, 0.1 * i]) for i in range(1, n)
    ]
    depths = [_depth_map(h, w, 2.0 + 0.2 * i, 0.3 * i) for i in range(n)]
    # all pointmaps are in the frame of the first view (poses[0] = I)
    pts = [_pointmap(depths[i], f, w / 2, h / 2, poses[i]) for i in range(n)]
    colors = np.linspace(-0.8, 0.8, n)

    processed = []
    for i in range(n):
        img = torch.full((1, 3, h, w), float(colors[i]))
        processed.append(
            {"img": img, "true_shape": np.array([[h, w]]), "idx": i, "instance": str(i)}
        )
    order_before = [p["idx"] for p in processed]

    def _fake_inference_mv(imgs, model, device, verbose=True):
        def pred(idx, key):
            return {
                key: torch.tensor(pts[idx][None], dtype=torch.float32),
                "conf": torch.full((1, h, w), 2.0),
            }

        return {
            "pred1": pred(imgs[0]["idx"], "pts3d"),
            "pred2s": [pred(im["idx"], "pts3d_in_other_view") for im in imgs[1:]],
        }

    fake_inference = types.ModuleType("mvdust3r.dust3r.inference")
    fake_inference.inference_mv = _fake_inference_mv
    monkeypatch.setitem(sys.modules, "mvdust3r", types.ModuleType("mvdust3r"))
    monkeypatch.setitem(sys.modules, "mvdust3r.dust3r", types.ModuleType("mvdust3r.dust3r"))
    monkeypatch.setitem(sys.modules, "mvdust3r.dust3r.inference", fake_inference)

    rec = object.__new__(mod.SceneFromViewsMvdust3r)
    rec.device = torch.device("cpu")
    rec.model = None
    rec.min_conf_thr = 0.0

    raw = rec.infer(processed)
    assert raw["view_order"] != list(range(n))  # the test must exercise a permutation
    # the caller's list and dicts are not modified
    assert [p["idx"] for p in processed] == order_before
    assert all(isinstance(p["true_shape"], np.ndarray) for p in processed)

    result = rec.postprocess_results(raw, images=[None] * n, processed_images=processed)
    for i in range(n):
        np.testing.assert_allclose(result.camera_poses[i], poses[i], atol=2e-3)
        np.testing.assert_allclose(result.depth_predictions[i], depths[i], atol=2e-3)
        expected_color = (colors[i] + 1) / 2 * 255
        assert abs(float(result.processed_images[i].mean()) - expected_color) <= 1.0


# ---------------------------------------------------------------------------
# VGGT
# ---------------------------------------------------------------------------


class _FakeVggtModel:
    """Returns VGGT-shaped predictions (batch dim B=1)."""

    def __init__(self, depth):
        self.depth = depth  # [S, H, W]

    def __call__(self, images):
        S, _, H, W = images.shape
        pose_enc = torch.zeros(1, S, 9)
        pose_enc[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        pose_enc[..., 2] = torch.arange(S, dtype=torch.float32)
        pose_enc[..., 7:9] = 1.0  # field of view
        depth = torch.tensor(self.depth, dtype=torch.float32)
        return {
            "pose_enc": pose_enc,
            "depth": depth[None, ..., None],  # [1, S, H, W, 1]
            "depth_conf": torch.ones(1, S, H, W) * 2,
            "world_points": torch.zeros(1, S, H, W, 3),
            "world_points_conf": torch.ones(1, S, H, W) * 2,
            "images": images[None],
        }


@pytest.mark.parametrize("num_views", [1, 2])
def test_vggt_infer_postprocess_shapes(num_views):
    mod = _import_or_skip("pyslam.scene_from_views.scene_from_views_vggt")
    _import_or_skip("vggt.utils.pose_enc")
    H = W = 28
    depth = np.stack([_depth_map(H, W, 2.0 + i) for i in range(num_views)])

    rec = object.__new__(mod.SceneFromViewsVggt)
    rec.device = torch.device("cpu")  # also checks that CPU inference works
    rec.model = _FakeVggtModel(depth)
    rec.conf_thres = 3.0
    rec.prediction_mode = "Depthmap Regression"
    rec.mask_black_bg = rec.mask_white_bg = rec.mask_sky = False

    images = torch.rand(num_views, 3, H, W)
    raw = rec.infer(images)
    result = rec.postprocess_results(raw, images=[None] * num_views, processed_images=images)

    assert len(result.camera_poses) == num_views
    assert all(np.asarray(T).shape == (4, 4) for T in result.camera_poses)
    for i in range(num_views):
        assert result.depth_predictions[i].shape == (H, W)
        np.testing.assert_allclose(result.depth_predictions[i], depth[i], atol=1e-5)


@pytest.mark.parametrize("num_views", [1, 2])
def test_vggt_robust_depth_shapes(num_views):
    mod = _import_or_skip("pyslam.scene_from_views.scene_from_views_vggt_robust")
    H = W = 14
    S = num_views
    depth = np.stack([_depth_map(H, W, 2.0 + i) for i in range(S)]).astype(np.float32)
    extrinsic = np.tile(np.eye(4)[:3][None], (S, 1, 1)).astype(np.float32)
    intrinsic = np.tile(np.eye(3)[None], (S, 1, 1)).astype(np.float32)
    predictions = {
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "depth": depth[..., None],  # [S, H, W, 1]
        "depth_conf": np.full((S, H, W), 2.0, np.float32),
        "world_points_from_depth": np.zeros((S, H, W, 3), np.float32),
        "images": np.random.rand(S, 3, H, W).astype(np.float32),
    }
    raw = {
        "predictions": predictions,
        "conf_thres": 3.0,
        "prediction_mode": "Depthmap Regression",
        "mask_black_bg": False,
        "mask_white_bg": False,
        "mask_sky": False,
    }
    rec = object.__new__(mod.SceneFromViewsVggtRobust)
    result = rec.postprocess_results(raw, images=[None] * S, processed_images=None)
    for i in range(S):
        assert result.depth_predictions[i].shape == (H, W)
        np.testing.assert_allclose(result.depth_predictions[i], depth[i])


# ---------------------------------------------------------------------------
# Fast3R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_pointcloud", [True, False])
def test_fast3r_geometry_depth_and_colors(monkeypatch, as_pointcloud):
    mod = _import_or_skip("pyslam.scene_from_views.scene_from_views_fast3r")
    lit = _import_or_skip("fast3r.models.multiview_dust3r_module")
    _import_or_skip("fast3r.dust3r.utils.image")

    H = W = 16
    f = 20.0
    poses = [np.eye(4), _pose(0.1, -0.2, 0.05, [0.4, 0.1, -0.2])]
    depths = [_depth_map(H, W, 2.0), _depth_map(H, W, 2.6, 1.0)]
    rgb = [np.array([0.8, 0.2, -0.4]), np.array([-0.6, 0.4, 0.9])]  # in [-1, 1]

    preds, views = [], []
    for T, d, c in zip(poses, depths, rgb):
        pts = _pointmap(d, f, W / 2, H / 2, T)
        conf = 2.0 + np.random.default_rng(0).random((H, W))
        preds.append(
            {
                "pts3d_local_aligned_to_global": torch.tensor(pts[None], dtype=torch.float32),
                "conf": torch.tensor(conf[None], dtype=torch.float32),
            }
        )
        img = torch.tensor(c, dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, H, W).clone()
        views.append({"img": img, "true_shape": np.array([[H, W]])})

    def _fake_estimate(preds_batched, views=None, niter_PnP=10, focal_length_estimation_method=""):
        return [poses], [[f, f]]

    monkeypatch.setattr(
        lit.MultiViewDUSt3RLitModule, "estimate_camera_poses", staticmethod(_fake_estimate)
    )

    rec = object.__new__(mod.SceneFromViewsFast3r)
    rec.min_conf_thr_percentile = 10
    raw = {
        "output_dict": {"preds": preds, "views": views},
        "niter_PnP": 10,
        "focal_length_estimation_method": "first_view_from_global_head",
        "min_conf_thr_percentile": 0,
    }
    result = rec.postprocess_results(raw, images=[], processed_images=[], as_pointcloud=as_pointcloud)

    if as_pointcloud:
        assert isinstance(result.global_point_cloud, trimesh.PointCloud)
        assert result.global_mesh is None
        global_colors = result.global_point_cloud.colors[:, :3]
        assert global_colors.shape[0] > 0
    else:
        # used to be swapped: the mesh ended up in global_point_cloud
        assert isinstance(result.global_mesh, trimesh.Trimesh)
        assert result.global_point_cloud is None

    for i, (d, c) in enumerate(zip(depths, rgb)):
        np.testing.assert_allclose(result.depth_predictions[i], d, atol=1e-4)
        # per-view clouds use the same BGR convention as the processed images / global cloud
        expected_bgr = ((c[::-1] + 1) * 127.5).astype(np.uint8)
        pc_colors = result.point_clouds[i].colors[:, :3]
        assert np.all(np.abs(pc_colors.astype(int) - expected_bgr.astype(int)) <= 1)
