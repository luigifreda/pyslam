import numpy as np
import pytest
import sys
import os

# Add lib directory to path to import the volumetric module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

volumetric = pytest.importorskip("volumetric")


def _zeros_points(n):
    return np.zeros((n, 3), dtype=np.float64)


def _zeros_colors(n):
    return np.zeros((n, 3), dtype=np.uint8)


def test_voxel_semantic_data_label_switch_and_confidence():
    """The simple voting voxel should switch label when confidence drops to zero."""
    grid = volumetric.VoxelSemanticGrid(0.1)

    points = _zeros_points(2)
    colors = _zeros_colors(2)
    class_ids = np.array([1, 2], dtype=np.int32)
    instance_ids = np.array([1, 2], dtype=np.int32)

    grid.integrate(points, colors, class_ids, instance_ids)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    assert len(voxels.object_ids) == 1
    assert voxels.object_ids[0] == 2
    assert voxels.class_ids[0] == 2
    # After two conflicting observations, confidence counter is 1 over count 2.
    assert voxels.confidences[0] == pytest.approx(0.5, abs=1e-3)


def test_probabilistic_voxel_favors_majority_label():
    """Probabilistic voxel should pick the most frequent label."""
    grid = volumetric.VoxelSemanticGridProbabilistic(0.1)

    points = _zeros_points(4)
    colors = _zeros_colors(4)
    class_ids = np.array([5, 5, 5, 6], dtype=np.int32)
    instance_ids = np.array([1, 1, 1, 2], dtype=np.int32)

    grid.integrate(points, colors, class_ids, instance_ids)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    assert len(voxels.object_ids) == 1
    assert voxels.object_ids[0] == 1
    assert voxels.class_ids[0] == 5
    # Majority label should have probability > 0.5
    assert voxels.confidences[0] > 0.5


def test_probabilistic_depth_decay_downweights_far_observations():
    """A far observation with different label should not override a close one."""
    grid = volumetric.VoxelSemanticGridProbabilistic(0.1)

    points = _zeros_points(2)
    colors = _zeros_colors(2)
    class_ids = np.array([7, 8], dtype=np.int32)
    instance_ids = np.array([3, 4], dtype=np.int32)
    depths = np.array([1.0, 20.0], dtype=np.float32)  # second point is far

    grid.integrate(points, colors, class_ids, instance_ids, depths)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    assert len(voxels.object_ids) == 1
    # Near observation should dominate despite the conflicting far observation.
    assert voxels.object_ids[0] == 3
    assert voxels.class_ids[0] == 7
    # Near observation should keep the highest probability; expect majority > 0.5
    assert voxels.confidences[0] > 0.5


def test_semantic_grid_preserves_labels_across_voxels():
    """Two spatially separated voxels should keep their own labels."""
    grid = volumetric.VoxelSemanticGrid(0.1)

    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float64)
    colors = _zeros_colors(2)
    class_ids = np.array([10, 20], dtype=np.int32)
    instance_ids = np.array([101, 202], dtype=np.int32)

    grid.integrate(points, colors, class_ids, instance_ids)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    paired = sorted(
        zip(voxels.points, voxels.object_ids, voxels.class_ids),
        key=lambda x: tuple(x[0]),
    )
    assert len(paired) == 2
    assert paired[0][1:] == (101, 10)
    assert paired[1][1:] == (202, 20)


def test_probabilistic_voxel_strong_majority_probability():
    """Strong majority of one label should yield high confidence for that label."""
    grid = volumetric.VoxelSemanticGridProbabilistic(0.1)

    majority_points = 12
    minority_points = 1
    total = majority_points + minority_points

    points = _zeros_points(total)
    colors = _zeros_colors(total)
    class_ids = np.array([5] * majority_points + [6], dtype=np.int32)
    instance_ids = np.array([1] * majority_points + [2], dtype=np.int32)

    grid.integrate(points, colors, class_ids, instance_ids)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    assert len(voxels.object_ids) == 1
    assert voxels.object_ids[0] == 1
    assert voxels.class_ids[0] == 5
    # With 12:1 evidence, probability should be well above 0.7
    assert voxels.confidences[0] > 0.7


def test_probabilistic_with_label_noise():
    """Uniform random points with label noise should remain robust."""
    rng = np.random.default_rng(0)
    grid = volumetric.VoxelSemanticGridProbabilistic(0.2)

    majority = 50
    noise = 5
    total = majority + noise

    # Keep all points in one voxel: sample inside a small cube
    points = rng.uniform(low=0.0, high=0.05, size=(total, 3)).astype(np.float64)
    colors = _zeros_colors(total)

    class_ids = np.array([11] * majority + [12] * noise, dtype=np.int32)
    instance_ids = np.array([111] * majority + [222] * noise, dtype=np.int32)

    # Shuffle to mix labels
    perm = rng.permutation(total)
    points = points[perm]
    colors = colors[perm]
    class_ids = class_ids[perm]
    instance_ids = instance_ids[perm]

    grid.integrate(points, colors, class_ids, instance_ids)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    assert len(voxels.object_ids) == 1
    assert voxels.object_ids[0] == 111
    assert voxels.class_ids[0] == 11
    # Probability should remain high despite 10% noise
    assert voxels.confidences[0] > 0.75


def test_voting_with_label_noise():
    """Voting voxel should keep majority label under noise."""
    rng = np.random.default_rng(1)
    grid = volumetric.VoxelSemanticGrid(0.2)

    majority = 30
    noise = 3
    total = majority + noise

    points = rng.uniform(low=0.0, high=0.05, size=(total, 3)).astype(np.float64)
    colors = _zeros_colors(total)

    class_ids = np.array([21] * majority + [22] * noise, dtype=np.int32)
    instance_ids = np.array([210] * majority + [220] * noise, dtype=np.int32)

    perm = rng.permutation(total)
    points = points[perm]
    colors = colors[perm]
    class_ids = class_ids[perm]
    instance_ids = instance_ids[perm]

    grid.integrate(points, colors, class_ids, instance_ids)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    assert len(voxels.object_ids) == 1
    assert voxels.object_ids[0] == 210
    assert voxels.class_ids[0] == 21
    # Voting confidence is counter/count = (majority - noise)/total
    expected_conf = (majority - noise) / float(total)
    assert voxels.confidences[0] == pytest.approx(expected_conf, abs=1e-2)


def test_probabilistic_joint_distribution_beats_marginals():
    """Joint label distribution should prefer the most observed pair, not just marginals."""
    grid = volumetric.VoxelSemanticGridProbabilistic(0.1)

    # Build observations where marginal winners differ from the joint winner:
    # (object=1, class=10) -> 3
    # (object=1, class=11) -> 3
    # (object=2, class=10) -> 4  (joint winner)
    pair_counts = {(1, 10): 3, (1, 11): 3, (2, 10): 4}
    total = sum(pair_counts.values())

    points = _zeros_points(total)
    colors = _zeros_colors(total)
    class_ids = []
    instance_ids = []
    for (obj_id, cls_id), count in pair_counts.items():
        instance_ids.extend([obj_id] * count)
        class_ids.extend([cls_id] * count)

    class_ids = np.array(class_ids, dtype=np.int32)
    instance_ids = np.array(instance_ids, dtype=np.int32)

    # Shuffle to avoid ordering bias
    rng = np.random.default_rng(42)
    perm = rng.permutation(total)
    points = points[perm]
    colors = colors[perm]
    class_ids = class_ids[perm]
    instance_ids = instance_ids[perm]

    grid.integrate(points, colors, class_ids, instance_ids)
    voxels = grid.get_voxels(min_count=1, min_confidence=0.0)

    assert len(voxels.object_ids) == 1
    assert voxels.object_ids[0] == 2
    assert voxels.class_ids[0] == 10

    # Confidence should match the softmax over log-evidence.
    base_log = 0.10536051565782628  # VoxelSemanticDataProbabilistic::BASE_LOG_PROB_PER_OBSERVATION
    log_probs = np.array([4 * base_log, 3 * base_log, 3 * base_log], dtype=np.float64)
    expected_prob = np.exp(log_probs[0]) / np.exp(log_probs).sum()
    assert voxels.confidences[0] == pytest.approx(expected_prob, rel=1e-4, abs=1e-4)


if __name__ == "__main__":
    test_voxel_semantic_data_label_switch_and_confidence()
    test_probabilistic_voxel_favors_majority_label()
    test_probabilistic_depth_decay_downweights_far_observations()
    test_semantic_grid_preserves_labels_across_voxels()
    test_probabilistic_voxel_strong_majority_probability()
    test_probabilistic_joint_distribution_beats_marginals()
    print("All tests passed!")
