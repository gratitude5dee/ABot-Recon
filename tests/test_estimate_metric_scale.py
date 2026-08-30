"""Floor selection in scripts/estimate_metric_scale.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "estimate_metric_scale.py"
spec = importlib.util.spec_from_file_location("estimate_metric_scale", SCRIPT)
ems = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ems
spec.loader.exec_module(ems)


def make_scene(rng, wall_points=6000, floor_points=2000):
    """A larger wall (x=2) and a smaller floor (z=0)."""
    floor = np.column_stack([
        rng.uniform(-2, 2, floor_points),
        rng.uniform(-2, 2, floor_points),
        rng.normal(0, 0.002, floor_points),
    ])
    wall = np.column_stack([
        np.full(wall_points, 2.0) + rng.normal(0, 0.002, wall_points),
        rng.uniform(-2, 2, wall_points),
        rng.uniform(0, 3, wall_points),
    ])
    return np.vstack([floor, wall])


def make_centers(n=20):
    """Cameras at fixed height 0.5, moving toward the wall (x varies)."""
    x = np.linspace(-1.5, 1.5, n)
    return np.column_stack([x, np.zeros(n), np.full(n, 0.5)])


def test_floor_beats_bigger_wall():
    rng = np.random.default_rng(3)
    points = make_scene(rng)
    centers = make_centers()
    planes = ems.candidate_planes(points, iterations=500, threshold=0.01, seed=0)
    # The wall really is the best-supported candidate...
    wall_normal = planes[0][0]
    assert abs(wall_normal[0]) > 0.99
    # ...but the floor is the plane the cameras keep a constant distance from.
    normal, _offset, _inliers, heights = ems.select_floor_plane(
        planes, centers, max_height_cv=0.05
    )
    assert abs(normal[2]) > 0.99
    assert heights.mean() == pytest.approx(0.5, abs=0.02)


def test_no_consistent_plane_is_a_hard_failure():
    rng = np.random.default_rng(3)
    # Wall only: camera distance to it varies with x, so nothing qualifies.
    points = make_scene(rng, wall_points=6000, floor_points=0)
    centers = make_centers()
    planes = ems.candidate_planes(points, iterations=500, threshold=0.01, seed=0)
    with pytest.raises(SystemExit, match="cannot identify the floor"):
        ems.select_floor_plane(planes, centers, max_height_cv=0.05)


def test_duplicate_hypotheses_collapse():
    rng = np.random.default_rng(3)
    points = make_scene(rng)
    planes = ems.candidate_planes(points, iterations=500, threshold=0.01, seed=0)
    # One slot per distinct surface: no two candidates are near-parallel at
    # nearly the same offset.
    for i, (n1, d1, _) in enumerate(planes):
        for n2, d2, _ in planes[i + 1:]:
            assert not (abs(n1.dot(n2)) > 0.995 and abs(abs(d1) - abs(d2)) < 0.03)
