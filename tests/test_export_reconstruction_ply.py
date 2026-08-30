import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script(name: str):
    script = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export = load_script("export_reconstruction_ply")
estimate = load_script("estimate_metric_scale")


def make_args(**overrides):
    defaults = dict(
        output=None,
        poses=None,
        points=None,
        colors=None,
        points_frame="auto",
        metadata=None,
        metric_scale=1.0,
        poses_output=None,
        confidence=None,
        confidence_threshold=None,
        point_stride=1,
        frame_stride=1,
        max_points=0,
        pose_stride=1,
        frustum_scale=0.15,
        bev_output=None,
        bev_size=1600,
        bev_plane="auto",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def identity_poses(count: int, spacing: float = 0.1) -> np.ndarray:
    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, 0, 3] = np.arange(count) * spacing
    return poses


def test_metric_scale_scales_world_points(tmp_path):
    maps = torch.arange(2 * 2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 2, 3)
    points_path = tmp_path / "world_points.pt"
    torch.save(maps, points_path)
    args = make_args(points=points_path, metric_scale=2.0)
    points, _ = export.prepare_points(args, None)
    expected, _ = export.prepare_points(make_args(points=points_path), None)
    np.testing.assert_allclose(points, expected * 2.0, rtol=1e-6)


def test_metric_scale_keeps_local_points_consistent_with_scaled_poses(tmp_path):
    maps = torch.rand(3, 2, 2, 3)
    points_path = tmp_path / "local_points.pt"
    torch.save(maps, points_path)
    poses = identity_poses(3)
    poses[:, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

    unscaled, _ = export.prepare_points(make_args(points=points_path), poses)

    scale = 3.5
    scaled_poses = poses.copy()
    scaled_poses[:, :3, 3] *= scale
    scaled, _ = export.prepare_points(
        make_args(points=points_path, metric_scale=scale), scaled_poses
    )
    np.testing.assert_allclose(scaled, unscaled * scale, rtol=1e-5)


def test_estimate_metric_scale_recovers_camera_height(tmp_path):
    rng = np.random.default_rng(1)
    floor = np.zeros((5000, 3))
    floor[:, :2] = rng.uniform(-1, 1, size=(5000, 2))
    clutter = rng.uniform(-1, 1, size=(500, 3)) + np.array([0, 0, 0.3])
    points = np.concatenate([floor, clutter]).astype(np.float64)
    normal, offset, inliers = estimate.dominant_plane(points, 200, 0.01, seed=0)
    assert inliers >= len(floor) * 0.95
    camera = np.array([[0.0, 0.0, 0.5]])
    height = np.abs(camera.dot(normal) + offset)[0]
    assert height == pytest.approx(0.5, abs=0.02)


def test_rejects_nonpositive_metric_scale(tmp_path, monkeypatch, capsys):
    poses_path = tmp_path / "camera_poses.npy"
    np.save(poses_path, identity_poses(2))
    output = tmp_path / "out.ply"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_reconstruction_ply.py",
            "--output", str(output),
            "--poses", str(poses_path),
            "--metric-scale", "0",
        ],
    )
    with pytest.raises(SystemExit):
        export.main()


def test_poses_output_writes_scaled_poses(tmp_path, monkeypatch):
    poses_path = tmp_path / "camera_poses.npy"
    np.save(poses_path, identity_poses(4, spacing=1.0))
    output = tmp_path / "out.ply"
    poses_out = tmp_path / "poses_m.npy"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_reconstruction_ply.py",
            "--output", str(output),
            "--poses", str(poses_path),
            "--metric-scale", "2.5",
            "--poses-output", str(poses_out),
        ],
    )
    export.main()
    written = np.load(poses_out)
    assert written.shape == (4, 4, 4)
    np.testing.assert_allclose(written[:, 0, 3], np.arange(4) * 2.5, rtol=1e-6)
    np.testing.assert_allclose(written[:, :3, :3], identity_poses(4)[:, :3, :3])
