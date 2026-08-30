#!/usr/bin/env python3
"""Estimate a SLAM-units -> meters scale from the camera height above the floor plane.

Monocular reconstructions are up-to-scale. When the camera rode at a known height
above a flat floor (e.g. a robot head camera), the dominant plane in the dense
points recovers that height in SLAM units, and the ratio gives a metric scale
usable with export_reconstruction_ply.py --metric-scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True, help="world_points.pt")
    parser.add_argument("--poses", type=Path, required=True, help="camera_poses*.npy")
    parser.add_argument("--confidence", type=Path, help="Optional confidence.pt")
    parser.add_argument(
        "--camera-height-m",
        type=float,
        required=True,
        help="True camera height above the floor in meters",
    )
    parser.add_argument("--ransac-iters", type=int, default=500)
    parser.add_argument("--inlier-threshold", type=float, default=0.01)
    parser.add_argument("--min-inlier-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dominant_plane(
    points: np.ndarray, iterations: int, threshold: float, seed: int
) -> tuple[np.ndarray, float, int]:
    rng = np.random.default_rng(seed)
    best_inliers = -1
    best_normal = None
    best_offset = 0.0
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        offset = -normal.dot(a)
        inliers = int((np.abs(points.dot(normal) + offset) < threshold).sum())
        if inliers > best_inliers:
            best_inliers, best_normal, best_offset = inliers, normal, offset
    if best_normal is None:
        raise SystemExit("Could not fit a plane: degenerate points")
    return best_normal, best_offset, best_inliers


def main() -> None:
    args = parse_args()
    if args.camera_height_m <= 0:
        raise SystemExit("--camera-height-m must be positive")

    points = torch.load(args.points, map_location="cpu", weights_only=True)
    points = points.reshape(-1, 3).numpy().astype(np.float64)
    valid = np.isfinite(points).all(axis=1)
    if args.confidence is not None:
        confidence = torch.load(args.confidence, map_location="cpu", weights_only=True)
        confidence = confidence.reshape(-1).numpy()
        valid &= confidence > np.percentile(confidence[np.isfinite(confidence)], 50)
    points = points[valid]
    if len(points) < 1000:
        raise SystemExit(f"Too few valid points to fit a plane: {len(points)}")

    poses = np.asarray(np.load(args.poses), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise SystemExit(f"Expected poses with shape [N,4,4], got {poses.shape}")
    centers = poses[:, :3, 3]

    normal, offset, inliers = dominant_plane(
        points, args.ransac_iters, args.inlier_threshold, args.seed
    )
    fraction = inliers / len(points)
    heights = np.abs(centers.dot(normal) + offset)
    scale = args.camera_height_m / heights.mean()

    print(f"plane inliers: {inliers:,}/{len(points):,} ({fraction:.1%})")
    print(
        "camera height above plane (SLAM units): "
        f"mean {heights.mean():.4f}, min {heights.min():.4f}, max {heights.max():.4f}"
    )
    print(f"metric scale: {scale:.6f}")
    if fraction < args.min_inlier_fraction:
        raise SystemExit(
            f"Dominant plane holds only {fraction:.1%} of points "
            f"(< {args.min_inlier_fraction:.0%}); scale estimate is unreliable"
        )


if __name__ == "__main__":
    main()
