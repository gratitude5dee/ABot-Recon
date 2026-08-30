#!/usr/bin/env python3
"""Estimate a SLAM-units -> meters scale from the camera height above the floor plane.

Monocular reconstructions are up-to-scale. When the camera rode at a known height
above a flat floor (e.g. a robot head camera), the floor plane in the dense
points recovers that height in SLAM units, and the ratio gives a metric scale
usable with export_reconstruction_ply.py --metric-scale.

The floor is not assumed to be the largest plane: walls can hold more points.
Among the top RANSAC candidates, the floor is the plane the cameras keep an
essentially constant distance from across the whole trajectory (the camera
rode at fixed height); planes whose camera distances vary too much are
rejected, and the run fails if no candidate is consistent.
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
    parser.add_argument(
        "--max-height-cv",
        type=float,
        default=0.05,
        help="Max coefficient of variation of camera-to-plane distance for a "
        "candidate plane to count as the floor",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def candidate_planes(
    points: np.ndarray,
    iterations: int,
    threshold: float,
    seed: int,
    max_candidates: int = 8,
) -> list[tuple[np.ndarray, float, int]]:
    """Top RANSAC planes as (normal, offset, inliers), best-supported first.

    Near-duplicates (parallel normals at nearly the same offset) collapse into
    the better-supported hypothesis so distinct surfaces each get one slot.
    """
    rng = np.random.default_rng(seed)
    candidates: list[tuple[np.ndarray, float, int]] = []
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        offset = -normal.dot(a)
        inliers = int((np.abs(points.dot(normal) + offset) < threshold).sum())
        for i, (n, d, existing) in enumerate(candidates):
            if abs(normal.dot(n)) > 0.995 and abs(abs(offset) - abs(d)) < 3 * threshold:
                if inliers > existing:
                    candidates[i] = (normal, offset, inliers)
                break
        else:
            candidates.append((normal, offset, inliers))
    if not candidates:
        raise SystemExit("Could not fit a plane: degenerate points")
    # In-place replacement can drift a slot toward a neighbour, so merge
    # near-duplicates once more, best-supported first.
    candidates.sort(key=lambda item: -item[2])
    merged: list[tuple[np.ndarray, float, int]] = []
    for normal, offset, inliers in candidates:
        if any(
            abs(normal.dot(n)) > 0.995 and abs(abs(offset) - abs(d)) < 3 * threshold
            for n, d, _ in merged
        ):
            continue
        merged.append((normal, offset, inliers))
    return merged[:max_candidates]


def select_floor_plane(
    candidates: list[tuple[np.ndarray, float, int]],
    centers: np.ndarray,
    max_height_cv: float,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """Pick the candidate the cameras stay a constant distance from.

    A fixed-height camera keeps a near-constant distance to the floor but a
    varying distance to walls it moves along; of the candidates whose distance
    spread stays under `max_height_cv`, the best-supported wins, and none
    qualifying is a hard failure.
    """
    best = None
    best_cv = np.inf
    for normal, offset, inliers in candidates:
        heights = np.abs(centers.dot(normal) + offset)
        mean = heights.mean()
        if mean < 1e-9:
            continue
        cv = heights.std() / mean
        best_cv = min(best_cv, cv)
        if cv > max_height_cv:
            continue
        if best is None or inliers > best[2]:
            best = (normal, offset, inliers, heights)
    if best is None:
        raise SystemExit(
            "No candidate plane keeps a consistent camera distance "
            f"(best coefficient of variation {best_cv:.3f} > {max_height_cv:g}); "
            "cannot identify the floor reliably"
        )
    return best


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.camera_height_m) or args.camera_height_m <= 0:
        raise SystemExit("--camera-height-m must be a positive finite number")

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

    planes = candidate_planes(
        points, args.ransac_iters, args.inlier_threshold, args.seed
    )
    normal, offset, inliers, heights = select_floor_plane(
        planes, centers, args.max_height_cv
    )
    fraction = inliers / len(points)
    scale = args.camera_height_m / heights.mean()

    print(f"floor plane inliers: {inliers:,}/{len(points):,} ({fraction:.1%})")
    print(
        "camera height above plane (SLAM units): "
        f"mean {heights.mean():.4f}, min {heights.min():.4f}, max {heights.max():.4f}"
    )
    print(f"metric scale: {scale:.6f}")
    if fraction < args.min_inlier_fraction:
        raise SystemExit(
            f"Floor plane holds only {fraction:.1%} of points "
            f"(< {args.min_inlier_fraction:.0%}); scale estimate is unreliable"
        )


if __name__ == "__main__":
    main()
