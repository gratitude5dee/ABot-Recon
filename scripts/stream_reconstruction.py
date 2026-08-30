#!/usr/bin/env python3
"""Rolling-window ABot-Recon streamer for God's Eye View.

``infer()`` runs one full causal pass and returns at the end — it is not an
incremental generator — so "streaming" means: watch the directory a recorder
(``record_g1_camera.py``, possibly rsync'd off the Orin) is filling with
zero-padded JPEGs, and every time new frames arrive reconstruct the most
recent window of them. After each window the streamer writes, into
``--output-dir``:

* ``pose_latest.json`` — the newest camera pose in the reconstruction's SLAM
  frame, rewritten atomically (tmp file + rename) so a reader never sees a
  torn record. This is the file the GEV bridge provider
  (``tools/robot-bridge/providers/unitree.mjs``) polls.
* ``points/batch_<seq>.ply`` — that window's decimated colored point chunk in
  the same binary PLY layout ``export_reconstruction_ply.py`` writes, small
  enough for a browser to fetch directly.

Read-only by construction: the only input is a directory of image files. This
process never talks to the robot.

Usage (off-board CUDA box, or on-Orin with ``--attention-backend sdpa``)::

    python scripts/stream_reconstruction.py \
        --image-dir /data/g1-frames --output-dir outputs/g1-live \
        --window 24 --min-new-frames 4 --attention-backend auto

Stop with Ctrl-C; SIGINT/SIGTERM finish the current window and exit cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_ID = "acvlab/ABot-Recon"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/g1-live"))
    parser.add_argument("--checkpoint", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention-backend", choices=("auto", "paged", "sdpa"), default="auto"
    )
    parser.add_argument(
        "--window", type=int, default=24, help="frames reconstructed per pass"
    )
    parser.add_argument(
        "--min-new-frames",
        type=int,
        default=4,
        help="new frames required before another pass runs",
    )
    parser.add_argument(
        "--point-stride", type=int, default=8, help="keep every Nth point per chunk"
    )
    parser.add_argument(
        "--max-points", type=int, default=200_000, help="hard cap per chunk"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=1.0, help="seconds between dir scans"
    )
    parser.add_argument(
        "--once", action="store_true", help="run a single window and exit (testing)"
    )
    return parser


def list_frames(image_dir: Path) -> list[Path]:
    """Sorted image files; sorted order is capture order for zero-padded names."""
    if not image_dir.is_dir():
        return []
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
    )


def select_window(
    frames: list[Path], window: int, last_newest: Path | None, min_new: int
) -> list[Path] | None:
    """The trailing window to reconstruct, or None when too little is new.

    ``last_newest`` is the newest frame of the previous pass; a pass runs only
    once at least ``min_new`` frames arrived after it, so a stalled recorder
    does not burn GPU re-reconstructing the same clip.
    """
    if len(frames) < 2:
        return None
    if last_newest is not None:
        try:
            newness = len(frames) - 1 - frames.index(last_newest)
            if newness < max(1, min_new):
                return None
        except ValueError:
            pass  # the recorder restarted; everything is new
    return frames[-window:] if window > 0 else frames


def pose_record_from_matrix(matrix: np.ndarray) -> dict:
    """SLAM-frame camera center and forward axis from one cam-to-world pose.

    Matches what GEV's ``npyPoses.parsePoseTrack`` reads from
    ``camera_poses.npy``: translation column for the center, third rotation
    column for the optical (forward) axis. Accepts ``4x4`` or ``3x4``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape not in ((4, 4), (3, 4)):
        raise ValueError(f"expected a 4x4 or 3x4 pose, received {m.shape}")
    center = m[:3, 3]
    forward = m[:3, 2]
    if not (np.isfinite(center).all() and np.isfinite(forward).all()):
        raise ValueError("pose contains non-finite values")
    return {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "forward": [float(forward[0]), float(forward[1]), float(forward[2])],
    }


def decimate_points(
    points: np.ndarray, colors: np.ndarray, stride: int, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten, drop non-finite points, then thin to the chunk budget."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(pts) != len(cols):
        raise ValueError(f"{len(pts)} points but {len(cols)} colors")
    keep = np.isfinite(pts).all(axis=1)
    pts, cols = pts[keep], cols[keep]
    step = max(1, int(stride))
    pts, cols = pts[::step], cols[::step]
    if max_points > 0 and len(pts) > max_points:
        step = int(np.ceil(len(pts) / max_points))
        pts, cols = pts[::step], cols[::step]
    return pts, cols


def atomic_write_json(path: Path, payload: dict) -> None:
    """Rewrite ``path`` so a concurrent reader sees the old or the new record,
    never a torn one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def write_chunk_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Vertex-only chunk in ``export_reconstruction_ply.write_binary_ply``'s
    layout (zero edges), rewritten here so chunk writing needs no torch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Generated by ABot-Recon stream_reconstruction.py\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element edge 0\n"
        "property int vertex1\nproperty int vertex2\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    data = np.empty(len(points), dtype=vertex_dtype)
    data["x"], data["y"], data["z"] = np.asarray(points, dtype=np.float32).T
    data["r"], data["g"], data["b"] = np.asarray(colors, dtype=np.uint8).T
    with path.open("wb") as handle:
        handle.write(header)
        data.tofile(handle)


def run(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image

    from abot_recon import ABotRecon
    from abot_recon.preprocessing import preprocess_image

    model = ABotRecon.from_pretrained(
        args.checkpoint,
        device=args.device,
        attention_backend=args.attention_backend,
        loop_closure=False,  # rolling windows are short; loop closure is offline work
    )

    stop = {"requested": False}

    def request_stop(_signum, _frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    points_dir = args.output_dir / "points"
    seq = 0
    last_newest: Path | None = None

    while not stop["requested"]:
        frames = list_frames(args.image_dir)
        window = select_window(frames, args.window, last_newest, args.min_new_frames)
        if window is None:
            if args.once:
                print("nothing to reconstruct", file=sys.stderr)
                return 1
            time.sleep(args.poll_interval)
            continue

        result = model.infer(
            window,
            output_local_points=True,
            output_world_points=True,
            output_confidence=False,
            loop_closure=False,
        )
        seq += 1
        last_newest = window[-1]

        poses = result.camera_poses.cpu().numpy()
        pose = pose_record_from_matrix(poses[-1])

        batch_name = None
        if result.world_points is not None:
            colors = []
            for frame_path in window:
                with Image.open(frame_path) as image:
                    tensor, _ = preprocess_image(image)
                colors.append(
                    (tensor.clamp(0, 1) * 255)
                    .round()
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .numpy()
                )
            pts, cols = decimate_points(
                result.world_points.cpu().numpy(),
                np.stack(colors),
                args.point_stride,
                args.max_points,
            )
            batch_name = f"points/batch_{seq:06d}.ply"
            write_chunk_ply(points_dir / f"batch_{seq:06d}.ply", pts, cols)

        atomic_write_json(
            args.output_dir / "pose_latest.json",
            {
                "seq": seq,
                "t": int(time.time() * 1000),
                "pose": pose,
                "batch": batch_name,
                "window": {"frames": len(window), "newest": window[-1].name},
            },
        )
        print(f"window {seq}: {len(window)} frames → {batch_name or 'pose only'}")
        if args.once:
            return 0
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.window < 2:
        raise SystemExit("--window must be at least 2")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
