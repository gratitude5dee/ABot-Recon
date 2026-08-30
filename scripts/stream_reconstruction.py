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

Every ``infer()`` call reconstructs its window in a fresh local frame, so
consecutive windows are stitched into one persistent stream frame: the poses
each pass produces for frames shared with the previous pass are aligned onto
the previous pass's (already-stitched) poses — Umeyama on the shared camera
centers when there are enough, a single shared pose pair otherwise — and the
resulting similarity is applied to the pass's poses and world points before
anything is published.

Frames are admitted only once they are *stable* (same inode/size/mtime across
two consecutive scans) and decodable, so a file the recorder is still writing
is never fed to the model; a frame that fails to decode is retried on later
scans instead of killing the daemon. Frame identity (not just the file name)
is tracked, so a recorder restart that rewrites the same names is detected
and treated as all-new footage.

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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Frame:
    """One admitted image file, identified by content identity, not just name:
    a recorder restart that rewrites ``000001.jpg`` yields a different Frame."""

    path: Path
    ino: int
    size: int
    mtime_ns: int

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.path.name, self.ino, self.size, self.mtime_ns)


def scan_frames(image_dir: Path) -> list[Frame]:
    """Sorted snapshot of the image files currently in the directory."""
    if not image_dir.is_dir():
        return []
    frames = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue  # deleted between listing and stat
        if not path.is_file():
            continue
        frames.append(
            Frame(path=path, ino=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        )
    return frames


def stable_frames(previous: dict[str, Frame], current: list[Frame]) -> list[Frame]:
    """Frames whose identity survived one full scan interval unchanged.

    A file the recorder is mid-write on either grows (size/mtime changes) or
    is brand new; both are held back until the next scan confirms them.
    """
    return [
        frame for frame in current if previous.get(frame.path.name) == frame
    ]


def select_window(
    frames: list[Frame], window: int, last_newest: Frame | None, min_new: int
) -> list[Frame] | None:
    """The next window to reconstruct, or None when too little is new.

    ``last_newest`` is the newest frame of the previous pass; a pass runs only
    once at least ``min_new`` frames arrived after it, so a stalled recorder
    does not burn GPU re-reconstructing the same clip. Identity comparison
    means a recorder restart that rewrote the same file names counts as
    all-new footage.

    When a backlog accumulated (more new frames than a window can absorb),
    the window advances at most ``window - overlap`` frames past the previous
    pass, so consecutive passes always share the frames coordinate stitching
    needs; the backlog is consumed over successive passes rather than jumped
    over.
    """
    if len(frames) < 2:
        return None
    if last_newest is not None:
        try:
            index = frames.index(last_newest)
        except ValueError:
            # The recorder restarted or rewrote files; everything is new and
            # there is nothing left to overlap with.
            return frames[-window:] if window > 0 else frames
        newness = len(frames) - 1 - index
        if newness < max(1, min_new):
            return None
        if window > 0:
            overlap = min(max(3, min_new), window - 1)
            end = min(len(frames), index + 1 + window - overlap)
            return frames[max(0, end - window):end]
    return frames[-window:] if window > 0 else frames


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with ``dst ≈ s·R·src + t``.

    Umeyama (1991). Raises ``ValueError`` when the point sets are too few or
    too degenerate (rank < 2) to determine a rotation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"expected matching Nx3 sets, received {src.shape} and {dst.shape}")
    n = len(src)
    if n < 3:
        raise ValueError("need at least 3 correspondences")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / n
    u, d, vt = np.linalg.svd(cov)
    if np.linalg.matrix_rank(cov) < 2:
        raise ValueError("degenerate correspondences (rank < 2)")
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1.0
    rotation = u @ s_fix @ vt
    var_src = (src_c**2).sum() / n
    if var_src <= 0:
        raise ValueError("zero-variance source points")
    scale = float((d * np.diag(s_fix)).sum() / var_src)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"non-positive similarity scale {scale}")
    translation = mu_dst - scale * rotation @ mu_src
    return scale, rotation, translation


def alignment_from_overlap(
    stream_poses: dict[tuple, np.ndarray],
    window_keys: list[tuple],
    new_poses: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity mapping this pass's local frame onto the persistent stream
    frame, from the poses of frames shared with the previous pass.

    Umeyama on the shared camera centers when there are ≥3 and they are
    non-degenerate; otherwise the newest shared pose pair fixes a rigid
    (scale-1) transform. With no overlap at all the pass keeps its own frame
    (identity) — the first pass, or a stream discontinuity after a recorder
    restart.
    """
    overlap = [
        (index, key) for index, key in enumerate(window_keys) if key in stream_poses
    ]
    if not overlap:
        return 1.0, np.eye(3), np.zeros(3)
    if len(overlap) >= 3:
        src = np.array([new_poses[i][:3, 3] for i, _ in overlap])
        dst = np.array([stream_poses[k][:3, 3] for _, k in overlap])
        try:
            return umeyama_similarity(src, dst)
        except ValueError:
            pass  # nearly-stationary camera; fall through to the pose pair
    index, key = overlap[-1]
    prev = stream_poses[key]
    new = np.asarray(new_poses[index], dtype=np.float64)
    rotation = prev[:3, :3] @ new[:3, :3].T
    translation = prev[:3, 3] - rotation @ new[:3, 3]
    return 1.0, rotation, translation


def apply_similarity_to_pose(
    pose: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Map a cam-to-world pose into the stream frame, keeping the rotation
    block orthonormal (scale moves only the center)."""
    pose = np.asarray(pose, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = rotation @ pose[:3, :3]
    out[:3, 3] = scale * rotation @ pose[:3, 3] + translation
    return out


def apply_similarity_to_points(
    points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return (scale * (rotation @ pts.T).T + translation).astype(np.float32)


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
    from PIL import Image, UnidentifiedImageError

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

    def decodable(frame: Frame) -> bool:
        try:
            with Image.open(frame.path) as image:
                image.verify()
            return True
        except (OSError, UnidentifiedImageError, SyntaxError) as error:
            print(f"skipping undecodable {frame.path.name}: {error}", file=sys.stderr)
            return False

    points_dir = args.output_dir / "points"
    seq = 0
    last_newest: Frame | None = None
    previous_scan: dict[str, Frame] = {}
    verified: set[tuple] = set()
    stream_poses: dict[tuple, np.ndarray] = {}

    scans = 0
    while not stop["requested"]:
        scans += 1
        scan = scan_frames(args.image_dir)
        candidates = stable_frames(previous_scan, scan)
        previous_scan = {frame.path.name: frame for frame in scan}
        frames = []
        for frame in candidates:
            if frame.key not in verified:
                if not decodable(frame):
                    continue  # retried next scan; the recorder may still fix it
                verified.add(frame.key)
            frames.append(frame)
        if len(verified) > 4096:
            verified = {frame.key for frame in frames}

        window = select_window(frames, args.window, last_newest, args.min_new_frames)
        if window is None:
            # Stability needs a confirming scan, so one-shot mode only
            # concludes "empty" after the second scan has had its chance.
            if args.once and scans >= 2:
                print("nothing to reconstruct", file=sys.stderr)
                return 1
            time.sleep(args.poll_interval)
            continue

        try:
            result = model.infer(
                [frame.path for frame in window],
                output_local_points=True,
                output_world_points=True,
                output_confidence=False,
                loop_closure=False,
            )
            colors = None
            if result.world_points is not None:
                colors = []
                for frame in window:
                    with Image.open(frame.path) as image:
                        tensor, _ = preprocess_image(image)
                    colors.append(
                        (tensor.clamp(0, 1) * 255)
                        .round()
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .numpy()
                    )
        except (OSError, UnidentifiedImageError) as error:
            # A frame vanished or went bad between admission and the pass;
            # drop its verification and try again on the next scan. Nothing
            # is committed until every read for the pass has succeeded.
            print(f"window failed to load, retrying: {error}", file=sys.stderr)
            verified.difference_update(frame.key for frame in window)
            time.sleep(args.poll_interval)
            continue
        seq += 1
        last_newest = window[-1]

        poses = result.camera_poses.cpu().numpy().astype(np.float64)
        window_keys = [frame.key for frame in window]
        scale, rotation, translation = alignment_from_overlap(
            stream_poses, window_keys, poses
        )
        aligned = [
            apply_similarity_to_pose(pose, scale, rotation, translation)
            for pose in poses
        ]
        stream_poses = dict(zip(window_keys, aligned))
        pose = pose_record_from_matrix(aligned[-1])

        batch_name = None
        if colors is not None:
            pts, cols = decimate_points(
                result.world_points.cpu().numpy(),
                np.stack(colors),
                args.point_stride,
                args.max_points,
            )
            pts = apply_similarity_to_points(pts, scale, rotation, translation)
            batch_name = f"points/batch_{seq:06d}.ply"
            write_chunk_ply(points_dir / f"batch_{seq:06d}.ply", pts, cols)

        atomic_write_json(
            args.output_dir / "pose_latest.json",
            {
                "seq": seq,
                "t": int(time.time() * 1000),
                "pose": pose,
                "batch": batch_name,
                "window": {"frames": len(window), "newest": window[-1].path.name},
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
