import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "stream_reconstruction.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stream_reconstruction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve __module__ through here
    spec.loader.exec_module(module)
    return module


stream = load_module()


def touch_frames(directory: Path, count: int, start: int = 1) -> list:
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(start, start + count):
        path = directory / f"{index:06d}.jpg"
        path.write_bytes(b"\xff\xd8fake")
        stat = path.stat()
        frames.append(
            stream.Frame(
                path=path, ino=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns
            )
        )
    return frames


def test_windows_wait_for_enough_new_frames(tmp_path):
    frames = touch_frames(tmp_path, 10)
    first = stream.select_window(frames, window=6, last_newest=None, min_new=4)
    assert first == frames[-6:]

    # Only 2 frames arrived since; below min_new means no pass.
    frames = frames + touch_frames(tmp_path, 2, start=11)
    assert stream.select_window(frames, 6, last_newest=first[-1], min_new=4) is None

    frames = frames + touch_frames(tmp_path, 2, start=13)
    again = stream.select_window(frames, 6, last_newest=first[-1], min_new=4)
    assert again == frames[-6:]

    # A recorder restart (previous newest gone) counts everything as new.
    gone = stream.Frame(path=Path("gone.jpg"), ino=0, size=0, mtime_ns=0)
    assert stream.select_window(frames[-3:], 6, last_newest=gone, min_new=4)


def test_rewritten_last_newest_counts_as_a_restart(tmp_path):
    frames = touch_frames(tmp_path, 4)
    old_newest = frames[-1]
    # The recorder restarted and rewrote the same names: same paths, new identity.
    rewritten = [
        stream.Frame(path=f.path, ino=f.ino, size=f.size + 1, mtime_ns=f.mtime_ns + 1)
        for f in frames
    ]
    assert stream.select_window(rewritten, 6, last_newest=old_newest, min_new=99)


def test_stable_frames_holds_back_files_still_changing(tmp_path):
    first = touch_frames(tmp_path, 3)
    prev = {frame.path.name: frame for frame in first}
    # Unchanged files pass; a grown file and a brand-new file are held back.
    grown = stream.Frame(
        path=first[2].path,
        ino=first[2].ino,
        size=first[2].size + 100,
        mtime_ns=first[2].mtime_ns + 1,
    )
    fresh = touch_frames(tmp_path, 1, start=4)[0]
    admitted = stream.stable_frames(prev, [first[0], first[1], grown, fresh])
    assert admitted == [first[0], first[1]]
    # Next scan, once nothing changed anymore, both are admitted.
    prev2 = {f.path.name: f for f in [first[0], first[1], grown, fresh]}
    assert stream.stable_frames(prev2, [first[0], first[1], grown, fresh]) == [
        first[0],
        first[1],
        grown,
        fresh,
    ]


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(7)
    src = rng.normal(size=(8, 3))
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    dst = 2.5 * (rotation @ src.T).T + np.array([1.0, -2.0, 3.0])
    scale, r, t = stream.umeyama_similarity(src, dst)
    assert np.isclose(scale, 2.5)
    assert np.allclose(r, rotation)
    assert np.allclose(t, [1.0, -2.0, 3.0])

    with pytest.raises(ValueError):
        stream.umeyama_similarity(src[:2], dst[:2])
    collinear = np.outer(np.arange(5.0), np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        stream.umeyama_similarity(collinear, collinear * 2)


def test_alignment_stitches_overlapping_windows_into_one_frame():
    keys = [("a",), ("b",), ("c",), ("d",)]
    # Previous pass placed a/b/c at x = 0, 1, 2 in the stream frame.
    stream_poses = {}
    for index, key in enumerate(keys[:3]):
        pose = np.eye(4)
        pose[0, 3] = float(index)
        stream_poses[key] = pose
    # New pass sees b/c/d in its own frame, offset by -1 (b at 0, c at 1, d at 2),
    # plus a slight vertical wobble so the centers aren't collinear.
    new_poses = []
    for index, wobble in enumerate((0.0, 0.3, 0.0)):
        pose = np.eye(4)
        pose[0, 3] = float(index)
        pose[1, 3] = wobble
        new_poses.append(pose)
    new_poses = np.array(new_poses)

    scale, rotation, translation = stream.alignment_from_overlap(
        stream_poses, keys[1:], new_poses
    )
    aligned_d = stream.apply_similarity_to_pose(new_poses[2], scale, rotation, translation)
    # d continues the stream: near x = 3, not restarting at 2.
    assert abs(aligned_d[0, 3] - 3.0) < 0.5


def test_alignment_single_overlap_uses_the_pose_pair():
    prev = np.eye(4)
    prev[:3, 3] = [5.0, 0.0, 0.0]
    stream_poses = {("x",): prev}
    new_poses = np.array([np.eye(4)])
    scale, rotation, translation = stream.alignment_from_overlap(
        stream_poses, [("x",)], new_poses
    )
    assert scale == 1.0
    aligned = stream.apply_similarity_to_pose(new_poses[0], scale, rotation, translation)
    assert np.allclose(aligned, prev)


def test_alignment_without_overlap_is_identity():
    scale, rotation, translation = stream.alignment_from_overlap(
        {}, [("a",)], np.array([np.eye(4)])
    )
    assert scale == 1.0
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, 0)


def test_single_frame_is_never_a_window(tmp_path):
    frames = touch_frames(tmp_path, 1)
    assert stream.select_window(frames, 6, None, 1) is None


def test_pose_record_matches_gev_column_convention():
    pose = np.eye(4)
    pose[:3, 3] = [1.0, -2.0, 3.5]
    record = stream.pose_record_from_matrix(pose)
    assert (record["x"], record["y"], record["z"]) == (1.0, -2.0, 3.5)
    assert record["forward"] == [0.0, 0.0, 1.0]  # third rotation column

    with pytest.raises(ValueError):
        stream.pose_record_from_matrix(np.zeros((2, 2)))
    bad = np.eye(4)
    bad[0, 3] = np.nan
    with pytest.raises(ValueError):
        stream.pose_record_from_matrix(bad)


def test_decimation_drops_nans_and_respects_the_budget():
    points = np.random.rand(1000, 3).astype(np.float32)
    points[7] = np.nan
    colors = np.full((1000, 3), 128, dtype=np.uint8)
    pts, cols = stream.decimate_points(points, colors, stride=2, max_points=100)
    assert len(pts) == len(cols) <= 100
    assert np.isfinite(pts).all()

    with pytest.raises(ValueError):
        stream.decimate_points(points, colors[:10], 1, 0)


def test_pose_json_is_replaced_atomically(tmp_path):
    target = tmp_path / "out" / "pose_latest.json"
    stream.atomic_write_json(target, {"seq": 1})
    stream.atomic_write_json(target, {"seq": 2})
    assert json.loads(target.read_text()) == {"seq": 2}
    assert list(target.parent.iterdir()) == [target], "no tmp file left behind"


def test_chunk_ply_is_the_exporter_layout(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    cols = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    path = tmp_path / "points" / "batch_000001.ply"
    stream.write_chunk_ply(path, pts, cols)

    data = path.read_bytes()
    header, _, body = data.partition(b"end_header\n")
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 2" in header
    assert b"element edge 0" in header
    x, y, z, r, g, b = struct.unpack_from("<fffBBB", body, 15)
    assert (x, y, z, r, g, b) == (1.0, 2.0, 3.0, 0, 255, 0)
    assert len(body) == 2 * 15
