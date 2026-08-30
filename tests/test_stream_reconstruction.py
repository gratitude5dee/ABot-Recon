import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "stream_reconstruction.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stream_reconstruction", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stream = load_module()


def touch_frames(directory: Path, count: int, start: int = 1) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(start, start + count):
        path = directory / f"{index:06d}.jpg"
        path.write_bytes(b"\xff\xd8fake")
        paths.append(path)
    return paths


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
    assert stream.select_window(frames[-3:], 6, last_newest=Path("gone.jpg"), min_new=4)


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
