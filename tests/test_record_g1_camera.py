import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "record_g1_camera.py"


def load_module():
    spec = importlib.util.spec_from_file_location("record_g1_camera", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so the module's dataclasses can resolve their own
    # postponed annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recorder = load_module()

JPEG = b"\xff\xd8\xff\xe0payload\xff\xd9"


def make_args(tmp_path, **overrides):
    args = recorder.build_parser().parse_args(["--output-dir", str(tmp_path / "clip")])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_frame_names_are_zero_padded_and_one_based():
    assert recorder.frame_filename(1) == "000001.jpg"
    assert recorder.frame_filename(1234, digits=4) == "1234.jpg"
    with pytest.raises(ValueError):
        recorder.frame_filename(0)


def test_frame_gate_decimates_a_faster_publisher():
    gate = recorder.FrameGate(10.0)
    accepted = [t for t in (0.0, 0.03, 0.06, 0.11, 0.15, 0.21) if gate.accept(t)]
    assert accepted == [0.0, 0.11, 0.21]


def test_frame_gate_does_not_bank_up_missed_deadlines():
    gate = recorder.FrameGate(10.0)
    assert gate.accept(0.0)
    # Publisher stalled for a second: the next sample is written once, and the
    # cadence re-anchors on it instead of firing a catch-up burst.
    assert gate.accept(1.0)
    assert not gate.accept(1.05)


def test_extract_jpeg_reads_video_and_image_shaped_samples():
    assert recorder.extract_jpeg(SimpleNamespace(video720p=list(JPEG))) == JPEG
    assert recorder.extract_jpeg(SimpleNamespace(data=bytearray(JPEG))) == JPEG
    # An empty keep-alive and a raw (unencoded) image are both "no JPEG here".
    assert recorder.extract_jpeg(SimpleNamespace(video720p=b"")) is None
    assert recorder.extract_jpeg(SimpleNamespace(data=b"\x89PNG")) is None
    assert recorder.extract_jpeg(SimpleNamespace(unrelated=1)) is None


def test_resolve_message_class_reports_bad_paths():
    assert recorder.resolve_message_class("json.JSONDecoder") is json.JSONDecoder
    with pytest.raises(ValueError):
        recorder.resolve_message_class("json")
    with pytest.raises(ValueError):
        recorder.resolve_message_class("json.NotAThing")


def test_validate_args_rejects_unusable_settings(tmp_path):
    with pytest.raises(ValueError):
        recorder.validate_args(make_args(tmp_path, fps=0))
    with pytest.raises(ValueError):
        recorder.validate_args(make_args(tmp_path, max_seconds=-1))
    with pytest.raises(ValueError):
        recorder.validate_args(make_args(tmp_path, digits=0))


def test_record_writes_ordered_frames_and_a_manifest(tmp_path):
    args = make_args(tmp_path, fps=10.0, max_seconds=1.0, max_frames=3)
    clock = {"t": 0.0}
    samples = []

    def subscribe(_args, on_sample):
        samples.append(on_sample)
        return lambda: samples.clear()

    def sleep(_seconds):
        clock["t"] += 0.1
        for callback in tuple(samples):
            callback(SimpleNamespace(video720p=JPEG))

    manifest = recorder.record(
        args, subscribe=subscribe, clock=lambda: clock["t"], sleep=sleep
    )

    assert manifest["frames"] == 3
    assert manifest["samples_received"] >= 3
    assert sorted(p.name for p in args.output_dir.iterdir()) == [
        "000001.jpg",
        "000002.jpg",
        "000003.jpg",
        "manifest.json",
    ]
    assert (args.output_dir / "000002.jpg").read_bytes() == JPEG
    written = json.loads((args.output_dir / "manifest.json").read_text())
    assert written["topic"] == recorder.DEFAULT_TOPIC
    assert written["read_only"] is True
    # The subscriber is closed even on the normal exit path.
    assert samples == []


def test_record_stops_on_a_signal_request(tmp_path):
    args = make_args(tmp_path, fps=100.0, max_seconds=60.0)
    clock = {"t": 0.0}
    stop = {"value": False}

    def subscribe(_args, on_sample):
        on_sample(SimpleNamespace(video720p=JPEG))
        return lambda: None

    def sleep(_seconds):
        clock["t"] += 0.02
        stop["value"] = True

    manifest = recorder.record(
        args,
        subscribe=subscribe,
        clock=lambda: clock["t"],
        sleep=sleep,
        stop_requested=lambda: stop["value"],
    )
    assert manifest["frames"] == 1
    assert manifest["duration_s"] >= 0


def test_list_fields_reports_the_sample_shape_without_writing(tmp_path):
    args = make_args(tmp_path, list_fields=True)
    clock = {"t": 0.0}

    def subscribe(_args, on_sample):
        on_sample(SimpleNamespace(video720p=JPEG, stamp=1))
        return lambda: None

    manifest = recorder.record(
        args, subscribe=subscribe, clock=lambda: clock["t"], sleep=lambda _s: None
    )
    assert set(manifest["sample_fields"]) == {"video720p", "stamp"}
    assert manifest["frames"] == 0
    assert list(args.output_dir.iterdir()) == []
