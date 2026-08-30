#!/usr/bin/env python3
"""Record JPEG frames from a Unitree G1 head camera into an ABot-Recon image dir.

Runs ON the robot's Jetson Orin (aarch64) and needs nothing from this
repository at runtime: only ``unitree_sdk2py`` and ``cyclonedds==0.10.2``.
Perception is read-only — the script subscribes to an image topic and never
publishes, never enters developer mode, and never commands a joint.

    # on the Orin, inside a disposable env
    pip install cyclonedds==0.10.2 unitree_sdk2py
    python record_g1_camera.py --output-dir /tmp/g1-clip --fps 10 --max-seconds 30

    # then, from the workstation
    rsync -a unitree@<orin>:/tmp/g1-clip/ ./g1-clip/
    ssh unitree@<orin> 'rm -rf /tmp/g1-clip'

Output is exactly what ``demo.py --image-dir`` expects: zero-padded JPEGs
(``000001.jpg``, ``000002.jpg``, ...) in capture order, plus ``manifest.json``
describing the topic, requested rate, and observed rate.

The DDS topic and message type differ between SDK builds, so both are flags.
The defaults match the ``unitree_sdk2py`` front-video sample; confirm them
against the SDK version installed on the robot before a session and override
with ``--topic``/``--message-class`` if they differ. ``--list-fields`` prints
the fields of one received sample, which is the quickest way to check that a
topic carries what this script expects.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TOPIC = "rt/frontvideostream"
DEFAULT_MESSAGE_CLASS = "unitree_go.msg.dds_.Go2FrontVideoData_"
# Candidate byte-payload field names, most specific first. The SDK's video
# samples expose `video720p`/`video360p`/`video180p`; sensor_msgs-shaped image
# messages expose `data`.
PAYLOAD_FIELDS = ("video720p", "video360p", "video180p", "video_data", "data")
JPEG_SOI = b"\xff\xd8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record G1 head-camera JPEGs for ABot-Recon (read-only perception).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Scratch frame dir")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="DDS image topic")
    parser.add_argument(
        "--message-class",
        default=DEFAULT_MESSAGE_CLASS,
        help="Dotted path of the IDL message class carrying the JPEG payload",
    )
    parser.add_argument(
        "--network-interface",
        default=None,
        help="Interface passed to ChannelFactoryInitialize (omit to use the SDK default)",
    )
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames written per second")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Stop after this many seconds of recording (keep clips short)",
    )
    parser.add_argument("--max-frames", type=int, default=600, help="Stop after this many frames")
    parser.add_argument("--digits", type=int, default=6, help="Zero-padding width of frame names")
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="Print the first sample's field names and exit without writing frames",
    )
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Reject arguments that would silently produce an unusable clip."""
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if not 1 <= args.digits <= 12:
        raise ValueError("--digits must be between 1 and 12")
    return args


def frame_filename(index: int, digits: int = 6) -> str:
    """Name of the ``index``-th written frame, 1-based like the demo examples."""
    if index < 1:
        raise ValueError("frame index is 1-based")
    return f"{index:0{digits}d}.jpg"


@dataclass
class FrameGate:
    """Decimates an arbitrary-rate DDS stream down to a fixed write rate.

    Pure and monotonic-clock driven so the write cadence stays independent of
    however fast the camera publishes.
    """

    fps: float
    _next_at: float | None = None
    # Float accumulation of the period makes an exactly-on-time sample land a
    # few ULP early; a microsecond of slack keeps the nominal rate.
    _slack_s: float = 1e-6

    def accept(self, now: float) -> bool:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self._next_at is None or now >= self._next_at - self._slack_s:
            period = 1.0 / self.fps
            # Anchor on `now` rather than the missed deadline so a slow or
            # stalled publisher cannot bank up a burst of writes.
            self._next_at = now + period
            return True
        return False


def resolve_message_class(dotted: str) -> type:
    """Import an IDL message class from its dotted path."""
    module_name, _, attribute = dotted.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(f"--message-class must be a dotted path, got {dotted!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as error:
        raise ValueError(f"{module_name} has no attribute {attribute}") from error


def message_fields(message: Any) -> list[str]:
    """Public field names of a received sample, for --list-fields."""
    annotations = getattr(type(message), "__annotations__", None)
    if annotations:
        return list(annotations)
    return [name for name in dir(message) if not name.startswith("_")]


def extract_jpeg(message: Any) -> bytes | None:
    """Pull a JPEG payload out of a DDS image sample.

    Handles both the SDK's video messages (a `video*` byte sequence) and
    sensor_msgs-shaped images whose `data` already holds an encoded JPEG.
    Returns None when the sample carries no JPEG — an empty keep-alive, or a
    raw (unencoded) image this script deliberately does not convert.
    """
    for field in PAYLOAD_FIELDS:
        payload = getattr(message, field, None)
        if payload is None:
            continue
        if isinstance(payload, (bytes, bytearray, memoryview)):
            data = bytes(payload)
        elif isinstance(payload, Iterable):
            try:
                data = bytes(bytearray(payload))
            except (TypeError, ValueError):
                continue
        else:
            continue
        if data.startswith(JPEG_SOI):
            return data
    return None


def build_manifest(
    *,
    topic: str,
    message_class: str,
    fps: float,
    digits: int,
    frames: int,
    received: int,
    duration_s: float,
) -> dict:
    """Provenance sidecar written next to the frames."""
    observed = frames / duration_s if duration_s > 0 else 0.0
    return {
        "source": "unitree-g1-head-camera",
        "topic": topic,
        "message_class": message_class,
        "requested_fps": fps,
        "observed_fps": round(observed, 3),
        "frames": frames,
        "samples_received": received,
        "duration_s": round(duration_s, 3),
        "filename_pattern": f"%0{digits}d.jpg",
        "read_only": True,
    }


def record(
    args: argparse.Namespace,
    *,
    subscribe: Callable[[argparse.Namespace, Callable[[Any], None]], Callable[[], None]],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    stop_requested: Callable[[], bool] = lambda: False,
) -> dict:
    """Drive the capture loop.

    `subscribe` installs a sample callback and returns a teardown callable, so
    the loop itself is testable without DDS.
    """
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate = FrameGate(args.fps)
    state = {"frames": 0, "received": 0}
    first_sample_fields: list[str] = []

    def on_sample(message: Any) -> None:
        state["received"] += 1
        if not first_sample_fields:
            first_sample_fields.extend(message_fields(message))
        if args.list_fields or state["frames"] >= args.max_frames:
            return
        if not gate.accept(clock()):
            return
        jpeg = extract_jpeg(message)
        if jpeg is None:
            return
        index = state["frames"] + 1
        path = args.output_dir / frame_filename(index, args.digits)
        path.write_bytes(jpeg)
        state["frames"] = index

    teardown = subscribe(args, on_sample)
    started = clock()
    try:
        while not stop_requested():
            elapsed = clock() - started
            if elapsed >= args.max_seconds:
                break
            if args.list_fields and first_sample_fields:
                break
            if state["frames"] >= args.max_frames:
                break
            sleep(0.02)
    finally:
        teardown()
    duration = clock() - started

    manifest = build_manifest(
        topic=args.topic,
        message_class=args.message_class,
        fps=args.fps,
        digits=args.digits,
        frames=state["frames"],
        received=state["received"],
        duration_s=duration,
    )
    manifest["sample_fields"] = first_sample_fields
    if not args.list_fields:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def dds_subscribe(
    args: argparse.Namespace,
    on_sample: Callable[[Any], None],
) -> Callable[[], None]:
    """Subscribe to the image topic via unitree_sdk2py. Read-only: no publisher."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

    if args.network_interface:
        ChannelFactoryInitialize(args.domain_id, args.network_interface)
    else:
        ChannelFactoryInitialize(args.domain_id)
    subscriber = ChannelSubscriber(args.topic, resolve_message_class(args.message_class))
    subscriber.Init(on_sample, 10)
    return subscriber.Close


def main(argv: list[str] | None = None) -> int:
    args = validate_args(build_parser().parse_args(argv))
    stopping = {"value": False}

    def request_stop(_signum, _frame):
        stopping["value"] = True

    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, request_stop),
        signal.SIGTERM: signal.signal(signal.SIGTERM, request_stop),
    }
    try:
        manifest = record(
            args,
            subscribe=dds_subscribe,
            stop_requested=lambda: stopping["value"],
        )
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    if args.list_fields:
        fields = manifest["sample_fields"]
        print(f"{args.topic}: {', '.join(fields) if fields else 'no sample received'}")
        return 0 if fields else 1
    print(json.dumps(manifest, indent=2))
    if manifest["frames"] == 0:
        print(
            f"No JPEG frames on {args.topic}. Re-run with --list-fields, or set "
            "--topic/--message-class for this SDK build.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
