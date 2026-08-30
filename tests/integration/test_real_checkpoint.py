import gc
import os
from pathlib import Path

import pytest
import torch

from abot_recon import ABotRecon


CHECKPOINT = os.environ.get("ABOT_RECON_CHECKPOINT")
IMAGE_DIR = os.environ.get("ABOT_RECON_IMAGE_DIR")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@pytest.mark.skipif(not CHECKPOINT, reason="set ABOT_RECON_CHECKPOINT for GPU integration")
@pytest.mark.parametrize("backend", ["paged", "sdpa"])
def test_real_checkpoint_strict_loads_for_both_backends(backend):
    model = ABotRecon.from_pretrained(
        Path(CHECKPOINT),
        device=os.environ.get("ABOT_RECON_DEVICE", "cuda"),
        attention_backend=backend,
        max_frames=128,
    )
    assert sum(parameter.numel() for parameter in model.model.parameters()) == 1_000_934_170
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not CHECKPOINT or not IMAGE_DIR,
    reason="set checkpoint and image directory for inference integration",
)
@pytest.mark.parametrize("backend", ["paged", "sdpa"])
def test_real_checkpoint_runs_dense_confidence_inference(backend):
    paths = sorted(
        path for path in Path(IMAGE_DIR).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )[:3]
    assert len(paths) == 3
    model = ABotRecon.from_pretrained(
        Path(CHECKPOINT),
        device=os.environ.get("ABOT_RECON_DEVICE", "cuda"),
        attention_backend=backend,
        max_frames=128,
    )
    result = model.infer(
        paths,
        output_points=True,
        output_confidence=True,
        loop_closure=False,
    )
    assert result.camera_poses.shape == (3, 4, 4)
    assert result.relative_poses.shape == (2, 4, 4)
    assert result.local_points.shape == (3, 280, 504, 3)
    assert result.world_points.shape == (3, 280, 504, 3)
    assert result.confidence.shape == (3, 280, 504)
    values = (result.camera_poses, result.local_points, result.world_points, result.confidence)
    assert all(torch.isfinite(value).all() for value in values)

    sparse = model.infer(
        paths,
        output_points=True,
        output_confidence=True,
        dense_output_indices=[0, 2],
        loop_closure=False,
    )
    assert sparse.camera_poses.shape == (3, 4, 4)
    assert sparse.relative_poses.shape == (2, 4, 4)
    assert sparse.local_points.shape == (2, 280, 504, 3)
    assert sparse.world_points.shape == (2, 280, 504, 3)
    assert sparse.confidence.shape == (2, 280, 504)
    assert sparse.metadata["dense_output_indices"] == [0, 2]
    torch.testing.assert_close(sparse.camera_poses, result.camera_poses, rtol=0, atol=0)
    torch.testing.assert_close(sparse.local_points, result.local_points[[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(sparse.world_points, result.world_points[[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(sparse.confidence, result.confidence[[0, 2]], rtol=1e-6, atol=1e-7)
    sparse_values = (sparse.camera_poses, sparse.local_points, sparse.world_points, sparse.confidence)
    assert all(torch.isfinite(value).all() for value in sparse_values)


@pytest.mark.skipif(
    not CHECKPOINT or not IMAGE_DIR,
    reason="set checkpoint and image directory for inference integration",
)
@pytest.mark.parametrize("quantization", ["none", "int8_dynamic"])
def test_real_checkpoint_runs_on_cpu_without_cuda(quantization):
    paths = sorted(
        path for path in Path(IMAGE_DIR).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )[:3]
    assert len(paths) == 3
    model = ABotRecon.from_pretrained(
        Path(CHECKPOINT),
        device="cpu",
        amp_dtype="fp32",
        quantization=quantization,
        max_frames=64,
    )
    result = model.infer(
        paths,
        output_points=True,
        output_confidence=True,
        loop_closure=False,
    )
    assert result.metadata["attention_backend"] == "sdpa"
    assert result.metadata["quantization"] == quantization
    assert result.camera_poses.shape == (3, 4, 4)
    assert result.camera_poses.device.type == "cpu"
    assert result.local_points.shape == (3, 280, 504, 3)
    assert result.confidence.shape == (3, 280, 504)
    assert torch.allclose(result.camera_poses[0], torch.eye(4), atol=1e-5)
    for value in (result.camera_poses, result.local_points, result.confidence):
        assert torch.isfinite(value).all()
    del model
    gc.collect()
