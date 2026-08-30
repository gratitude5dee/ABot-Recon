"""Paged-KV window bookkeeping on CPU, where FlashInfer is never available."""

import torch
import torch.nn.functional as F

from abot_recon.modeling.streaming.paged_kv import PagedKVCacheManager


TPF = 4
HEADS = 2
DIM = 8
WINDOW = 3


def build_manager(**overrides) -> PagedKVCacheManager:
    kwargs = dict(
        num_layers=2,
        tpf=TPF,
        num_heads=HEADS,
        head_dim=DIM,
        num_reference_frames=0,
        local_window_frames=WINDOW,
        num_summary_tokens=0,
        memory_mode="window",
        max_total_frames=16,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return PagedKVCacheManager(**kwargs)


def test_cpu_manager_falls_back_to_gather_sdpa_without_flashinfer():
    manager = build_manager()

    assert manager.force_fp32 is True
    assert manager.prefill_wrapper is None
    assert manager.storage_dtype is torch.float32


def test_cpu_attention_matches_dense_sdpa_over_the_visible_window():
    torch.manual_seed(0)
    manager = build_manager()
    frames = [
        (torch.randn(TPF, HEADS, DIM), torch.randn(TPF, HEADS, DIM)) for _ in range(5)
    ]
    for key, value in frames:
        manager.append_frame(0, key, value)
    query = torch.randn(TPF, HEADS, DIM)

    actual = manager.compute_attention(0, query)

    visible = frames[-WINDOW:]
    keys = torch.cat([key for key, _ in visible]).permute(1, 0, 2).unsqueeze(0)
    values = torch.cat([value for _, value in visible]).permute(1, 0, 2).unsqueeze(0)
    expected = F.scaled_dot_product_attention(
        query.permute(1, 0, 2).unsqueeze(0), keys, values
    )
    torch.testing.assert_close(actual, expected.squeeze(0).permute(1, 0, 2))


def test_window_recycles_pages_instead_of_growing():
    manager = build_manager()
    for _ in range(8):
        manager.append_frame(0, torch.zeros(TPF, HEADS, DIM), torch.zeros(TPF, HEADS, DIM))

    stats = manager.get_stats(0)
    assert stats == {
        "frames": 8,
        "reference": 0,
        "window": WINDOW,
        "summary_pages": 0,
        "summary_tokens": 0,
        "free_recyc": manager._n_recyclable_pages - WINDOW,
        "free_summary": 0,
    }
    assert manager.visible_frame_indices(0) == [5, 6, 7]
