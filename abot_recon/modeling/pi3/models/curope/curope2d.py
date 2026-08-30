# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch

try:
    from . import curope as _kernels  # run `python setup.py build_ext --inplace`
except ImportError:  # CPU-only installs, or the extension was never built
    _kernels = None


def kernels_available() -> bool:
    """Whether the compiled CUDA operator can be used."""
    return _kernels is not None


def rope_2d_torch(tokens, positions, base=100.0, F0=1.0):
    """Pure-PyTorch 2D rotary encoding matching ``_kernels.rope_2d``.

    input:
        * tokens: batch_size x nheads x ntokens x dim
        * positions: batch_size x ntokens x 2 (y and x position of each token)
    The CUDA kernel folds ``F0`` into the inverse frequencies, so ``F0=-1``
    applies the inverse rotation, and it rounds every intermediate value to the
    token dtype; both are reproduced here.
    """
    dim = tokens.shape[-1] // 2
    inv_freq = F0 / (
        base ** (torch.arange(0, dim, 2, device=tokens.device).float() / dim)
    )
    steps = torch.arange(int(positions.max()) + 1, device=tokens.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", steps, inv_freq).to(tokens.dtype)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos, sin = freqs.cos(), freqs.sin()

    def apply(values, indices):
        embedded_cos = torch.nn.functional.embedding(indices, cos)[:, None]
        embedded_sin = torch.nn.functional.embedding(indices, sin)[:, None]
        left, right = values.chunk(2, dim=-1)
        rotated = torch.cat((-right, left), dim=-1)
        return values * embedded_cos + rotated * embedded_sin

    y, x = tokens.chunk(2, dim=-1)
    return torch.cat((apply(y, positions[:, :, 0]), apply(x, positions[:, :, 1])), dim=-1)


class cuRoPE2D_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tokens, positions, base, F0=1):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        _kernels.rope_2d(tokens, positions, base, F0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_res):
        positions, base, F0 = ctx.saved_tensors[0], ctx.saved_base, ctx.saved_F0
        # CUDA kernel expects layout (B, N, H, D) with stride(2)==D, stride(3)==1
        g = grad_res.contiguous()
        _kernels.rope_2d(g, positions, base, -F0)
        ctx.mark_dirty(g)
        return g, None, None, None


class cuRoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0

    def forward(self, tokens, positions):
        if not (tokens.is_cuda and kernels_available()):
            return rope_2d_torch(tokens, positions, self.base, self.F0)
        # Attention passes (B, nheads, ntokens, dim); kernel expects (B, N, H, D) contiguous.
        # A plain transpose(1, 2) does not satisfy stride(2)==D when ntokens>1.
        x = tokens.transpose(1, 2).contiguous()
        x = cuRoPE2D_func.apply(x, positions, self.base, self.F0)
        return x.transpose(1, 2).contiguous()
