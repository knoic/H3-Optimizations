'''Single-pass fused SwiGLU kernel for H3 DiT MLP execution.

Computes SwiGLU:
    gate, up = x.chunk(2, dim=-1)
    return silu(gate) * up
in a single pass, eliminating redundant DRAM reads and writes while preserving
100% bitwise/mathematical equivalence.
'''

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_swiglu_kernel(
        x_ptr,
        y_ptr,
        stride_xm,
        stride_xd,
        stride_ym,
        stride_yd,
        M,
        W,
        BLOCK_SIZE: tl.constexpr,
        DTYPE_CODE: tl.constexpr,  # 0: bfloat16, 1: float16, 2: float32
    ):
        row_idx = tl.program_id(0)
        col_block = tl.program_id(1)
        if row_idx >= M:
            return

        cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < W

        gate_ptrs = x_ptr + row_idx * stride_xm + cols * stride_xd
        up_ptrs = x_ptr + row_idx * stride_xm + (cols + W) * stride_xd

        gate_f32 = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
        up_f32 = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

        # In float32 for high-precision sigmoid, matching native PyTorch silu rounding
        silu_f32 = gate_f32 * tl.sigmoid(gate_f32)

        y_ptrs = y_ptr + row_idx * stride_ym + cols * stride_yd
        if DTYPE_CODE == 0:  # bfloat16
            silu_bf16 = silu_f32.to(tl.bfloat16)
            up_bf16 = up_f32.to(tl.bfloat16)
            res = silu_bf16 * up_bf16
            tl.store(y_ptrs, res, mask=mask)
        elif DTYPE_CODE == 1:  # float16
            silu_fp16 = silu_f32.to(tl.float16)
            up_fp16 = up_f32.to(tl.float16)
            res = silu_fp16 * up_fp16
            tl.store(y_ptrs, res, mask=mask)
        else:  # float32
            res = silu_f32 * up_f32
            tl.store(y_ptrs, res, mask=mask)


def is_fused_swiglu_available() -> bool:
    '''Whether Triton-accelerated fused SwiGLU is available.'''
    return TRITON_AVAILABLE


def _fused_swiglu_triton(
    x: torch.Tensor,
    out: torch.Tensor,
    width: int,
) -> torch.Tensor:
    M = x.numel() // x.shape[-1]
    W = width
    BLOCK_SIZE = 1024

    if x.dtype == torch.bfloat16:
        dtype_code = 0
    elif x.dtype == torch.float16:
        dtype_code = 1
    else:
        dtype_code = 2

    stride_xm = x.stride(-2) if x.ndim >= 2 else 0
    stride_xd = x.stride(-1)
    stride_ym = out.stride(-2) if out.ndim >= 2 else 0
    stride_yd = out.stride(-1)

    grid = (M, triton.cdiv(W, BLOCK_SIZE))
    _fused_swiglu_kernel[grid](
        x,
        out,
        stride_xm,
        stride_xd,
        stride_ym,
        stride_yd,
        M,
        W,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE_CODE=dtype_code,
    )
    return out


def fused_swiglu_inplace_(x: torch.Tensor) -> torch.Tensor:
    '''SwiGLU written into the gate half of ``x``; returns that half as a view.

    Bit-identical to :func:`swiglu_inplace_` without allocating extra memory.
    ``x`` is consumed: only the returned view is valid.
    '''
    width = int(x.shape[-1]) // 2
    gate = x[..., :width]
    if (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.is_contiguous()
        and (x.ndim == 2 or x.ndim == 3)
    ):
        return _fused_swiglu_triton(x, gate, width)

    F.silu(gate, inplace=True).mul_(x[..., width:])
    return gate


def fused_swiglu(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    '''Out-of-place fused SwiGLU.'''
    width = int(x.shape[-1]) // 2
    if out is None:
        out_shape = list(x.shape)
        out_shape[-1] = width
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    if (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.is_contiguous()
        and out.is_contiguous()
        and (x.ndim == 2 or x.ndim == 3)
    ):
        return _fused_swiglu_triton(x, out, width)

    gate = x[..., :width]
    up = x[..., width:]
    silu_gate = F.silu(gate)
    return torch.mul(silu_gate, up, out=out)
