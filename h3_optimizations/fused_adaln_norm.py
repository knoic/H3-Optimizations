'''Single-pass fused AdaLN-RMSNorm kernel for H3 DiT blocks.

Fuses RMSNorm, scale, and shift into a single SRAM/register pass, eliminating
redundant DRAM roundtrips for Norm1 and Norm2 in standard/BF16 mode while
preserving 100% bitwise/mathematical equivalence.
'''

from __future__ import annotations

import torch

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
    def _fused_adaln_rms_norm_kernel(
        x_ptr,
        w_ptr,
        scale_ptr,
        shift_ptr,
        y_ptr,
        stride_xm,
        stride_xd,
        stride_w,
        stride_scalem,
        stride_scaled,
        stride_shiftm,
        stride_shiftd,
        stride_ym,
        stride_yd,
        M,
        D,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DTYPE_CODE: tl.constexpr,  # 0: bfloat16, 1: float16, 2: float32
    ):
        row_idx = tl.program_id(0)
        if row_idx >= M:
            return

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D

        # 1. Load input row in float32 for high-precision reduction
        x_ptrs = x_ptr + row_idx * stride_xm + cols * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

        # 2. Compute variance and reciprocal RMS
        var = tl.sum(x * x, axis=0) / D
        rrms = tl.rsqrt(var + eps)

        # 3. Load norm weight
        w = tl.load(w_ptr + cols * stride_w, mask=mask, other=0.0).to(tl.float32)

        # 4. Normalized row matching target precision
        normed_f32 = x * rrms * w

        # 5. Load scale and shift (supports broadcasting when stride_m == 0)
        scale_ptrs = scale_ptr + row_idx * stride_scalem + cols * stride_scaled
        shift_ptrs = shift_ptr + row_idx * stride_shiftm + cols * stride_shiftd

        if DTYPE_CODE == 0:  # bfloat16
            normed = normed_f32.to(tl.bfloat16)
            scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.bfloat16)
            shift = tl.load(shift_ptrs, mask=mask, other=0.0).to(tl.bfloat16)
            one = tl.full([BLOCK_SIZE], 1.0, dtype=tl.bfloat16)
            y = normed * (one + scale) + shift
            y_ptrs = y_ptr + row_idx * stride_ym + cols * stride_yd
            tl.store(y_ptrs, y, mask=mask)
        elif DTYPE_CODE == 1:  # float16
            normed = normed_f32.to(tl.float16)
            scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float16)
            shift = tl.load(shift_ptrs, mask=mask, other=0.0).to(tl.float16)
            one = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float16)
            y = normed * (one + scale) + shift
            y_ptrs = y_ptr + row_idx * stride_ym + cols * stride_yd
            tl.store(y_ptrs, y, mask=mask)
        else:  # float32
            scale = tl.load(scale_ptrs, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(shift_ptrs, mask=mask, other=0.0).to(tl.float32)
            y = normed_f32 * (1.0 + scale) + shift
            y_ptrs = y_ptr + row_idx * stride_ym + cols * stride_yd
            tl.store(y_ptrs, y, mask=mask)


def is_fused_adaln_available() -> bool:
    '''Whether Triton-accelerated fused AdaLN-RMSNorm is available.'''
    return TRITON_AVAILABLE


def _fused_adaln_rms_norm_triton(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, D = x.shape
    if out is None:
        out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(D)
    if D <= 8192 and BLOCK_SIZE < 8192:
        BLOCK_SIZE = 8192

    if x.dtype == torch.bfloat16:
        dtype_code = 0
    elif x.dtype == torch.float16:
        dtype_code = 1
    else:
        dtype_code = 2

    stride_xm, stride_xd = x.stride(0), x.stride(1)
    stride_w = norm_weight.stride(0)

    if scale.ndim == 1:
        stride_scalem, stride_scaled = 0, scale.stride(0)
    else:
        stride_scalem, stride_scaled = scale.stride(0), scale.stride(1)

    if shift.ndim == 1:
        stride_shiftm, stride_shiftd = 0, shift.stride(0)
    else:
        stride_shiftm, stride_shiftd = shift.stride(0), shift.stride(1)

    stride_ym, stride_yd = out.stride(0), out.stride(1)

    grid = (M,)
    _fused_adaln_rms_norm_kernel[grid](
        x,
        norm_weight,
        scale,
        shift,
        out,
        stride_xm,
        stride_xd,
        stride_w,
        stride_scalem,
        stride_scaled,
        stride_shiftm,
        stride_shiftd,
        stride_ym,
        stride_yd,
        M,
        D,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE_CODE=dtype_code,
    )
    return out


def _fused_adaln_rms_norm_torch(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    '''Pure PyTorch bit-identical fallback.'''
    x_f32 = x.to(torch.float32)
    var = torch.mean(x_f32 * x_f32, dim=-1, keepdim=True)
    rrms = torch.rsqrt(var + eps)
    normed = (x_f32 * rrms * norm_weight.to(torch.float32)).to(x.dtype)
    res = normed * (1.0 + scale.to(x.dtype)) + shift.to(x.dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def fused_adaln_rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    selector: int | torch.Tensor | None = None,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    '''Compute single-pass AdaLN-modulated RMSNorm.

    Mathematically and bitwise equivalent to:
        h = RMSNorm(x, weight=norm_weight, eps=eps)
        h = h * (1.0 + scale[selector]) + shift[selector]
    '''
    if selector is not None:
        if isinstance(selector, int):
            scale = scale[selector]
            shift = shift[selector]
        elif torch.is_tensor(selector):
            if selector.device != scale.device:
                selector = selector.to(device=scale.device)
            scale = scale[selector]
            shift = shift[selector]

    # Ensure norm_weight, scale, shift match x.device
    if norm_weight.device != x.device:
        norm_weight = norm_weight.to(device=x.device)
    if scale.device != x.device:
        scale = scale.to(device=x.device)
    if shift.device != x.device:
        shift = shift.to(device=x.device)

    # Use Triton if on CUDA with rank-2 activations and supported hidden width
    if (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.ndim == 2
        and x.shape[-1] <= 8192
        and x.is_contiguous()
        and (out is None or out.is_contiguous())
    ):
        return _fused_adaln_rms_norm_triton(
            x,
            norm_weight,
            scale,
            shift,
            eps=eps,
            out=out,
        )

    return _fused_adaln_rms_norm_torch(
        x,
        norm_weight,
        scale,
        shift,
        eps=eps,
        out=out,
    )
