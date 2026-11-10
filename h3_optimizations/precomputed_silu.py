'''Precomputed SiLU tensor wrapper for eliminating redundant activations.

In H3's block loop, ``t_emb`` is passed to ``block.adaln_proj(t_emb)`` for
every one of the 50 DiT blocks.  Each ``AdalnProj.forward`` call contains::

    x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)

Because ``t_emb`` is a constant tensor that never changes across the block
loop (it encodes the denoising timestep, which is fixed for the entire forward
pass), ``nn.functional.silu(t_emb)`` is computed 50+ times with identical
input and output per denoising step.  At 20 inference steps that is 1 000+
redundant SiLU kernel launches, each allocating a fresh output tensor.

This module provides :func:`precompute_silu` which wraps ``t_emb`` in a
lightweight :class:`PrecomputedSiluTensor` subclass.  The subclass intercepts
``F.silu`` calls via ``__torch_function__`` and returns a single pre-computed
result, making every subsequent ``nn.functional.silu(t_emb)`` a zero-cost
attribute lookup instead of a GPU kernel dispatch.

The optimisation is **100 % bitwise identical** to the original behaviour:
the returned value is exactly ``F.silu(t_emb)`` computed once on the original
data.  It is also zero-VRAM-increase: the cached tensor has the same shape and
dtype as a single silu output (typically ``[M, 512]``, ~3 KB), which is a
negligible and constant overhead that exists only for the duration of the
forward pass.
'''

import torch
import torch.nn.functional as F


class PrecomputedSiluTensor(torch.Tensor):
    '''Tensor subclass that short-circuits ``F.silu`` with a cached result.

    The cached silu output is stored on the instance as ``_cached_silu``.
    When ``F.silu`` (or equivalently ``nn.functional.silu``) is called on this
    tensor without ``inplace=True``, the cached result is returned immediately.

    All other operations (``F.linear``, arithmetic, indexing, …) fall through
    to the standard :meth:`torch.Tensor.__torch_function__` implementation and
    behave identically to a plain tensor.
    '''

    _cached_silu: torch.Tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is F.silu:
            kw = kwargs or {}
            # Only intercept the out-of-place call; inplace=True is rare and
            # falls through so the in-place behaviour is never accidentally
            # skipped.
            if not kw.get('inplace', False):
                # Support both positional (F.silu(x)) and keyword
                # (F.silu(input=x)) call styles.
                inp = args[0] if args else kw.get('input')
                if isinstance(inp, cls):
                    cached = getattr(inp, '_cached_silu', None)
                    if cached is not None:
                        return cached
        return super().__torch_function__(func, types, args, kwargs or {})


def precompute_silu(t_emb: torch.Tensor) -> torch.Tensor:
    '''Wrap *t_emb* so that subsequent ``F.silu(t_emb)`` calls are free.

    Computes ``F.silu(t_emb)`` exactly once and attaches the result to the
    returned :class:`PrecomputedSiluTensor` wrapper.  Every call to
    ``F.silu`` on the returned object returns the pre-computed value directly
    without dispatching to a GPU kernel.

    If *t_emb* is not a :class:`torch.Tensor` (defensive guard for unexpected
    call sites), the original object is returned unchanged.

    Parameters
    ----------
    t_emb:
        The timestep embedding tensor, typically ``[M, t_dim]`` BF16/FP32.

    Returns
    -------
    torch.Tensor
        A :class:`PrecomputedSiluTensor` wrapping the same storage with the
        silu cache pre-filled, or *t_emb* unchanged on non-Tensor input.
    '''
    if not isinstance(t_emb, torch.Tensor):
        return t_emb
    wrapped = t_emb.as_subclass(PrecomputedSiluTensor)
    # Compute silu on the underlying base tensor to avoid a recursive call
    # through __torch_function__ before _cached_silu is set.
    wrapped._cached_silu = F.silu(t_emb.detach() if t_emb.requires_grad else t_emb)
    return wrapped
