'''Unit tests for PrecomputedSiluTensor and precompute_silu.

Verifies:
  1. F.silu returns the cached result (bitwise identical, same tensor object).
  2. nn.functional.silu is equivalently intercepted.
  3. Keyword-argument call style F.silu(input=wrapped) is handled.
  4. inplace=True falls through to normal PyTorch behaviour (not intercepted).
  5. All other tensor operations (F.linear, arithmetic, indexing) are transparent.
  6. precompute_silu on a non-Tensor returns the input unchanged.
  7. apply_silu=False path: wrapped tensor works as direct input to F.linear.
  8. Multiple wraps of distinct base tensors hold independent caches.
  9. BF16 and FP32 dtypes both work correctly.
 10. The optimisation is bitwise identical to computing silu 50 times in a row.
'''

import unittest

import torch
import torch.nn.functional as F

from h3_optimizations.precomputed_silu import (
    PrecomputedSiluTensor,
    precompute_silu,
)


class PrecomputedSiluTensorTests(unittest.TestCase):

    # ------------------------------------------------------------------
    # Core interception behaviour
    # ------------------------------------------------------------------

    def test_silu_returns_cached_result_data_ptr(self):
        base = torch.randn(3, 512)
        wrapped = precompute_silu(base)
        result = F.silu(wrapped)
        self.assertEqual(
            result.data_ptr(),
            wrapped._cached_silu.data_ptr(),
            'F.silu should return the exact cached tensor (same data_ptr)',
        )

    def test_silu_bitwise_identical_to_eager(self):
        torch.manual_seed(42)
        base = torch.randn(5, 512)
        wrapped = precompute_silu(base)
        self.assertTrue(
            torch.equal(F.silu(wrapped), F.silu(base)),
            'Cached silu must be bitwise identical to eager F.silu(base)',
        )

    def test_nn_functional_silu_intercepted(self):
        import torch.nn.functional as nnF
        base = torch.randn(2, 128)
        wrapped = precompute_silu(base)
        result = nnF.silu(wrapped)
        self.assertEqual(result.data_ptr(), wrapped._cached_silu.data_ptr())

    def test_keyword_argument_call_intercepted(self):
        base = torch.randn(2, 64)
        wrapped = precompute_silu(base)
        result = F.silu(input=wrapped)
        self.assertEqual(result.data_ptr(), wrapped._cached_silu.data_ptr())

    # ------------------------------------------------------------------
    # inplace safety
    # ------------------------------------------------------------------

    def test_inplace_silu_not_intercepted(self):
        '''inplace=True must NOT use the cache - it should modify the tensor.'''
        base = torch.randn(2, 8)
        wrapped = precompute_silu(base.clone())
        original_values = wrapped.clone()
        F.silu(wrapped, inplace=True)
        self.assertFalse(
            torch.equal(wrapped, original_values),
            'inplace silu must modify the tensor in-place',
        )

    # ------------------------------------------------------------------
    # Transparency: other ops must behave identically to plain tensors
    # ------------------------------------------------------------------

    def test_linear_transparent(self):
        torch.manual_seed(0)
        base = torch.randn(4, 8)
        wrapped = precompute_silu(base)
        weight = torch.randn(16, 8)
        bias = torch.randn(16)
        out_wrapped = F.linear(wrapped, weight, bias)
        out_plain = F.linear(base, weight, bias)
        self.assertTrue(
            torch.equal(out_wrapped, out_plain),
            'F.linear must be transparent on a wrapped tensor',
        )

    def test_arithmetic_transparent(self):
        base = torch.randn(3, 4)
        wrapped = precompute_silu(base)
        self.assertTrue(torch.equal(wrapped + 1.0, base + 1.0))
        self.assertTrue(torch.equal(wrapped * 2.0, base * 2.0))
        self.assertTrue(torch.equal(wrapped - base, torch.zeros_like(base)))

    def test_indexing_transparent(self):
        base = torch.randn(5, 8)
        wrapped = precompute_silu(base)
        self.assertTrue(torch.equal(wrapped[1:3], base[1:3]))
        self.assertTrue(torch.equal(wrapped[:, :4], base[:, :4]))

    def test_shape_dtype_device_transparent(self):
        base = torch.randn(3, 512, dtype=torch.bfloat16)
        wrapped = precompute_silu(base)
        self.assertEqual(wrapped.shape, base.shape)
        self.assertEqual(wrapped.dtype, base.dtype)
        self.assertEqual(wrapped.device, base.device)

    # ------------------------------------------------------------------
    # apply_silu=False path: wrapped tensor passed directly to F.linear
    # ------------------------------------------------------------------

    def test_apply_silu_false_path_linear(self):
        '''When adaln_proj skips silu, the wrapped tensor is passed to linear.'''
        torch.manual_seed(1)
        base = torch.randn(3, 512)
        wrapped = precompute_silu(base)
        weight = torch.randn(6144, 512)
        bias = torch.randn(6144)
        out_wrapped = F.linear(wrapped, weight, bias)
        out_plain = F.linear(base, weight, bias)
        self.assertTrue(
            torch.equal(out_wrapped, out_plain),
            'apply_silu=False path must produce identical output',
        )

    # ------------------------------------------------------------------
    # precompute_silu helper
    # ------------------------------------------------------------------

    def test_non_tensor_input_returned_unchanged(self):
        for val in (None, 42, 'string', [1, 2, 3]):
            result = precompute_silu(val)
            self.assertIs(result, val, 'Non-Tensor input should be returned unchanged')

    def test_cached_silu_bitwise_equal_to_silu(self):
        torch.manual_seed(99)
        base = torch.randn(4, 256)
        wrapped = precompute_silu(base)
        self.assertTrue(
            torch.equal(wrapped._cached_silu, F.silu(base)),
            '_cached_silu must be bitwise identical to F.silu(base)',
        )

    def test_bfloat16_dtype(self):
        base = torch.randn(2, 512, dtype=torch.bfloat16)
        wrapped = precompute_silu(base)
        result = F.silu(wrapped)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result, F.silu(base)))

    def test_independent_caches_per_wrap(self):
        base1 = torch.randn(2, 8)
        base2 = torch.randn(2, 8)
        wrapped1 = precompute_silu(base1)
        wrapped2 = precompute_silu(base2)
        self.assertNotEqual(
            wrapped1._cached_silu.data_ptr(),
            wrapped2._cached_silu.data_ptr(),
            'Different wraps must hold independent caches',
        )

    # ------------------------------------------------------------------
    # End-to-end: 50-block simulation (the actual use case)
    # ------------------------------------------------------------------

    def test_50_block_silu_all_identical_to_eager(self):
        '''Simulate the adaln_proj silu call across 50 blocks and verify parity.'''
        torch.manual_seed(2024)
        # t_emb shape matches a typical H3 forward: [M, 512] with M=3 unique timesteps
        t_emb_base = torch.randn(3, 512)
        wrapped = precompute_silu(t_emb_base)

        # Simulate 50 blocks each calling F.silu(t_emb)
        cached_calls = [F.silu(wrapped) for _ in range(50)]

        # All must be the exact same cached tensor (zero allocation)
        for i, result in enumerate(cached_calls):
            self.assertEqual(
                result.data_ptr(),
                wrapped._cached_silu.data_ptr(),
                f'Block {i}: expected cached tensor, got different allocation',
            )

        # And each must be bitwise identical to the eager computation
        eager = F.silu(t_emb_base)
        for i, result in enumerate(cached_calls):
            self.assertTrue(
                torch.equal(result, eager),
                f'Block {i}: cached silu not bitwise identical to eager F.silu',
            )


if __name__ == '__main__':
    unittest.main()
