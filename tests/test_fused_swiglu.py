'''Unit tests for fused SwiGLU kernel and inplace execution.'''

import unittest
import torch
import torch.nn.functional as F

from h3_optimizations.fused_swiglu import (
    fused_swiglu,
    fused_swiglu_inplace_,
    is_fused_swiglu_available,
)


class FusedSwiGLUTests(unittest.TestCase):
    def test_inplace_bitwise_parity(self):
        torch.manual_seed(20261110)
        M, W = 64, 24576
        x = torch.randn(M, 2 * W, dtype=torch.bfloat16)

        # Reference PyTorch eager
        ref_x = x.clone()
        ref_gate = ref_x[..., :W]
        F.silu(ref_gate, inplace=True).mul_(ref_x[..., W:])

        # Actual fused inplace
        act_x = x.clone()
        act_gate = fused_swiglu_inplace_(act_x)

        self.assertEqual(act_gate.data_ptr(), act_x.data_ptr())
        self.assertTrue(
            torch.equal(ref_gate, act_gate),
            'fused_swiglu_inplace_ must match PyTorch silu.mul_ bit-identically',
        )

    def test_out_of_place_bitwise_parity(self):
        torch.manual_seed(20261110)
        M, W = 32, 24576
        x = torch.randn(M, 2 * W, dtype=torch.bfloat16)

        ref_gate, ref_up = x.chunk(2, dim=-1)
        ref_out = F.silu(ref_gate).mul_(ref_up)

        act_out = fused_swiglu(x)
        self.assertTrue(
            torch.equal(ref_out, act_out),
            'fused_swiglu out-of-place must match PyTorch chunk silu*up',
        )

    def test_custom_output_buffer(self):
        torch.manual_seed(20261110)
        M, W = 16, 12288
        x = torch.randn(M, 2 * W, dtype=torch.bfloat16)
        out_buf = torch.empty(M, W, dtype=torch.bfloat16)

        res = fused_swiglu(x, out=out_buf)
        self.assertIs(res, out_buf)

        ref_gate, ref_up = x.chunk(2, dim=-1)
        ref_out = F.silu(ref_gate).mul_(ref_up)
        self.assertTrue(torch.equal(out_buf, ref_out))


if __name__ == '__main__':
    unittest.main()
