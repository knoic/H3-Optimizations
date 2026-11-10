'''Unit tests for single-pass fused AdaLN-RMSNorm kernel and dispatch.'''

import unittest
import torch
import torch.nn as nn

from h3_optimizations.fused_adaln_norm import (
    fused_adaln_rms_norm,
    is_fused_adaln_available,
)


class FusedAdaLNNormTests(unittest.TestCase):
    def test_scalar_selector_bitwise_parity(self):
        torch.manual_seed(20261110)
        M, D = 64, 7168
        num_selectors = 4
        x = torch.randn(M, D, dtype=torch.bfloat16)
        norm = nn.RMSNorm(D, eps=1e-6)
        norm.weight.data = torch.randn(D, dtype=torch.bfloat16)
        shift = torch.randn(num_selectors, D, dtype=torch.bfloat16)
        scale = torch.randn(num_selectors, D, dtype=torch.bfloat16)

        for selector in range(num_selectors):
            # Baseline PyTorch RMSNorm + manual scale/shift
            h_ref = norm(x.clone())
            h_ref.mul_(1.0 + scale[selector])
            h_ref.add_(shift[selector])

            # Fused AdaLN-RMSNorm
            h_fused = fused_adaln_rms_norm(
                x,
                norm.weight,
                scale,
                shift,
                selector=selector,
                eps=norm.eps,
            )

            self.assertTrue(
                torch.equal(h_ref, h_fused),
                f'Mismatch with scalar selector {selector}',
            )

    def test_tensor_selector_bitwise_parity(self):
        torch.manual_seed(20261110)
        M, D = 128, 7168
        num_selectors = 3
        x = torch.randn(M, D, dtype=torch.bfloat16)
        norm = nn.RMSNorm(D, eps=1e-6)
        norm.weight.data = torch.randn(D, dtype=torch.bfloat16)
        shift = torch.randn(num_selectors, D, dtype=torch.bfloat16)
        scale = torch.randn(num_selectors, D, dtype=torch.bfloat16)
        selector = torch.randint(0, num_selectors, (M,), dtype=torch.long)

        # Baseline
        h_ref = norm(x.clone())
        h_ref.mul_(1.0 + scale[selector])
        h_ref.add_(shift[selector])

        # Fused
        h_fused = fused_adaln_rms_norm(
            x,
            norm.weight,
            scale,
            shift,
            selector=selector,
            eps=norm.eps,
        )

        self.assertTrue(
            torch.equal(h_ref, h_fused),
            'Mismatch with per-row tensor selector',
        )

    def test_out_parameter_in_place_write(self):
        torch.manual_seed(20261110)
        M, D = 32, 7168
        x = torch.randn(M, D, dtype=torch.bfloat16)
        norm = nn.RMSNorm(D, eps=1e-6)
        norm.weight.data = torch.randn(D, dtype=torch.bfloat16)
        shift = torch.randn(1, D, dtype=torch.bfloat16)
        scale = torch.randn(1, D, dtype=torch.bfloat16)

        out_buffer = torch.empty((M, D), dtype=torch.bfloat16)
        res = fused_adaln_rms_norm(
            x,
            norm.weight,
            scale,
            shift,
            selector=0,
            eps=norm.eps,
            out=out_buffer,
        )

        self.assertIs(res, out_buffer)
        h_ref = norm(x.clone())
        h_ref.mul_(1.0 + scale[0])
        h_ref.add_(shift[0])
        self.assertTrue(torch.equal(out_buffer, h_ref))


if __name__ == '__main__':
    unittest.main()
