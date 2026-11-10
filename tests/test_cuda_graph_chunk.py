'''Unit tests for chunk-level static execution and CUDA graph replay parity.'''

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_optimizations.fused_adaln_norm import fused_adaln_rms_norm
from h3_optimizations.fused_swiglu import fused_swiglu_inplace_


class ChunkExecutionTests(unittest.TestCase):
    def test_mlp_chunk_eager_vs_graph_parity(self):
        torch.manual_seed(20261110)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rows, hidden, ffn = 128, 512, 1024

        x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
        norm_weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        scale = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
        shift = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
        gate = torch.randn(1, hidden, device=device, dtype=torch.bfloat16)
        fc1_weight = torch.randn(ffn * 2, hidden, device=device, dtype=torch.bfloat16)
        fc2_weight = torch.randn(hidden, ffn, device=device, dtype=torch.bfloat16)

        def step(target_x):
            h = fused_adaln_rms_norm(
                target_x,
                norm_weight,
                scale,
                shift,
                selector=0,
                eps=1e-6,
            )
            expanded = F.linear(h, fc1_weight)
            activated = fused_swiglu_inplace_(expanded)
            out = F.linear(activated, fc2_weight)
            target_x.addcmul_(out, gate[0])
            return target_x

        # Ground truth: eager execution
        x_eager = x.clone()
        step(x_eager)

        # Static buffer execution
        x_static = x.clone()
        static_buf = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)
        static_buf.copy_(x_static)
        step(static_buf)
        x_static.copy_(static_buf)

        self.assertTrue(
            torch.equal(x_eager, x_static),
            'Static buffer copy path must match eager step bit-identically',
        )

        # If CUDA is available, test real CUDA Graph capture and replay
        if torch.cuda.is_available() and hasattr(torch.cuda, 'CUDAGraph'):
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(s):
                for _ in range(3):
                    step(static_buf)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                step(static_buf)
            torch.cuda.current_stream(device=device).wait_stream(s)

            x_graph = x.clone()
            static_buf.copy_(x_graph)
            g.replay()
            x_graph.copy_(static_buf)

            # Note: Float operations in graph match step exactly
            diff = (x_eager.float() - x_graph.float()).abs().max().item()
            self.assertLessEqual(diff, 1e-4)


if __name__ == '__main__':
    unittest.main()
