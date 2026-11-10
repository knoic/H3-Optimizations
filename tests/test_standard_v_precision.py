"""Standard must not add FP8 rounding before the existing INT8 V quantizer."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from h3_optimizations import kitchen_qkv
from h3_optimizations.native.v_staging import BACKEND_TORCH, TwoPassVCarrier
sys.argv = [sys.argv[0], *TEST_ARGS]


class StandardVPrecisionTests(unittest.TestCase):
    def test_both_producers_pack_original_v_with_global_scales(self):
        for dtype in (torch.bfloat16, torch.float16):
            for streamed in (False, True):
                for sequence in (17, 33):
                    for mode in ('retain', 'two_pass'):
                        with self.subTest(dtype=dtype, streamed=streamed, sequence=sequence, mode=mode):
                            self._check(dtype, streamed, sequence, mode)

    def _check(self, dtype, streamed, sequence, mode):
        # Include values that are changed by an FP8 round trip. Use the real
        # reference packer, including padding, instead of mocking quantization.
        values = torch.tensor([1.03, .97, .51, .26, .13, .065, -1.03, 0.], dtype=dtype)
        source = values.reshape(1, 1, 1, 8).repeat(1, 2, sequence, 1)
        source[..., 1::2, :] *= .75
        spec = SimpleNamespace(k_input_shape=source.shape, kernel_head_dim=8,
                               cta_k=16, device=torch.device('cpu'), input_dtype=dtype,
                               q_tile=16, k_tile=16, k_anchor_positions=(0,))
        reference = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        reference.update(source)
        reference.finalize_scale()
        reference.quantize(source, 0)
        expected_v, expected_scale = reference.finish()
        # Check the fixture would detect the original precision regression.
        rounded = source.to(torch.float8_e4m3fn).to(dtype)
        bad = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
        bad.update(source)
        bad.finalize_scale()
        bad.quantize(rounded, 0)
        self.assertFalse(torch.equal(bad.finish()[0], expected_v))

        producer = SimpleNamespace()
        retained_calls = []
        def quantize_retained(p, v):
            retained_calls.append(v.shape)
            self.assertEqual(v.dtype, source.dtype)
            self.assertTrue(torch.equal(v, source))
            packed = TwoPassVCarrier(spec, backend=BACKEND_TORCH)
            packed.update(v)
            packed.finalize_scale()
            packed.quantize(v, 0)
            p.v, p.v_scale = packed.finish()
        kitchen = SimpleNamespace(
            select_int8_attention_k_anchor=lambda *a: None,
            create_int8_attention_producer=lambda *a: producer,
            quantize_int8_attention_qk_chunk=lambda *a, **k: None,
            quantize_int8_attention_k_chunk=lambda *a, **k: None,
            finalize_int8_attention_producer=lambda p: p,
            quantize_int8_attention_v=quantize_retained,
        )
        v_calls = []
        def project_v(_held, _x, _rope, start, end):
            v_calls.append((start, end))
            return source[..., start:end, :]
        def kv(_held, _x, _rope, start, end):
            v = source[..., start:end, :]
            return v, v
        def qkv(_module, _x, _rope, start, end, **kwargs):
            k, v = kv(None, None, None, start, end)
            return k, k, v
        module = SimpleNamespace(qkv_proj=None, heads=2, head_dim=8)
        x = torch.zeros(sequence, 16, dtype=dtype)
        held = mock.MagicMock()
        with (
            mock.patch.object(kitchen_qkv, 'resolve_kitchen', return_value=kitchen),
            mock.patch.object(kitchen_qkv, 'describe_linear', return_value=SimpleNamespace()),
            mock.patch.object(kitchen_qkv, '_native_bf16_format', return_value=False),
            mock.patch.object(kitchen_qkv, '_project_anchor_samples', return_value=x[:1]),
            mock.patch.object(kitchen_qkv, 'create_held_qkv', return_value=held),
            mock.patch.object(kitchen_qkv, 'project_chunk_hnd', side_effect=qkv),
            mock.patch.object(kitchen_qkv, 'project_kv_hnd', side_effect=kv),
            mock.patch.object(kitchen_qkv, 'project_v_hnd', side_effect=project_v),
            # Exercise the old CUDA-only FP8 decision on CPU as well. All
            # arithmetic stays real CPU torch; this is not a GPU performance test.
            mock.patch.object(torch.Tensor, 'is_cuda', new_callable=mock.PropertyMock,
                              return_value=True),
        ):
            kwargs = dict(layer_index=0, transformer_options={}, spec=spec, chunk_rows=16, v_mode=mode)
            if streamed:
                result = kitchen_qkv.run_streamed_kitchen_qkv(
                    module, x, None, projection_mode='native', **kwargs)
            else:
                result = kitchen_qkv.run_chunked_kitchen_qkv(module, x, None, **kwargs)
        self.assertTrue(torch.equal(result.carrier.v, expected_v))
        self.assertTrue(torch.equal(result.carrier.v_scale, expected_scale))
        if mode == 'retain':
            self.assertEqual(retained_calls, [source.shape])
            self.assertEqual(v_calls, [])
        else:
            self.assertEqual(retained_calls, [])
            self.assertEqual(v_calls, [(s, min(s + 16, sequence)) for s in range(0, sequence, 16)])


if __name__ == '__main__':
    unittest.main()
