'''Unit tests for PCIe weight double-buffering and asynchronous prefetching.'''

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

# Provide fallback stubs if comfy is not installed in the test environment
if 'comfy' not in sys.modules:
    comfy = types.ModuleType('comfy')
    comfy.__path__ = []
    comfy.options = types.ModuleType('comfy.options')
    comfy.options.enable_args_parsing = lambda: None
    comfy.model_management = types.ModuleType('comfy.model_management')
    comfy.model_management.NUM_STREAMS = 2
    comfy.patcher_extension = types.ModuleType('comfy.patcher_extension')
    class CallbacksMP:
        ON_LOAD = 'on_load'
    comfy.patcher_extension.CallbacksMP = CallbacksMP
    comfy.model_prefetch = types.ModuleType('comfy.model_prefetch')
    comfy.model_prefetch.make_prefetch_queue = lambda *a, **kw: None
    comfy.model_prefetch.prefetch_queue_pop = lambda *a, **kw: None
    comfy.ldm = types.ModuleType('comfy.ldm')
    comfy.ldm.__path__ = []
    comfy.ldm.common_dit = types.ModuleType('comfy.ldm.common_dit')
    comfy.ldm.minimax = types.ModuleType('comfy.ldm.minimax')
    comfy.ldm.minimax.__path__ = []
    comfy.ldm.minimax.model = types.ModuleType('comfy.ldm.minimax.model')
    comfy.ldm.minimax.model.time_shift_sigma = lambda *a, **kw: 1.0
    comfy.ldm.minimax.model.MiniMaxH3Model = type('MiniMaxH3Model', (), {})
    comfy.ops = types.ModuleType('comfy.ops')
    comfy.ops.disable_weight_init = lambda: None
    comfy.quant_ops = types.ModuleType('comfy.quant_ops')
    comfy.quant_ops.QuantizedTensor = type('QuantizedTensor', (), {})
    comfy.quant_ops.TensorWiseINT8Layout = type('TensorWiseINT8Layout', (), {})
    comfy.quant_ops.get_layout_class = lambda *a, **kw: type('Layout', (), {})

    sys.modules['comfy'] = comfy
    sys.modules['comfy.options'] = comfy.options
    sys.modules['comfy.model_management'] = comfy.model_management
    sys.modules['comfy.patcher_extension'] = comfy.patcher_extension
    sys.modules['comfy.model_prefetch'] = comfy.model_prefetch
    sys.modules['comfy.ldm'] = comfy.ldm
    sys.modules['comfy.ldm.common_dit'] = comfy.ldm.common_dit
    sys.modules['comfy.ldm.minimax'] = comfy.ldm.minimax
    sys.modules['comfy.ldm.minimax.model'] = comfy.ldm.minimax.model
    sys.modules['comfy.ops'] = comfy.ops
    sys.modules['comfy.quant_ops'] = comfy.quant_ops

if 'comfy_api' not in sys.modules:
    comfy_api = types.ModuleType('comfy_api')
    comfy_api.__path__ = []
    comfy_api.latest = types.ModuleType('comfy_api.latest')
    class FakeInput:
        def __init__(self, *args, **kwargs):
            self.options = kwargs.get('options', [])
            self.default = kwargs.get('default')
            self.tooltip = kwargs.get('tooltip', '')
            self.id = args[0] if args else ''
    class FakeSchema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    class FakeIO:
        ComfyNode = object
        Schema = FakeSchema
        Model = types.SimpleNamespace(Input=FakeInput, Output=lambda: None)
        Combo = types.SimpleNamespace(Input=FakeInput)
        Int = types.SimpleNamespace(Input=FakeInput)
        NodeOutput = lambda *args, **kwargs: types.SimpleNamespace(args=args, kwargs=kwargs)
    class FakeUI:
        PreviewText = lambda text: text
    comfy_api.latest.io = FakeIO
    comfy_api.latest.ui = FakeUI
    sys.modules['comfy_api'] = comfy_api
    sys.modules['comfy_api.latest'] = comfy_api.latest

import torch
import torch.nn as nn

from h3_optimizations.memory.async_prefetch import (
    AsyncBlockDoubleBuffer,
    _prefetch_block_weights,
    _evict_block_weights,
)
from h3_optimizations import aimdo_limiter


class SimpleBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.norm(self.fc(x)) + x


class AsyncPrefetchTests(unittest.TestCase):
    def test_cpu_fallback_matches_sequential_execution_and_pop_calls(self):
        torch.manual_seed(42)
        dim = 32
        num_blocks = 4
        blocks = [SimpleBlock(dim) for _ in range(num_blocks)]
        x_init = torch.randn(2, 10, dim)

        # 1. Sequential execution
        x_seq = x_init.clone()
        for block in blocks:
            x_seq = block(x_seq)

        # 2. AsyncBlockDoubleBuffer on CPU
        mock_queue = mock.MagicMock()
        mock_pop_calls = []

        with mock.patch('h3_optimizations.memory.async_prefetch._safe_prefetch_pop',
                        side_effect=lambda q, dev, blk, **kw: mock_pop_calls.append(blk)):
            double_buffer = AsyncBlockDoubleBuffer(
                blocks=blocks,
                device='cpu',
                prefetch_queue=mock_queue,
                prefetch_scope={'malloc_scope': 'block'},
            )
            self.assertFalse(double_buffer.enabled)  # disabled on CPU

            x_buf = x_init.clone()
            for i, block in double_buffer.iterate():
                x_buf = block(x_buf)

        # Parity check: 100% bitwise identical
        self.assertTrue(torch.equal(x_seq, x_buf))

        # Check pop calls: block 0, 1, 2, 3, then None
        expected_pop_blocks = list(blocks) + [None]
        self.assertEqual(mock_pop_calls, expected_pop_blocks)

    def test_empty_blocks_exits_cleanly(self):
        mock_pop_calls = []
        with mock.patch('h3_optimizations.memory.async_prefetch._safe_prefetch_pop',
                        side_effect=lambda q, dev, blk, **kw: mock_pop_calls.append(blk)):
            double_buffer = AsyncBlockDoubleBuffer(
                blocks=[],
                device='cpu',
                prefetch_queue=object(),
            )
            count = 0
            for i, block in double_buffer.iterate():
                count += 1
        self.assertEqual(count, 0)
        self.assertEqual(mock_pop_calls, [None])

    def test_single_block_fallback(self):
        block0 = SimpleBlock(16)
        mock_pop_calls = []
        with mock.patch('h3_optimizations.memory.async_prefetch._safe_prefetch_pop',
                        side_effect=lambda q, dev, blk, **kw: mock_pop_calls.append(blk)):
            double_buffer = AsyncBlockDoubleBuffer(
                blocks=[block0],
                device='cpu',
                prefetch_queue=object(),
            )
            self.assertFalse(double_buffer.enabled)
            yielded = []
            for i, block in double_buffer.iterate():
                yielded.append((i, block))
        self.assertEqual(yielded, [(0, block0)])
        self.assertEqual(mock_pop_calls, [block0, None])

    def test_exception_safety_guarantees_cleanup_pop(self):
        blocks = [SimpleBlock(16), SimpleBlock(16)]
        mock_pop_calls = []
        with mock.patch('h3_optimizations.memory.async_prefetch._safe_prefetch_pop',
                        side_effect=lambda q, dev, blk, **kw: mock_pop_calls.append(blk)):
            double_buffer = AsyncBlockDoubleBuffer(
                blocks=blocks,
                device='cpu',
                prefetch_queue=object(),
            )
            with self.assertRaises(RuntimeError):
                for i, block in double_buffer.iterate():
                    if i == 0:
                        raise RuntimeError('Test failure in block 0')

        # None must be popped in finally
        self.assertIn(None, mock_pop_calls)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for async double buffering test')
    def test_cuda_async_double_buffering_parity(self):
        device = torch.device('cuda:0')
        torch.manual_seed(123)
        dim = 64
        num_blocks = 6
        blocks = [SimpleBlock(dim).to(device) for _ in range(num_blocks)]
        x_init = torch.randn(2, 16, dim, device=device)

        # 1. Sequential ground truth
        x_expected = x_init.clone()
        for block in blocks:
            x_expected = block(x_expected)

        # 2. AsyncBlockDoubleBuffer execution
        double_buffer = AsyncBlockDoubleBuffer(
            blocks=blocks,
            device=device,
            enabled=True,
        )
        self.assertTrue(double_buffer.enabled)

        x_actual = x_init.clone()
        for i, block in double_buffer.iterate():
            x_actual = block(x_actual)

        # 100% Bitwise Parity verification
        self.assertTrue(torch.equal(x_expected, x_actual))

    def test_mock_cuda_double_buffering_stream_pipeline(self):
        class MockEvent:
            def __init__(self, name):
                self.name = name
                self.records = []
            def record(self, stream):
                self.records.append(stream)

        class MockStream:
            def __init__(self, name):
                self.name = name
                self.waited_events = []
                self.synchronized = False
            def wait_event(self, event):
                self.waited_events.append(event)
            def synchronize(self):
                self.synchronized = True
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        compute_stream = MockStream('compute')
        dma_stream = MockStream('dma')
        event_counter = [0]
        all_events = []

        def make_event(*a, **kw):
            idx = event_counter[0]
            event_counter[0] += 1
            ev = MockEvent('event_%d' % idx)
            all_events.append(ev)
            return ev

        num_blocks = 4
        blocks = [SimpleBlock(16) for _ in range(num_blocks)]
        action_log = []

        with mock.patch('torch.cuda.is_available', return_value=True), \
             mock.patch('torch.cuda.current_stream', return_value=compute_stream), \
             mock.patch('torch.cuda.Stream', return_value=dma_stream), \
             mock.patch('torch.cuda.stream', side_effect=lambda s: s), \
             mock.patch('torch.cuda.Event', side_effect=make_event), \
             mock.patch('h3_optimizations.memory.async_prefetch._prefetch_block_weights',
                        side_effect=lambda blk, dev: action_log.append(('prefetch', blk))), \
             mock.patch('h3_optimizations.memory.async_prefetch._evict_block_weights',
                        side_effect=lambda blk, dev: action_log.append(('evict', blk))):

            double_buffer = AsyncBlockDoubleBuffer(
                blocks=blocks,
                device='cuda:0',
                enabled=True,
            )
            self.assertTrue(double_buffer.enabled)

            for i, block in double_buffer.iterate():
                action_log.append(('compute', block))

        # Check action sequencing:
        # Prologue prefetches block 0 and block 1
        self.assertEqual(action_log[0], ('prefetch', blocks[0]))
        self.assertEqual(action_log[1], ('prefetch', blocks[1]))
        # In iteration 0: prefetches block 2, computes block 0, evicts block 0
        self.assertEqual(action_log[2], ('prefetch', blocks[2]))
        self.assertEqual(action_log[3], ('compute', blocks[0]))
        self.assertEqual(action_log[4], ('evict', blocks[0]))
        # In iteration 1: prefetches block 3, computes block 1, evicts block 1
        self.assertEqual(action_log[5], ('prefetch', blocks[3]))
        self.assertEqual(action_log[6], ('compute', blocks[1]))
        self.assertEqual(action_log[7], ('evict', blocks[1]))
        # In iteration 2: no more blocks to prefetch, computes block 2, evicts block 2
        self.assertEqual(action_log[8], ('compute', blocks[2]))
        self.assertEqual(action_log[9], ('evict', blocks[2]))
        # In iteration 3: computes block 3, evicts block 3
        self.assertEqual(action_log[10], ('compute', blocks[3]))
        self.assertEqual(action_log[11], ('evict', blocks[3]))

        # Check synchronization:
        # compute_stream waited for each block ready event
        self.assertEqual(len(compute_stream.waited_events), num_blocks)
        # dma_stream waited for each compute done event
        self.assertEqual(len(dma_stream.waited_events), num_blocks)
        # dma_stream was synchronized in finally
        self.assertTrue(dma_stream.synchronized)

    def test_aimdo_limiter_double_buffer_residency(self):
        self.assertIn('double_buffer', aimdo_limiter.RESIDENCY_BLOCKS)
        self.assertEqual(aimdo_limiter.RESIDENCY_BLOCKS['double_buffer'], 2)

        class FakePatcher:
            def __init__(self):
                self.callbacks = {}
            def is_dynamic(self):
                return True
            def clone(self):
                return self
            def remove_callbacks_with_key(self, *a):
                pass
            def add_callback_with_key(self, cat, key, cb):
                self.callbacks.setdefault(cat, {})[key] = [cb]

        patcher = FakePatcher()
        with mock.patch.object(aimdo_limiter, 'is_minimax_h3', return_value=True):
            aimdo_limiter.ensure_double_buffer_residency(patcher, min_blocks=2)
            self.assertIn(aimdo_limiter.CALLBACK_KEY,
                          patcher.callbacks.get(aimdo_limiter.CallbacksMP.ON_LOAD, {}))


if __name__ == '__main__':
    unittest.main()
