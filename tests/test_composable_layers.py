'''Independent layer combinations, cleanup, and shared-plan contracts.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
from h3_optimizations import composable_memory_node as node
from h3_optimizations import apply
from h3_optimizations.memory import patch as block_patch, final_layer
from h3_optimizations.plan import H3OptimizationPlan, STATUS_KEY
from h3_optimizations.status import format_memory_status
sys.argv = [sys.argv[0], *TEST_ARGS]


class LayerTests(unittest.TestCase):
    def test_four_combinations_have_independent_ownership(self):
        for standard in (False, True):
            for low in (False, True):
                with self.subTest(standard=standard, low=low):
                    request = node.layer_request(standard, low, 1024)
                    self.assertEqual(request.standard_optimizations, standard)
                    self.assertEqual(request.attention_v_memory, 'two_pass' if low else 'retain')
                    self.assertEqual(request.mlp_memory == 'off', not standard)
                    self.assertEqual(request.qkv_streaming, 'auto' if standard or low else 'off')
                    self.assertNotEqual(request.fused_qkv, 'force_quant')
                    self.assertEqual(request.embedding_memory, 'auto' if standard else 'stock')

    def test_shared_plan_applied_once_for_each_enabled_combination(self):
        for standard, low in ((True, False), (False, True), (True, True)):
            model = SimpleNamespace(model_options={})
            with mock.patch.object(node, 'apply_plan', return_value=model) as install:
                node.H3ComposableMemoryOptimization.execute(model, standard, low)
            install.assert_called_once()
            plan = install.call_args.args[1]
            self.assertEqual(plan.memory, node.layer_request(standard, low))

    def test_both_off_is_identity_and_preserves_external_model(self):
        model = object()
        with mock.patch.object(node, 'apply_plan') as install:
            result = node.H3ComposableMemoryOptimization.execute(model, False, False)
        self.assertIs(result.result[0], model)
        install.assert_not_called()

    def test_low_only_removes_owned_standard_components(self):
        plan = H3OptimizationPlan(memory=node.layer_request(False, True))
        patcher = object()
        with (
            mock.patch.object(apply, 'clear_memory_patch') as blocks,
            mock.patch.object(apply, 'clear_final_layer_chunking') as final,
            mock.patch.object(apply, 'clear_embedding_memory') as embedding,
            mock.patch.object(apply, 'install_final_layer') as install_final,
            mock.patch.object(apply, 'install_memory_patch') as install_blocks,
            mock.patch.object(apply, 'resolve_mlp_provider', return_value='off'),
        ):
            self.assertEqual(apply._install_mlp(patcher, plan, None, None), ('off', 0))
        for clear in (blocks, final, embedding):
            clear.assert_called_once_with(patcher)
        install_final.assert_not_called()
        install_blocks.assert_not_called()

    def test_clear_blocks_preserves_foreign_and_restores_owned_forward(self):
        stock = lambda *args: None
        owned = lambda *args: None
        owned._h3_optimizations_memory = True
        owned._h3_optimizations_memory_original = stock
        foreign = lambda *args: None
        model = SimpleNamespace(object_patches={block_patch.key_for(0): owned,
                                                block_patch.key_for(1): foreign}, model_options={})
        with mock.patch.object(block_patch, 'get_h3_blocks', return_value=[SimpleNamespace(forward=stock)] * 2):
            block_patch.clear(model)
        self.assertNotIn(block_patch.key_for(0), model.object_patches)
        self.assertIs(model.object_patches[block_patch.key_for(1)], foreign)

    def test_clear_final_layer_preserves_foreign_forward(self):
        foreign = lambda *args: None
        model = SimpleNamespace(object_patches={final_layer.FINAL_LAYER_KEY: foreign})
        self.assertFalse(final_layer.clear_chunking(model))
        self.assertIs(model.object_patches[final_layer.FINAL_LAYER_KEY], foreign)

    def test_clear_final_layer_removes_owned_chunking(self):
        stock = lambda x, t_emb, video_seg, audio_seg: (x, None)
        layer = SimpleNamespace(forward=stock)
        owned = lambda *args: None
        setattr(owned, final_layer.OWNER_MARKER, True)
        setattr(owned, final_layer.ORIGINAL_MARKER, stock)
        model = SimpleNamespace(object_patches={final_layer.FINAL_LAYER_KEY: owned})
        with mock.patch.object(final_layer, 'get_minimax_h3_model', return_value=SimpleNamespace(final_layer=layer)):
            self.assertTrue(final_layer.clear_chunking(model))
        self.assertNotIn(final_layer.FINAL_LAYER_KEY, model.object_patches)

    def test_clear_final_layer_keeps_independent_cube_order(self):
        stock = lambda x, t_emb, video_seg, audio_seg: (x, None)
        layer = SimpleNamespace(forward=stock)
        cube = SimpleNamespace(resolve=lambda rows: (None, True))
        owned = lambda *args: None
        setattr(owned, final_layer.OWNER_MARKER, True)
        setattr(owned, final_layer.ORIGINAL_MARKER, stock)
        setattr(owned, final_layer.CUBE_STATE_MARKER, cube)
        model = SimpleNamespace(object_patches={final_layer.FINAL_LAYER_KEY: owned})
        with mock.patch.object(final_layer, 'get_minimax_h3_model', return_value=SimpleNamespace(final_layer=layer)):
            self.assertTrue(final_layer.clear_chunking(model))
        restored = model.object_patches[final_layer.FINAL_LAYER_KEY]
        self.assertIs(getattr(restored, final_layer.CUBE_STATE_MARKER), cube)
        self.assertIsNone(getattr(restored, final_layer.SIGNATURE_MARKER))

    def test_external_sla_no_projector_is_reported_as_not_applied(self):
        status = {'attention': {'selected': 'existing'}, 'mlp': {'provider': 'off'},
                  'layers': {'standard': False, 'low_vram': True},
                  'fused_qkv': {'provider': 'standard_h3_qkv', 'projector': None,
                                'v_memory_requested': 'two_pass', 'v_memory': None,
                                'reason': 'preserved external SLA full-Q semantics'}}
        model = SimpleNamespace(model_options={'transformer_options': {STATUS_KEY: status}})
        text = format_memory_status(model)
        self.assertIn('Standard off; Low VRAM on', text)
        self.assertIn('not applied', text)
        self.assertIn('external SLA', text)

    def test_signature_distinguishes_standard_and_low_only(self):
        self.assertNotEqual(node.layer_request(True, True).signature, node.layer_request(False, True).signature)


if __name__ == '__main__':
    unittest.main()
