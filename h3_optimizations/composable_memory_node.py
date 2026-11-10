'''Independent standard and low-VRAM layers sharing one reconciled H3 plan.'''

from dataclasses import replace

from comfy_api.latest import io, ui

from .apply_policy import apply_plan
from .memory_migration_node import (
    _memory_request_for_modes,
    KITCHEN_V_MEMORY_MODE_RETAIN,
    KITCHEN_V_MEMORY_MODE_TWO_PASS,
    PRECISION_MODE_PRESERVE_NATIVE,
    QKV_STREAMING_MODE_AUTO,
    QKV_STREAMING_MODE_OFF,
)
from .node_constants import DEFAULT_CHUNK_ROWS, NODE_CATEGORY
from .plan import (
    EMBEDDING_MEMORY_STOCK, FUSED_QKV_OFF, MLP_MEMORY_AUTO, MLP_MEMORY_OFF,
    MAX_CHUNK_ROWS, MIN_CHUNK_ROWS, QKV_STREAMING_OFF, read_plan,
)
from .status import format_memory_status


def layer_request(standard, low_vram, chunk_rows=DEFAULT_CHUNK_ROWS):
    if not isinstance(standard, bool) or not isinstance(low_vram, bool):
        raise ValueError('optimization layer switches must be booleans')
    request = _memory_request_for_modes(
        fused_qkv='auto', mlp_memory=MLP_MEMORY_AUTO, chunk_rows=chunk_rows,
        precision_mode=PRECISION_MODE_PRESERVE_NATIVE,
        qkv_streaming_mode=QKV_STREAMING_MODE_AUTO if standard or low_vram else QKV_STREAMING_MODE_OFF,
        kitchen_v_memory_mode=KITCHEN_V_MEMORY_MODE_TWO_PASS if low_vram else KITCHEN_V_MEMORY_MODE_RETAIN,
    )
    if not standard:
        request = replace(
            request, standard_optimizations=False, mlp_memory=MLP_MEMORY_OFF,
            embedding_memory=EMBEDDING_MEMORY_STOCK,
        )
    if not standard and not low_vram:
        request = replace(request, fused_qkv=FUSED_QKV_OFF, qkv_streaming=QKV_STREAMING_OFF)
    return request


class H3ComposableMemoryOptimization(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3ComposableMemoryOptimization',
            display_name='H3 Optimization Layers', category=NODE_CATEGORY,
            description='Standard and low-VRAM execution are independent layers. Both may be enabled together. '
                        'Preserves native precision and the existing attention consumer.',
            inputs=[
                io.Model.Input('model'),
                io.Boolean.Input('standard_optimizations', display_name='Standard optimizations', default=True,
                                 tooltip='Bounded MLP/FinalLayer, embedding lifetime and compatible streamed QKV.'),
                io.Boolean.Input('low_vram_optimizations', display_name='Low VRAM layer', default=False,
                                 tooltip='Two-pass V on compatible attention paths. Works alone or on top of Standard. '
                                         'Requires extra projection work; unsupported external consumers stay unchanged.'),
                io.Int.Input('chunk_rows', display_name='Activation chunk rows', default=DEFAULT_CHUNK_ROWS,
                             min=MIN_CHUNK_ROWS, max=MAX_CHUNK_ROWS, step=256, advanced=True),
            ], outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, standard_optimizations=True, low_vram_optimizations=False,
                chunk_rows=DEFAULT_CHUNK_ROWS):
        request = layer_request(standard_optimizations, low_vram_optimizations, chunk_rows)
        if not standard_optimizations and not low_vram_optimizations:
            return io.NodeOutput(model, ui=ui.PreviewText(
                'Optimization layers: Standard off; Low VRAM off. Input model unchanged.',
            ))
        # One request owns shared QKV machinery. Never nest separate patch stacks.
        plan = read_plan(model).with_memory(request)
        patched = apply_plan(model, plan)
        return io.NodeOutput(patched, ui=ui.PreviewText(format_memory_status(patched)))
