# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end export of a macOS-style LLM, mirroring the coreai-models pipeline.

This test is coreai-torch's guard rail for the macOS LLM export path that
``coreai_models.export.macos.export_to_coreai`` drives. It reproduces that
pipeline against a tiny, randomly initialized model with the Qwen3-0.6B
*architecture* (fused QKV projection, fused Q/K RMSNorm, GQA, SiLU-gated MLP,
tied embeddings) scaled down to a few dozen hidden units so it exports and runs
in seconds:

    torch.export -> run_decompositions(coreai decomp table)
      -> remove_functionalization -> TorchConverter.add_pytorch_module(
             externalize_modules=..., state_names=...)
      -> register custom lowerings -> to_coreai()

The pieces that live in coreai-models rather than here — the mutable KV-cache
custom op, the ``remove_functionalization`` FX pass, and the MLIR lowering for
the immutable slice update — are reproduced below under a test-private
``coreai_torch_test`` operator namespace so this file stays self-contained and
cannot collide with a coreai-models install in the same interpreter.

What is actually asserted:
  * the exported IR keeps each composite (``rms_norm``, ``rope``,
    ``scaled_dot_product_attention``) as a private noinline graph and annotates
    the KV caches as mutable;
  * running prefill followed by two decode steps on the Core AI runtime
    reproduces PyTorch's logits, with the KV cache carried across calls.
"""

import operator
import platform
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from coreai._compiler.dialects import coreai
from coreai.runtime import NDArray
from torch import Tensor, fx
from torch._higher_order_ops.auto_functionalize import (
    AutoFunctionalized,
    AutoFunctionalizedV2,
)

from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
from coreai_torch.composite_ops import SDPA, RMSNorm, RMSNormImpl, RoPE

from ..utils import filecheck_pattern

# The pipeline compiles and executes a real asset, so it needs the runtime.
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="macOS LLM export is exercised on macOS only",
)

# Runner-visible state names, matching coreai_models._constants.
KEY_CACHE_NAME = "keyCache"
VALUE_CACHE_NAME = "valueCache"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TinyQwen3Config:
    """Qwen3-0.6B shaped config, scaled down so export and execution stay cheap.

    Qwen3-0.6B is hidden=1024 / 28 layers / 16 q heads / 8 kv heads /
    head_dim=128 / intermediate=3072 / vocab=151936. Every structural ratio is
    kept here (GQA with 2x more query heads than KV heads, head_dim not equal
    to hidden/n_heads, tied embeddings); only the magnitudes shrink.
    """

    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 16
    intermediate_size: int = 128
    vocab_size: int = 48
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_position_embeddings: int = 32
    tie_word_embeddings: bool = True


@dataclass(frozen=True)
class TraceSpec:
    """Shapes for the single ``torch.export`` trace, mirroring coreai-models.

    ``cache_seq_len`` is the length the caches are *traced* at (it only bounds
    peak trace memory); ``max_context_length`` is the upper bound of the dynamic
    sequence and cache dims at inference.
    """

    max_context_length: int = 32
    cache_seq_len: int = 16
    query_len: int = 4
    offset: int = 2


# ---------------------------------------------------------------------------
# KV-cache custom ops (test-private copies of coreai_models.primitives._ops)
# ---------------------------------------------------------------------------

_NS = "coreai_torch_test"


@torch.library.custom_op(f"{_NS}::mutable_cache_update_and_fetch", mutates_args=["x"])
def mutable_cache_update_and_fetch(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
    layer_idx: int,
    seq_len: int,
) -> Tensor:
    """Write ``update`` into ``x[begin:end]`` then fetch the populated prefix.

    ``x`` is the 5D cache ``(n_layers, 1, n_kv_heads, max_seq, head_dim)`` and
    ``update`` the 4D per-layer values. Returns
    ``x[layer_idx, :, :, :seq_len, :]``. ``begin``/``end`` are tensors rather
    than int lists because a custom op cannot take dynamic index lists.
    """
    update = update.unsqueeze(0)
    begins = torch.split(begin, 1, dim=0)
    ends = torch.split(end, 1, dim=0)
    slices = tuple(slice(b.item(), e.item()) for b, e in zip(begins, ends, strict=True))
    x[slices] = update
    fetched = x.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len)
    # Clone: the returned slice would otherwise alias the mutated cache.
    return fetched.squeeze(0).clone()


@mutable_cache_update_and_fetch.register_fake
def _mutable_cache_update_and_fetch_fake(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
    layer_idx: int,
    seq_len: int,
) -> Tensor:
    out_shape = list(x.shape)
    out_shape[-2] = seq_len
    out_shape.pop(0)  # squeeze the layer dim
    return torch.empty(out_shape, dtype=x.dtype)


@torch.library.custom_op(f"{_NS}::immutable_slice_update", mutates_args=())
def immutable_slice_update(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
) -> Tensor:
    """Non-mutating slice update, the export-time stand-in for the cache write."""
    result = x.clone()
    begins = torch.split(begin, 1, dim=0)
    ends = torch.split(end, 1, dim=0)
    slices = tuple(slice(b.item(), e.item()) for b, e in zip(begins, ends, strict=True))
    result[slices] = update
    return result


@immutable_slice_update.register_fake
def _immutable_slice_update_fake(
    x: Tensor,
    update: Tensor,
    begin: Tensor,
    end: Tensor,
) -> Tensor:
    return torch.empty(x.shape, dtype=x.dtype)


class KVCache:
    """Paged-by-layer KV cache held in two 5D tensors passed in as state."""

    #: Sequence dim of the 5D cache tensors.
    SEQ_DIM = 3

    def __init__(self, k_cache: Tensor, v_cache: Tensor) -> None:
        self._k_cache = k_cache
        self._v_cache = v_cache

    @staticmethod
    def zeros(
        config: TinyQwen3Config, dtype: torch.dtype, seq_len: int
    ) -> tuple[Tensor, Tensor]:
        shape = (
            config.num_hidden_layers,
            1,
            config.num_key_value_heads,
            seq_len,
            config.head_dim,
        )
        return torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)

    def _bounds(
        self, cache: Tensor, layer_idx: int, offset: int, update_len: int
    ) -> tuple[Tensor, Tensor]:
        """Per-dim begin/end index tensors for one layer's write."""

        def _t(value: int) -> Tensor:
            return torch.tensor((value,), dtype=torch.int32)

        begin = torch.cat([_t(layer_idx), _t(0), _t(0), _t(offset), _t(0)])
        end = torch.cat(
            [
                _t(layer_idx + 1),
                _t(cache.size(1)),
                _t(cache.size(2)),
                # offset is symbolic, so this end index stays traced.
                torch.tensor((offset + update_len,), dtype=torch.int32),
                _t(cache.size(4)),
            ]
        )
        return begin, end

    def update_and_fetch(
        self,
        layer_idx: int,
        offset: int,
        key: Tensor,
        value: Tensor,
        seq_len: int,
    ) -> tuple[Tensor, Tensor]:
        torch._check_is_size(layer_idx)
        torch._check_is_size(offset)
        torch._check_is_size(seq_len)
        torch._check(seq_len <= self._k_cache.size(self.SEQ_DIM))

        out = []
        for cache, update in ((self._k_cache, key), (self._v_cache, value)):
            begin, end = self._bounds(cache, layer_idx, offset, update.size(-2))
            out.append(
                mutable_cache_update_and_fetch(
                    x=cache,
                    update=update,
                    begin=begin,
                    end=end,
                    layer_idx=layer_idx,
                    seq_len=seq_len,
                )
            )
        return out[0], out[1]


# ---------------------------------------------------------------------------
# Model — Qwen3 architecture over the composite ops
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """SiLU-gated feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Up before gate, matching the macOS-tuned order in coreai-models.
        up = self.up_proj(x)
        gate = nn.functional.silu(self.gate_proj(x))
        return self.down_proj(up * gate)


class Attention(nn.Module):
    """GQA with a fused QKV projection and a fused Q/K RMSNorm."""

    def __init__(self, config: TinyQwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        qkv_out = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, qkv_out, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        # One norm covering the query and key heads, as Qwen3 exports on macOS.
        self.qk_norm = RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            n_heads=self.n_heads + self.n_kv_heads,
        )
        self.rope = RoPE(base=config.rope_theta)
        self.sdpa = SDPA(is_causal=True)

    def forward(self, x: Tensor, position_ids: Tensor, cache: KVCache) -> Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        query_key = self.qk_norm(query_key)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)

        query_key = self.rope(
            query_key, position_ids=position_ids.narrow(-1, offset, query_len)
        )
        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv_heads)

        key, value = cache.update_and_fetch(
            self.layer_idx, offset, key, value, seq_len=seq_len
        )

        attn = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * self.head_dim)
        )
        return self.o_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, config: TinyQwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, x: Tensor, position_ids: Tensor, cache: KVCache) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), position_ids, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class TinyQwen3ForCausalLM(nn.Module):
    """Randomly initialized Qwen3-shaped causal LM with a stateful KV cache."""

    def __init__(self, config: TinyQwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        keyCache: Tensor,  # noqa: N803 — runner-visible state name
        valueCache: Tensor,  # noqa: N803
    ) -> Tensor:
        cache = KVCache(keyCache, valueCache)
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.lm_head(self.norm(h))


def build_model(
    config: TinyQwen3Config, dtype: torch.dtype = torch.float32
) -> TinyQwen3ForCausalLM:
    """A deterministically seeded model with small random weights.

    Weights are scaled down from the default init so the residual stream stays
    in a range where fp32 torch and the Core AI runtime agree tightly.
    """
    torch.manual_seed(0)
    model = TinyQwen3ForCausalLM(config).to(dtype)
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                param.mul_(0.5)
    return model.eval()


# ---------------------------------------------------------------------------
# Export contract
# ---------------------------------------------------------------------------


def build_reference_inputs(
    config: TinyQwen3Config, spec: TraceSpec, dtype: torch.dtype
) -> dict[str, Any]:
    """Trace-time tensors, keyed in ``forward`` signature order."""
    total_len = spec.query_len + spec.offset
    k_cache, v_cache = KVCache.zeros(config, dtype, spec.cache_seq_len)
    return {
        "input_ids": torch.randint(
            1, config.vocab_size, (1, spec.query_len), dtype=torch.int32
        ),
        "position_ids": torch.arange(total_len, dtype=torch.int32).unsqueeze(0),
        KEY_CACHE_NAME: k_cache,
        VALUE_CACHE_NAME: v_cache,
    }


def build_dynamic_shapes(spec: TraceSpec) -> dict[str, Any]:
    """Dynamic sequence dims, keyed like :func:`build_reference_inputs`."""
    max_ctx = spec.max_context_length
    cache_dim = torch.export.Dim("cache_seq", min=spec.cache_seq_len, max=max_ctx)
    return {
        "input_ids": {1: torch.export.Dim("seq_ids", max=max_ctx - 2)},
        "position_ids": {
            1: torch.export.Dim("seq_pos", min=spec.query_len, max=max_ctx - 1)
        },
        KEY_CACHE_NAME: {KVCache.SEQ_DIM: cache_dim},
        VALUE_CACHE_NAME: {KVCache.SEQ_DIM: cache_dim},
    }


# ---------------------------------------------------------------------------
# remove_functionalization — test-private copy of coreai_models.export.mlir_ops
# ---------------------------------------------------------------------------


def _autofunc_base_tensor(node: fx.Node) -> Any:
    """The mutated base tensor an auto-functionalized node wraps.

    ``AutoFunctionalizedV2`` indirects it through ``_all_bases[_x_base_index]``;
    v1 passes it directly as the ``x`` kwarg.
    """
    if isinstance(node.target, AutoFunctionalizedV2):
        return node.kwargs["_all_bases"][node.kwargs["_x_base_index"]]
    return node.kwargs["x"]


def _copy_provenance(dst: fx.Node, src: fx.Node) -> None:
    dst.meta["val"] = src.meta["val"]
    dst.meta["nn_module_stack"] = src.meta.get("nn_module_stack", {})
    dst.meta["stack_trace"] = src.meta.get("stack_trace", "")
    dst.stack_trace = src.stack_trace


def remove_functionalization(program: torch.export.ExportedProgram) -> None:
    """Rewrite auto-functionalized cache updates into Core AI-representable ops.

    ``torch.export`` wraps every mutable custom op in a higher-order
    ``auto_functionalized`` node, which Core AI cannot represent (the op itself
    is an argument). Each one is replaced by
    ``unsqueeze -> immutable_slice_update -> slice(layer) -> slice(seq) ->
    squeeze``: getitem 0 of the fused op is the fetched slice feeding SDPA,
    getitem 1 is the mutated 5D cache.
    """
    graph = program.graph_module.graph
    autofuncs = [
        n
        for n in graph.nodes
        if isinstance(n.target, (AutoFunctionalized, AutoFunctionalizedV2))
    ]
    assert autofuncs, "expected auto-functionalized cache updates in the graph"

    retired: dict[str, fx.Node] = {}  # getitem name -> replacement node

    for autofunc in autofuncs:
        op_name = autofunc.args[0].name()
        assert op_name == f"{_NS}::mutable_cache_update_and_fetch", (
            f"unexpected auto-functionalized op {op_name}"
        )

        getitem_by_idx = {
            user.args[1]: user
            for user in autofunc.users
            if user.target is operator.getitem
        }
        getitem_fetched = getitem_by_idx.get(0)  # 4D slice -> SDPA
        getitem_cache = getitem_by_idx.get(1)  # 5D cache -> state handle
        assert getitem_cache is not None, (
            f"{autofunc.name}: no getitem at index 1, the cache write would be "
            "dead-code eliminated"
        )

        update = autofunc.kwargs["update"]
        layer_idx = autofunc.kwargs["layer_idx"]
        seq_len = autofunc.kwargs["seq_len"]

        with graph.inserting_before(autofunc):
            unsqueezed = graph.call_function(
                torch.ops.aten.unsqueeze.default, args=(update, 0)
            )
            unsqueezed.meta["val"] = update.meta["val"].unsqueeze(0)

            updated = graph.call_function(
                getattr(torch.ops, _NS).immutable_slice_update.default,
                args=(
                    _autofunc_base_tensor(autofunc),
                    unsqueezed,
                    autofunc.kwargs["begin"],
                    autofunc.kwargs["end"],
                ),
            )
            _copy_provenance(updated, getitem_cache)

            layer_slice = graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(updated, 0, layer_idx, layer_idx + 1),
            )
            layer_slice.meta["val"] = updated.meta["val"].narrow(0, layer_idx, 1)

            seq_slice = graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(layer_slice, KVCache.SEQ_DIM, 0, seq_len),
            )
            seq_len_val = (
                seq_len.meta["val"] if isinstance(seq_len, fx.Node) else seq_len
            )
            seq_slice.meta["val"] = layer_slice.meta["val"].narrow(
                KVCache.SEQ_DIM, 0, seq_len_val
            )

            squeezed = graph.call_function(
                torch.ops.aten.squeeze.dims, args=(seq_slice, [0])
            )
            if getitem_fetched is not None:
                _copy_provenance(squeezed, getitem_fetched)

        if getitem_fetched is not None:
            getitem_fetched.replace_all_uses_with(squeezed)
            retired[getitem_fetched.name] = squeezed
        getitem_cache.replace_all_uses_with(updated)
        retired[getitem_cache.name] = updated

    # Repoint the graph signature at the replacements before erasing the nodes,
    # otherwise the user-input-mutation outputs dangle.
    for spec in program.graph_signature.output_specs:
        if spec.arg.name in retired:
            spec.arg.name = retired[spec.arg.name].name

    for name in retired:
        graph.erase_node(next(n for n in graph.nodes if n.name == name))
    for autofunc in autofuncs:
        graph.erase_node(autofunc)

    program.graph_module.recompile()


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

#: Composites the macOS pipeline keeps as named graphs instead of inlining.
EXTERNALIZE_SPECS = [
    ExternalizeSpec(
        target_class=RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    ),
    ExternalizeSpec(
        target_class=RoPE,
        composite_op_name="rope",
        composite_attrs=["scale", "base", "dims", "interleaved"],
    ),
    ExternalizeSpec(
        target_class=SDPA,
        composite_op_name="scaled_dot_product_attention",
        composite_attrs=["scale", "is_causal", "window_size"],
    ),
]


def _lower_immutable_slice_update(
    values_map: dict[str, Any], node: fx.Node, location: Any
) -> Any:
    """Lower ``immutable_slice_update`` to ``coreai.slice_update``."""
    operands = []
    for arg in node.args[:4]:
        if isinstance(arg, fx.Node):
            operands.append(values_map[arg.name])
        else:
            data = arg.detach().cpu().numpy() if isinstance(arg, Tensor) else arg
            operands.append(coreai.constant(data, loc=location))
    x, update, begin, end = operands
    return coreai.slice_update(x, begin, end, [1] * x.type.rank, update)


def export_to_coreai(
    model: nn.Module,
    reference_inputs: dict[str, Any],
    dynamic_shapes: dict[str, Any],
) -> Any:
    """Run the macOS export pipeline and return the converted ``AIProgram``."""

    def export_fn(module: nn.Module) -> torch.export.ExportedProgram:
        with torch.no_grad():
            program = torch.export.export(
                module, args=(), kwargs=reference_inputs, dynamic_shapes=dynamic_shapes
            )
        program = program.run_decompositions(get_decomp_table())
        remove_functionalization(program)
        return program

    converter = TorchConverter()
    converter.add_pytorch_module(
        model,
        export_fn=export_fn,
        externalize_modules=EXTERNALIZE_SPECS,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    converter.register_torch_lowering(f"{_NS}::immutable_slice_update.default")(
        _lower_immutable_slice_update
    )
    # ``to_coreai`` already runs the pre-compilation rewrite, which is what
    # ``AIProgram.optimize()`` does in the coreai-models pipeline.
    return converter.to_coreai()


def _export_tiny_llm(
    config: TinyQwen3Config | None = None,
    spec: TraceSpec | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[TinyQwen3ForCausalLM, Any]:
    """Build and export the tiny model; returns ``(model, coreai_program)``."""
    config = config or TinyQwen3Config()
    spec = spec or TraceSpec()
    model = build_model(config, dtype)
    reference_inputs = build_reference_inputs(config, spec, dtype)
    program = export_to_coreai(model, reference_inputs, build_dynamic_shapes(spec))
    return model, program


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.ir
def test_macos_llm_ir() -> None:
    """The exported IR keeps the composites external and the caches mutable.

    Each composite becomes its own ``private noinline`` graph carrying a
    ``composite_declaration``, so the backend can pattern-match a fused kernel
    instead of the decomposed math, and both KV caches carry a
    ``MutableBuffers.buffer_mutation`` annotation.

    Only ``layers.0``'s composites are checked: whether identical call sites are
    deduplicated into one graph, and whether mutated inputs become
    ``!coreai.handle`` + ``!coreai.token`` or stay plain tensors, differ between
    coreai builds and are not part of this contract.
    """
    _, program = _export_tiny_llm()
    ir = str(program)

    filecheck_pattern(
        ir,
        check_file="""
            // CHECK-LABEL: module {
            // CHECK:   coreai.graph private noinline @layers.0.self_attn.qk_norm.rmsnorm_impl_{{[0-9a-f]+}}(
            // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm"
            // CHECK:   coreai.graph private noinline @layers.0.self_attn.rope_{{[0-9a-f]+}}(
            // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rope"
            // CHECK-SAME: base = 1.000000e+06
            // CHECK:   coreai.graph private noinline @layers.0.self_attn.sdpa_{{[0-9a-f]+}}(
            // CHECK-SAME: composite_decl = #coreai.composite_declaration<"scaled_dot_product_attention"
            // CHECK-SAME: is_causal = true
            // CHECK:   coreai.graph private noinline @layers.0.input_layernorm.rmsnorm_impl_{{[0-9a-f]+}}(
            // CHECK-SAME: composite_decl = #coreai.composite_declaration<"rms_norm"
            // CHECK:   coreai.graph @main(
            // CHECK-SAME: {coreai.name = "input_ids"}
            // CHECK-SAME: {coreai.name = "position_ids"}
            // CHECK-SAME: MutableBuffers.buffer_mutation = "keyCache"
            // CHECK-SAME: MutableBuffers.buffer_mutation = "valueCache"
            // CHECK-SAME: coreai.name = "logits"
            // CHECK: }
        """,
    )

    # One slice_update per (layer, cache): the lowering of the KV-cache write
    # that `remove_functionalization` turned into `immutable_slice_update`.
    expected_writes = 2 * TinyQwen3Config().num_hidden_layers
    assert ir.count("coreai.slice_update") == expected_writes


class _CacheChannel:
    """Drives the KV caches through whichever runtime channel the program uses.

    ``state_names=`` is one contract with two valid lowerings. Some coreai
    builds rewrite mutated inputs into ``!coreai.handle`` state, so the caches
    are passed via ``state=`` and mutated in place; others leave them as
    ordinary tensor inputs whose updated values come back as outputs and must
    be fed forward. Which one applies is read off the compiled function.
    """

    def __init__(self, rt_func: Any, k_cache: Tensor, v_cache: Tensor) -> None:
        self.stateful = bool(rt_func.desc.state_names)
        self._arrays = {
            KEY_CACHE_NAME: NDArray(k_cache.numpy().copy()),
            VALUE_CACHE_NAME: NDArray(v_cache.numpy().copy()),
        }

    def inputs(self) -> dict[str, NDArray]:
        return {} if self.stateful else dict(self._arrays)

    def state(self) -> dict[str, NDArray]:
        # The same NDArrays every call, so in-place mutation accumulates.
        return self._arrays if self.stateful else {}

    def absorb(self, outputs: dict[str, Any]) -> None:
        """Carry the updated caches forward; a no-op when mutated in place."""
        if self.stateful:
            return
        for name in self._arrays:
            self._arrays[name] = outputs[name]

    def numpy(self, name: str) -> np.ndarray:
        return self._arrays[name].numpy()


async def test_macos_llm_numerics() -> None:
    """Prefill plus two decode steps match PyTorch, with the cache carried over.

    The runtime cache is sized at the full context (32) while the trace used
    16, so this also covers the dynamic cache dim.
    """
    config = TinyQwen3Config()
    spec = TraceSpec()
    model, program = _export_tiny_llm(config, spec)

    context_len = spec.max_context_length
    k_cache, v_cache = KVCache.zeros(config, torch.float32, context_len)

    # (prefill of 4 tokens, then two single-token decode steps)
    steps = [4, 1, 1]
    torch.manual_seed(1)
    all_ids = torch.randint(1, config.vocab_size, (1, sum(steps)), dtype=torch.int32)

    with TemporaryDirectory(suffix=".aimodel") as tmp:
        asset = program.save_asset(Path(tmp))
        async with asset.executable() as ai_model:
            rt_func = ai_model.load_function("main")
            caches = _CacheChannel(rt_func, k_cache, v_cache)

            consumed = 0
            for step_idx, step_len in enumerate(steps):
                input_ids = all_ids[:, consumed : consumed + step_len]
                consumed += step_len
                position_ids = torch.arange(consumed, dtype=torch.int32).unsqueeze(0)

                # Torch mutates its caches in place, mirroring the runtime.
                expected = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    keyCache=k_cache,
                    valueCache=v_cache,
                )

                outputs = await rt_func(
                    inputs={
                        "input_ids": NDArray(input_ids.numpy()),
                        "position_ids": NDArray(position_ids.numpy()),
                        **caches.inputs(),
                    },
                    state=caches.state(),
                )
                caches.absorb(outputs)

                np.testing.assert_allclose(
                    outputs["logits"].numpy(),
                    expected.detach().numpy(),
                    rtol=1e-4,
                    atol=1e-3,
                    err_msg=f"logits mismatch on step {step_idx} (len {step_len})",
                )

            # The runtime caches must have tracked torch's writes across all
            # three calls, not just produced matching logits. Guard against a
            # vacuous all-zeros-vs-all-zeros pass: the written positions must be
            # non-zero, and everything past them untouched.
            assert k_cache[:, :, :, :consumed, :].abs().max() > 0, (
                "torch never wrote the KV cache"
            )
            assert k_cache[:, :, :, consumed:, :].abs().max() == 0, (
                "torch wrote past the current sequence length"
            )
            for name, expected_cache in (
                (KEY_CACHE_NAME, k_cache),
                (VALUE_CACHE_NAME, v_cache),
            ):
                np.testing.assert_allclose(
                    caches.numpy(name),
                    expected_cache.numpy(),
                    rtol=1e-4,
                    atol=1e-3,
                    err_msg=f"{name} diverged from torch",
                )
