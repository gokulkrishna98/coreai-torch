# Externalization Design

Externalization extracts a PyTorch `nn.Module` subgraph — an `RMSNorm`, an attention
block, a custom layer — from a parent model and lowers it into a standalone
`coreai.graph noinline` that the backend can recognize and optimize independently
(e.g. as a composite op for Core ML).

**The core problem:** once a model passes through `torch.export`, a quantizer, or any
FX pass, `nn_module_stack` metadata is often incomplete or erased.  Externalization
solves this by replacing each target submodule's `forward` with a
`torch.library.custom_op` *before* export.  The custom op bakes an opaque
`call_function` node into the FX graph that is invisible to decompositions, quantizers,
and other passes — it survives all transformations and unambiguously marks the subgraph
boundary.

---

## Two Workflows

Both workflows share the same underlying spine — mark, export, `_subexport_and_restore` —
implemented once in `coreai_torch/externalize.py`. They differ only in **who calls
`export_fn`/the external tool** and **who calls `_subexport_and_restore`**.

### Workflow A — `add_pytorch_module` (self-contained)

Use when you own the export step and have no external FX-level tools in the pipeline.
`TorchConverter` handles marking, re-export, sub-export, and cleanup internally —
under the hood, `to_coreai()` calls the same public `_patch_model_for_externalization(module,
targets)` you'd call yourself in Workflow B, then calls
`_subexport_and_restore(module, whole_ep, progress_bar=...)`, wiring its own progress
bar through for per-submodule reporting.

```python
coreai_program = (
    TorchConverter()
    .add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, args=sample),
        externalize_modules=[
            ExternalizeSpec(RMSNorm, composite_op_name="rms_norm",
                            composite_attrs=["eps"]),
        ],
    )
    .to_coreai()
)
```

### Workflow B — `_patch_model_for_externalization` (decoupled)

Use when an external tool (quantizer, pruner, compiler pass) owns the
`ExportedProgram`.  Mark the model first, run the tool, then call
`_subexport_and_restore(model, ep)` yourself to get back the exported submodules
and pass that list to the converter — this is the same `_subexport_and_restore` call
Workflow A makes internally, just invoked explicitly with no progress bar.

```python
from coreai_torch import _patch_model_for_externalization, _subexport_and_restore, TorchConverter

_patch_model_for_externalization(model, [
    ExternalizeSpec(RMSNorm, composite_op_name="rms_norm", composite_attrs=["eps"]),
])

# Pseudocode — substitute your actual quantizer/tool API here:
ep = quantizer.prepare(model).calibrate(data).finalize()
# custom op nodes survive quantization — model.forward is still patched here

# Sub-export each marked submodule (Phases 2-3) and restore the model,
# via try/finally, regardless of whether sub-export succeeds.
_externalized_exported_programs = _subexport_and_restore(model, ep)

coreai_program = (
    TorchConverter()
    .add_exported_program(ep, _externalized_exported_programs=_externalized_exported_programs)
    .to_coreai()
)
```

`_subexport_and_restore` always runs inside its own `try/finally`, so no manual
restore call is needed — by the time it returns, the model is back to its
pre-`_patch_model_for_externalization` state, whether or not sub-export succeeded.

---

## Pipeline Overview

Both workflows execute the same four phases, and both funnel Phases 2–3 through the
exact same code — the standalone `_subexport_and_restore(model, ep)` function. The
only difference is *where* Phase 1 (mark) and the export step happen, and who calls
`_subexport_and_restore`:

```
  Workflow A (add_pytorch_module)          Workflow B (_patch_model_for_externalization)
  ══════════════════════════════           ═════════════════════════════════════

  ┌─ inside to_coreai() ──────────┐        ┌─ user code ──────────────────────┐
  │                                │        │                                    │
  │  _patch_model_for_externalization(     │        │  _patch_model_for_externalization(         │
  │      module, targets)          │        │      model, targets)               │
  │         │                      │        │         │                          │
  │         ▼                      │        │         ▼                          │
  │  patches applied in-place;     │        │  patches applied in-place;         │
  │  stamps stored on submodules   │        │  stamps stored on submodules       │
  │         │                      │        │         │                          │
  │         ▼                      │        │         ▼                          │
  │  export_fn(module)             │        │  quantizer / export_fn(model)      │
  │         │                      │        │         │                          │
  │         ▼                      │        │         ▼                          │
  │  _subexport_and_restore(        │        │  _subexport_and_restore(            │
  │      module, whole_ep,         │        │      model, ep)                    │
  │      progress_bar=self._bar)   │        │      (no progress_bar)             │
  │         │                      │        │         │                          │
  └─────────┼──────────────────────┘        └─────────┼──────────────────────────┘
            │                                         │
            ▼                                         ▼
  self._externalized_exported_programs             _externalized_exported_programs = [...]
     = [...]  (stashed directly              │
        on the converter)                    ▼
            │                       add_exported_program(
            │                           ep, _externalized_exported_programs=[...])
            │                                         │
            └─────────────────────┬───────────────────┘
                                   ▼
                    ┌─ to_coreai(): Phase 4 ────────────┐
                    │  build coreai.graph noinline      │
                    │  register coreai.invoke           │
                    │    lowerings per FX node name      │
                    └────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      coreai.Program          │
                    │  ├── @main                   │
                    │  │     coreai.invoke          │
                    │  │       @rms_norm_abc123     │
                    │  ├── @rms_norm_abc123  ◄──── │── noinline subgraph
                    │  └── @rms_norm_def456  ◄──── │── second call site
                    └──────────────────────────────┘
```

Both branches call the exact same standalone `_subexport_and_restore(model, ep)` —
the only differences are *who* calls `export_fn`/the external tool (the converter vs.
the caller), and whether a `progress_bar` is threaded through (Workflow A only,
since it runs inside `to_coreai()`'s own progress-bar context).

Inside `_subexport_and_restore` (Phases 2-3), it first rediscovers marked submodules by
walking `model` for the stamps `_patch_model_for_externalization` left behind, then walks them
in **shallowest-first** order (`_PreparedModules.__iter__`) and, for each one, finds its
call-site nodes in the FX graph, extracts everything needed to re-export the
submodule standalone (Phase 2), then runs `torch.export.export` + `run_decompositions()`
on each one (Phase 3) — detailed below.

---


## Phase 1: Mark

**API:** `_patch_model_for_externalization(model, targets)` → `None` — called
directly by the user (Workflow B) or by `_run_externalize_pipeline_from_module`
inside `to_coreai()` (Workflow A); both paths call the exact same public function.
**Internal:** `_mark_externalize(model, targets)` + `_prepare_module(model, submodule)`

Patches are applied immediately when `_patch_model_for_externalization` is called — whether
that's the user calling it explicitly (Workflow B) or `to_coreai()` calling it at the
start of `_run_externalize_pipeline_from_module` (Workflow A). All patch details are
stamped directly onto each matched submodule instance — there is no separate handle
object tracking marked submodules; `_subexport_and_restore` rediscovers them later by
walking the model for these stamps (`_find_marked_submodules`).

```
  model.named_modules()
  │
  ├── encoder.norm   isinstance(mod, RMSNorm) ✓
  │       │
  │       │  1. name     = "encoder.norm"
  │       │  2. op_name  = "encoder_norm"          (dots → underscores)
  │       │  3. register torch.library.custom_op(
  │       │         "coreai_torch_ext::encoder_norm",
  │       │          impl = original_forward         ← real weights, used by torch.export
  │       │     )
  │       │  4. register_fake("coreai_torch_ext::encoder_norm",
  │       │          original_forward               ← shape inference for FakeTensor tracing
  │       │     )
  │       │  5. submodule.forward = patched_forward  ← calls custom_op, not original
  │       │  6. stamp on submodule:
  │       │         _original_forward  = <original>
  │       │         _externalize_name  = "encoder.norm"
  │       │         _externalize_op_name = "encoder_norm"
  │       │         _externalize_config  = ExternalizeSpec(...)
  │       │
  │       ▼
  │   submodule.forward now wraps coreai_torch_ext::encoder_norm
  │
  ├── decoder.norm   isinstance(mod, RMSNorm) ✓
  │       └── (same process → coreai_torch_ext::decoder_norm)
  │
  └── lm_head        isinstance(mod, RMSNorm) ✗  (skipped)

returns None   ← each matched submodule is stamped with _externalize_name
                 (and friends) in place, for later rediscovery
```

After marking, any call to `model(...)` or `torch.export.export(model, ...)` will
produce FX graphs containing opaque `call_function` nodes instead of the submodule
body.  Quantizers and decompositions cannot see through them.

> **Training note:** The autograd registration re-runs `original_forward` under
> `torch.enable_grad()` on every backward pass to reconstruct the inner graph.
> This means backward cost is roughly 1.5× a normal backward (forward runs twice),
> and stateful submodules (BatchNorm running stats, dropout, RNG) are observed
> twice per training step.  `_patch_model_for_externalization` is **not** a transparent
> drop-in for training loops with stateful submodules; it is designed for
> inference and QAT (where the exported model is eval-mode).

---

## Phase 2: Prepare

**API:** inside `_subexport_and_restore(model, ep)` — shared by both workflows
**Internal:** `_PreparedModules.__iter__` → `_prepare_module_export(submodule, ep)`

Walks marked submodules in **shallowest-first** order and, for each one, finds its
call-site nodes in the FX graph and extracts everything needed to re-export the
submodule standalone.

```
  Marked modules (sorted by path.count(".")):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  depth 1: "encoder.norm"   depth 1: "decoder.norm"   (depth 2 after)   │
  └─────────────────────────────────────────────────────────────────────────┘

  For each module (shallowest first):

  ep.graph
  ┌────────────────────────────────────────────────────────┐
  │  placeholder %x                                        │
  │  ...                                                   │
  │  %norm_out = call_function[                            │
  │      target=coreai_torch_ext.encoder_norm.default      │◄── found by op name
  │  ](args=(%x, %weight))                                 │
  │  ...                                                   │
  └────────────────────────────────────────────────────────┘
          │
          │  per call-site node:
          │    fake_inputs    = (Tensor[2,8,16,fp32], Tensor[16,fp32])
          │                     extracted from FakeTensor metadata on node.args
          │    dynamic_shapes = ({0: batch_dim}, None)
          │                     reconstructed from SymInt in FakeTensors
          │    name           = "encoder.norm_3f8a1b2c"   (UUID suffix)
          │    op_name        = "encoder_norm"
          │    module_path    = "encoder.norm"
          │
          ▼
  _PreparedModule(name, op_name, module_path, module, fake_inputs,
                  dynamic_shapes, composite_op_name, composite_decl_attrs,
                  source_nodes=[node])

  Note: submodule.forward is RESTORED to _original_forward here, so
  torch.export.export(submodule, ...) in Phase 3 sees the real implementation.
```

Before walking modules, `_subexport_and_restore` calls
`_drop_missing_call_sites(model, ep)`: for each **top-level** marked submodule with
no custom-op node anywhere in `ep` (e.g. the model was exported before marking, or
before the tool ran), it emits a `UserWarning` and restores that submodule's
patch immediately (and any of its nested marked children — their call sites only
exist inside the missing parent's own sub-export). It rediscovers marked submodules
itself via `_find_marked_submodules(model)` and returns nothing — the next
`_find_marked_submodules(model)` call (in `_PreparedModules`) naturally excludes
whatever it just restored.

One `_PreparedModule` is created **per call-site node**.  If the same module is called
twice, two `_PreparedModule` objects are created with different UUID suffixes, producing
two distinct `coreai.graph` symbols at runtime.

---

## Phase 3: Export Submodules

**API:** inside `_subexport_and_restore(model, ep)` — shared by both workflows
**Internal:** `_torch_export_module(prep)` → `_finalize_module_export(prep, inner_ep)`

Re-exports each submodule standalone using the fake inputs and dynamic shapes captured
in Phase 2.

```
  _PreparedModule
  ├── module        = <RMSNorm instance, forward restored>
  ├── fake_inputs   = (Tensor[2,8,16,fp32], Tensor[16,fp32])
  └── dynamic_shapes = ({0: batch_dim}, None)
          │
          ▼
  torch.export.export(
      module,
      args=fake_inputs,
      dynamic_shapes=dynamic_shapes,
  )
          │
          │  Optional-arg handling: if the call site passed None for a
          │  positional arg (e.g. SDPA's mask), fake_inputs are passed
          │  as kwargs keyed by forward parameter names so torch.export
          │  receives the correct signature.
          │
          ▼
  ExportedProgram (submodule only)
          │
          ├── run_decompositions()   ← always decomposed, regardless of parent
          │
          ├── derive composite I/O names from graph_signature:
          │     USER_INPUT  → forward parameter name  (e.g. "x")
          │     PARAMETER   → attribute target path   (e.g. "weight")
          │     BUFFER      → attribute target path
          │     USER_OUTPUT → "output" or "output_0", "output_1", ...
          │
          ▼
  _ExternalizedExportedProgram(
      name             = "encoder.norm_3f8a1b2c",
      op_name          = "encoder_norm",
      exported_program = <decomposed EP>,
      composite_op_name      = "rms_norm",
      composite_decl_attrs   = {"eps": 1e-5},
      composite_input_names  = ["x", "weight"],
      composite_output_names = ["output"],
      source_nodes           = ["encoder_norm_default"],  ← FX node name string
  )
```

If `torch.export.export(module, ...)` raises here, `_subexport_and_restore` wraps it as
`RuntimeError("Internal error: failed to export submodule '...'. This is a
coreai-torch bug.")` — by this point `exported_program` (the whole model) already
contained the call site, so a failure re-exporting it standalone indicates a
coreai-torch bug rather than a caller mistake, for either workflow. When called with
`progress_bar=...` (Workflow A only), each completed call site advances a
`"Externalizing submodules"` progress bar; Workflow B's direct call omits it and
runs silently.

After all `_ExternalizedExportedProgram` objects are built, `_subexport_and_restore` runs
`_restore_externalized(_find_marked_submodules(model))` in a `finally` block:

```
  finally:
      _restore_externalized(_find_marked_submodules(model))
          │
          └── for each mod still marked:
                  mod.forward = mod._original_forward
                  del mod._original_forward
                  del mod._externalize_name
                  del mod._externalize_op_name
                  del mod._externalize_config   (if present)
```

The model is now in exactly the state it was before `_patch_model_for_externalization`.

---

## Phase 4: Emit Core AI IR

**API:** inside `to_coreai()`
**Internal:** `_perform_externalization(context)` in `converter.py`

Processes `_ExternalizedExportedProgram` objects in **deepest-first** order so that inner (nested)
graphs are emitted before the outer graphs that reference them.

```
  _ExternalizedExportedProgram list (sorted deepest-first by name.count(".")):
  ["encoder.norm_abc", "decoder.norm_def", ...]

  For each _ExternalizedExportedProgram:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  converter.exported_program ← temporarily swapped to submodule EP      │
  │                                                                         │
  │  Plain externalization:                                                 │
  │    coreai.graph noinline @encoder_norm_abc123(%x, %weight) { ... }     │
  │                                                                         │
  │  Composite op (composite_op_name is set):                               │
  │    coreai.graph private noinline @encoder_norm_abc123(...)              │
  │        attributes { composite_decl = #coreai.composite_declaration<    │
  │            name = "rms_norm",                                           │
  │            inputs = ["x", "weight"],                                    │
  │            outputs = ["output"],                                        │
  │            attributes = {eps = 1.0e-5}                                 │
  │        >} { ... }                                                       │
  │                                                                         │
  │  Register per-node lowering:                                            │
  │    _externalized_lowerings["encoder_norm_default"] =                    │
  │        lambda: coreai.invoke @encoder_norm_abc123(...)                  │
  │                                                                         │
  │  converter.exported_program ← restored to parent EP                    │
  └─────────────────────────────────────────────────────────────────────────┘

  Lower parent EP — node dispatch checks _externalized_lowerings first:

  FX node name in _externalized_lowerings?
  ┌── YES → emit  %0 = coreai.invoke @encoder_norm_abc123(%x, %w)
  └── NO  → lower normally (aten.add, aten.mm, ...)
```

**Before vs. after:**

```
  Parent FX graph                        Core AI IR

  placeholder %x                         %x = coreai.graph_input
  placeholder %weight                    %w = coreai.graph_input
  %norm = coreai_torch_ext               %0 = coreai.invoke
            .encoder_norm(%x, %weight)         @encoder_norm_abc123(%x, %w)
  %out  = aten.linear(%norm, ...)        %1 = coreai.linear(%0, ...)
  output %out                            coreai.return %1
```

---

## Ordering Invariants

Two ordering rules keep nested externalization correct:

```
  model
  ├── block                    ← ExternalizeSpec(Block)
  │   ├── attn                 ← ExternalizeSpec(Attention)
  │   └── norm                 ← ExternalizeSpec(RMSNorm)
  └── lm_head

  Phase 2 — shallowest first (find call-site nodes in parent EP before children):
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  "block" (depth 1)  →  "block.attn" (depth 2)  →  "block.norm" (depth 2) │
  └────────────────────────────────────────────────────────────────────────────┘
  Parent's FX graph is searched for "block"'s node before the children's graphs
  are searched — child custom ops live inside block's subgraph, not the top-level.

  Phase 4 — deepest first (emit child coreai.graph before parent references it):
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  @block_attn_abc  →  @block_norm_def  →  @block_ghi                        │
  └─────────────────────────────────────────────────────────────────────────────┘

  Resulting coreai.Program:

    @block_attn_abc123  { ... }                     ← inner, emitted first
    @block_norm_def456  { ... }                     ← inner
    @block_ghi789       { coreai.invoke @block_attn_abc123(...)   ← outer
                          coreai.invoke @block_norm_def456(...) }
    @main               { coreai.invoke @block_ghi789(...) }
```

---

## Data Structures

```
  _patch_model_for_externalization(model, targets)
          │
          │  stamps directly on each matched submodule:
          │    _original_forward, _externalize_name,
          │    _externalize_op_name, _externalize_config
          │
          └──► returns None  (state lives entirely on the model)

                           │
                    _subexport_and_restore(model, ep)
                           │
                           │  every phase below rediscovers marked
                           │  submodules itself via
                           │  _find_marked_submodules(model) —
                           │  no list is threaded through
                           ▼

  Per call site: _PreparedModule  ──(torch.export)──►  _ExternalizedExportedProgram
  ┌──────────────────────────────┐                    ┌───────────────────────────────┐
  │ name           str           │                    │ name              str         │
  │ op_name        str           │                    │ op_name           str         │
  │ module_path    str           │                    │ exported_program  EP          │
  │ module         nn.Module     │                    │ composite_op_name str|None    │
  │ fake_inputs    tuple[Tensor] │                    │ composite_decl_attrs  dict    │
  │ dynamic_shapes tuple         │                    │ composite_input_names  list   │
  │ composite_op_name  str|None  │                    │ composite_output_names list   │
  │ composite_decl_attrs  dict   │                    │ source_nodes      list[str]   │
  │ source_nodes   list[fx.Node] │                    └───────────────────────────────┘
  └──────────────────────────────┘                              │
                                                                │
                                                     returns list[_ExternalizedExportedProgram]
                                                                │
                                                                ▼
                                        TorchConverter().add_exported_program(
                                            ep, _externalized_exported_programs=[...])
                                                                │
                                                                ▼
                                                        _perform_externalization()
                                                                │
                                                                ▼
                                                       coreai.graph noinline
```

---

## Why `torch.library.custom_op`?

| Approach | Problem |
|---|---|
| `nn_module_stack` metadata | Erased by decompositions, quantizers, and compile passes |
| Module hooks | Don't survive `torch.export` |
| Custom FX passes | Fragile — depend on graph structure, not module identity |
| **`custom_op` patch** | Opaque node; survives all FX transforms; identity is the op name |

The custom op acts as a **tombstone in the FX graph**: the boundary is preserved
regardless of what passes run.  The fake implementation satisfies shape inference;
the real implementation (with weights) is used by `torch.export.export` when
sub-exporting each submodule in Phase 3.

---

## File Map

| File | Role |
|---|---|
| `coreai_torch/externalize.py` | Phases 1–3, public API: `_patch_model_for_externalization`, `_subexport_and_restore` (standalone, shared by both workflows), `ExternalizeSpec` |
| `coreai_torch/converter.py` | `add_pytorch_module` (Workflow A: `_run_externalize_pipeline_from_module` calls `_patch_model_for_externalization` + `_subexport_and_restore` internally), `add_exported_program(_externalized_exported_programs=...)` (Workflow B), Phase 4 IR emission (`_perform_externalization`) |
| `coreai_torch/_utils.py` | `_find_all_custom_op_nodes`, `_fake_inputs_from_node`, `_dynamic_shapes_from_node`, `_ancestor_paths`, `_sanitize_op_name`, `_resolve_name` |
| `tests/test_externalize.py` | IR-level (`@pytest.mark.ir`) and numerical tests for both workflows |
| `tests/utils.py` | `convert_via_module` / `convert_via_markers` test helpers |
| `docs/guides/externalization.ipynb` | User-facing notebook guide |
