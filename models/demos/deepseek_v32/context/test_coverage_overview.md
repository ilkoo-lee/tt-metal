# Test coverage overview (DSA: DeepSeek V3.2 & GLM-5.1)

Companion to [`migration.md`](migration.md) / [`migration_log.md`](migration_log.md) / [`test_spec.md`](test_spec.md).

**Status: the MLA-layer test migration is DONE.** DeepSeek V3.2 and GLM-5.1 are onboarded onto the v3.1
test infra as `TestVariant`s, the sparse-MLA tests live in v3.1 `test_mla.py` (variant-driven), and CI
jobs run them on Galaxy + LoudBox. This doc is the current coverage map + the per-test budget.

**Why "migration" was mostly wiring, not porting:** after the G1.3 unification the V3.2/GLM MLA *is*
v3.1's `ttMLA` — the indexer + sparse SDPA activate when the weights carry indexer keys
(`INDEXER_WEIGHT_NAMES`). So the device path is the shared `run_mla_inference`; onboarding meant making
the **weights** (MLACPU → v3-dict via `WEIGHT_NAME_MAP`) and the **truth** (MLACPU, not the dense
reference) variant-driven, plus a few capability flags.

---

## Coverage overview

### v3.1 suite (`deepseek_v3_d_p/tests/`) — the shared home
Driven by the **variant registry**: `model_variants.py` defines `TestVariant` (config, HF repo, reference
classes, PCC thresholds, + DSA capabilities) and `TEST_VARIANTS = {DSV3, KIMI_V2_6, DSV32, GLM51}`. Test
bodies branch on capabilities (`has_indexer`, `tp_cap`, `config_builder`, `cpu_model_args`,
`reference_kind`), never on model name. Shared config lives in `dsa_plugin.py` (tier + group markers,
`--ds-*` knobs) and `dsa_reference.py` (the MLACPU CPU-reference substrate).

| File / test | Models | Group | Truth | Notes |
|---|---|---|---|---|
| `test_mla.py::test_ds_mla` | DSV3 | accuracy | CPU ref + golden trace | dense; seq 100k–128k; line/ring/fabric2d |
| `test_mla.py::test_kimi_mla` | Kimi | accuracy | CPU ref | dense; seq 5k/25k |
| `test_mla.py::test_mla_chunked_prefill` | DSV3, Kimi | feature_chunking | CPU / GPU-trace | dense; 50k production, rotation, multi-user |
| `test_mla.py::test_dsa_mla` | **DSV32, GLM51** | accuracy, mesh | **MLACPU** | sparse; seq {256(dev),2k,4k} × box-adaptive SP×TP |
| `test_mla.py::test_dsa_mla_determinism` | **DSV32, GLM51** | determinism | none | sparse seq4k × 3 runs, PCC==1.0 |
| `test_mla.py::test_dsa_mla_chunked` | **DSV32, GLM51** | feature_chunking, feature_cache | **MLACPU chunked** | sparse; 4k/chunk1k, aligned |
| `test_prefill_block*.py`, `test_prefill_transformer*.py`, `test_kv_cache_table.py` | DSV3, Kimi | accuracy/determinism/perf | CPU + golden trace | dense only (DSA block/e2e is out of P0 scope) |

### v3.2 bespoke suite (`deepseek_v32/tests/`) — stays here (trace-validated)
Validated against recorded **vLLM trace bundles**, which the v3.1 CPU-reference machinery cannot
produce, so these don't move onto the variant infra. They carry the same tier/group markers.

| File | Models | Group | Truth | Why bespoke |
|---|---|---|---|---|
| `test_vs_gpu_ref.py` | DSV32, GLM51 | accuracy (`gate`) | vLLM trace bundles | official-GPU parity (indexer logits/topk, sparse output, k_pe frame) |
| `test_indexer_chunked.py` | DSV32 | feature_chunking (`dev`) | self (chunk == single-shot) | indexer self-consistency |
| `test_mla_perf.py` | DSV32 | perf | none (Tracy) | hardcoded mesh/§7; local Tracy harness |
| `test_mla.py` (CPU-ref) | DSV32 | accuracy (`gate`) | MLACPU | **now redundant** with v3.1 `test_dsa_mla*` → retire next |

> The DSA ttnn kernels (`indexer_score`, `sparse_sdpa`, `topk_large_indices`) are covered by their own
> op unit tests; the old `ops.py` wrapper + `test_ops_*` were folded away (inlined into `TtIndexer`/`ttMLA`).

### Two kinds of "reference" — don't conflate
- **CPU PyTorch reference** — the in-repo reference *model* on CPU: `DSv3RefAttention`/`KimiRef*` (v3.1,
  dense) and **`MLACPU`/`IndexerCPU`** (DSA, sparse). Disk-cached. This is what the DSA tests use.
- **vLLM/GPU-recorded trace** — recorded output of a real GPU run (safetensors). v3.1's "golden traces"
  and the v3.2/GLM bit_sculpt bundles are the same category, differing only in model/scope/layout.

---

## Test groups → parameters (sparse by design)

Groups are by **intent** (orthogonal to tier). Each sweeps **one** axis and pins a representative value
on the others (covering-array style, never the full cartesian product). New models add a row, not a
cross-product.

| Group | Marker | Tests (DSA) | Swept axis | Pinned |
|---|---|---|---|---|
| accuracy | `accuracy` | `test_dsa_mla` | seq {256,2k,4k} | 1 mesh per box, random weights |
| determinism | `determinism` | `test_dsa_mla_determinism` | (3 runs) | seq4k, 1 mesh |
| feature:chunking | `feature_chunking` | `test_dsa_mla_chunked`, `test_indexer_chunked` | chunk scenario | seq4k/chunk1k, 1 mesh |
| feature:cache | `feature_cache` | `test_dsa_mla_chunked` (KV un-rotate) | — | as chunking |
| mesh | `mesh` | `test_dsa_mla` | SP×TP shapes that fit the box | seq2k |
| perf | `perf` | `test_mla_perf` (bespoke) | 1 scenario | fixed mesh, LB+GLX |

**Mesh sparsity is hardware-driven** (`mesh_utils`): each box emits exactly its SP×TP shapes — QuietBox
`{1, 1×4, 2×2}`, LoudBox `{8×1, 4×2, 2×4}`, Galaxy `{8×4}`. GLM is capped at tp≤2 (`tp_cap`,
sparse_sdpa needs `n_heads/tp ≥ 32`), so it skips tp>2 meshes (e.g. Galaxy 8×4 → GLM runs on LB only).

## Tier map (dev ⊆ gate ⊆ nightly)

| Tier | Marker | What | Where |
|---|---|---|---|
| dev | `dev` | fast inner loop, no cold truth — `test_dsa_mla` seq256, `test_dsa_mla_determinism`, `test_indexer_chunked` | per-edit + in CI |
| gate | `gate` | full correctness (cached truths) — `test_dsa_mla` all seqs, `test_dsa_mla_chunked`, `test_vs_gpu_ref` | pre-commit / CI |
| nightly | `nightly` | cold-truth + scale sweeps (full layer/mesh) | big-box nightly |

Select by intent × tier, e.g. `-m "accuracy and gate"`, or by `-k "<variant> and <mesh>"` (what the CI
jobs use).

---

## CI jobs (live)

The DSA tests use **random weights + on-the-fly MLACPU truth + R1 / hand-built GLM config** → no staged
V3.2/GLM weights or traces are needed, so the jobs **run** (not skip-gated). The cold CPU truth at
seq≤4k is seconds (cached after first run).

| Pipeline (box) | Job | Selection | Covers |
|---|---|---|---|
| galaxy-deepseek-prefill (Galaxy 8×4) | `(Galaxy) DeepSeek Prefill - DSA MLA (V3.2)` | `-m gate -k "deepseek_v32 and sp8xtp4"` | DSV32 accuracy + chunked on 8×4 (GLM auto-skips, tp=4) |
| blackhole-e2e (LoudBox) | `bh-lb-deepseek-dsa` | `-m gate -k "glm_5_1 and sp4xtp2"` + `"deepseek_v32 and sp2xtp4"` | GLM accuracy+chunked (its tp≤2 home); DSV32 asymmetric mesh |

The bespoke `test_vs_gpu_ref` (trace bundles) is not yet CI-wired — its traces live on exabox; staging
them is the flip-on step (it's the gold-standard official-GPU gate).

## Per-test budget (measured, 8× Blackhole, 2026-06-24)

Per-test **call** time on the `line` fabric (the topology this 8-chip box trains reliably; see the topology
note below). Add a one-time **JIT kernel compile of ~1–2 min** per fresh mesh shape, and cold MLACPU truth
(+seconds, cached to `$*_MLA_REF_CACHE`).

| `test_sparse_mla.py` case (line) | deepseek_v32 | glm_5_1 |
|---|---|---|
| `test_sparse_mla_accuracy` seq256 (dense-equiv) | 7.5 s | 7.1 s |
| `test_sparse_mla_accuracy` seq2k (dense-equiv) | 8.5 s | 8.7 s |
| `test_sparse_mla_accuracy` seq4k (sparse) | 12.5 s | 11.8 s |
| `test_sparse_mla_determinism` seq4k ×3 (dev) | 16.7 s | 14.1 s |
| `test_sparse_mla_chunked` 4k/c1k | 11.9 s | 10.4 s |

| `test_sparse_mla_vs_trace.py` (`-m trace`, per layer, line) | time |
|---|---|
| `test_mla_output_device_vs_reference` | 10.5 s |
| `test_indexer_device_vs_reference` / `test_kv_cache_device_vs_reference` | ~6.5 s each |
| host (`*_host_vs_reference`, CPU) | ~0.5–2 s |
| full DS L0 set (12 tests, incl compile) | ~300 s |

| `test_sparse_mla_perf.py` (`-m perf`, box-adaptive: profiles a per-chip Galaxy slice) | time |
|---|---|
| `test_mla_chunked_perf_impl` (device profile; LoudBox SP2×TP4 / Galaxy SP8×TP4) | ~20 s |
| `test_mla_chunked_perf` (tracy wrapper) | ~44 s |

Measured metric (LoudBox proxy, 1280-tok chunk @ 51200-tok cache, SP2×TP4): **13.5 ms critical-path
device-kernel / 163 op calls** (Galaxy target: 5120-tok chunk @ 51200, SP8×TP4 — same per-chip shapes).

A full LB `bh-lb-deepseek-dsa` selection (4 tests/variant, line) runs in **~1–2 min** + compile; both LB and
Galaxy DSA jobs fit the 30–40 min budgets with large headroom. (v3.1 baseline: the galaxy pipeline wall-clock
is the longest parallel job ≈ 27 min — the transformer-determinism variants — not these MLA jobs.)

**Marker-gated skips are instant:** `-m`-less collection of the perf + trace suites skips all 106 tests in
**0.11 s** (no device), via the `is_marker_explicitly_selected` autouse gate.

## Topology / mesh support (8-chip LoudBox vs Galaxy)

Verified on the 8-chip box: the migrated tests pass for **both variants on the `line` fabric**. The other
configs fail to *initialize fabric* on a partial-Galaxy box (environmental, not test logic):
- **`ring` / `fabric2d`** → `TT_FATAL: Could not find any forwarding direction` (ring is intermittent —
  some cases train, some don't). They target a full Galaxy.
- **`2x2` sub-mesh** → `Fabric Router Sync: Timeout` (a sub-mesh smaller than the physical box doesn't train).
- **`8x4`** = Galaxy production (32 chips) — not openable on a LoudBox.
- **`4x2`** was added to `SPARSE_MESH_PARAMS` (the LoudBox full-box TP=2 shape) so **GLM (tp≤2) is
  exercisable on a LoudBox** — with the original `[(8,4),(2,4),(2,2)]`, GLM only mapped to the broken `2x2`.

**Perf invocation** (mirrors `tests/nightly/blackhole/sdpa`): run the **gate by node-id** —
`pytest -m perf …::test_mla_chunked_perf`. The gate opens no device itself; it drives the impl as a tracy
subprocess via `run_device_profiler` with `CI=false` patched in, and the impl is `skipif(CI=="true")`. So
only the subprocess opens the board — no parent `CHIP_IN_USE` lock, no deadlock. (Avoid a bare
`pytest -m perf <file>`: it also collects `test_mla_chunked_perf_impl`, which would open the device in the
parent — the reference avoids this by running the gate node-id-scoped, not via a bulk marker.)

---

## What v3.1 can't cover (stays bespoke)
1. **`test_vs_gpu_ref.py`** — official-GPU/vLLM parity needs recorded trace bundles, not a CPU-reproducible
   model. The gold-standard gate; stays in the v3.2 suite.
2. **`test_mla_perf.py`** — Tracy-driven, hardcoded mesh/§7; perf harnesses are scenario-specific.
3. **GLM `AutoConfig`** can't load `glm_moe_dsa` → the hand-built config travels with the variant
   (`config_builder=glm_hf_config`). Permanent quirk, handled in the registry.

## Gaps / follow-ups
1. **DSA block / full-model e2e** — only the MLA layer is covered for V3.2/GLM (P0 scope); v3.1 has block+e2e.
2. **Sparse partial-chunk / rotation** — `test_dsa_mla_chunked` uses aligned chunks; the rotation/padding
   semantics on the sparse path are unresolved (the original "how padding works" TODO). The 50k-production
   `test_mla_chunked_prefill` is intentionally not extended to sparse (dense refs wrong above topk; its
   scenarios exceed the MLACPU window of 16384 DS / 8192 GLM).
3. **`test_vs_gpu_ref` not CI-wired** — pending trace staging on exabox.
4. **No CI perf gate for the DSA path** — `test_mla_perf` is local-only (`@skipif(CI)`).
5. **Retire the redundant v3.2 `test_mla.py`** CPU-reference tests now that `test_dsa_mla*` cover them.

## Ideas
- **Replace the `sparse_mla` host fallback** with device `sparse_sdpa` in the heavy device tests — the
  biggest single CI speedup at large scale.
- **Add a DSA transformer-block / e2e test** (V3.2/GLM through `TtPrefillBlock`).
- **Promote a trimmed `test_mla_perf` scenario** off `@skipif(CI)` with a regression threshold.
- **Cross-stack k_pe interop** (`--ds-kpe-layout vllm`) in CI once a vLLM-written cache is staged.
