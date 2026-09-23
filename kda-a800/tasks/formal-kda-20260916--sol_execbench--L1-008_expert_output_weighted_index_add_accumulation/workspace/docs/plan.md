# Plan — L1/008 Expert Output Weighted Index-Add Accumulation

Task ID: `sol_execbench :: L1/008_expert_output_weighted_index_add_accumulation`
Target GPU: NVIDIA A800 (`sm_80`, Ampere). HBM2e ~2.0 TB/s, L2 = 40 MB.
Entry point: `solution/solution.py` exposing `run(final_hidden_states, expert_outputs, token_indices) -> output`.
Primary implementation: **Triton**. PyTorch only for tensor metadata / launch plumbing. **No** Torch/CPU/NumPy/CUDA-extension computational fallback; a failing Triton kernel is invalid and is never replaced by a fallback.

This plan operationalizes `docs/draft.md`. It defines the sequential candidate ladder, the exact source to build at each step, correctness gates, performance hypotheses, decision rules, stopping criteria, and the evidence (`candidates.jsonl`) schema. **No candidate is implemented or evaluated in this turn.**

---

## 0. Recap of the problem (authoritative)

For `i in [0, N)`: `output[token_indices[i], :] += expert_outputs[i, :]`, over a fresh
`output = clone(final_hidden_states)`. Pure scatter-add (weights already folded into `expert_outputs`).

- `M = batch_size*seq_len` (rows), `H = 3072` (const), `N = 8*M` (num selected tokens).
- dtypes: `final_hidden_states` bf16 `[M,H]`, `expert_outputs` bf16 `[N,H]`, `token_indices` int64 `[N]`, `output` bf16 `[M,H]`.
- Indices are in-range (`randint(0, M)`); duplicates expected (~8 collisions/row avg). No index masking needed; only hidden-tail masking.
- Feedback workloads (fixed 5): W1 (2×256, M=512), W2 (32×256, M=8192 — dominant), W3 (16×256, M=4096), W4 (2×1024, M=2048), W5 (4×512, M=2048). tol: `rtol=0.05`, `atol≈0.087–0.10` per workload.
- Cost model: memory-bandwidth + atomic-contention bound; zero real FLOPs. `expert_outputs` read (`8*M*H*2` bytes) is the irreducible ~8/10 of traffic and is identical for every correct implementation. The only levers are **atomic cost** (native fp32 atomic vs torch bf16 CAS loop) and **L2 residency** of the accumulator + number of passes/launches.

---

## 1. Strategy overview

Primary bet (from draft §2, §4): torch `index_add_` on bf16 uses a per-element bit-level CAS loop
(`sm_80` has no native bf16 global atomic-add); contention amplifies retries. We replace it with an
**fp32-scratch accumulator + native `red.global.add.f32` atomics** (single instruction, cheap under
contention) and round to bf16 once at the end. This is also **more numerically accurate** than the
reference (draft §3.2), so correctness risk is low and biased toward safety.

Two axes of exploration, executed in phases:

1. **Algorithmic form** (Phase A): fp32-scratch B0 vs B1, plus a reference-parity bf16-direct
   variant (A) as an L2-residency comparison point on small workloads.
2. **Tuning** (Phase B): given the winning form, sweep the scatter kernel's tile shape (full-row vs
   tiled hidden blocks), `BLOCK_H`, `num_warps`, `num_stages`, and load/vectorization choices.
3. **Refinement** (Phase C): per-workload–aware hybrids only if the evidence shows a form that wins
   large workloads but regresses the launch-bound small workload (W1), or vice versa.

We change **exactly one meaningful thing per candidate ID**. Each candidate is an immutable source
version; a new source hash ⇒ a new candidate ID (never reuse).

---

## 2. Shared implementation contract (applies to every candidate)

All candidates share this host-side skeleton in `solution/solution.py`; only the kernel(s) and launch
config differ between candidates.

- `run(...)`:
  - Read shapes/strides/device/dtype from inputs (PyTorch, metadata only — allowed).
  - Assert `expert_outputs.shape[0] == token_indices.shape[0]` and `expert_outputs.shape[1] == final_hidden_states.shape[1]` (defensive; no compute).
  - Allocate **fresh** output/scratch with `torch.empty` (never mutate inputs in place — the harness may reuse inputs across timing repeats).
  - Make tensors contiguous only if needed (they are row-major contiguous per the generator; avoid an unnecessary copy — check `.is_contiguous()` and only `.contiguous()` if false).
  - Launch Triton kernel(s); return the bf16 `output`.
- Pointer math uses **int64** row offsets (`idx*H`) to stay general (max offset ~25.2M fits int32 for these workloads, but int64 is safe and free here).
- **Hidden-tail masking**: keep `mask = offs_h < H` on every hidden load/store/atomic even though
  `H=3072` divides cleanly by candidate block sizes — safety over micro-optimization until proven costly.
- **Full row coverage**: init/finalize grids must cover all `M` rows so untouched rows equal `base`
  exactly (draft §3.4).
- No autotune first-call timing risk: Phase A/B use **fixed** launch configs chosen per candidate so
  the evaluator never times an autotune warmup. (Autotune considered only as an explicit later option
  in §5, Phase B-alt, with the tuning key fixed so it resolves once.)

---

## 3. Candidate ladder (sequential, immutable)

Each row = one candidate = one source version = one `feedback` evaluation (5 workloads = 1 eval).
"Parent" = the candidate this one is derived from. Advance only per the decision gates in §6.

### Phase A — establish a correct, winning algorithmic form

| ID   | Parent | Change / Hypothesis | Kernels | Expected |
|------|--------|---------------------|---------|----------|
| c001 | —      | **Option B0**: fp32 scratch `buf[M,H]`; init `buf=base` (cast-copy kernel); scatter `buf[idx]+=src.to(f32)` via native fp32 `tl.atomic_add`; finalize cast `buf→bf16 output`. Fixed config: scatter grid `(N, cdiv(H,BLOCK_H))`, `BLOCK_H=1024`, `num_warps=4`. **Baseline correctness + first speedup from native atomics.** | 3 | Correct on all 5; geomean > 1.0 driven by W2/W3. |
| c002 | c001   | **Option B1**: init `buf=0` (write-only memset kernel or `torch.zeros`? → use a Triton memset to stay Triton-native), scatter same, finalize `out = round(base + buf)` (fused read base + read buf, write out). Same fixed config as c001. **Fewer init bytes; tests init-copy vs memset+fused-add net cost.** | 3 | Correct; geomean ≥ c001 or reveals which init wins. |
| c003 | best(c001,c002) | **Option A (bf16-direct, comparison point)**: fill `output=base` (cast-copy = identity copy), then `tl.atomic_add` bf16 `src` directly into bf16 `output` (no fp32 scratch). Same fixed tile config. **Tests whether L2-resident bf16 (3–50 MB) beats fp32 (6–100 MB) on small/L2-bound W1 despite bf16 CAS atomics.** First confirm Triton lowers bf16 `atomic_add` on `sm_80`; if it errors/does not lower, mark c003 invalid and skip (do NOT fallback). | 2 | Likely wins W1 (L2-resident), likely loses W2 (CAS contention). Informs Phase C hybrid. |

Rationale for order: c001 secures a correct, rule-clean, accurate baseline with the strongest a-priori
win (native atomics). c002 is a cheap byte/launch refinement of the same idea. c003 is a targeted
probe of the small-workload regime, not expected to win overall but valuable evidence for a possible
hybrid.

### Phase B — tune the winning form's scatter kernel (the dominant cost)

Base = winning Phase-A form (call it `F`). One knob per candidate. Concrete first sweep values
(adjust the list only in response to evidence, and only via new IDs):

| ID   | Parent | Change / Hypothesis |
|------|--------|---------------------|
| c004 | F      | **Full-row scatter**: one program per source row, `BLOCK_H=3072` (or the smallest power-of-two ≥3072 handled via masking, e.g. process 3072 in one tile), grid `(N,)`. One `idx` load per row (vs redundant loads when tiled). Hypothesis: fewer idx loads + coalesced 6 KB streamed loads help; risk lower occupancy. |
| c005 | best-so-far | **BLOCK_H sweep**: set the hidden tile to the next candidate value from {512, 768, 1536} (pick the one most likely to raise occupancy/coalescing given c004 result). |
| c006 | best-so-far | **num_warps sweep**: try {2, 8} around the current best (one value per ID if two are worth testing → c006, c007). |
| c007 | best-so-far | **num_stages / pipelining**: try `num_stages ∈ {1,2,3}` around best (scatter is atomic-bound; more stages may or may not help the streamed src load). |
| c008 | best-so-far | **Load/cast placement & vectorization**: load `src` bf16 then `.to(tl.float32)`; ensure contiguous vectorized loads (`tl.load` on aligned `offs_h`); optionally 2 hidden tiles per program to amortize idx load. |

(IDs beyond c008 are allocated on demand as the sweep dictates; the *shape* of the sweep is fixed here,
the exact next value is chosen from evidence. Each distinct config = a new sequential ID.)

### Phase C — refinement / hybrids (only if evidence justifies)

| ID (as needed) | Parent | Change / Hypothesis |
|----------------|--------|---------------------|
| cNNN | best | **Per-size dispatch**: if c003 (bf16-direct) wins W1 while an fp32 form wins W2/W3, dispatch on accumulator footprint vs L2 (e.g. choose bf16-direct when `M*H*2 < ~L2_fraction`, else fp32-scratch). Single source, deterministic threshold from measured crossover — still one immutable kernel set. |
| cNNN | best | **Finalize fusion**: if B1 finalize is a measurable cost, try fusing base-add into the scatter epilogue or reducing passes further. |
| cNNN | best | **Init/finalize tiling**: tune the init/finalize kernels' `BLOCK`/`num_warps` (they are pure streaming copies; usually saturate BW easily but verify they aren't a W1 launch-overhead tax). |

---

## 4. Correctness checks (gate before accepting any candidate)

Correctness is judged **only** by `./scripts/evaluate_candidate.sh feedback cNNN` (the official
evaluator applies each workload's `max_atol`/`max_rtol`). Before every evaluation, complete this
static pre-check (reason it through; do not build a private harness, do not run CUDA/profiler/nvidia-smi):

1. **Fresh output / no input mutation**: `output` (and `buf`) are `torch.empty`-allocated; inputs are
   never written. (Prevents corruption across timing repeats.)
2. **Full initialization**: every one of the `M` rows is set to `base` (B0/A) or the finalize covers
   all `M` rows (B1). Untouched rows must equal `base` exactly. Grid coverage + tail mask verified.
3. **fp32 accumulation → round once**: our result ≈ true sum rounded to bf16, whose deviation from the
   reference (repeatedly-rounded bf16 CAS sum) is ≤ the reference's own bf16 accumulation error, which
   is within `atol + rtol*|ref|` for the given tolerances (draft §3.2–3.3). Stress case: near-zero,
   high-collision rows (where `rtol*|ref|≈0`, only `atol` protects) — still within a few bf16 ulps.
4. **Index arithmetic**: int64 row-base offsets; no index masking (in-range by construction); hidden
   tail masked.
5. **dtype/shape of return**: bf16 `[M,H]`, contiguous, on the input device.
6. **Triton-only compute**: no Torch reduction/scatter/sort/argsort/bincount used for the actual
   accumulation. `torch.empty`/shape/stride/`is_contiguous` are metadata-only and allowed.

A candidate that fails correctness on **any** of the 5 workloads is invalid (decision = `reject`);
we do not weaken tolerances (cannot — fixed) and never add a fallback. If a Triton kernel fails to
compile/run (e.g. bf16 atomic unsupported in c003), the candidate is invalid and we move on.

---

## 5. Performance hypotheses (explicit, falsifiable)

- **H1 (primary win):** Native fp32 atomics beat torch bf16 CAS atomics enough to overcome the
  +~300 MB of init/finalize passes on the dominant W2 ⇒ c001 geomean > 1.0, largest gain on W2/W3.
  *Falsified if* c001 geomean ≤ 1.0 → pivot to Option A / bf16-direct or reconsider passes.
- **H2 (init form):** memset+fused-add (B1) moves fewer bytes than init-copy (B0) and wins net, but
  costs an extra base read at finalize — near a wash; decided empirically by c002 vs c001.
- **H3 (L2 residency):** On W1 (accumulator + src fit largely in 40 MB L2), the bf16-direct form (A)
  avoids fp32 scratch traffic and may win despite CAS atomics; on W2 (fp32 scratch 100 MB ≫ L2) the
  RMW spills to HBM and native fp32 atomics dominate. ⇒ c003 wins W1, loses W2. If confirmed, motivates
  the Phase-C per-size hybrid.
- **H4 (tile shape):** Full-row scatter (one idx load/row) reduces redundant idx traffic and improves
  coalescing vs tiled; net effect depends on occupancy. Decided by c004.
- **H5 (occupancy):** `BLOCK_H`/`num_warps`/`num_stages` trade coalescing vs occupancy vs pipeline
  depth; a memory-bound streaming scatter typically prefers moderate warps (4–8) and shallow pipelines.
- **H6 (launch overhead on W1):** Fewer kernel launches help the launch-bound small workload; watch
  whether B1's memset (or extra passes) regresses W1 relative to a 2-kernel form.

Ranking metric = **geometric mean speedup across the 5 feedback workloads**, subject to all 5 passing
correctness. Track per-workload speedups too, so W1 (launch-bound) and W2 (BW/L2-bound) regressions are
visible even when geomean improves.

Phase B-alt (only if manual sweep is inconclusive and budget allows): a single autotuned candidate with
an explicit config list and a tuning key fixed on `H` (constant ⇒ resolves once), accepting that the
first call incurs a one-time tune; evaluate its impact on measured timing before trusting it.

---

## 6. Decision gates & lineage rules

- **Accept & branch**: a candidate becomes the new "best-so-far" (parent for the next candidate) iff
  it passes all 5 workloads AND its geomean > current best geomean by a meaningful margin
  (> ~1% and outside run-to-run noise). Record decision = `accept`.
- **Keep-as-evidence (no branch)**: passes correctness but does not beat best (or is a probe like
  c003). Decision = `reject-for-branching` / `evidence-only`; parent for the next candidate stays the
  prior best. Its data still informs later hypotheses (e.g. per-size hybrid).
- **Invalid**: fails correctness or fails to compile/run. Decision = `invalid`. Never patched into a
  fallback; a fresh idea gets a new ID.
- **One knob per ID**: never bundle an algorithmic change with a tuning change in one candidate.
- **Immutability**: once evaluated, a candidate's source is frozen. Any edit ⇒ new ID. Never rewrite an
  earlier `candidates.jsonl` record.
- **Phase transitions**: only enter Phase B after a correct, >1.0-geomean form exists in Phase A.
  Only enter Phase C if per-workload evidence shows a genuine size-dependent crossover (H3) or a
  measurable finalize/init tax (H2/H6).

---

## 7. Stopping / convergence criteria

Stop and write `SEARCH_COMPLETE` (with a written reason) when the **first** of these holds:

1. **Convergence**: the last 2–3 accepted candidates improve geomean by < ~1% each (plateau), and the
   remaining planned knobs are exhausted or predicted (from evidence) to be within noise.
2. **Evaluation budget**: cumulative `feedback` evaluations approach 100 (leave margin; do not exceed).
3. **Token budget**: approaching the 1,000,000 soft limit → begin wrapping up; hard-stop well before
   1,500,000 (normal) / 1,650,000 (absolute). Prefer to converge and document rather than burn budget
   on marginal sweeps.
4. **No viable path**: if fp32-scratch and bf16-direct and their tunings all fail to beat torch
   (geomean ≤ 1.0) and no untested hypothesis remains, stop and report the best valid candidate with a
   negative-result rationale.

`final` (16-workload) evaluation is **operator-only** and is never run without explicit approval, even
after `SEARCH_COMPLETE`.

---

## 8. Evidence format (`candidates.jsonl`)

Append **exactly one** JSON object per evaluated candidate, in evaluation order. Never edit/rewrite a
prior line. Required fields (per CLAUDE.md workflow item 7):

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "phase": "A",
  "hypothesis": "fp32-scratch B0 + native fp32 atomics beats torch bf16 CAS; correct on all 5.",
  "config": {"form": "B0-fp32-scratch", "kernels": 3, "BLOCK_H": 1024, "num_warps": 4, "num_stages": 2, "grid": "(N, cdiv(H,BLOCK_H))"},
  "validation": {
    "pre_checks": ["fresh-output", "full-init", "fp32-round-once", "int64-offsets", "tail-mask"],
    "correctness": "pass|fail",
    "per_workload_pass": {"ada14c03": true, "8c5c94b5": true, "b61ee9a7": true, "2c1d8396": true, "8b677344": true}
  },
  "results": {
    "per_workload_speedup": {"ada14c03": 0.0, "8c5c94b5": 0.0, "b61ee9a7": 0.0, "2c1d8396": 0.0, "8b677344": 0.0},
    "geomean_speedup": 0.0
  },
  "decision": "accept|reject-for-branching|evidence-only|invalid",
  "decision_reason": "…",
  "cumulative_evaluations": 1,
  "skill_usage": {"KernelWiki": "not-used (sm_80; Hopper/Blackwell-only skill out of scope)"},
  "notes": "observations feeding the next hypothesis"
}
```

Rules for the evidence log:
- Speedups are the evaluator's reported per-workload speedups vs the reference; `geomean_speedup` is the
  geometric mean over the 5. Use `uuid` keys (or the short prefixes above consistently) for per-workload
  entries.
- `source_sha256` is computed on `solution/solution.py` immediately before the evaluation that produced
  the record (proves the immutable version that was measured).
- `cumulative_evaluations` increments by 1 per `feedback` run (5 workloads = 1).
- `skill_usage` documents that `KernelWiki` is out of scope here (A800/`sm_80` Ampere; the skill covers
  Hopper/Blackwell only). If that judgment ever changes, record the actual usage.
- Keep `notes` short and forward-looking (what the result implies for the next candidate).

---

## 9. Execution checklist (per candidate, next turns — not this turn)

1. Confirm parent and the single knob being changed; allocate the next sequential ID.
2. Write/modify `solution/solution.py` to the immutable source for this candidate.
3. Run the static correctness pre-check (§4) by reasoning over the source.
4. Evaluate: `./scripts/evaluate_candidate.sh feedback cNNN` (only this launcher).
5. Append one complete record to `candidates.jsonl` (§8), including source hash + decision.
6. Apply the decision gate (§6): branch or keep-as-evidence; pick the next candidate.
7. Check stopping criteria (§7); if converged, write `SEARCH_COMPLETE` with the reason.

**This turn ends here: plan only. No candidate is implemented or evaluated.**

---

## 10. Decision log

- **c001 (Option B0, fp32-scratch, 3 kernels, BLOCK_H=1024, num_warps=4)** — `accept`, new best-so-far.
  Correct on all 5 workloads; geomean **1.929x** (W1 1.730, W2 2.047, W3 2.016, W4 1.925, W5 1.942).
  Confirms **H1** (native fp32 atomics beat torch bf16 CAS despite +~300 MB init/finalize passes).
  Per-workload pattern confirms the expected split: BW/L2-bound W2/W3 gain most (~2.05x), launch-bound
  W1 gains least (1.73x). Parent for next candidate = c001.
  - Next: **c002 = Option B1** (Triton memset init `buf=0`, scatter, finalize `out = round(base + buf)`).
    Hypothesis H2/H6: fewer init bytes + fewer distinct read passes may lift W1 without hurting W2.
    Single knob changed vs c001 (init form + finalize fusion of the same algorithm family).

- **Reorder decision (before c002 build):** dropped B1 from the c002 slot. Byte accounting shows B1 is
  **exactly byte-neutral** vs B0 — it removes one `base` read (M*H*2) from init but adds the identical
  `base` read (M*H*2) to finalize, with the same 3 launches. So H2 is essentially pre-falsified (a wash)
  and B1 is a low-information eval. Meanwhile the **scatter kernel is the dominant cost**: it streams
  `expert_outputs` (8*M*H*2) and issues all N*H fp32 atomic RMWs into a fp32 buffer that exceeds the 40 MB
  L2 on W2/W3 (~1.6 GB of RMW traffic on W2 — the real bottleneck). Therefore c002 pulls forward the
  planned Phase-B **c004 full-row scatter (H4)** as the higher-value single knob: grid `(N,)`, one int64
  `idx` load per source row amortized over `NUM_H=cdiv(H,BLOCK_H)=3` hidden tiles via `tl.static_range`,
  giving 3× fewer idx loads and better per-program ILP/accumulator locality. init/finalize kernels and all
  launch configs (`BLOCK_H=1024`, `num_warps=4`) are unchanged vs c001 — exactly one meaningful knob moved.
  Risk to watch: full-row lowers program count 3× (65536 vs 196608 on W2); if MLP/occupancy is the limiter
  this could regress. If so, revert to tiled and instead sweep `num_warps`/`BLOCK_H` (still Phase B).

- **c002 (Option B0 fp32-scratch, full-row scatter grid `(N,)`, `NUM_H=3` via `tl.static_range`, else
  identical to c001)** — `reject-for-branching`. Correct on all 5; geomean **1.9130x** < c001 1.9287x
  (~0.8% regression), and slower on *every* workload (W1 1.696/1.730, W2 2.039/2.047, W3 2.006/2.016,
  W4 1.924/1.925, W5 1.921/1.942). **H4 falsified**: the full-row grid cut the scatter program count 3×
  (W2: 65536 vs 196608), losing occupancy / latency-hiding on the memory+atomic-bound scatter — that
  costs more than the (negligible, N·8 B) idx-load traffic it saved. Best-so-far stays **c001**; parent
  for the next candidate = c001.
  - Next: **c003 = occupancy sweep on the c001 tiled form** — set scatter `num_warps=8` (only the scatter
    launch changes; init/finalize stay `num_warps=4`, `BLOCK_H=1024`). Hypothesis H5: more warps/program
    give the SM more in-flight atomic RMWs + streamed src loads to hide latency on the dominant scatter,
    without reducing the program count. One meaningful knob vs c001.
