# Plan: L1/020 Vision Patch Merger — Executable KDA Optimization Plan

Companion to `docs/draft.md`. This is the executable, sequential search plan: candidate
lineage, per-candidate correctness checks, performance hypotheses, stopping criteria, and the
`candidates.jsonl` evidence format. No code is written or evaluated in this turn.

Constants (fixed): `C=1536`, `merge=2`, `E=6144=4C`, `O=3584`, `eps=1e-6`.
Metric: geomean speedup over the 15 feedback workloads; every selected workload must pass
correctness within its per-workload `atol/rtol`.

---

## 0. Ground rules the search obeys

- One immutable candidate = one source version = one full `feedback` run = one eval. Budget: **100
  evals**. Token soft/normal/hard = **9M / 10M / 11M**.
- Evaluate ONLY via `./scripts/evaluate_candidate.sh feedback cNNN`. Never run CUDA, `nvidia-smi`,
  the external evaluator, or an alternate correctness harness.
- Profiling ONLY via `./scripts/ncu_profile.sh …` (ncu-report-skill), and **never** while an eval
  is running (foreign process on the locked GPU ⇒ rc 3, wastes the eval). Serialize strictly.
- Triton-only compute (both Linears are `tl.dot`). PyTorch allowed only for shape math, index
  tensors, output allocation. A failing Triton kernel is invalid — **no** Torch/CPU/NumPy fallback.
- Candidate IDs are immutable: any source/config/launch change ⇒ new ID; never reuse an ID for
  changed source; never rewrite earlier `candidates.jsonl` records.
- `final` only on explicit operator approval.

---

## 1. Candidate lineage strategy

Tree search, one live branch at a time. Keep the **best correct** candidate as the running
parent; a change is adopted only if it (a) stays correct on all 15 and (b) improves geomean beyond
noise (see §5). Rejected changes are recorded but not extended.

```
c001  correctness baseline (3 kernels, simple, conservative configs)
  └─ c002  GEMM tiling/stages/warps tuning (M-keyed), grouped-M raster
       └─ c003  epilogue fusion polish (bias+erf-GELU fold), rounding-mode test
            ├─ c004  fused LN+shuffle prologue into GEMM1 (drop [M,6144] intermediate)
            │     └─ c005  small-M path: split-K / low-occupancy handling
            │           └─ c006  sync-free device-side index (drop D2H)
            └─ (fallbacks if a branch regresses: re-parent to last best)
```

Each phase below lists Goal / Change-from-parent / Hypothesis / Pass-gate / Next. IDs are assigned
strictly in evaluation order (the numbering above is indicative; the *actual* next ID is always
`max(existing)+1`). Only ONE new source version exists on disk at a time.

---

## 2. Phase C1 — correctness baseline (`c001`)

**Goal.** A correct, simple, Triton-only implementation passing all 15 workloads; establish the
baseline geomean. Optimize nothing yet beyond obvious fusion that is also the simplest to write.

**Design (Option A, 3 kernels — see draft §4.1):**
1. **Index build (host-light):** read `grid_thw` once via `.tolist()` (single tiny D2H sync,
   `[G,3]`, G≤8). Compute `M = Σ T·(H/2)·(W/2)`. Build an int32 tensor `src_row[M*4]` (or a
   per-output-row gather map) mapping each of the `4·C` sub-blocks of every merged row to its
   global input row, using the bijection from draft §1.2:
   `o_global = base_o[g] + (t·Hm+hm)·Wm + wm`, `s = 2a+b`, input row
   `r = base_i[g] + ((t·H + (2hm+a))·W + (2wm+b))`. Built with vectorized torch ops on-device.
2. **`ln_shuffle` kernel:** grid over merged rows (or over covered input rows). Each program
   LN-normalizes one `[1536]` input row in **fp32** (mean, population var `/C`, `1/sqrt(var+eps)`,
   `*w+b` in fp32), casts to bf16, stores to the correct `out_shuffled[o, s·C : s·C+C]`.
   Produces `[M,6144]` bf16.
3. **`gemm1_gelu` kernel:** `[M,6144] = shuffled @ fc1_weightᵀ + fc1_bias`, bf16×bf16→fp32 acc;
   epilogue: round acc→bf16 (=`hidden_fc1`), upcast, **exact erf GELU** in fp32
   (`0.5·x·(1+erf(x·0.7071067811865476))`), round→bf16. B-tile loaded from `fc1_weight[n,k]`
   (row-major `[out,in]`, stride-N=K) to realize `x@Wᵀ`.
4. **`gemm2` kernel:** `[M,3584] = gelu @ fc2_weightᵀ + fc2_bias`, fp32 acc → bf16.

Conservative configs to start: `BM=64, BN=128, BK=64, num_stages=3, num_warps=4`, grouped-M
raster. Correctness over speed here.

**Pre-eval static checks (do before spending the eval):**
- Hand-verify the shuffle bijection on `T=1,H=W=2` (1 merged row; the four `(a,b)` blocks land in
  columns `0,1,2,3 ×1536`).
- LN: fp32 accum, `unbiased=False` (÷C), eps inside sqrt, weight/bias in fp32, cast bf16 before GEMM.
- Weight orientation realizes `x@Wᵀ` for both fc1 `[6144,6144]` and fc2 `[3584,6144]`.
- GELU symbol resolves: try `tl.math.erf`; fallback `tl.extra.libdevice.erf`. Confirm importable.
- `M`, `base_i`, `base_o` derived from `grid_thw`, handling `ΣT·H·W ≤ N` (trailing rows unused).
- Output dtype bf16, shape `[M,3584]`.

**Pass-gate:** all 15 workloads PASS. If any fails, do NOT branch — fix and re-issue under the
**next** ID (c001 stays as the failed record). First suspects on failure: GELU approx, LN
variance/eps, weight transpose, shuffle index ordering, boundary grid (`H=W=2`).

**Next:** on pass, record baseline geomean; proceed to C2.

---

## 3. Phase C2 — GEMM tuning (compute-bound wins)

**Goal.** Raise the two GEMMs toward cuBLAS efficiency on large-M workloads (M ≥ ~1024:
3b69084d, 24f153b1, b52d9dc5, 42de85c1, 1f411501, ffadc1e3, c60491d2, ea969b1d).

**Change-from-parent:** M-keyed config heuristics (a *small curated* set, not a broad autotune, to
bound JIT/compile wall-clock across 15 shapes). Candidate configs:
`{128×256, 256×128, 128×128}` tiles, `BK∈{64,128}`, `num_stages∈{3,4}`, `num_warps∈{8}`, grouped-M
raster with tuned `GROUP_M` for L2 weight reuse. Keep fp32 acc.

**Hypothesis (H-C2):** the compute-bound workloads are GEMM-floor-limited; better tiling + deeper
pipeline + grouped raster narrows the gap to cuBLAS and lifts geomean on the large-M half. Per
KernelWiki `pattern-compute-bound`: pipeline stages + epilogue overlap are the levers.

**Evidence to confirm regime (optional, after c001 correct, never during an eval):** `ncu_profile.sh`
on M=16384 (ea969b1d) → expect high tensor-core utilization, DRAM not saturated. If tensor-core
util already high and near cuBLAS, deprioritize further GEMM tuning.

**Pass-gate:** all 15 still PASS; geomean improves beyond noise (§5). Adopt best config set as new
parent; discard the rest (recorded).

---

## 4. Phase C3 — fusion & memory-bound / small-M wins

Multiple sub-steps, each its own candidate ID, sequential, only extending the current best.

**C3a — epilogue fusion polish + rounding-mode test.** Ensure bias+erf-GELU are folded into the
FC1 store (no separate GELU pass). Empirically test whether reference-matched intermediate bf16
rounding is *required* or whether keeping fp32 through GELU still passes tol with more headroom
(draft §3.3). Hypothesis (H-C3a): fusion removes a `[M,6144]` read+write round trip (~200 MB at
M=16384) — helps both regimes; rounding choice is correctness-neutral within tol. Pass-gate: all
15 PASS, geomean ≥ parent.

**C3b — fused LN+shuffle prologue into GEMM1 (Option B, draft §4.1).** Fold LN+scatter into
GEMM1's A-tile load: each `[BM,BK]` A-tile gathers the correct input rows via the index map and
LN-normalizes on the fly; drop the materialized `[M,6144]` intermediate (2 kernels total).
K-tiling aligns to the four `C=1536` sub-blocks (`BK | 1536`). Hypothesis (H-C3b): removes two full
`[M,6144]` HBM round-trips — largest benefit at large M; also removes one launch (helps small M).
Risk: more complex A-load; guard correctness carefully. Pass-gate: all 15 PASS, geomean ≥ parent.
If it regresses or is too fragile, re-parent to C3a.

**C3c — small-M / low-occupancy path.** For M ≤ ~512 (38e61f30, 133134ee, 9d5b4822, 9376d72b,
67d44c8f, 31eaac46) few M-tiles under-fill the ~132 SMs. Try: smaller `BM` with finer `BN` to
raise tile count, and/or **split-K** over `K=6144` (two-stage reduction or atomics) to spread the
weight read across more CTAs and hide HBM latency; stream weights once. Hypothesis (H-C3c): these
workloads are weight-memory-bound + occupancy-starved (KernelWiki `pattern-memory-bound`,
`pattern-tail-effect`); split-K/occupancy raises SM utilization and geomean on the small half
without hurting large M. Confirm memory-bound via `ncu_profile.sh` on M=256 first (DRAM
throughput high, tensor-core util low). Pass-gate: all 15 PASS; geomean ≥ parent (watch for
large-M regression from split-K overhead — keep split-K M-gated).

**C3d — sync-free device-side index (draft §4.2 v1).** Remove the single D2H sync by passing
`grid_thw` + device cumsum prefixes and locating each program's grid via a ≤8-iter in-kernel scan.
Hypothesis (H-C3d): eliminates the last host sync; measurable only on the smallest workloads where
launch/sync overhead dominates. Adopt only if it helps small-M and is correctness-neutral.

---

## 5. Decision rule (adopt / reject / stop)

For each evaluated candidate:
- **Invalid** if any workload fails correctness OR the run errors (rc≠0) OR profiling
  contaminated the eval (rc 3). Record as invalid; do not branch from it.
- **Adopt** (becomes new parent) if: all 15 PASS **and** geomean improves by a **meaningful margin
  ≥ ~1.5%** over the current best. (Coarse timing warmup2/10-iter has run-to-run jitter; treat
  sub-~1% deltas as noise. If a change is plausibly beneficial but lands within noise, prefer the
  simpler source and note it.)
- **Reject** if all PASS but geomean is flat/worse: keep parent, record the negative result, move
  to the next planned lever. Avoid re-testing a lever already shown flat.

Re-parenting: if a branch regresses, the next candidate extends the last adopted (best) parent, not
the regressed one.

---

## 6. Stopping criteria → `SEARCH_COMPLETE`

Stop and write `SEARCH_COMPLETE` (with the reason) when any holds:
1. **Convergence:** 3 consecutive adopted-or-tested candidates yield < ~1.5% cumulative geomean
   gain, and the remaining planned levers are exhausted or judged low-value by profiling evidence.
2. **Profiling ceiling:** ncu shows large-M workloads at high tensor-core utilization near cuBLAS
   AND small-M workloads DRAM-bandwidth-bound near their weight-streaming SOL — i.e. little
   headroom on either half.
3. **Budget:** approaching the 100-eval cap or the token soft limit (9M) with no active promising
   lever. Leave margin; never blow past the hard limit.

`SEARCH_COMPLETE` names the best candidate ID, its geomean, and the reason. `final` is run ONLY
after explicit operator approval.

---

## 7. Evidence format (append-only `candidates.jsonl`)

One JSON object per **evaluated** candidate, appended in evaluation order; earlier records are
never rewritten. Fields:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "correctness baseline: 3-kernel LN-scatter + two tl.dot GEMMs, exact erf GELU",
  "change_from_parent": "initial implementation",
  "validation": {
    "static_checks": ["shuffle bijection T1H2W2", "LN fp32 unbiased eps-in-sqrt",
                      "weight x@W^T orientation", "erf symbol resolved", "M from grid_thw"],
    "all_pass": true
  },
  "per_workload": [
    {"uuid": "3b69084d", "num_patches": 4096, "M": 1024, "pass": true, "speedup": 0.0},
    "... one entry per 15 feedback workloads ..."
  ],
  "geomean": 0.0,
  "eval_return_code": 0,
  "decision": "adopt|reject|invalid",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki"],
  "notes": "baseline; suspects-on-failure list; profiling refs if any"
}
```

Rules: `per_workload` covers all 15 with pass flag + speedup; `geomean` is the reported
geometric-mean speedup; `decision` follows §5; `cumulative_evals` monotonic; `skills_used` records
KernelWiki / ncu-report-skill usage; profiling artifacts (if produced) referenced by path under
`profile/`. If an eval is contaminated (rc 3) or errors, still record it as `invalid` with the
return code so the budget accounting stays accurate.

---

## 8. Per-turn operating procedure

1. Ensure only ONE source version on disk (`solution/solution.py`); confirm it is the intended
   candidate before evaluating.
2. Run pre-eval static checks for the phase.
3. `./scripts/evaluate_candidate.sh feedback cNNN` (never with profiling active).
4. Parse per-workload pass/speedup + geomean; apply §5 decision.
5. Append the `candidates.jsonl` record (§7).
6. If profiling is needed, do it **after** the eval finishes, via `./scripts/ncu_profile.sh`, on
   1 small-M + 1 large-M representative; record findings.
7. Decide next lever per the lineage (§1); assign the next sequential ID.
8. Re-check budget/tokens against §6; if a stop condition holds, write `SEARCH_COMPLETE`.
