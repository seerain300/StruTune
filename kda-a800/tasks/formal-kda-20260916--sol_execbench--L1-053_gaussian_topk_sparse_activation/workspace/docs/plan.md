# Plan — L1/053 `gaussian_topk_sparse_activation` (executable optimization plan)

Companion to `docs/draft.md`. This is the actionable, sequential plan: candidate
lineage, per-candidate build recipe, correctness gates, performance hypotheses,
stopping criteria, and the evidence (`candidates.jsonl`) format. **No candidate is
implemented or evaluated in this turn.**

Target: NVIDIA A800 (`sm_80`, Ampere). Submission: `solution/solution.py` exposing
`run(inputs, target_sparsity) -> Tensor`. Primary compute in Triton; PyTorch only for
metadata / launch plumbing / host-side scalar. Metric: geomean speedup over reference
across the 5 fixed feedback workloads, with **every scored workload required to pass
correctness**.

---

## 0. Ground rules (operational)

- **Only measurement channel:** `./scripts/evaluate_candidate.sh feedback cNNN`
  (trusted launcher → official evaluator). 5 fixed workloads = **one** evaluation.
- **No local execution** of CUDA / profiler / `nvidia-smi` / raw evaluator / any
  alternate correctness harness. All pre-evaluation checking is **static reasoning**.
- **No fallback.** A failing Triton kernel is invalid; fix numerics/config, never swap
  in Torch/CPU/NumPy/CUDA-extension compute.
- **Immutable candidates.** Any meaningful change to source, launch config, or compile
  constants ⇒ **new** candidate ID. Never reuse an ID for changed source.
- **Budget:** 100 evaluations; token soft 1.0M / normal 1.5M / absolute 1.65M. Converge
  early and deliberately — every eval must test a distinct, justified hypothesis.
- **`final` (12-workload) only with explicit operator approval.** Never run otherwise.
- **Isolation.** Work only in this workspace; only external knowledge source is
  `KernelWiki` (see §9 — not applicable to Ampere, so not planned for use).

---

## 1. Candidate mechanics & lineage

### 1.1 How a candidate is produced
1. Write the immutable kernel version into `solution/solution.py` (the entry point the
   evaluator reads). One source version corresponds to exactly one candidate ID.
2. Do a **static correctness review** (§3 checklist) — pass all items before spending an
   evaluation.
3. Run `./scripts/evaluate_candidate.sh feedback cNNN`. The launcher locks the source
   hash under the candidate ID; that ID is now immutable.
4. Append exactly one JSON record to `candidates.jsonl` (§7). Never rewrite prior lines.

### 1.2 Lineage / branching policy
- Candidates form a tree rooted at `c001`. Each record names its `parent` and the single
  changed dimension (the "one knob" rule).
- **One variable per candidate** wherever practical (config sweep = one knob at a time),
  so speedup deltas are attributable.
- Keep the **best valid candidate so far** (`best_id`) as the running baseline. New
  candidates branch from `best_id` unless explicitly exploring an alternative design.
- A candidate that regresses geomean or breaks correctness is a **dead branch**: record
  it, do not build further on it, revert to `best_id`.

### 1.3 Decision labels (per candidate)
`accept` (new best) · `keep-baseline` (valid but not better; parent stays best) ·
`reject` (regression) · `invalid` (correctness fail / compile fail / spill-crash).

---

## 2. Design recap (from draft, the implementation targets)

Row-wise reduction + broadcast elementwise map: `M = B·S` independent rows of length
`H`. Per row: `mean`, population `std` (divisor **H**, `unbiased=False`), then
`out = relu(x - (mean + std·Z))`, where `Z = _ndtri(target_sparsity)` is a **single
host-precomputed float32 scalar** (the reference `_ndtri` rational approx, replicated
exactly). bf16 in → f32 compute → bf16 out.

Named designs (from draft §6):
- **(A)** single-block row-resident, in-register two-pass, `BLOCK_H = next_pow2(H)`.
  **1 read + 1 write** (memory-optimal). Primary.
- **(B)** blocked two-pass over HBM (small BLOCK, loop twice). 2 read + 1 write.
  Fallback for register pressure at H=12288.
- **(C)** blocked single-pass sum+sumsq. Lower priority (cancellation risk).
- **(D)** split-row reduction. Not needed (M ≥ 512 fills the GPU). Skip.

Host wrapper responsibilities: validate/`contiguous` inputs, view as `[M, H]`, allocate
bf16 output, **early-return `inputs` unchanged when `target_sparsity == 0.0`**, compute
`Z` in float32, launch `grid=(M,)`, reshape back to `[B, S, H]`.

---

## 3. Correctness checklist (static gate — run before EVERY evaluation)

Numerical / semantic:
- [ ] Output dtype **bf16**, exact shape `[B, S, H]`, contiguous, same device.
- [ ] Stats computed in **float32** (upcast bf16 load to f32 before reduce).
- [ ] Variance divisor is **H** (population / `unbiased=False`), not `H-1`.
- [ ] Two-pass variance: mean first, then `sum((x-mean)^2)/H` (avoid `E[x²]-E[x]²`
      cancellation near the ReLU boundary — dominant correctness risk, draft §4.1).
- [ ] `Z = _ndtri(target_sparsity)` replicates the reference rational approximation
      **exactly** (same constants, same 3-region branch, float32), evaluated once on host
      and passed as a scalar kernel arg.
- [ ] Threshold `thr = mean + std·Z`; output `maximum(x - thr, 0)` then `.to(bf16)`.
- [ ] **Masked lanes** (`BLOCK_H > H`): load `other=0.0`; force `(x-mean)` term to 0 on
      masked lanes via `tl.where` **before squaring**; store with row mask (padding never
      written, never pollutes mean/var).
- [ ] **Early exit:** `target_sparsity == 0.0` returns `inputs` directly (bit-exact
      identity, no kernel launch). (No feedback WL uses 0.0, but guard for final/robustness.)
- [ ] Indexing: `row*H + offs` uses 64-bit-safe offsets (WL1: M·H ≈ 1.5e8 elements — fits
      int32, but confirm `pid*H` product does not overflow int32: 18608·8192 ≈ 1.52e8 < 2.1e9 OK;
      still prefer int64 offset arithmetic if cheap).

Build / launch:
- [ ] `grid = (M,)`; one program per row (or documented rows-per-program variant).
- [ ] `BLOCK_H = next_pow2(H)` as `constexpr` (3 distinct H ⇒ ≤3 recompiles; acceptable).
- [ ] `num_warps` / `num_stages` explicit and recorded in the candidate record.
- [ ] No Torch compute on the tensor path; only metadata/launch/host-scalar.

If any box is uncertain, resolve by reasoning **before** spending the evaluation.

---

## 4. Performance hypotheses (what each design predicts)

The op is **HBM-bandwidth bound** (reduction + one elementwise pass, negligible FLOPs).
A800 HBM ≈ 1.6–2.0 TB/s.

- **H1 — Fusion is the main win.** The reference materializes an f32 copy and runs
  several kernels (upcast, mean, std, sub, relu, downcast), moving ≈20–30·MH bytes.
  Design (A) moves ≈4·MH bytes (bf16 read once + bf16 write once) in one launch ⇒
  expected **~5–8× traffic reduction**, bounding achievable speedup similarly (minus
  reduction/launch overhead). Expect a strong multi-× geomean from c001 alone.
- **H2 — WL1 dominates geomean weight** (largest bytes, ~305 MB in). Optimize A800
  occupancy for WL1 (H=8192) first; it is the primary lever.
- **H3 — Register pressure at H=12288** (BLOCK_H=16384) may cause spills in design (A),
  hurting WL2/WL3. Raising `num_warps` (16/32) lowers per-thread element count and
  register footprint; if spills persist, design (B) for large H trades +50% traffic for
  occupancy — a net win only if (A) is spill-bound.
- **H4 — `num_stages` pipelining** of the load can hide latency for the streaming pass;
  benefit likely small for a pure-streaming kernel but cheap to test.
- **H5 — Small-H amortization** (WL4, H=4096): multiple rows per program may improve
  utilization; low expected upside since M=8192 already fills the GPU.

Expected speedup ordering of levers: **fusion (H1) ≫ warps@H=12288 (H3) > stages (H4)
≈ rows-per-program (H5)**.

---

## 5. Sequential candidate roadmap

Each entry is a distinct immutable ID with a single hypothesis. Later IDs are contingent
on earlier evidence (branch from `best_id`). Only c001 is fully specified now; subsequent
IDs are chosen from the actual evaluator feedback.

- **c001 — Correctness anchor / baseline (Design A).**
  One row/program, `BLOCK_H=next_pow2(H)`, f32 in-register two-pass, host `_ndtri`
  scalar, heuristic `num_warps` (start 8; consider 16 for H≥12288 via a simple
  H→warps heuristic, documented). **Gate: 5/5 correctness.** Establishes baseline
  geomean. If any WL fails, next candidate fixes numerics (§4 risks) before any perf work.

- **c002+ — `num_warps` sweep (H3).** Branch from best valid. Vary `num_warps` only
  (e.g. 4→8→16→32), especially for the H=12288 path. Pick the value maximizing geomean
  without correctness loss. One warps value per candidate.

- **c003+ — `num_stages` sweep (H4).** From best. Vary `num_stages` (1→2→3→4) only.

- **c004+ — Register-pressure fallback (Design B) for large H, IF c001–c003 show
  spill-bound behavior on WL2/WL3.** H-dependent dispatch: keep (A) for H≤8192, use (B)
  blocked two-pass for H=12288. Only pursued if evidence (relative WL2/WL3 speedup lagging
  WL1/WL4) supports it.

- **c005+ — Small-H amortization (H5) / vectorization / eviction hints.** From best.
  Rows-per-program for WL4; explicit 128-bit vectorized bf16 access; `eviction_policy`
  hints. One knob per candidate.

- **c006+ — Opportunistic:** single-pass sum+sumsq (Design C) only if it holds tolerance
  and reduces traffic vs the chosen design; otherwise skip.

Iterate strictly while geomean improves (§8). Abandon a lever after 1–2 non-improving
candidates on it.

---

## 6. Per-candidate execution loop (the repeatable procedure)

For each candidate cNNN:
1. **Pick one hypothesis** and the single knob it changes (from §5, branching off `best_id`).
2. **Write** `solution/solution.py` for that version.
3. **Static review** — complete the §3 checklist; resolve every uncertainty by reasoning.
4. **Evaluate** exactly once: `./scripts/evaluate_candidate.sh feedback cNNN`.
5. **Read evidence** — per-workload pass/fail and speedup; compute geomean over the 5.
6. **Decide** (`accept`/`keep-baseline`/`reject`/`invalid`); update `best_id` on `accept`.
7. **Append** one JSON record to `candidates.jsonl` (§7). Never edit earlier lines.
8. **Check stopping criteria** (§8). If met, write `SEARCH_COMPLETE`. Else go to 1.

Correctness watch-list to inspect in each evaluator output: **WL1** (dominates geomean)
and **WL2/WL3** (H=12288, highest design-A register pressure).

---

## 7. Evidence format — `candidates.jsonl`

Append **one JSON object per evaluated candidate**, one line, never rewritten. Schema:

```json
{
  "id": "c001",
  "parent": null,
  "stage": "feedback",
  "source_sha256": "<hash of solution/solution.py as evaluated>",
  "design": "A: single-block row-resident two-pass",
  "config": {"BLOCK_H": "next_pow2(H)", "num_warps": 8, "num_stages": 2},
  "hypothesis": "Fused 1-read/1-write bf16 kernel with f32 two-pass stats and host _ndtri Z is correct and yields multi-x speedup over the multi-kernel f32 reference.",
  "static_validation": {
    "dtype_bf16_out": true, "divisor_H": true, "two_pass_var": true,
    "ndtri_exact": true, "masked_lanes": true, "early_exit_0": true
  },
  "results": [
    {"uuid": "75640ea8-8bb2-546f-b44d-c8d51baee3e4", "wl": 1, "pass": true, "speedup": null},
    {"uuid": "30b3bded-7830-54c5-a242-75faeae6e052", "wl": 2, "pass": true, "speedup": null},
    {"uuid": "151250c8-7c37-56d8-9d87-6edc9f5fd531", "wl": 3, "pass": true, "speedup": null},
    {"uuid": "eb88804d-33eb-5840-93b8-235b7f88961a", "wl": 4, "pass": true, "speedup": null},
    {"uuid": "75e273d1-070a-5864-a5e2-11f455340620", "wl": 5, "pass": true, "speedup": null}
  ],
  "all_pass": true,
  "geomean_speedup": null,
  "decision": "accept",
  "best_after": "c001",
  "cumulative_evals": 1,
  "skills_used": [],
  "notes": "Fill speedup/geomean/pass from the official evaluator output; nulls are placeholders to overwrite with real values before appending."
}
```

Rules:
- `geomean_speedup` = geometric mean of the 5 per-workload speedups **only if all pass**;
  if any workload fails, mark `all_pass: false` and treat geomean as invalid for ranking.
- `cumulative_evals` is monotonic across the file (1, 2, 3, …); it must never exceed 100.
- `source_sha256`, per-workload `pass`/`speedup`, and geomean are copied from the actual
  evaluator output — never fabricated. Placeholder `null`s in this doc are illustrative
  only and must be replaced with measured values in the real record.
- `skills_used` lists any skill consulted for that candidate (expected empty; see §9).

---

## 8. Stopping / convergence criteria

Stop and write `SEARCH_COMPLETE` (with the reason) when **any** holds:
1. **Converged:** best geomean improves by **< ~2%** across **2 consecutive** tuning
   candidates (diminishing returns across the remaining levers in §5).
2. **Levers exhausted:** warps, stages, large-H design, and small-H amortization have all
   been tried from the current best without improvement.
3. **Roofline reached:** measured geomean is near the bandwidth-bound expectation of
   design (A) (≈ the H1 estimate) such that further tuning cannot plausibly help.
4. **Budget guard:** approaching the evaluation cap (≤ a small margin of 100) or the token
   soft limit (1.0M) — stop, keep the best valid candidate.

On stop: `best_id` is the submission candidate. Do **not** run `final` without explicit
operator approval. `SEARCH_COMPLETE` states: best id, its geomean, per-WL correctness,
number of evals used, and why search ended.

---

## 9. Skill usage

- **KernelWiki** — scope is NVIDIA **Blackwell (SM100)/Hopper (SM90)**. This task is
  **A800 / Ampere (`sm_80`)**, which is out of scope; therefore **not planned for use**.
  The relevant Triton idioms here (LayerNorm-style fused reduction + elementwise, warps/
  stages tuning, register-pressure management for wide rows) are standard and handled by
  static reasoning. If ever consulted, log it in that candidate's `skills_used`.
- No other skills, subagents, MCP tools, web search, or external agents (per CLAUDE.md).

---

## 10. Risk register & rollback

| Risk | Trigger to watch | Response |
|------|------------------|----------|
| Threshold error near ReLU boundary | any WL correctness fail | verify f32 two-pass, divisor H, exact `_ndtri`; do not touch perf until fixed |
| Register spill at H=12288 (Design A) | WL2/WL3 speedup lags WL1/WL4 | raise `num_warps`; else Design B for large H (c004+) |
| Masked-lane pollution of variance | WL with H not power-of-2 fails | `other=0` load, `tl.where` before square, masked store |
| `Z` (std_multiplier) mismatch | systematic small bias across WLs | re-check `_ndtri` constants/branches vs reference, float32 |
| Fallback temptation on failure | kernel compile/spill error | never fall back; fix Triton; a failing kernel stays `invalid` |
| Wasted evaluations | multi-knob change, no attributable delta | enforce one-knob-per-candidate; branch from `best_id` |
| ID reuse for changed source | edit after evaluating | always allocate a new ID for any source/config/launch change |

Rollback = branch the next candidate from the last `accept` (`best_id`); the regressing
branch is recorded and abandoned.

---

## 11. Immediate next actions (subsequent turns, not this one)

1. Implement **c001** (Design A) into `solution/solution.py`; complete §3 checklist.
2. Evaluate c001; require 5/5 correctness; record per §7; set `best_id`.
3. Proceed through §5 levers one knob per candidate, honoring §8 stopping criteria.
4. On convergence, write `SEARCH_COMPLETE`; hold `final` for operator approval.

---

## 12. Decision log

### c001 — accept (new best). `best_id = c001`. cumulative_evals = 1.
- Design A (row-resident two-pass), `num_warps = 16 if H>=12288 else 8`, `num_stages=2`.
- **5/5 PASSED, geomean 24.35×**, arithmetic mean 26.53×.
- Per-WL speedup: WL1 12.20× (H=8192, largest, dominant weight), WL2 35.81× (H=12288),
  WL3 40.00× (H=12288), WL4 19.45× (H=4096), WL5 25.17× (H=8192).
- Correctness margins healthy (WL1 max_rel 0.167 under the 99% matched-ratio rule; all
  others ≤ 0.017). No register-spill penalty visible at H=12288 (WL2/WL3 are the fastest).
- **H3 (large-H register pressure) appears NOT to be the bottleneck** — H=12288 rows are
  already the fastest. The bottleneck is the **large-M / H=8192 regime (WL1)**: sol_ms
  0.353 ms, the single biggest absolute cost and the lowest speedup.

**Interpretation for next lever.** WL1 has M=18608 rows × H=8192. At one row/program with
`num_warps=8`, BLOCK_H=8192 ⇒ 1024 elems/thread — heavy per-thread serial work and low ILP
for a pure streaming kernel. WL1/WL5 (both H=8192, num_warps=8) are the two lowest speedups;
WL4 (H=4096, num_warps=8) is also low. Hypothesis: raising `num_warps` for the H≤8192 path
(fewer elements/thread, more parallel loads in flight) should lift WL1/WL4/WL5 without hurting
the already-fast H=12288 path.

### Next candidate — c002 (planned): `num_warps` sweep on the H≤8192 path.
- Single knob: raise `num_warps` for H≤8192 (e.g. 8→16), keep H=12288 at 16, `num_stages=2`.
- Success = WL1/WL4/WL5 speedup up, no correctness loss, geomean > 24.35×.
- If no gain, try `num_stages` (§5 c003+) or rows-per-program for small H before converging.

### c002 — accept (new best). `best_id = c002`. cumulative_evals = 2.
- Single knob vs c001: **uniform `num_warps=16`** (H≤8192 raised 8→16). `num_stages=2`.
- **5/5 PASSED, geomean 24.93× (+2.4% over c001 24.35×)**, arith mean 27.42×.
- Per-WL: WL1 12.23× (≈flat, sol 0.3532→0.3524 ms), WL2 37.26× (↑ from 35.81), WL3 42.50×
  (↑ from 40.00), WL4 19.16× (↓ slightly from 19.45, sol 0.0825→0.0837 ms), WL5 25.95×
  (↑ from 25.17). Correctness margins identical to c001.
- **Key insight:** the gain came from H=12288 (WL2/WL3) and WL5, **not** WL1. WL1
  (M=18608 × H=8192, sol 0.352 ms, 12.2×) is **insensitive to `num_warps`** → it is
  effectively **memory-bandwidth bound / near roofline**, not ILP-bound. WL4 (H=4096)
  regressed slightly at warps=16 (reduced per-SM row residency at small H).

**Interpretation.** The dominant workload (WL1) is at/near its bandwidth roofline, so
further config tuning cannot move it much; remaining geomean upside lives in the
smaller/faster WLs (already ≥19×) and is bounded. Two low-risk levers remain worth one
eval each: (c003) `num_stages` sweep (pipeline the streaming load — may still help the
non-WL1 workloads), and (c004) small-H rows-per-program to recover WL4's residency.

### Next candidate — c003 (planned): `num_stages` sweep.
- Single knob from c002: vary `num_stages` (try 3, and/or 4) with `num_warps=16` fixed.
- Success = geomean > 24.93× with 5/5 pass. If flat/worse, WL1 roofline is confirmed and
  the search is near convergence (§8 criteria 1/3) → likely `SEARCH_COMPLETE` after
  exhausting the small-H residency lever (c004).
