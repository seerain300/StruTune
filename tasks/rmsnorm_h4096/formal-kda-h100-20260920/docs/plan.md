# Plan — `rmsnorm_h4096` (executable KDA optimization plan)

This plan operationalizes `docs/draft.md`. It defines the exact candidate lineage,
per-candidate specs, correctness checks, performance hypotheses, decision rules, stopping
criteria, and the evidence/record format. **No code or evaluation happens in this turn.**

---

## 0. Ground rules (recap, binding)

- Compute is **Triton only**. PyTorch used solely for metadata/launch. No Torch/CPU/NumPy/
  CUDA-extension/alternate fallback. A failing Triton kernel is invalid → fix, don't bypass.
- Entry point: `solution/solution.py` exposing `run(hidden_states, weight)` → bf16 output.
- Candidates are **immutable and sequential**: `c001`, `c002`, … Any meaningful source/
  config/launch change ⇒ new id. Never reuse an id for changed source. Never rewrite an
  earlier `candidates.jsonl` record.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback <id>`. One full pass over
  the feedback workloads = one candidate evaluation = one unit of the 100-eval budget.
- **Never** run `final` without explicit operator approval.
- Profiling **only** through `ncu-report-skill` on a workspace-local harness, **never**
  overlapping an evaluation (foreign process on the locked GPU ⇒ rc 3, wasted eval).
- Budgets: 100 evals; tokens soft 1.0 M / normal 1.5 M / absolute 1.65 M.
- **Note on this session:** local `Bash`/`python` are denied. Every candidate is
  validated by *static review* (§4) before spending an evaluation; the evaluator is the
  sole correctness/perf oracle.

---

## 1. Objective & metric model

- Maximize **geometric mean speedup** vs the reference over the feedback workloads, with a
  **hard correctness gate on every workload** (any fail ⇒ candidate invalid, geomean N/A).
- The feedback set has **14** shapes: 9 small/latency-bound (batch 1,7,15,16,34,63,64,79,
  170) and 5 large/bandwidth-bound (8804,10827,11832,14418,14509). (Docs say "five"; the
  file has 14 and `TASK.md` says the full set runs — we count one full pass = one eval.)
- **Geomean is dominated (9/14) by small batches.** Therefore two objectives, in order:
  1. **Minimize host/launch + per-row latency** (moves the 9 small factors).
  2. **Saturate HBM bandwidth** (moves the 5 large factors).
  A change that helps one regime must not regress the other below its current best, or it
  is rejected.

---

## 2. Candidate lineage strategy

Tree, not a chain: each experimental candidate branches from the **current best valid
parent**, changing **exactly one lever** so the evaluation attributes cause→effect.
Lineage recorded via `parent` + `source_sha256` in every record.

```
c001  baseline (correctness + perf reference)         [parent: none]
 ├─ c002  num_warps sweep A                            [parent: c001]
 ├─ c003  num_warps sweep B                            [parent: c001]
 │        → pick best of {c001,c002,c003} = BEST_W
 ├─ c004  large-batch amortization (ROWS_PER_BLOCK / num_stages)  [parent: BEST_W]
 ├─ c005  batch-adaptive launch (small vs large regime)           [parent: best so far]
 ├─ c006  cache/eviction hints (weight L2 reuse)                  [parent: best so far]
 └─ ...   only if profiling/evidence motivates a distinct hypothesis
```

Rules:
- **One lever per candidate.** Never combine two untested changes in one id.
- Branch from the **best valid** candidate unless deliberately re-testing a lever in a new
  context (state the reason in the record).
- If a candidate regresses or fails correctness, **abandon that branch**; next candidate
  branches from the prior best, not the failed node.
- Stop generating speculative variants once §6 stopping criteria are met.

---

## 3. Per-candidate specifications

### c001 — correctness-first fused baseline  *(parent: none)*
- **Design:** one Triton program per row. `grid = (batch,)`. `BLOCK_H = 4096` as
  `tl.constexpr` (= hidden_size), so **no masking** on the hidden axis; `grid == batch`
  so no batch-axis masking.
- **Math (exact reference reproduction):** load `x` bf16 → cast fp32; `sum_sq = tl.sum(x*x)`
  in fp32; `inv_rms = tl.rsqrt(sum_sq / 4096 + 1e-5)`; load `weight` bf16 → fp32; `y =
  (x * inv_rms) * w_fp32`; cast `y` → bf16 once (RNE); store. **Reuse the loaded `x`
  register tile** for the output (single global load of `x`).
- **Host `run()`:** `out = torch.empty_like(hidden_states)`; compute `batch` from
  `.shape[0]`; single launch; pass `EPS`/`N` as constants. No `.contiguous()`, `.float()`,
  `.cuda()`, reshape, or per-call allocation beyond `out`.
- **Config:** default `num_warps` (start 8 for a 4096-wide row), `num_stages` default.
- **Purpose:** lock correctness on all 14 shapes; establish the perf/geomean baseline.
- **Success:** all workloads pass correctness; geomean ≥ 1.0 (expected well above, since we
  replace ~5–6 eager passes with one fused read/write).

### c002 / c003 — `num_warps` sweep  *(parent: c001)*
- Change **only** `num_warps`. Sweep set {4, 8, 16} (c001 already covers one value; pick the
  two not yet tested). Everything else identical to c001.
- **Hypothesis:** more warps ⇒ more threads on a lone 4096-row ⇒ lower per-row latency
  (helps small batches); too many may cut occupancy on large batches. Find the sweet spot.
- **Decision:** `BEST_W` = arg max geomean among {c001,c002,c003} that passes correctness.

### c004 — large-batch amortization  *(parent: BEST_W)*
- Change **one** of: `ROWS_PER_BLOCK` (each program strides over K rows, K∈{2,4,8}) *or*
  `num_stages` (software-pipeline the load, e.g. 2–4). Pick the single most promising per
  reasoning/profiling; if both are of interest, they become separate candidates.
- **Hypothesis:** for batch ~10⁴, amortizing scheduling and improving weight/L2 reuse
  raises effective bandwidth; shrinking the grid (14509→few k) reduces launch/sched cost.
- **Guard:** must NOT regress small-batch geomean below BEST_W. If it does, keep only if it
  becomes part of a batch-adaptive launch (c005).

### c005 — batch-adaptive launch  *(parent: best-so-far)*
- In `run()`, select launch params from `batch`: e.g. `ROWS_PER_BLOCK`/`num_warps` = large
  for big batch, one-row-per-program for small batch. Purely host-side selection;
  deterministic (branch on a Python int, no data-dependent divergence).
- **Hypothesis:** serve both regimes optimally in one candidate → best combined geomean.
- **Note:** distinct launch config by batch is a legitimate single lever ("adaptive
  launch"), recorded as one candidate.

### c006 — cache / eviction hints  *(parent: best-so-far)*
- Add `eviction_policy` hints: `weight` load `evict_last` (keep hot in L2 across programs);
  `x`/`y` `evict_first` (stream, avoid polluting L2) — only if profiling shows L2 pressure
  on large batches.
- **Hypothesis:** better weight reuse / less cache pollution ⇒ higher large-batch BW.

### c00N — evidence-driven only
Further candidates only if profiling (ncu) or evaluator results reveal a concrete,
addressable bottleneck (e.g. lone-row starvation at batch=1 → split-hidden reduction).
Each is a single-lever branch from the current best with an explicit hypothesis.

---

## 4. Correctness checks (static, run before every evaluation)

Since local execution is denied, each candidate passes this checklist **before** spending
an eval; the evaluator then confirms:

1. **fp32 accumulation:** load bf16 → `.to(tl.float32)`; square/sum/mean in fp32. No bf16
   arithmetic anywhere.
2. **Exact op order:** `inv_rms = rsqrt(sum_sq / 4096 + 1e-5)` — divide by N, then `+eps`,
   then rsqrt (matches reference). `EPS = 1e-5` literal.
3. **Weight promotion:** `weight` cast bf16→fp32 before the multiply; `y = (x*inv_rms)*w_fp32`.
4. **Single final cast:** keep product in fp32; one `.to(tl.bfloat16)` (RNE) at store.
5. **Indexing/bounds:** `grid == batch`, `BLOCK_H == 4096 == N` ⇒ no OOB, no mask needed;
   row base = `pid * N`; contiguous `arange(0, N)` load/store (coalesced).
6. **Host purity:** output `empty_like` (bf16, same device/layout); no unintended
   cast/copy/contiguous; single launch; shapes/dtypes match the contract signature.
7. **Determinism:** no atomics/cross-block reduction in baseline; any split-reduction
   variant must combine in fp32 deterministically.
8. **Constants specialized:** `N`/`BLOCK_H` as `tl.constexpr` so masking is compiled out.

**Correctness oracle:** `evaluate_candidate.sh feedback <id>` gates every workload. A pass
= numerical equivalence within tolerance on all shapes. A single fail ⇒ candidate invalid.

---

## 5. Performance hypotheses (explicit, testable)

| # | Lever | Hypothesis | Expected regime | Candidate |
|---|---|---|---|---|
| H1 | Fusion (1 read+1 write vs ~5–6 passes) | Large geomean gain, esp. large batch (BW) + fewer launches (small batch) | both | c001 |
| H2 | `num_warps` ↑ | Lower per-row latency; helps small batch; may cut large-batch occupancy | small ↑ | c002/c003 |
| H3 | `ROWS_PER_BLOCK` / `num_stages` | Amortize sched + weight reuse ⇒ higher large-batch BW | large ↑ | c004 |
| H4 | Batch-adaptive launch | Best of both regimes in one kernel | both | c005 |
| H5 | Eviction/cache hints | Weight stays in L2, x/y stream ⇒ large-batch BW ↑ | large ↑ | c006 |

Each hypothesis is confirmed/refuted by the **per-workload** deltas (not just geomean):
did the targeted regime improve without regressing the other?

---

## 6. Decision rules & stopping criteria

**Accept a candidate as new best** iff: (a) all 14 workloads pass correctness, AND (b)
geomean > current best geomean by a **meaningful margin (> ~1%)**, AND (c) no single
workload regresses by more than a small noise band (~2–3%) unless the geomean gain clearly
justifies it. Otherwise keep the previous best; abandon the branch.

**Coarse-run noise caveat:** feedback uses warmup 2 / 10 iters (coarse), so small deltas
(≲1–2%) are within noise — do not chase them; require a clear margin to accept.

**Stop (write `SEARCH_COMPLETE`) when any of:**
- Improvement has **converged**: ~2–3 consecutive candidates yield no meaningful geomean
  gain (each ≲1%), across the levers that reasoning/profiling identify as relevant.
- Kernel is demonstrably at the **bandwidth ceiling** on large batches (ncu shows achieved
  DRAM BW near device peak) and small batches are launch-overhead-bound with no further
  host-side reduction available.
- Approaching the **evaluation budget** (reserve margin) or the **token soft limit** (1.0 M):
  consolidate on the best valid candidate and stop.
`SEARCH_COMPLETE` states the reason and names the best candidate id + its geomean.

**Never** launch `final` without operator approval; when asked/approved, run it on the best
valid candidate only.

---

## 7. Evidence format (`candidates.jsonl`, one JSON object per evaluated candidate)

Append-only; never rewrite prior records. Each record includes:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<iso8601>",
  "hypothesis": "fused single-pass bf16 read/write, fp32 accumulate; establish baseline",
  "levers_changed": ["baseline"],
  "config": {"num_warps": 8, "num_stages": null, "rows_per_block": 1, "block_h": 4096},
  "validation": {
    "static_checks": "fp32 accum; mean+eps+rsqrt order; fp32 weight; single RNE cast; no mask; empty_like; single launch",
    "correctness": "pass|fail (per evaluator)"
  },
  "per_workload": [
    {"batch": 1, "uuid": "864d596c-...", "pass": true, "speedup": 0.0},
    {"batch": 7, "uuid": "33bf737d-...", "pass": true, "speedup": 0.0}
    /* ... all evaluated workloads ... */
  ],
  "geomean": 0.0,
  "decision": "accept-as-best | reject-regression | invalid-correctness | abandon-branch",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki?", "ncu-report-skill?"],
  "notes": "regime deltas, next lever to try"
}
```

Required fields per CLAUDE.md #7: parent, source hash, hypothesis, validation, per-workload
result, geomean, decision, cumulative evaluation count, skill usage. All present above.

**Reporting discipline:** record **per-workload** speedups (both regimes), not only the
geomean, so each hypothesis in §5 can be judged. Note whether a skill (KernelWiki for
arch-specific tuning; ncu-report-skill for profiling) was actually consulted.

---

## 8. Execution order (sequential checklist)

1. Implement **c001** → static checklist §4 → `evaluate_candidate.sh feedback c001` →
   append record. Establish baseline geomean.
2. `num_warps` sweep **c002/c003** → pick `BEST_W`.
3. Large-batch **c004** (single lever) → accept/reject per §6.
4. **c005** batch-adaptive launch if regimes conflict.
5. **c006** cache hints if profiling shows L2 pressure.
6. Evidence-driven **c00N** only if a concrete bottleneck remains.
7. Convergence → `SEARCH_COMPLETE` (reason + best id/geomean).
8. `final` only on explicit operator approval, best valid candidate.

**Between steps:** never overlap profiling and evaluation; reserve budget/token margin;
one lever per candidate; branch from the best valid parent.

**Next step (separate turn):** implement `solution/solution.py` for **c001** only, run the
static checklist, then evaluate. No other candidate until c001's record is appended.

---

## 9. Decision log (append-only)

- **c001 (eval #1) — accept-as-best. geomean 4.10x, 14/14 pass.** Device confirmed:
  **NVIDIA H100 80GB HBM3** (not A800 — portable memory-bound design was the right call).
  Tolerance: atol=0.01, rtol=0.01, matched_ratio=0.99. Fused baseline validated the whole
  numerical approach (max rel err 7.81e-3, well inside tol). Regime split as predicted:
  small batches 1–170 ≈ **2.38–2.70x** (launch/latency-bound, dominate 9/14 of the
  geomean); large batches 8804–14509 ≈ **8.88–9.73x** (bandwidth-bound). Small-batch
  sol times are ~0.029–0.036 ms and nearly flat across batch 1→170 → almost pure fixed
  overhead (launch + kernel prologue), not compute. **Next lever = c002/c003 num_warps
  sweep** (H2): fewer warps (e.g. 4) may cut per-row prologue/latency on the tiny batches
  that dominate the geomean; will branch from c001 changing only `num_warps`. Large-batch
  amortization (c004) is secondary since it only moves 5/14 factors.
- **c002 (eval #2) — accept-as-best. geomean 4.24x (+3.4% vs c001), 14/14 pass.** Lever:
  `num_warps` 8→16 only. H2 confirmed: small batches rose to **2.57–2.92x** (from
  2.38–2.70x) — the launch/latency-dominated shapes that drive the geomean improved. Large
  batches dipped slightly (14418 9.40 vs 9.66; 8804 8.64 vs 8.88) as occupancy fell, but
  that only touches 5/14 factors so net geomean gained. New best = **c002 (num_warps=16)**.
  Next: **c003 = num_warps=4** (opposite direction) to bracket the optimum and confirm 16 is
  best; then consider large-batch amortization (c004) since large-batch sol_ms rose at 16
  warps.
- **c003 (eval #3) — reject-regression. geomean 3.88x < c002 4.24x.** Lever: `num_warps`
  16→32 (I probed the *upper* edge rather than 4, since the geomean-dominating small batches
  were still rising and 4 could only hurt them). Result **brackets the optimum**:
  8→4.10, 16→4.24, 32→3.88 ⇒ **peak at num_warps=16**. Regime picture now monotonic/clear:
  small batches want MORE warps (8:2.38–2.70 → 16:2.57–2.92 → 32:2.62–2.94, 16→32 marginal),
  large batches want FEWER (8:8.88–9.73 → 16:8.64–9.54 → 32:6.58–7.05 — sharp fall as 1024
  threads/block crush occupancy). Classic small-vs-large **regime conflict**. Best stays
  **c002**. A uniform num_warps=4 test is redundant (only worsens small batches; 8 is already
  the known large-batch best). **Next c004 = batch-adaptive launch**: choose num_warps from
  batch in `run()` — 16 for small batch (c002 optimum), 8 for large batch (c001 optimum) — to
  capture best-of-both and resolve the conflict.
