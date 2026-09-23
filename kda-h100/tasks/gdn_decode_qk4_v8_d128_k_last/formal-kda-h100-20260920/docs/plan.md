# Plan — `gdn_decode_qk4_v8_d128_k_last` (H100 / sm_90)

Executable, sequential KDA optimization plan. Builds directly on `docs/draft.md`
(§ references below point there). No candidate is implemented or evaluated in this
turn — this file only defines the roadmap, gates, and evidence format.

---

## 0. Objective & operating budget

- **Goal:** maximise geometric-mean speedup over the reference across the **full 54-workload
  feedback set** while **every** selected workload passes correctness. Geomean weights all 54
  equally, so small-batch (B=1..8 = 25/54) counts as much as large-batch bandwidth (draft §2).
- **Primary implementation:** Triton kernel behind `solution/solution.py::run(...)`. PyTorch only
  for metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-ext/alternate fallback — a failed Triton
  path is an invalid candidate, never substituted.
- **Budgets:** ≤100 candidate evaluations; tokens soft 9M / normal 10M / absolute 11M.
- **One immutable source version per candidate ID.** Any meaningful source/config/launch change ⇒
  new ID. Never reuse an ID for changed source. Append one JSON record per evaluated candidate to
  `candidates.jsonl`; never rewrite earlier records.
- **Evaluate only** via `./scripts/evaluate_candidate.sh feedback <id>` (full 54-set = 1 eval).
  `final` only with explicit operator approval.

---

## 1. Candidate lineage strategy

### 1.1 Discipline
- Sequential, one candidate at a time. Implement → self-review (§2 checklist) → evaluate → record →
  decide → branch. Never run two candidates or edit source between an eval launch and its result.
- **Parent selection:** each new candidate declares a `parent` = the best *valid* (all-54-pass)
  ancestor by geomean, unless the candidate is an intentional exploration off a non-best parent
  (state that explicitly in the hypothesis).
- **One variable at a time** where practical: change either the algorithm form, the tiling/grid, or
  the launch config per candidate so an eval delta is attributable. Bundle changes only when they are
  logically inseparable, and say so.
- **Keep the best valid candidate pinned** in notes at all times so a regression is a cheap revert.

### 1.2 Lineage skeleton (concrete, may prune/extend from evidence)

| ID | Parent | Kind | One-line hypothesis |
|----|--------|------|---------------------|
| c001 | — | correctness baseline | Single fused launch, grid `(B·Hv, V/BLOCK_V)`, **faithful** math (materialise `new_row` then dot), f32 accum, stable softplus/sigmoid. All 54 pass; establish baseline geomean. |
| c002 | c001 | small-B occupancy | Tune `BLOCK_V`/grid so B=1..8 launch enough blocks (≥ a few × 132 SMs); measure small-batch geomean lift. |
| c003 | best | autotune-by-batch | `@triton.autotune` (or wrapper heuristic) selecting `BLOCK_V`/`num_warps` keyed on `B` so both small- and large-B regimes get their best config in one source. |
| c004 | best | large-B bandwidth | Vectorised/wide (128-bit) contiguous-K state loads + streaming/`evict_first` cache hint on the read-once state; target DRAM-SoL at B=32..64. |
| c005 | best | `num_warps`/`num_stages` sweep | Pick warp count so each thread owns a few contiguous K elements; shallow-pipeline check. |
| c006 | best | fused output formula | Replace materialise-then-dot with `out = scale*(g*(q·S_row)+delta*(q·k))` (draft §1.5); must still pass — de-risked *after* baseline is green. |
| c007 | best | per-block work vs occupancy | Fuse multiple heads / whole `(h,v)` of a batch element per program at small B to cut launch overhead; A/B vs c003. |
| c008+ | best | targeted from ncu | Address whichever bottleneck profiling reveals (occupancy, load width, tail, register pressure). |

Only c001 is fixed up front; c002+ are *directions* — the exact next candidate is chosen from the
prior evaluation + profiling evidence, not pre-committed. Prefer the change with the largest expected
geomean impact for the fewest evals.

---

## 2. Correctness checks (pre-evaluation gate for every candidate)

Run this checklist by inspection **before** spending an evaluation:

1. **Shapes:** `output` `[B,1,Hv=8,V=128]` bf16; `new_state` `[B,Hv=8,V=128,K=128]` f32. Singleton
   T handled in wrapper.
2. **Dtypes / accumulation:** bf16 inputs loaded then cast to f32; **all** reduction/state math in
   f32; cast to bf16 only on the final `output` store. `new_state` stays f32.
3. **Head mapping:** v-head `h` reads q/k-head `h//2` (GVA `repeat_interleave(2)`; draft §1.2).
4. **Layout:** read `state[b,h,v,:]` and write `new_state[b,h,v,:]` (k contiguous). **No physical
   transpose** reintroduced (draft §1.8).
5. **Gates:** `x=a+dt_bias`; `g=exp(-exp(A_log)*softplus(x))` with **stable softplus**
   `relu(x)+log1p(exp(-|x|))` (matches `F.softplus`); `beta=sigmoid(b)` via stable primitive
   (draft §4.1–4.3).
6. **Math form:** c001 uses faithful `sk=dot(k,S_row); old_v=g*sk; delta=beta*(v-old_v);
   new_row=g*S_row+k*delta; out=scale*dot(q,new_row)` (draft §1.5). Fused form only from c006.
7. **Scale:** applied once to the output (`scale=1/√128` in all feedback workloads, but read from the
   argument — do not hard-code).
8. **State-optional / strides:** if `state` is None allocate f32 zeros in the wrapper (no in-kernel
   branch); address with the tensors' actual strides or `.contiguous()` in the wrapper. Verify
   contiguity assumption.
9. **Single launch:** one kernel launch covers the whole batch — no Python loop over batch/heads;
   lean `run()` (precompute strides/grid once, no per-call graph building or redundant allocs;
   draft §5.4).
10. **Determinism/interference:** no reliance on global RNG; nothing that spawns extra GPU processes.

The **official feedback evaluation is the sole correctness oracle** (draft §6). If a candidate fails
any workload, it is invalid regardless of speed; diagnose from the failure signature, fix in a *new*
ID, and record the failure.

---

## 3. Performance hypotheses (falsifiable, each mapped to a candidate)

- **H1 (bottleneck):** the op is memory-bandwidth-bound at large B and occupancy/launch-bound at
  small B (draft §1.6). *Test:* ncu on a B=64 and a B=1 workload — expect high DRAM throughput vs SoL
  at B=64 and low launched-block/occupancy at B=1. Governs whether we chase bandwidth or occupancy.
- **H2 (small-B occupancy):** at B=1..8, larger `BLOCK_V` under-fills the 132 SMs; smaller `BLOCK_V`
  (more blocks) improves small-batch geomean. *Test:* c002 sweep; expect monotone improvement until
  block count ≫ SM count, then flatten.
- **H3 (batch-dependent optimum):** best `BLOCK_V`/`num_warps` differ between small and large B, so a
  single static config is suboptimal; autotune/heuristic-by-B (c003) beats any fixed config on
  geomean. *Test:* c003 vs best static.
- **H4 (large-B bandwidth):** wide 128-bit coalesced loads of the contiguous-K row + streaming cache
  policy push B=32..64 toward DRAM SoL (draft §5.3, `pattern-memory-bound`). *Test:* c004; expect
  large-B per-workload speedup approach the ~19 µs floor, DRAM throughput ↑ in ncu.
- **H5 (compute is not the bottleneck):** the fused output formula (c006) saves flops but should
  *not* materially change large-B time (already BW-bound); it may help only if it reduces state
  *traffic* or register pressure. *Test:* c006 delta ≈ 0 at large B ⇒ confirms H1; keep only if it
  helps small B and still passes.
- **H6 (launch overhead):** at B=1..8 host-side launch cost is a real fraction of runtime (Triton
  caveat, draft §5.4). *Test:* keep `run()` minimal; c007 fusing more work per block should help small
  B if H6 holds.

Falsification is informative: e.g., if c002 gives no small-B lift, H2 is wrong and we redirect to H6
(launch overhead) rather than more tiling sweeps.

---

## 4. Profiling protocol (optimization guidance only)

- Use **only** `./scripts/ncu_profile.sh <ncu args> -o profile/rN python <harness>` via the
  ncu-report-skill workflow. Never invoke `ncu`/CUDA/`nvidia-smi` directly.
- **Strict serialisation:** never profile while an evaluation is running, and never on the same GPU
  concurrently — a foreign process during timing ⇒ return code 3 ⇒ discarded measurement ⇒ one wasted
  evaluation. Finish one fully before starting the other.
- Profile sparingly and purposefully: confirm H1 early (one small-B + one large-B run), then only when
  an eval result is ambiguous about *why*. Prefer `--set basic`/section-limited runs to keep it cheap.
- Key metrics to read: DRAM throughput vs SoL, achieved occupancy, launched blocks/waves, memory-load
  width/coalescing, register pressure/spills. Profiling never gates correctness.

---

## 5. Stopping criteria

Stop and write `SEARCH_COMPLETE` when improvement has genuinely converged, i.e. **any** of:

1. **Convergence:** ≥3 consecutive new candidates each improve geomean by <1% over the running best
   (diminishing returns), and the best is near the modelled bandwidth SoL at large B.
2. **SoL saturation:** ncu shows large-B workloads at ≳80–85% DRAM SoL and small-B limited by launch
   overhead we cannot remove without CUDA graphs (evaluator-controlled) — no structural lever left.
3. **Budget:** approaching the evaluation cap (100) or the token soft limit (9M) — leave margin to
   record evidence and (with operator approval) run `final`.

On stop: ensure the best valid candidate is clearly identified, `candidates.jsonl` is complete, and
`SEARCH_COMPLETE` states the reason (which criterion) and the winning candidate ID + geomean. Do not
run `final` without explicit operator approval.

---

## 6. Evidence format (one JSON object appended per evaluated candidate)

Append to `candidates.jsonl` immediately after each evaluation; never edit prior lines. Schema:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "faithful baseline; establish correctness + baseline geomean",
  "change_from_parent": "initial implementation",
  "validation": {
    "preeval_checklist": "pass",
    "notes": "shapes/dtype/head-map/layout/gates verified by inspection"
  },
  "results": {
    "all_pass": true,
    "geomean_speedup": 0.00,
    "per_workload": [
      {"uuid": "901e5104-...", "batch_size": 1, "pass": true, "speedup": 0.00}
    ],
    "by_batch_geomean": {"1": 0.0, "4": 0.0, "8": 0.0, "16": 0.0, "32": 0.0, "48": 0.0, "64": 0.0}
  },
  "decision": "keep-as-best | reject | branch-parent-for-next",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "profiling": {"done": false, "report": null, "key_findings": null},
  "next": "c002: small-B BLOCK_V/grid occupancy sweep"
}
```

- `geomean_speedup` and `per_workload[*].speedup` are taken verbatim from the evaluator output
  (reference_time / candidate_time convention as the controller reports).
- `by_batch_geomean` is our own grouping of the reported per-workload numbers to track which regime a
  change helped — essential for attributing H2/H3/H4.
- If a candidate is invalid (a workload failed or the Triton kernel errored), record `all_pass:false`,
  the failing uuid(s)/error signature, `decision:"reject"`, and the corrective direction in `next`.
- Record `cumulative_evals` monotonically; keep a running note of the current best valid `(id, geomean)`.

---

## 7. First action next turn

Implement **c001** exactly as specified (§1.2 row, §2 checklist, faithful math), self-review against
the §2 checklist, then run `./scripts/evaluate_candidate.sh feedback c001` **once**, and append its
record to `candidates.jsonl` in the §6 format. Do not profile in the same window as that evaluation.

---

## 8. Decision log

### c001 — evaluated (eval #1). VALID, 54/54 pass, geomean **207.10x**. **Current best.**
Key evidence beyond raw numbers: our absolute solution time is **nearly flat (~0.055–0.071 ms) across
every batch size 1→64**, while the reference grows ~linearly with B. So the reported speedup is set
almost entirely by our (roughly fixed) wall time, and the per-batch geomean rises monotonically
(B=1 ≈21.7x → B=64 ≈1032x) only because the reference gets slower.

Implications that refine H1/H2/H6 (draft §1.6, plan §3):
- The **fixed ~60 µs floor dominates all 54 workloads**. At B=64 the pure state traffic (128 MB
  read+write) is only ~38 µs, so ~20+ µs is fixed launch/wrapper/compile-dispatch overhead — and at
  B≤8 essentially the *entire* time is that overhead. Cutting the fixed cost multiplies the geomean
  across **every** workload simultaneously → highest-leverage direction.
- Therefore reprioritise: **c002 = shrink the fixed overhead** before chasing large-B bandwidth.
  Candidate wrapper cost is suspect — `run()` currently does `reshape().contiguous().float()` copies
  for state (a full B·8·128·128 f32 copy every call) plus several small contiguous copies. Passing the
  evaluator's tensors directly with stride args (avoiding the state copy) is likely the single biggest
  win and is a pure launch-plumbing change (still Triton-only).
- Only after overhead is minimised do occupancy (BLOCK_V/num_warps for small B) and large-B
  vectorised/streamed loads matter. Confirm the overhead-vs-bandwidth split with one small-B + one
  large-B ncu run, strictly serialised with evaluation.

Next candidate: **c002** — remove avoidable host-side copies in `run()` (esp. the state `.contiguous().float()`)
and pass strides to the kernel; keep math identical to c001 so the eval delta is pure overhead.

### c002 — evaluated (eval #2). VALID, 54/54 pass, geomean **280.48x**. **New current best** (parent c001).
Pure wrapper change (kernel byte-identical; error columns unchanged). Dropped the per-call
`reshape().contiguous().float()` copies — most importantly the full `B·8·128·128` f32 **state copy** —
and allocated `output` directly as `[B,1,Hv,V]`. Result: **+35% geomean (207→280) from launch-plumbing
alone**, confirming H6/overhead was the dominant cost. Sol time fell ~0.062→0.045 ms (B=1) and
~0.064→0.052 ms (B=64).

Refined picture: solution wall time is **still nearly flat (~0.040–0.054 ms) across B=1→64**. Even at
B=64, where pure state traffic (~128 MB) is only ~38 µs, we are not yet bandwidth-bound — the residual
**fixed launch/dispatch + kernel-prologue cost still dominates every workload**. So the highest-leverage
lever remains cutting fixed overhead / improving small-B occupancy, *not* large-B bandwidth tuning yet.

Next candidate: **c003** — one structural change to reduce launch/prologue cost or raise small-B
occupancy: e.g. collapse the 2-D grid `(B·Hv, V/BLOCK_V)` into a single 1-D grid (cheaper launch,
same work) and/or lower `BLOCK_V` so B=1 fills >132 SMs; optionally `@triton.autotune` keyed on B.
Keep the delta-rule math identical so the eval delta is attributable. Before committing to large-B
bandwidth work, consider one small-B + one large-B ncu run (strictly serialised with evaluation) to
quantify the fixed-overhead vs. bandwidth split.

### c003 — evaluated (eval #3). VALID (54/54 pass) but geomean **197.76x → REGRESSION vs c002 (280.48x).** **Rejected.**
Change: 1-D grid + `@triton.autotune` over `(BLOCK_V∈{1,2,4,8,16}, num_warps∈{1,2,4})` keyed on
`n_bh`. Kernel math identical (error columns unchanged).

Why it regressed (**important lesson**): this kernel is **host-launch-latency bound**, and
`@triton.autotune` adds real per-call Python work on the hot path — key hashing, config-cache lookup,
and the `grid` lambda evaluation — which directly inflates the very launch-latency floor that dominates
all 54 workloads. Sol time rose ~0.045→0.065 ms across every B; there was also a 0.218 ms outlier at
wl#16 consistent with tuning-cache disturbance during timed iters. So: **any added Python/dispatch in
`run()` costs geomean**, and autotune is the wrong tool for a fixed-cost-dominated launch. The 1-D grid
itself was neutral/harmless; the autotune wrapper was the culprit.

**Decision:** revert to **c002 as the base (best = 280.48x)**. Firm constraint going forward: keep the
`run()` hot path as lean as c002 — no autotune, no extra Python.

Next candidate: **c004** — pick ONE static change off c002, no added hot-path Python:
(a) manual batch-adaptive `BLOCK_V`/`num_warps` chosen with a couple of int compares on `B` (negligible
overhead) to fill SMs at small B / coalesce better at large B; or
(b) fused output formula `out = scale*(g*(q·S) + delta*(q·k))` to cut per-row compute (draft §1.5/§4.5);
or (c) first spend one small-B + one large-B ncu run (strictly serialised with evaluation) to measure
how much of the ~0.045 ms floor is pure launch latency vs. kernel execution, so we stop guessing.
Given c001→c002→c003 all point to launch-latency dominance, option (c) is attractive before spending
more evals — but any change must keep math identical unless it is explicitly the fused-formula test.

### c004 — evaluated (eval #4). VALID, 54/54 pass, geomean **288.48x**. **New current best** (parent c002).
Single-variable change off c002: **removed the 8 per-call `.contiguous()` host round-trips** in `run()`
(the evaluator supplies contiguous inputs, so they were no-ops). Kernel byte-identical (error columns
match c001/c002 exactly). Geomean 280.48→288.48 (**+2.9%**) — even trivial Python removed from the hot
path lifts geomean, reconfirming the **launch-latency-bound** model.

Caveat on this run's numbers: reference times drifted ~7% higher than c002's run (B=64 ref ~71 ms vs
~66 ms), which inflates the large-B speedups somewhat. The most trustworthy signal is **small-B**, whose
speedup is set by our own (ref-stable) fixed time: B=1 geomean 29.7→31.1 improved cleanly. Sol floor now
~0.044–0.046 ms at small B.

**Running best = c004 (288.48x).** Firm constraints reconfirmed: lean `run()`, no autotune, no hot-path
Python.

Next candidate: **c005 = profile, not guess.** We have shaved host overhead about as far as trivial
edits allow; before spending more evals on tiling/compute guesses, run **one small-B + one large-B**
`./scripts/ncu_profile.sh` measurement (strictly serialised with evaluation — never concurrent, or the
controller discards the timing with rc 3) to split the ~0.044 ms floor into launch-latency vs. actual
kernel execution.
- If kernel exec ≪ 0.044 ms at small B → the residual floor is pure Triton/CUDA launch latency we
  cannot remove without CUDA graphs (evaluator-controlled) → we are near **convergence** (stopping
  criterion §5.2) and should stop soon.
- If kernel exec is a real fraction (esp. at large B, where 128 MB traffic ≈ 38 µs) → a static
  batch-adaptive `BLOCK_V`/`num_warps` (int compares only) or the fused output formula is worth one
  more candidate.
Either way, the profiling result decides whether c006 is a real optimization or whether we write
`SEARCH_COMPLETE`.
