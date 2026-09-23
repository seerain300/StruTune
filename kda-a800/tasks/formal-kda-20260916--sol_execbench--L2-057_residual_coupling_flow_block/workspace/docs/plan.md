# Plan — L2/057 `residual_coupling_flow_block`

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`,
`task/definition.json`, and `task/feedback_workloads.jsonl`. Target: **A800 / sm_80
(Ampere)**. Compute must be **Triton** (PyTorch only for metadata / launch plumbing /
constant weight packing). Metric: **geometric-mean speedup** over the reference with a
per-workload correctness gate. No Torch/CPU/NumPy/CUDA-ext computational fallback.

This document is the contract for implementation. **No candidate is implemented or
evaluated in this turn.** Implementation begins in the next turn with `c001`.

---

## 0. Fixed facts to build against

Constants: `channels=192`, `hidden=192`, `half=96`, `kernel_size=5`, `pad=2`, `n_layers=4`.
Per layer `i`: `x0=x[:,:96]`, `x1=x[:,96:]`,
`h = conv2_i(relu(conv1_i(relu(conv0_i(x0)))))`; `h*=x_mask`; `x1 ± h`;
`x = cat([x0,x1]); x *= x_mask`. Forward `+`, layers 0→3; reverse `−`, layers 3→0.
Conv1d SAME (`padding=2`), zero-padded. Output `[B,192,T]` float32, contiguous.

**Algebraic key (from draft §1.2):** `x0` is never mutated (mask-multiply only), so all
four `transform_i` read the *same* `x0` and are mutually independent:

```
delta  = Σ_{i=0..3} ( transform_i(x0) * x_mask )
x1_out = x1_in  ±  delta            # + forward, − reverse
x0_out = x0_in                       # unchanged
out    = cat([x0_out, x1_out]) * x_mask
```

Order-independence makes forward/reverse a pure sign flip. Mask is built as all-ones
and workloads only override `reverse`, so `mask ≡ 1` in practice; we still apply the mask
in the epilogue (idempotent, correct for binary masks) and validate numerically rather
than hard-coding mask=1 into the math.

### Workloads (feedback = one evaluation over all five)

| WL | uuid prefix | B  | T    | reverse | B·T     | atol   | rtol | match |
|----|-------------|----|------|---------|---------|--------|------|-------|
| 1  | 0e8f2568    | 8  | 1721 | true    | 13 768  | 0.011  | 1e-5 | 0.98  |
| 2  | f274d392    | 4  | 541  | false   | 2 164   | 0.010  | 1e-5 | 0.98  |
| 3  | 3c3868a4    | 16 | 2048 | true    | 32 768  | 0.012  | 1e-5 | 0.98  |
| 4  | 84fb0bc5    | 2  | 293  | true    | 586     | 0.0097 | 1e-5 | 0.98  |
| 5  | ea5ae433    | 64 | 8192 | true    | 524 288 | 0.012  | 1e-5 | 0.98  |

Span of B·T ≈ 586 → 524k (3 orders of magnitude): small WLs (4,2) are launch/occupancy
bound; WL5 is compute bound; WL1/WL3 mid. WL4 has the tightest atol (0.0097) — the
numerical watch-point.

---

## 1. Mechanics (how a candidate is built, snapshotted, evaluated)

1. **Working source:** `solution/solution.py`, exposing `run(...)` with the exact
   signature from `task/definition.json` (all 26 tensor/scalar args, keyword-compatible).
2. **Immutable snapshot per candidate:** before evaluating `cNNN`, copy the exact working
   source to `runs/candidates/cNNN/solution.py` and record its `sha256`. Never edit a
   snapshot after evaluation. A new candidate ID is required for *any* meaningful source /
   config / launch change (block sizes, autotune space, fusion strategy, dtype policy).
   Never reuse an ID for changed source.
3. **Evaluate only** via:
   ```bash
   ./scripts/evaluate_candidate.sh feedback cNNN
   ```
   This runs the official evaluator over all five fixed workloads = one candidate
   evaluation. No direct CUDA / profiler / `nvidia-smi` / evaluator / alternate harness.
4. **Record** one JSON line per evaluated candidate to `candidates.jsonl` (append-only;
   never rewrite earlier records) — schema in §7.
5. `final` is operator-only and never run without explicit approval.

Budget: 100 evaluations; token soft 1.0M / normal 1.5M / absolute 1.65M. Plan targets
convergence in well under ~12 evaluations to stay far inside budget.

---

## 2. Sequential candidate roadmap (lineage)

Lineage is a tree rooted at `c001`; the **standing champion** = best *passing* candidate by
geomean. Each step changes exactly one lever so a regression is attributable. Immutability
keeps every passing ancestor available as a fallback champion.

```
c001 (correct baseline, per-transform Triton conv, TF32)      ── root
  └─ c002 grouped/batched 4 transforms (fewer launches)       ← if c001 passes
       └─ c003 fixed-good block sizes (hand-tuned, no autotune churn)
            └─ c004 autotune (size-bucketed configs)
                 ├─ c005 deeper fusion conv0→conv1 (if mid-WL memory-bound)
                 └─ c006 persistent/2-stage large-tile variant for WL5
  └─ cNNN FP32-accumulate fallback  ← branch only if a WL fails TF32 gate
```

### c001 — correctness-first baseline (root)
- **Goal:** pass all 5 WLs; establish an immutable correct reference and first speedup.
- **Design:** implement the 12 convs directly, but using the independence result is
  optional here — simplest correct path is per-transform: for each of 4 transforms run
  conv0→bias→relu, conv1→bias→relu, conv2→bias (no relu), accumulate into `delta`, then
  one fused combine kernel (`x1 ± delta*mask`, `x0*mask`, write `out`).
- **Conv realization:** implicit-GEMM over `(batch, time-tile, cout-tile)` with the
  `k∈{0..4}` tap loop and zero-pad masking on time (`t+k-2` bounds), `tl.dot` TF32
  (inputs→TF32, FP32 accumulate). Layout `[B,C,T]` (no transpose). Fuse bias+relu in
  epilogue; conv2 epilogue has **no relu**.
- **Why first:** minimal surface area to get the padding offset, bias/relu order, reverse
  sign, and shapes provably right before any restructuring.

### c002 — grouped/batched transforms
- **Change:** pack the 4 transforms into 3 wide/grouped convs (draft §1.2/§5.3):
  `conv0` dense `96→768` on shared `x0`; `conv1` grouped(4) `768→768`; `conv2` grouped(4)
  `768→384` then sum the 4 groups → `delta[B,96,T]`, fused combine. Weight packing via
  `torch.cat/stack` on the provided constant weights (trivial data movement, once per
  call — not "computational").
- **Hypothesis:** collapses ~12 conv launches + ~28 elementwise/cat kernels into ~3 conv
  passes + 1 combine → large win on small WLs (2,4) and less traffic everywhere.

### c003 — fixed hand-chosen block sizes
- **Change:** lock in good `BLOCK_T`, `BLOCK_COUT`, `num_warps`, `num_stages` from
  reasoning (small tiles for WL4/2, larger for WL5), **without** autotune, to isolate the
  effect of tiling from autotune overhead/caching.

### c004 — autotune (size-bucketed)
- **Change:** wrap conv kernels in `@triton.autotune` keyed on `(B, T, reverse)` (or size
  buckets) with `BLOCK_T∈{32,64,128,256}`, `BLOCK_COUT∈{32,64,128}`, `num_warps∈{2,4,8}`,
  `num_stages∈{2,3,4}`. Confirm autotune improves over c003 net of compile/warm cost.

### c005 — deeper fusion (conditional)
- **Trigger:** only if per-WL timings suggest mid-size WLs are memory-bound on the `H0`
  round-trip. Fuse conv0→relu→conv1→relu in one kernel keeping hidden in regs/SMEM.

### c006 — large-tile / pipelined WL5 variant (conditional)
- **Trigger:** only if WL5 remains the geomean drag. Larger tiles + software pipelining
  (`num_stages≥3`); consider persistent scheduling.

### FP32-accumulate fallback (conditional branch)
- **Trigger:** any WL fails the correctness gate on precision (watch WL4). Rebuild the
  failing lineage node with `allow_tf32=False` (FP32 accumulate) — new ID, keep the TF32
  champion if it still passes other WLs.

Only conditional candidates that are actually triggered are implemented; the roadmap is a
menu, not a mandatory sequence.

---

## 3. Correctness checks (pre-evaluation, done by reasoning)

Because ad-hoc CUDA runs are disallowed, each candidate is checked *before* evaluation:

1. **Signature/shapes:** `run` accepts all 26 args by keyword; returns `[B,192,T]` float32
   contiguous. Trace shapes through every conv (`96→192→192→96` per transform / grouped
   `96→768→768→384→96`).
2. **Padding offset:** conv tap uses input index `t + k − 2` for `k∈{0..4}`; mask/skip when
   `<0` or `≥T` (zero-pad). This is the highest-risk off-by-one.
3. **Bias & ReLU order:** conv0→(+bias)→relu; conv1→(+bias)→relu; conv2→(+bias)→**no
   relu**. Verify no stray relu after conv2.
4. **Reverse sign:** `sign=+1` forward (WL2), `−1` reverse (WL1,3,4,5); folded as
   `x1 + sign*(mask*delta)`. A flipped sign silently fails 4/5 WLs.
5. **Mask:** applied to `delta` and to final output; `x0` passed through ×mask. Binary mask
   ⇒ commutes with the reference's raw-vs-masked `x0` feeding; do not hard-code mask=1.
6. **Group→channel/weight indexing (c002+):** each output-channel block maps to the correct
   input group and weight slice; verify the pack order of `torch.cat/stack` matches the
   per-transform indexing used in the kernel.
7. **Order independence (c002+):** relies on `x0` being unchanged across layers — true by
   construction (only mask-multiply writes x0). Re-verify whenever restructuring.
8. **dtype/contiguity:** accumulate in FP32; cast output to float32; ensure output stride
   matches `[B,C,T]` row-major.

**Correctness gate is authoritative** via the official evaluator (per-WL atol/rtol/
match-ratio). Any candidate failing a WL is invalid regardless of speed; never trade
correctness below the gate. A failing Triton kernel is never replaced with a Torch/CPU
fallback — it is fixed or abandoned.

---

## 4. Performance hypotheses (falsifiable, checked via per-WL eval timing)

- **H1 (fusion/launch):** collapsing the reference's ~40 kernels into ~3 convs + 1 combine
  yields the largest speedups on small WLs (WL4, WL2) where launch overhead dominates.
  *Test:* c002 vs c001 on WL2/WL4.
- **H2 (grouping/traffic):** grouped convs reduce global-memory round-trips vs per-transform,
  helping all WLs modestly. *Test:* c002 geomean ≥ c001.
- **H3 (TF32 sufficiency):** TF32 (matching cuDNN default) meets tolerances on all WLs incl.
  WL4. *Test:* c001 correctness on WL4; if it fails, trigger FP32 branch.
- **H4 (tiling):** small tiles + many CTAs win on WL4/WL2 (occupancy); large tiles +
  pipelining win on WL5 (throughput); size-bucketed autotune captures both. *Test:* c004 vs
  c003 per-WL.
- **H5 (compute bound WL5):** WL5 (~1.5 TFLOP) is throughput-limited; TF32 `tl.dot` and
  larger tiles are the lever there, not fusion. *Test:* c006 (if triggered) on WL5.
- **H6 (deeper fusion):** eliminating the `H0` round-trip helps only if mid WLs are
  memory-bound. *Test:* c005 vs c004 on WL1/WL3; keep only if it wins.

Any hypothesis that fails its test → abandon that lever; do not carry dead complexity
forward.

---

## 5. Stopping criteria (convergence)

Stop and write `SEARCH_COMPLETE` (with reason) when **any** holds:
- **Convergence:** best passing geomean improves < ~2% across 2 consecutive new candidates,
  and remaining ideas are conditional-only with no supporting evidence (no memory-bound /
  throughput signal from per-WL timings).
- **Diminishing menu:** all triggered levers explored; untried candidates are conditionals
  whose triggers never fired.
- **Budget guard:** approaching evaluation budget (≤ ~5 of 100 left) or token soft limit
  (≈1.0M) — finalize the standing champion.
- **Hard failure of further ideas:** no untried lever with a plausible >few-% upside.

On stop: the standing champion (best passing geomean, all 5 WLs pass) is the submission
candidate; `final` awaits explicit operator approval.

---

## 6. Champion selection rule

- Only candidates that **pass all 5 WLs** are eligible.
- Among eligible, rank by **geometric-mean speedup**; tie-break by worst-WL speedup, then by
  simplicity (prefer fewer kernels / no autotune churn).
- Champion only changes when a new candidate strictly beats it on geomean while passing all
  WLs. Immutability preserves prior champions as fallbacks.

---

## 7. Evidence format (append one JSON object per evaluated candidate)

Appended to `candidates.jsonl` (append-only; never rewrite). Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hex of runs/candidates/c001/solution.py>",
  "timestamp": "2026-09-17T..Z",
  "hypothesis": "Correct per-transform Triton baseline (TF32) passes all WLs.",
  "changes_vs_parent": "root",
  "validation": {
    "signature_ok": true, "padding_offset_ok": true, "bias_relu_order_ok": true,
    "reverse_sign_ok": true, "mask_ok": true, "grouping_ok": true, "dtype_ok": true
  },
  "results": [
    {"uuid": "0e8f2568", "wl": 1, "B": 8,  "T": 1721, "reverse": true,  "passed": true, "speedup": 0.0, "max_atol_obs": 0.0, "match_ratio": 0.0},
    {"uuid": "f274d392", "wl": 2, "B": 4,  "T": 541,  "reverse": false, "passed": true, "speedup": 0.0, "max_atol_obs": 0.0, "match_ratio": 0.0},
    {"uuid": "3c3868a4", "wl": 3, "B": 16, "T": 2048, "reverse": true,  "passed": true, "speedup": 0.0, "max_atol_obs": 0.0, "match_ratio": 0.0},
    {"uuid": "84fb0bc5", "wl": 4, "B": 2,  "T": 293,  "reverse": true,  "passed": true, "speedup": 0.0, "max_atol_obs": 0.0, "match_ratio": 0.0},
    {"uuid": "ea5ae433", "wl": 5, "B": 64, "T": 8192, "reverse": true,  "passed": true, "speedup": 0.0, "max_atol_obs": 0.0, "match_ratio": 0.0}
  ],
  "all_passed": true,
  "geomean_speedup": 0.0,
  "decision": "keep|reject|champion",
  "reason": "...",
  "cumulative_evaluations": 1,
  "skill_usage": "KernelWiki not used (A800/sm_80 out of scope); no external tools."
}
```

Rules: numeric fields filled from the evaluator output (do not invent numbers — the `0.0`
placeholders above are template only; record actual values or `null` if the evaluator does
not report a field). `decision` and `reason` reflect the champion rule (§6). Never modify a
prior line; every evaluated candidate gets exactly one line.

---

## 8. Skill usage note

- `KernelWiki` covers **Blackwell (SM100) / Hopper (SM90)**; this task is **A800 / sm_80
  (Ampere)** — out of scope, so not consulted. Recorded as "not used" in each evidence line.
- No web / subagent / MCP / external agents / profiler / nvidia-smi, per isolation rules.

---

## 8b. Decision log

### c001 — REJECTED (eval #1, all 5 WLs INCORRECT_NUMERICAL)
- **Result:** 0/5 passed. Systematic huge errors on every WL: `max_abs ≈ 4.5e3–6.1e3`,
  `max_rel ≈ 8.6e4–2.5e5`. Uniform scale across WLs ⇒ structural formula error, not an
  off-by-one/boundary bug.
- **Root cause — the draft's central simplification (§1.2 / draft §1.3) is WRONG.** The
  evaluator supplies a **non-trivial random `x_mask`** (workloads mark `x_mask` as
  `type:random`; only `reverse` is overridden). The reference does `x = x * x_mask` at the
  **end of every layer**, which re-masks the `x0` half before it feeds the next transform.
  So the four transforms see **progressively masked, different** `x0` inputs — they are
  **not independent**, and layer order **does** matter. `delta = Σ_i transform_i(raw x0)`
  is therefore invalid. (Also note the evaluator's reported `required_matched_ratio` is
  0.99, not the 0.98 seen in the workload file.)
- **What was correct:** conv implicit-GEMM tap offset (`t+k-2`), bias-before-ReLU,
  no-ReLU-after-conv2, reverse sign, TF32 dot, layout/strides — errors are not consistent
  with those being wrong.
- **Mandate for all future candidates:** implement the **exact sequential reference**. Do
  NOT assume `mask==1` and do NOT collapse the 4 transforms. The independence/grouping
  optimization (old c002) is **dropped**.

### Revised lineage (supersedes §2 roadmap)
```
c002  EXACT sequential reference in Triton (correctness-first)   ← new root
        per layer i in order (0..3 fwd / 3..0 rev):
          x0 = work[:, :96]        # current (already mask-applied) buffer
          h  = relu(conv0_i(x0)); h = relu(conv1_i(h)); h = conv2_i(h)
          h *= mask
          x1 = work[:, 96:] (+/-) h
          work[:, :96] = x0 * mask ; work[:, 96:] = x1 * mask
        output = final work buffer
  └─ c003  fuse epilogues / reduce buffer round-trips (once c002 passes)
       └─ c004  hand-tuned block sizes
            └─ c005  autotune (size-bucketed)
                 └─ c006  deeper conv0->conv1 fusion / WL5 large-tile (conditional)
  └─ cNNN  FP32-accumulate fallback (only if a WL fails TF32 gate)
```
Reuse c001's `_conv1d_kernel` (it is correct); the fix is purely in the host `run`
orchestration: iterate layers in true order over a mutable work buffer, applying the mask
after every layer, instead of summing independent transforms.

### c002 — REJECTED (eval #2, all 5 WLs INCORRECT_NUMERICAL, but structure now correct)
- **Result:** 0/5 passed, yet `max_abs` collapsed from c001's ~5.5e3 to **~17–22** across
  every WL (a ~250× improvement) purely from switching to the exact sequential per-layer
  mask application. This confirms (a) `x_mask` is non-trivial random, and (b) the sequential
  structure/order is right. `max_rel` ~140–360.
- **Root cause — precision, not formula.** The residual error is *uniform* across all five
  WLs (17.1, 17.2, 20.5, 21.0, 21.8) irrespective of `B·T` — so it is a global numeric gap,
  not a size-dependent tiling/boundary bug. The reference conv runs effectively at FP32
  precision for the *values* being compared; the gate's `rtol=1e-5` is ~200× tighter than
  TF32's ~2e-3 relative error, and the random mask amplifies magnitudes over 4 chained
  layers, so large elements exceed `atol=0.011 + rtol·|ref|` → match-ratio < 0.99.
- **Mandate:** the exact-sequential orchestration in c002 is the correct baseline structure.
  The remaining lever is **precision**. Next candidate must use **FP32 accumulate**
  (`ALLOW_TF32=False`, and ensure `tl.dot` operands are true FP32, not TF32-rounded).

### Revised lineage (supersedes above)
```
c003  EXACT sequential reference, FP32 conv (ALLOW_TF32=False)   ← correctness target
  └─ c004  once correct: fuse/reduce buffer round-trips, tune block sizes for speed
       └─ c005  autotune (size-bucketed) / WL5 large-tile
```
Note: full FP32 `tl.dot` on Ampere has no TF32 tensor-core path, so it lowers to FMA and
will be slower than TF32 — acceptable only if required to pass the gate. If FP32 passes but
is too slow, explore mixed strategies (e.g. TF32 with a residual FP32 correction, or
splitting mantissa) to recover speed while staying within tolerance.

### c003 — CHAMPION (eval #3, first VALID: 5/5 pass, geomean 0.503x)
- **Result:** all 5 WLs pass. Precision fix (TF32→FP32, single lever) closed the gate
  exactly as diagnosed: `max_abs` dropped from c002's ~20 to ≤0.018. Per-WL max_abs:
  0.0142/0.0089/0.0156/0.0089/0.0176 (all within atol + rtol·|ref| for ≥99% of elements).
- **Speed (the problem):** geomean **0.503x = a SLOWDOWN vs cuDNN**. Per-WL speedups:
  WL1 0.88x, WL2 0.73x, WL4 0.58x, WL3 0.49x, **WL5 0.17x**. Latencies (sol vs ref ms):
  WL1 10.6/9.4, WL2 2.10/1.53, WL3 25.2/12.4, WL4 1.73/1.01, **WL5 390.8/68.1**.
- **Diagnosis:** WL5 (B=64, T=8192, compute-bound, ~1.5 TFLOP) dominates the geomean drag.
  Full-FP32 `tl.dot` lowers to FMA (no tensor cores) → ~5.7x slower than cuDNN's TF32 conv
  on the big WL. The correctness-mandated FP32 is directly at odds with throughput. Smaller
  WLs are closer to parity (launch/occupancy bound, where our fusion helps).

### Revised roadmap after c003 (champion = c003)
The tension: we need FP32 accuracy (rtol=1e-5) but FP32 `tl.dot` has no tensor-core path on
A800. Two independent lever families, tried one candidate at a time:

```
c003  CHAMPION (FP32 exact-sequential, geomean 0.503x)
  ├─ c004  tiling/launch tuning of the SAME FP32 kernels:
  │         larger BLOCK_T / BLOCK_CO for WL5 throughput, num_warps/num_stages,
  │         fewer launches (fuse combine into conv2 epilogue). Pure-speed, keeps FP32.
  │         Test: does WL5 improve without regressing small WLs? geomean > 0.503x?
  └─ c005  accuracy-preserving tensor-core path: 3xTF32 (split each FP32 operand into
            hi/lo TF32 halves, 3 tl.dot terms) to reclaim TF32 tensor-core throughput
            while matching FP32 to ~rtol 1e-6. Big potential win on WL5. Higher risk;
            only after c004 establishes the tiling baseline. Must re-verify the gate.
```
Guiding: WL5 is the geomean lever (0.17x). c004 first (low-risk tuning of the proven-correct
FP32 kernel); c005 (3xTF32) if c004 can't get WL5 competitive. Keep c003 as fallback champion.

### c004 — ABANDONED (never validly evaluated: autotune too slow → eval killed)
- c004 wrapped the FP32 conv in `@triton.autotune` over 10 configs keyed on
  `(B,T,CIN,COUT)`. With 3 distinct conv signatures × 5 workloads = 15 unique keys, each
  benchmarking up to 10 FP32 configs (some with 256×64 tiles that compile slowly), the
  autotune warm-up ran 20+ minutes and produced no result before the process was torn down.
  `runs/candidates/c004/feedback.log` is empty; no `feedback.json` → **no valid evaluation**.
- Lesson: `triton.autotune` is impractical here (autotune cost dwarfs the tiny per-call work,
  and risks the eval timeout). Use **deterministic host-selected tile sizes** instead.
- c004's autotune source is immutable/snapshotted; not reusing the ID. Superseded by c005.

### c005 — FP32 exact-sequential with deterministic size-adaptive tiling (no autotune)
- Same proven-correct FP32 math as c003 (single passing champion). Single lever: pick
  `(BLOCK_T, BLOCK_CO, BLOCK_K, num_warps, num_stages)` in host code from `B*T`:
  large tiles for WL5 (throughput), small tiles for WL4/WL2 (occupancy). Deterministic →
  fast compile (few constexpr combos), no autotune benchmarking. Target: lift WL5 above
  0.17x without regressing small WLs; must keep 5/5 correctness.
- Note: with `cudnn.allow_tf32=False` the reference conv is also FP32 (WL5 ref≈68ms ≈ FP32
  roofline for ~1.5 TFLOP at ~19.5 TF/s), so my 390ms is ~5.7x off cuDNN's FP32 efficiency —
  a tiling gap, not a tensor-core gap. Bigger tiles should close much of it.

### c005 — REJECTED (eval #4, 5/5 pass but geomean 0.258x — REGRESSION vs c003 0.503x)
- **Result:** all 5 pass (max_abs identical to c003 → pure perf change), but geomean HALVED.
  Per-WL vs c003: WL1 0.88→0.26x, WL3 0.49→0.13x, WL2 0.73→0.46x, WL4 0.58→0.43x, WL5
  0.174→0.173x (393 vs 390ms, unchanged).
- **Falsified hypothesis:** larger tiles do NOT help. Two hard findings:
  1. c003's **small tiling (BLOCK_T=64, BLOCK_CO=32, num_warps=4)** is a sweet spot for the
     mid/small WLs; going to 128/64/warps=8/stages=3 causes ~3× slowdown (FP32 register
     pressure → low occupancy).
  2. **WL5 is insensitive to tile shape** in this range (~390ms both) → it is compute-
     throughput bound at ~3.8 TFLOP/s, ~5× below cuDNN's FP32 efficiency. The bottleneck is
     the many small-K (BLOCK_K=32) `tl.dot` calls, not the tile shape.
- **c003 remains champion (geomean 0.503x).**

### Revised roadmap after c005 (champion still = c003)
```
c003  CHAMPION (FP32, tiles 64/32, warps=4, BLOCK_K=32; geomean 0.503x)
  └─ c006  KEEP c003's tile shape (64/32/warps=4) but raise BLOCK_K 32 -> full CIN
            (single-shot large-K tl.dot instead of CIN/32 small-K dots). Raises FMA
            efficiency on compute-bound WLs (WL5/WL3). Must re-verify 5/5.
       └─ c007  if c006 helps WL5 but WL5 still lags: 3xTF32 tensor-core path
                 (split FP32 operands into hi/lo TF32, 3 dots) to reclaim tensor
                 cores while meeting rtol=1e-5. Higher risk; re-verify gate.
```
Rationale: c005 proved the lever is NOT outer tile shape but contraction efficiency /
tensor-core usage. c006 is the low-risk next step (only BLOCK_K changes); c007 (3xTF32) is
the high-upside fallback if FP32 FMA simply cannot reach cuDNN throughput. Keep c003 as the
standing fallback champion throughout.

## 9. Immediate next actions (next turn, not this one)

1. Implement `solution/solution.py` for **c001** (per-transform Triton implicit-GEMM,
   TF32, fused bias/relu + combine; reverse sign; padding offset).
2. Run the §3 correctness checklist by reasoning.
3. Snapshot to `runs/candidates/c001/solution.py`, record sha256.
4. Evaluate: `./scripts/evaluate_candidate.sh feedback c001`.
5. Append the c001 evidence line to `candidates.jsonl`; set champion if it passes.
6. Proceed down the lineage (c002 …) per §2, one lever per candidate.
