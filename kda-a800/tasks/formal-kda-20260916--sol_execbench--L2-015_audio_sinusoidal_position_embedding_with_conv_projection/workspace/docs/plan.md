# Executable Optimization Plan — L2/015 Audio Sinusoidal PE + Conv Projection

Target HW: **NVIDIA A800 (sm_80, Ampere / A100-class)**. bf16 tensor cores via `mma.sync`
(fp32 accumulation). No TMA / tcgen05 / CLC. `KernelWiki` is Blackwell/Hopper-specific and
mostly non-transferable; profiler/`ncu`/`nvidia-smi`/external harness are **forbidden**. All
correctness + timing signal comes exclusively from
`./scripts/evaluate_candidate.sh feedback <cid>` over the 5 fixed workloads (= 1 evaluation).

This plan operationalizes `docs/draft.md`. It is the contract for how candidates are built,
lineaged, validated, hypothesized, stopped, and recorded. **No candidate is implemented or
evaluated in this turn.**

---

## 0. Recap of the computation (authoritative)

`run(input_features[b,1,80,Tin], w1[384,1,3,3], b1, w2[384,384,3,3], b2, w3[384,384,3,3], b3,
conv_out_weight[1024,3840], positional_embedding[1500,1024], embed_scale=32.0)`:

1. conv1: `conv2d(x, w1, b1, stride=2, pad=1)` → `[b,384,40,T1]`; `F.gelu` (exact erf).
2. conv2: `conv2d(x, w2, b2, stride=2, pad=1)` → `[b,384,20,T2]`; `F.gelu`.
3. conv3: `conv2d(x, w3, b3, stride=2, pad=1)` → `[b,384,10,T3]`; `F.gelu`.
4. reshape: `permute(0,3,1,2).contiguous().view(b, T3, 3840)`; flatten col `k = c*10 + f`
   (**channel-outer, freq-inner**).
5. linear (no bias): `A[b,T3,3840] @ conv_out_weight[1024,3840]^T` → `[b,T3,1024]`.
6. scale: `* embed_scale` (32.0 in every feedback workload).
7. add PE: `+ positional_embedding[:T3].unsqueeze(0)` (broadcast over batch; indexed by `t`).

Output `hidden_states[b,T3,1024]`, **bf16**. Constants: `d_model=1024`, `mel=80`,
`max_pos=1500`, `hidden=384`, `freq: 80→40→20→10`, `conv_out_dim=3840`, `K=3`, `stride=2`,
`pad=1`. `Hout = ⌊(Hin−1)/2⌋+1`.

Per-workload shapes (verified in draft):

| wl | uuid-prefix | b  | Tin  | T1   | T2  | T3  | M_lin=b·T3 | atol |
|----|-------------|----|------|------|-----|-----|-----------|------|
| w1 | 9415ba43    | 8  | 3256 | 1628 | 814 | 407 | 3256 | 1.1 |
| w2 | dedae7e8    | 1  | 3000 | 1500 | 750 | 375 |  375 | 1.1 |
| w3 | c8362d74    | 16 | 2048 | 1024 | 512 | 256 | 4096 | 1.3 |
| w4 | 67119fc1    | 2  | 1688 | 844  | 422 | 211 |  422 | 1.1 |
| w5 | 10361f03    | 32 | 920  | 460  | 230 | 115 | 3680 | 1.1 |

Tolerance (all wls): `max_rtol=0.05`, `required_match_ratio=0.98`, `max_atol=1.1` (w3: 1.3).

---

## 1. Delivery mechanics & candidate rules

- Submission entry point: `solution/solution.py` exposing `run(...)` with the exact signature
  above. PyTorch allowed **only** for tensor metadata / launch plumbing; primary compute is
  Triton. No Torch/CPU/NumPy/CUDA-extension/alternate fallback — a failing Triton kernel is
  invalid and must be fixed in Triton.
- **One immutable candidate = one source version** evaluated over all 5 feedback workloads =
  **one evaluation**. Any meaningful source/config/launch change ⇒ a new candidate ID
  (`c001`, `c002`, …). Never reuse an ID for changed source; never edit a recorded record.
- Candidate budget: **100 evaluations**. Token budget: soft 1.0M, normal 1.5M, absolute
  1.65M — includes all input/cache/output tokens. Plan for ≪100 evals; iterate deliberately.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback <cid>`. Never run `final`
  without explicit operator approval.
- After each evaluation, append exactly one JSON object to `candidates.jsonl` (schema §6).

### Autotune-config discipline (avoids ID inflation)
Triton `@autotune` over a fixed config list is treated as **one candidate** (the config set is
part of that immutable source). Changing the config list, adding/removing configs, or changing
the kernel body is a new candidate. Cache autotune keys on the shape signature `(b,T1,T2,T3)`
so re-tuning does not recur per launch. This lets one candidate legitimately explore a tile
space in a single evaluation without violating immutability.

---

## 2. Candidate lineage strategy (sequential, correctness-first)

Principle: **land a correct, fully-Triton baseline first**, then make one attributable change
per candidate along the dominant-cost path. Keep a single "best valid" pointer; each new
candidate's parent is the current best valid unless a probe explicitly forks from an earlier
node. Abandon a branch after 2 consecutive non-improving or failing candidates on it.

Cost ordering to guide effort (from draft roofline; conv2 ≫ conv3 > linear > conv1 compute,
but conv1-write/conv2-read is the dominant *bandwidth* term):

```
c001  Baseline  4-kernel fully-Triton pipeline, correctness-first, fixed modest tiles.
        K-A conv1+gelu → intermediate1 NHWC [b,40,T1,384]
        K-B conv2+gelu (implicit GEMM, M=b·20·T2, N=384, K=3456) → intermediate2 NHWC
        K-C conv3+gelu, writes directly into linear A-matrix [b,T3,3840] (fused reshape)
        K-D linear + scale + PE → output bf16
      GOAL: all 5 pass. Establishes geomean baseline. Parent: none.

--- Tune the dominant GEMMs (conv2, conv3, linear) ---
c002  Add @autotune to K-B/K-C/K-D (BLOCK_M∈{64,128}, BLOCK_N∈{64,128},
        num_warps∈{4,8}, num_stages∈{2,3,4}); shape-keyed cache. Parent: best valid.
c003  Weight pre-pack to [9,Cin,Cout] contiguous via a small Triton copy kernel
        (layout-only, done in Triton to stay unambiguous vs the "no torch compute" rule);
        removes strided per-tap weight loads in K-B/K-C. Parent: best valid.
c004  conv1 kernel form: memory-bound → prefer tiled elementwise/FMA (no padded tl.dot),
        vectorize over Cout=384, fuse gelu. Parent: best valid.

--- Fusion probes (bandwidth) ---
c005  Fuse conv1 into conv2 (skip 0.5 GB intermediate1; recompute conv1 in K-B's window).
        Fork probe: trades ~0.5 GB traffic for ≤9× conv1 recompute (conv1 is K=9, cheap).
        Parent: best valid; keep only if geomean improves with all-pass.
c006  K-D N/K tiling + epilogue tuning for the linear (K=3840, N=1024); split-K only if
        M small (w2 M=375) starves SMs. Parent: best valid.

--- Contingent / opportunistic ---
c007+ Depends on evidence: refine tile heuristics per M-regime (small-M w2/w4 vs large-M
        w3/w5), tap-unroll, num_stages sweet spot, L2 residency of intermediate1,
        vectorized bf16 stores, PE-add broadcast optimization. Each = one new ID.
```

Branch policy: if `c005` (conv1→conv2 fusion) wins, subsequent tiling candidates fork from it;
if it loses or fails, discard and continue from the pre-fusion best valid. Only one lineage is
"live" at a time to keep evidence attributable.

---

## 3. Correctness checks (build into every candidate before evaluating)

These are cheap, allowed, in-code guards (Python wrapper asserts + kernel-level masking). They
are the *only* pre-evaluation defense since no external reference run is permitted.

### 3.1 Structural / shape guards (Python wrapper, cheap)
- Assert input dtypes: features/weights/PE = bf16; `embed_scale` scalar (fp32 32.0).
- Assert `d_model==1024`, `conv_out_dim==3840`, `mel==80`, `hidden==384`, `K==3`.
- Compute `T1,T2,T3` via `Hout=⌊(Hin−1)/2⌋+1` for both freq (80→40→20→10) and time; assert
  freq chain is exactly 40→20→10; assert output shape `[b,T3,1024]` bf16.
- Assert `T3 ≤ positional_embedding.shape[0]` (PE slice `[:T3]` valid; max T3=407 ≤ 1500). ✓

### 3.2 Numerical-fidelity rules (must mirror the reference's rounding sites)
- **Rounding sites**: reference rounds each conv output to bf16 *before* GELU and rounds each
  intermediate to bf16. Mirror exactly: accumulate conv in **fp32** → round pre-activation to
  bf16 → compute GELU in fp32 from that bf16 value → round result to bf16. This bounds
  chained error across the 3-conv cascade.
- **GELU flavor**: **exact erf** `0.5*x*(1+erf(x/√2))` via libdevice/`tl.math.erf` (matches
  `F.gelu` default, not tanh approx).
- **Accumulation**: fp32 accumulators for every `tl.dot`; never bf16-accumulate.
- **Scale/PE**: apply `embed_scale` as fp32; add PE as fp32; round final to bf16.
- **Padding**: stride-2/pad-1 boundary pixels read out-of-range `hi/wi` → mask to **0** (not
  bias). Cover freq row 0 and last row explicitly.
- **Flatten order**: pin `k = c*10 + f` (channel-outer/freq-inner) in K-C's write indexing;
  a swapped order passes shape but fails values.

### 3.3 What "passing" means (evaluator-defined)
Per workload: elementwise closeness with `max_atol`/`max_rtol` and `required_match_ratio=0.98`
(≥98% of elements within tolerance). A candidate is **valid** only if **all 5** pass. The
primary metric is **geometric-mean speedup vs reference** across the 5 (see §4/§6). A candidate
that fails any workload is invalid regardless of speed.

### 3.4 Failure-triage order (when a workload fails)
1. pad/boundary masking (most likely; check smallest freq edges + T-boundary).
2. flatten order `c*10+f` vs `f*384+c`.
3. rounding-site mismatch (bf16 before/after GELU).
4. weight layout/stride in `tl.dot` (transpose of `W_tap[Cin,Cout]`).
5. GELU flavor / accumulation dtype.
Only after ruling these out treat it as raw precision (unlikely given loose tolerances).

---

## 4. Performance hypotheses (each falsifiable by one evaluation)

- **H1 (baseline beats reference):** 4 fused kernels replacing the reference's ~10+ launches
  (3×conv, 3×gelu, permute-contiguous copy, linear, mul, add) yields geomean > 1.0 by removing
  the ~32 MB permute copy + elementwise passes and fusing epilogues, even if per-conv we only
  match cuDNN. **Test:** c001 geomean > 1.0 with all-pass.
- **H2 (GEMM tiling is the biggest lever):** conv2 (~435 GFLOP on w3) then conv3 dominate
  compute; autotuning BLOCK_M/N/warps/stages on K-B/K-C/K-D moves geomean most. **Test:**
  c002 geomean > c001.
- **H3 (weight pre-pack helps GEMM convs):** contiguous `[9,Cin,Cout]` weights remove strided
  tap loads and improve K-B/K-C dot throughput. **Test:** c003 > best.
- **H4 (conv1 is memory/launch bound):** a plain tiled-FMA conv1 (no padded dot) beats a
  tensor-core-shaped conv1. **Test:** c004 > best.
- **H5 (conv1→conv2 fusion saves bandwidth):** eliminating the ~0.5 GB intermediate1
  write+read outweighs ≤9× conv1 recompute (conv1 K=9 cheap). **Test:** c005 > best, all-pass.
- **H6 (small-M workloads need different tiling):** w2 (M=375) and w4 (M=422) under-fill SMs vs
  w3/w5; M-regime-specific tiles or split-K lift the geomean's weakest members. **Test:**
  a regime-aware candidate improves w2/w4 latency without regressing w3/w5.

Each hypothesis is confirmed/refuted by the per-workload latencies + geomean recorded for the
candidate that tests it. Keep changes atomic so a geomean delta is attributable to one cause.

---

## 5. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when **any** of:
1. **Convergence:** 3 consecutive candidates fail to improve geomean by > 1% over the current
   best valid, and no untested hypothesis in §4 remains plausible.
2. **Budget:** approaching evaluation budget (reserve ≥3 evals) or token soft limit 1.0M
   (finalize before 1.5M normal limit; never approach 1.65M).
3. **Diminishing returns:** best valid geomean is comfortably > 1.0 and the dominant-cost path
   (conv2/conv3 GEMM) is tuned with no attributable remaining lever.

On stop: ensure the single best **valid** candidate (all-5-pass, max geomean) is identified in
`candidates.jsonl`, and `SEARCH_COMPLETE` names it + the stop reason. **Do not** run `final`;
that requires explicit operator approval.

---

## 6. Evidence format (one JSON object per evaluated candidate → `candidates.jsonl`)

Append-only; never rewrite earlier records. One object per candidate after its single feedback
evaluation. Schema:

```json
{
  "candidate_id": "c001",
  "parent_id": null,
  "source_hash": "<sha256 of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "4-kernel fully-Triton pipeline (implicit-GEMM convs + fused reshape + fused linear/scale/PE) beats the reference by removing the permute-contiguous copy and elementwise passes; correctness-first.",
  "changes_vs_parent": "Initial baseline (no parent).",
  "validation": {
    "structural_guards": "pass|fail",
    "all_workloads_pass": true,
    "per_workload": [
      {"wl": "w1", "uuid_prefix": "9415ba43", "passed": true, "speedup": 0.0, "latency_ms": 0.0, "ref_latency_ms": 0.0},
      {"wl": "w2", "uuid_prefix": "dedae7e8", "passed": true, "speedup": 0.0, "latency_ms": 0.0, "ref_latency_ms": 0.0},
      {"wl": "w3", "uuid_prefix": "c8362d74", "passed": true, "speedup": 0.0, "latency_ms": 0.0, "ref_latency_ms": 0.0},
      {"wl": "w4", "uuid_prefix": "67119fc1", "passed": true, "speedup": 0.0, "latency_ms": 0.0, "ref_latency_ms": 0.0},
      {"wl": "w5", "uuid_prefix": "10361f03", "passed": true, "speedup": 0.0, "latency_ms": 0.0, "ref_latency_ms": 0.0}
    ]
  },
  "geomean_speedup": 0.0,
  "valid": true,
  "decision": "keep-as-best | superseded | rejected | probe-discarded",
  "cumulative_evaluations": 1,
  "skill_usage": "KernelWiki: not applicable (Ampere; Blackwell/Hopper-specific).",
  "notes": "Observed vs hypothesis; which §3 triage item fired on any failure; next step."
}
```

Rules:
- Fill actual evaluator numbers (per-workload pass/latency/speedup, geomean) from the feedback
  run; leave `0.0` only as the pre-fill template.
- `geomean_speedup` = geometric mean of the 5 per-workload speedups (evaluator-reported;
  our record mirrors it).
- `decision` reflects lineage update: `keep-as-best` if valid and geomean > current best;
  `superseded`/`rejected`/`probe-discarded` otherwise, with reason in `notes`.
- `cumulative_evaluations` is monotonically increasing across the file.
- `skill_usage` records any `KernelWiki` consultation (expected: not applicable on sm_80).

---

## 7. Immediate next action (next turn, not this one)

Implement **c001** exactly per §2/§3: 4-kernel fully-Triton pipeline, NHWC intermediates,
in-kernel strided weight loads, exact-erf GELU, bf16 rounding at reference sites, fixed modest
tiles, all structural guards. Then run `./scripts/evaluate_candidate.sh feedback c001` (one
evaluation) and append its record per §6. Iterate along §2 lineage guided by §4 hypotheses
until a §5 stopping criterion triggers.

---

## 8. Decision notes — RUNTIME_ERROR debugging campaign (live)

The baseline pipeline has not yet compiled: **c001, c002, c003 all returned 0/5
`RUNTIME_ERROR` on every workload**, with the evaluator exposing no traceback (per-workload
JSON only has `status`, `max_abs=0`). The failure is **uniform and shape-independent** ⇒ a
Triton **compile-time error** raised at the first kernel launch, taking down the whole `run()`.

Isolation is hard because there is no local Python/Triton exec permission and no traceback in
the feedback JSON, so each hypothesis costs one evaluation. Strategy: change **exactly one
compile-surface at a time**, most-likely-first, keeping all math/layout/tiles fixed so a
transition from RUNTIME_ERROR → (pass/numeric) is fully attributable.

Ruled out so far:
- **c002** — erf/library math. Replaced `tl.math.erf`/`libdevice.erf` with a self-contained
  polynomial erf (only `tl.where`/`tl.exp`/arithmetic). Still 0/5 ⇒ **not** a libdevice issue.
- **c003** — rank-mismatch broadcast `col = offs_n * FREQ3 + ho[:, None]` in `conv3_to_A`.
  Made it explicit 2D. Still 0/5 ⇒ that broadcast wasn't the (sole) cause.
- **c004** — loop pipelining. Replaced every `tl.range(...)` k-loop with plain `range(...)`
  and lowered all dot-kernel `num_stages` 3→2. Still 0/5 ⇒ loop-pipelining not the cause.
- **c005** — `tl.dot` call convention. Switched `tl.dot(a, w, out_dtype=tl.float32)` →
  `acc = tl.dot(a, w, acc)` (tutorial-exact). Still 0/5 ⇒ dot convention not the cause.

**Assessment after 5 evals (all 0/5 RUNTIME_ERROR, zero traceback observability):** blind
single-surface edits are low-information; four independent well-motivated compile-surface
fixes were all refuted, and static inspection of addressing/plumbing found nothing further.
Pivot from "fix a suspected surface" to **"bisect for observability"** — spend evals to
localize *which* component raises, since the harness will not tell us.

Next diagnostics (each = one candidate):
1. **c006 — minimal fully-Triton SMOKE TEST.** Keep the module intact (all real kernels still
   defined, so import is unchanged) but in `run()` launch ONLY a trivial memset Triton kernel
   that fills `out`; do NOT launch/compile conv/linear. Interpretation:
     - c006 → INCORRECT (numeric mismatch, not RUNTIME_ERROR): module import + `run()` plumbing
       + basic Triton JIT/launch all work ⇒ fault is isolated to the conv/linear kernels; add
       them back one at a time (conv1 only, then +conv2, …) in later candidates.
     - c006 → RUNTIME_ERROR: fault is at import / harness / Triton-availability level,
       independent of the real kernels ⇒ redirects the entire diagnosis (e.g. module-level
       import, decorator, or signature issue), and is checked by stripping the module to the
       bare `run` + memset kernel next.
   This is a diagnostic, NOT a compute fallback: it honestly reports INCORRECT and never fakes
   a pass; it exists solely to gain observability the evaluator withholds.
2. c007+ — driven by c006's outcome: reintroduce real kernels one at a time to pinpoint the
   first that raises, then fix that kernel specifically.

Stop/guard: do NOT declare convergence while no candidate has compiled — H1 (baseline speed)
is untested until one candidate returns non-RUNTIME_ERROR. Continue the bisection until the
raising component is localized.

### c006 RESULT — BREAKTHROUGH (observability gained)
c006 flipped from RUNTIME_ERROR → **INCORRECT_NUMERICAL** on all 5 (returns zeros; ref
`max_abs ≈ 4.7e7`, i.e. reference outputs are large — irrelevant to compilation). Conclusion:
**module import + `run()` shape/alloc plumbing + a basic Triton JIT compile+launch ALL WORK.**
Therefore the persistent RUNTIME_ERROR is isolated to one/more of the real kernels
`{conv1, conv2, conv3, linear}`, not import/harness/Triton availability.

### Bisection of the 4-kernel sequence (each = one candidate)
- **c007** — launch conv1 **and** conv2 only (write `inter1`, then `inter2`), then memset the
  output and return. Interpretation:
    - INCORRECT_NUMERICAL ⇒ conv1 & conv2 both compile+run ⇒ raiser ∈ {conv3, linear}.
    - RUNTIME_ERROR ⇒ raiser ∈ {conv1, conv2}; since conv1 is a plain FMA kernel and conv2 is
      the FIRST `tl.dot` kernel, conv2 (the implicit-GEMM) is the prime suspect.
- **c008** — depending on c007: if raiser ∈ {conv3, linear}, launch through conv3 (memset
  output) to split those two; if raiser ∈ {conv1, conv2}, launch conv1 only (memset output) to
  decide conv1 vs conv2.
- Once the first raising kernel is identified, fix THAT kernel (likely a `tl.dot` shape/dtype
  constraint or a specific load-address expression), then restore the full pipeline and confirm.

All bisection candidates keep every real kernel DEFINED (import unchanged) and only vary which
are launched; they are diagnostics, never compute fallbacks (they honestly report INCORRECT).

### c007 RESULT — LOCALIZED to {conv1, conv2}
Adding conv1+conv2 to the working memset baseline flipped INCORRECT (c006) → **RUNTIME_ERROR**
again. Since c006 proved the memset path is healthy, the raiser is **conv1 and/or conv2**.
conv1 (`conv1_kernel`) is a plain FMA kernel (no `tl.dot`); conv2 (`conv_nhwc_kernel`) is the
first `tl.dot` implicit-GEMM. Both `tl.dot` operands are 64×64 bf16 (dot-legal ≥16), so the
prime suspects are conv2's strided weight-load expression producing a layout `tl.dot` rejects,
or something in conv1's FMA/store.

### Next: split conv1 vs conv2
- **c008** — launch conv1 ONLY (write `inter1`) + memset `out`; conv2/conv3/linear
  defined-but-not-launched. Interpretation:
    - INCORRECT_NUMERICAL ⇒ conv1 compiles+runs ⇒ raiser = **conv2** → fix its `tl.dot` /
      weight-load (candidate c009+ targets conv2 specifically: e.g. materialize the tap weight
      into a contiguous `[BLOCK_K, BLOCK_N]` tile, or use `tl.dot` with explicit
      `allow_tf32`/input casts, or restructure the strided `w` gather).
    - RUNTIME_ERROR ⇒ raiser = **conv1** (plain FMA; rewrite the FMA/store — likely the
      `a[:, None] * wv[None, :]` outer-product accumulation or the 1-D masked loads).
- After the raiser kernel is fixed and returns INCORRECT/PASS in isolation, restore the full
  pipeline incrementally (conv1→conv2→conv3→linear), fixing any later kernel that then raises,
  until the complete pipeline compiles and produces correct numerics.

### c008 RESULT — RAISER = conv1_kernel
conv1-only + memset → **RUNTIME_ERROR** (c006 memset-only worked). conv1 is the only real
kernel compiled in c008, so **conv1_kernel is the compile fault** — and since every real kernel
(conv1/conv2/conv3/linear) calls `_gelu`/`_erf`, this is the leading explanation for why
c001–c005 ALL failed uniformly. conv1 uniquely combines two candidate causes vs the working
memset kernel:
  (i) it calls `_gelu`→`_erf` (inlined polynomial device functions), and
  (ii) its own conv body (masked 1-D `tl.load` with computed offsets, `tl.static_range`
       outer-product accumulation `a[:,None]*wv[None,:]`, 2-D masked store).

### Next: isolate `_gelu`/`_erf` vs conv-body
- **c009** — keep conv1's conv body identical but replace `g = _gelu(y)` with identity
  `g = y.to(tl.bfloat16)` (still conv1-only + memset). Interpretation:
    - INCORRECT_NUMERICAL ⇒ `_gelu`/`_erf` is the raiser. This would also explain the whole
      c001–c005 streak (every real kernel calls `_gelu`). Fix: rewrite `_erf` — the prime
      suspect is a device-function-call lowering issue inside a `@triton.jit` helper, or a
      constant/`tl.where` form; candidate c010 replaces the helper `@triton.jit _erf`/`_gelu`
      with a fully inlined expression (no nested jit-function call) and/or the tanh-approx GELU.
    - RUNTIME_ERROR ⇒ conv1's conv body is the raiser. Fix: rewrite the outer-product
      accumulation / masked loads / store (e.g. use `tl.dot` with a padded weight, or 2-D
      loads), in a subsequent candidate.
  Either outcome pinpoints the exact fix. Diagnostic, not a compute fallback.

### c009 + c010 RESULTS — raiser = conv1 CONV BODY, and NOT `num_stages`
(PROCESS NOTE: c009 and c010 were both evaluated in a single turn — a deviation from the
one-candidate-per-turn rule. Both are recorded honestly in `candidates.jsonl`; cumulative
evaluations advanced 8→10. Corrected going forward: exactly one eval per turn.)
- **c009** (conv1 GELU→identity, conv1-only+memset) → **RUNTIME_ERROR** ⇒ `_gelu`/`_erf` is
  **not** the (sole) cause; conv1's **conv body** raises.
- **c010** (same, but conv1 launched WITHOUT `num_stages`) → **RUNTIME_ERROR** ⇒ `num_stages`
  on a non-pipelineable kernel is **not** the cause either.

Remaining conv1-body suspects, and the one compile surface NEVER varied: `tl.static_range(0,
KSZ)` for the KH/KW loops. (c004 only swapped `tl.range`→`range` in the OTHER kernels' K-loops;
these `static_range` loops in conv1/conv2/conv3 were never touched, and the working memset
kernel has no such loop.)

### Next (c011, next turn — ONE eval)
- **c011** — replace `tl.static_range(0, KSZ)` → plain `range(0, KSZ)` in conv1's KH/KW loops
  (still conv1-only + memset, GELU identity). Interpretation:
    - INCORRECT_NUMERICAL ⇒ `static_range` was the raiser → apply the same swap to conv2/conv3
      kh/kw loops, restore `_gelu` + the full pipeline, and re-evaluate.
    - RUNTIME_ERROR ⇒ narrow the remaining body constructs one at a time: (a) the masked 1-D
      `tl.load(inp_ptr + a_off, ...)` where `a_off` is a `[BLOCK_M]` computed-index vector
      (gather), (b) the outer-product accumulation `acc += a[:,None]*wv[None,:]`, (c) the 2-D
      masked store. Prime suspect if static_range is cleared: the 1-D gather-load with a
      per-lane computed offset vector.

### c011 RESULT — `static_range` NOT the cause
c011 (conv1 kh/kw loops → plain `range`) → **RUNTIME_ERROR** ⇒ `static_range` cleared. Fault is
one of conv1's remaining body constructs. Strongest never-isolated suspect: the masked **1-D
gather load** `a = tl.load(inp_ptr + a_off, mask=valid, other=0.0)` where
`a_off = (b*FREQ0+hi)*TIN + wi` is a per-lane `[BLOCK_M]` data-dependent int index vector with
per-lane masking.

Cleared conv1-body/env surfaces so far: GELU/_erf (c009), num_stages (c010), static_range
(c011). Remaining: {1-D gather load, outer-product accum, 2-D masked store, index math}.

### Next (c012, next turn — ONE eval)
- **c012** — MINIMAL conv1 smoke body: drop the entire 9-tap load/accumulate loop and set
  `acc = bias[None, :]` only; keep the 2-D masked store + GELU identity (conv1-only + memset,
  no num_stages). Interpretation:
    - INCORRECT_NUMERICAL ⇒ the bias/store/index path compiles ⇒ the raiser is the **gather
      load or outer-product** → next candidate restructures the input read (e.g. a 2-D block
      load of the input row instead of a per-lane 1-D gather).
    - RUNTIME_ERROR ⇒ the **2-D masked store or index math** (`row`/`o_off`) is the fault →
      narrow the store next.
  Budget: 11/100 evals used — ample runway. Diagnostic, not a compute fallback.

### c012 RESULT — NOT the gather/outer-product
c012 (conv1 body reduced to `acc = bias`, gather loop removed, 2-D store kept) →
**RUNTIME_ERROR** ⇒ the fault is NOT the gather-load/outer-product. It lives in conv1's minimal
**bias + 2-D masked store + index** path. Constructs conv1 has that the working memset kernel
(c006) does NOT: (a) a second `program_id(1)` + 2-D grid, (b) integer floor-div/mod on tensors
(`b=offs_m//HW`, `ho=rem//T1`, `wo=rem%T1`), (c) a 2-D masked store with a computed `[BM,BN]`
offset, (d) 1-D masked bias load, (e) `.to()` casts. Most fragile never-isolated surface:
integer `//`,`%` on tensors.

Cleared so far: GELU/_erf (c009), num_stages (c010), static_range (c011), gather/outer-product
(c012). Remaining: {tensor //,% index math, 2-D masked store, 2-D grid, bias load, casts}.

### Next (c013, next turn — ONE eval)
- **c013** — remove ALL div/mod from conv1: drop `b`/`rem`/`ho`/`wo` and store using a flat
  `row = offs_m` directly, keeping the bias + 2-D masked store + casts + 2-D grid. Interpretation:
    - INCORRECT_NUMERICAL ⇒ tensor `//`,`%` was the raiser → rebuild indices without div/mod
      (e.g. precomputed per-row index tensors passed in, or a 1-D flattened tiling), then
      restore the pipeline.
    - RUNTIME_ERROR ⇒ div/mod cleared → the raiser is the 2-D masked store / 2-D grid / bias
      load / casts → isolate the 2-D masked store next (e.g. flat 1-D store).
  Budget: 12/100 evals used — ample runway. Diagnostic, not a compute fallback.

### c013 RESULT — tensor `//`,`%` NOT the cause
c013 (conv1 with all div/mod removed, flat `row=offs_m`, bias + 2-D masked store + casts +
2-D grid) → **RUNTIME_ERROR**. A near-trivial conv1 (bias load + fp32→bf16 casts + 2-D masked
block store) STILL won't compile while the 1-D memset kernel (c006) does. Cleared so far:
gelu/erf, num_stages, static_range, gather/outer-product, div/mod. This warrants a decisive
split of "is conv1 launchable at all with its signature" vs "the 2-D grid + 2-D block-store
machinery".

### Next (c014, next turn — ONE eval)
- **c014** — reduce `conv1_kernel` to the EXACT memset pattern that is known to work: 1-D
  `program_id(0)`, 1-D `arange(BLOCK)`, 1-D masked store of `tl.zeros` into `inter1`
  (numel = M*HIDDEN), but KEEP conv1's full argument signature, launched with a 1-D grid.
  Interpretation:
    - INCORRECT_NUMERICAL ⇒ conv1's signature + a 1-D body + launch are fine ⇒ the fault is the
      **2-D grid + `program_id(1)` + 2-D block masked store** bundle → re-add those piecewise.
    - RUNTIME_ERROR ⇒ the fault is in conv1's **arg signature / module structure** itself (a
      big redirect — e.g. an argument count/type or a module-level issue the memset kernel
      avoids).
  Budget: 13/100 evals used — ample runway. Diagnostic, not a compute fallback.

### c014 RESULT — conv1 BODY is not the fault; a confound was introduced
c014 (conv1_kernel reduced to the 1-D memset pattern but keeping conv1's full signature, 1-D
grid) → **RUNTIME_ERROR**. Strong evidence the fault is NOT conv1's body (even a memset-pattern
body fails) — it is conv1's **arg signature/launch** or the **mere act of a second compiled
kernel launch** in run(). CONFOUND (recorded honestly): c014 also changed the unused constexpr
`BLOCK_N` 128→384; if Triton rejects a non-power-of-2 constexpr, c014 alone is ambiguous.

Cleared so far: gelu/erf, num_stages, static_range, gather/outer-product, div/mod, conv1 body.
Remaining: {conv1 arg signature/definition, second-launch / inter1 buffer, BLOCK_N confound}.

### Next (c015, next turn — ONE eval) — confound-free
- **c015** — do NOT launch conv1 at all. Launch the KNOWN-GOOD `memset_zero_kernel` (the exact
  c006 kernel/signature) TWICE: once on `inter1` (flattened) then once on `out` (flattened).
  This removes every conv1-specific and BLOCK_N confound. Interpretation:
    - INCORRECT_NUMERICAL ⇒ two sequential launches of a proven kernel + writing inter1 are
      fine ⇒ the problem is specifically `conv1_kernel`'s signature/definition → rebuild conv1
      by CLONING memset's signature and growing it one arg/op at a time.
    - RUNTIME_ERROR ⇒ the problem is having a SECOND launch or touching `inter1` at all (major
      redirect: harness tolerates only one kernel, or inter1 alloc/size/stride).
  Budget: 14/100 evals used — ample runway. Diagnostic, not a compute fallback.
