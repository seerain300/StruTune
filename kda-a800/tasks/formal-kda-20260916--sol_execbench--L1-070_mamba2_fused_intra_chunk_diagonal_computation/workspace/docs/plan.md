# Plan — L1/070 Mamba2 Fused Intra-Chunk Diagonal Computation

Executable optimization plan derived from `docs/draft.md`, `task/definition.json`, and the
five fixed feedback workloads. Target: **NVIDIA A800 (`sm_80`, Ampere)**, Triton-only compute.
This document is the contract for how candidates are built, checked, evaluated, and retired.
No candidate is implemented or evaluated in this turn.

---

## 0. Operating constraints (recap, binding)

- **Triton-only compute.** PyTorch only for shape/stride/dtype/empty-alloc/launch. No
  Torch/CPU/NumPy/CUDA-extension/alternate computational fallback. A failing Triton kernel is
  **invalid** — it is never patched with a non-Triton path.
- **Submission entry point:** `solution/solution.py` exposing
  `run(hidden_states, A_cumsum, B, C) -> Y_diag`.
- **Immutable, sequential candidates** `c001, c002, …`. One kernel source version over all 5
  feedback workloads = **one** evaluation. Any meaningful source/config/launch change ⇒ **new**
  candidate ID; never reuse an ID for changed source; never rewrite an earlier `candidates.jsonl`
  record.
- **Only sanctioned signal:** `./scripts/evaluate_candidate.sh feedback cNNN`. No direct
  CUDA / profiler / `nvidia-smi` / external evaluator / alternate correctness harness.
- **Budgets:** ≤100 candidate evaluations; token soft 1.0M / normal 1.5M / absolute 1.65M.
- **Metric:** geometric-mean speedup across the 5 feedback workloads; **every** selected
  workload must pass correctness (a single fail ⇒ candidate invalid regardless of speed).
- **Skills:** `KernelWiki`/profiling skills are Blackwell/Hopper/B200-specific and out of scope
  for `sm_80`; additionally blocked by the "no profiler" rule. Not invoked. Record skill usage
  as "none" in each record.
- **Final:** `./scripts/evaluate_candidate.sh final <id>` only on explicit operator approval.

---

## 1. Reference math (frozen spec the kernel must reproduce)

Constants: `CHUNK_SIZE(s)=128`, `NUM_HEADS(h)=32`, `HEAD_DIM(d)=128`, `N_GROUPS(g)=8`,
`STATE_SIZE(n)=128`, `heads_per_group=4`, group `g = h//4`. Variable axes: `batch B`,
`num_chunks C`. All inputs/outputs `bfloat16`; reference arithmetic is **fp32**.

Per independent `(b, c, h)` tile, with `i` = query row, `j` = key/value row, both `∈[0,128)`:

- `Cmat = C[b,c,:,g,:]` `[i,n]`; `Bmat = B[b,c,:,g,:]` `[j,n]`; `a = A_cumsum[b,h,c,:]` `[128]`;
  `X = hidden_states[b,c,:,h,:]` `[j,d]`.
- **Decay mask (segment_sum, re-cumsum of the input):** with prefix `cA[i]=Σ_{k≤i} a[k]`,
  `L[i,j] = exp(cA[i]-cA[j])` for `i≥j`, else `0` (lower-triangular incl. diagonal; `i==j → 1`).
- **Scores:** `G = Cmat @ Bmatᵀ` `[i,j]` — head-independent, **shared by the 4 heads of a group**.
- **Masked weights:** `M = G ∘ L` (fp32).
- **Output:** `Y = M @ X` `[i,d]`, stored bf16 into `Y_diag[b,c,:,h,:]` (same layout as `X`).

Two `128×128×128` matmuls per output tile; `B·C·32` tiles total.

### 1.1 Feedback workloads (fixed)
| uuid (prefix) | B | C | B·C | head-tiles (B·C·32) | group-tiles (B·C·8) |
|---|---|---|---|---|---|
| 314bcaf9 | 2 | 4 | 8  | 256 | 64  |
| 019db6f8 | 1 | 3 | 3  | 96  | 24  |
| 6c45c19e | 4 | 4 | 16 | 512 | 128 |
| 46445623 | 4 | 7 | 28 | 896 | 224 |
| cccd1cb3 | 1 | 4 | 4  | 128 | 32  |

Tolerance (all five): `max_atol=1e-5`, `max_rtol=0.05`. Smallest workload (24 group-tiles /
96 head-tiles) is the occupancy-risk case for any grid-shrinking optimization; largest (896
head-tiles ≈ 2–8 waves on 108 SMs) is the throughput case.

---

## 2. Candidate lineage strategy

**One variable per candidate.** Each ID changes exactly one axis vs. its parent so any
correctness or speed delta is attributable. Lineage is a **greedy chain**: the accepted best so
far is the parent of the next probe; rejected probes do not become parents. Config/tile/precision
knobs baked into the kernel source are treated as source changes ⇒ new IDs (no autotuner that
would make "one immutable kernel" ambiguous across workloads).

Planned spine (each step gated on the previous passing correctness; exact later IDs depend on
outcomes — this is the priority order, not a promise of count):

- **c001 — correctness anchor (per-`(b,c,h)`).** Grid `B·C·32`; faithful mapping of §1. fp32
  accumulators, exp-of-difference, no max-subtraction, bf16 cast only at store + matmul-2
  operands. `num_warps=8`, `num_stages=2–3`. Goal: **pass all 5**, establish baseline geomean.
  Nothing else matters until this is green.
- **c002 — per-group `G` fusion.** One program per `(b,c,g)` computes `G=Cmat@Bmatᵀ` once, loops
  the 4 heads: `M_h=G∘L_h`, `Y_h=M_h@X_h`. Cuts matmul-1 and B/C loads 4×; grid → `B·C·8`.
  Hypothesis: net win from reduced compute/traffic; **watch occupancy on (1,3)=24 tiles**.
- **c003 — matmul-2 precision knob.** If c001/c002 pass comfortably, keep bf16 tensor-core
  (fast). Only if any workload is marginal/fails, branch a candidate with fp32/TF32 matmul-2 as
  a correctness-preserving fallback. (Precision is a correctness lever first, perf second.)
- **c004 — KV tiling + causal skip (no renorm).** `BLOCK_M`∈{64,128}, kv loop in blocks with
  strict-upper-triangular block skip; `i≥j` mask only on the diagonal block. Lowers register
  pressure and skips ~half the score work. Compare vs full-128.
- **c005 — warp/stage/block sweep.** From the best of c002/c004, probe `num_warps∈{4,8}`,
  `num_stages∈{2,3,4}`, block sizes — each concrete config is its own ID. Pick the config that
  is robust across **all 5** (must be one immutable kernel).
- **c006 — load-layout / block-pointer tuning.** `make_block_ptr` / precomputed strides; keep
  the contiguous 128-wide inner dim innermost (`X`/`Y` row stride `32*128=4096`, contiguous in
  `d`; `B`/`C` row stride `8*128=1024`, contiguous in `n`; `A_cumsum` stride 1 in `s`). Probe
  vector-load coalescing and `X`/`Y` traffic (the memory-bound bottleneck).
- **c007+ — epilogue/occupancy refinement.** Combine the winners (group fusion × best
  tiling × best config), ensure no `[128,128]` intermediate touches HBM, and squeeze remaining
  occupancy/register wins. Stop per §5.

Deviations allowed: if c001 fails, the immediate next ID is a **minimal correctness fix** (e.g.
`A_cumsum` axis-order bug, exp Inf/NaN handling, precision bump) before any perf work — the
spine resumes only once an anchor passes.

---

## 3. Per-candidate execution procedure (executable checklist)

For each candidate `cNNN`:

1. **Pre-flight static audit** (before writing/finalizing source) — the §4 correctness checklist.
2. **Write source** to `solution/solution.py` (single immutable version for this ID). Record the
   source hash (e.g. `sha256sum solution/solution.py`) into the record.
3. **Evaluate once:** `./scripts/evaluate_candidate.sh feedback cNNN`. This runs all 5 feedback
   workloads = one evaluation; increment cumulative eval count.
4. **Parse result:** per-workload pass/fail + timing/speedup, geomean speedup.
5. **Decide** (§6): accept (new parent) / reject / fix. Never mutate `cNNN`'s source after eval;
   any change ⇒ `c(N+1)`.
6. **Append** exactly one JSON object to `candidates.jsonl` (append-only, §7).
7. **Loop** to the next planned probe, or stop per §5.

Rule: **never** run more than one distinct source under the same ID; **never** edit an earlier
record; **never** run `final` without operator approval.

---

## 4. Correctness checks (analysis-first, evaluator-confirmed)

Direct execution/profiling is forbidden, so correctness is enforced by static audit + the
feedback evaluator. Before each candidate, re-verify:

### 4.1 Index & layout audit
- Group map `g = h//4` for B/C; **but** `A_cumsum` is `[b, h, c, s]` — note the `h`/`c` axes are
  **swapped vs. `X`/`Y`** `[b, c, s, h, d]`. This axis-order is the top-priority pitfall.
- `Cmat`/`Bmat` indexed **per group** `g`; `a` and `X` indexed **per head** `h`.
- `L` lower-triangular **incl. diagonal**: `i==j → exp(0)=1`; `i<j → 0`.
- `G = Cmat @ Bmatᵀ`: `i` indexes `C`'s row, `j` indexes `B`'s row, reduce over `n`.
- `Y = M @ X`: reduce over `j`; output `[i,d]`.
- Output tensor exact shape `[B, C, 128, 32, 128]`, dtype `bfloat16`, **same strides/layout as
  `hidden_states`**.
- All dims are multiples of 128 ⇒ no masked/ragged load or store on the 128-lanes.

### 4.2 Numeric-faithfulness checklist
- fp32 accumulation for both matmuls and for `L`/`M`.
- **exp-of-difference** `exp(cA[i]-cA[j])` — never factored `exp(cA[i])·exp(-cA[j])`
  (overflow/underflow even when the difference is bounded).
- **No** softmax/max-subtraction (there is no normalization; subtraction would not cancel).
- The intra-chunk cumsum re-applies `cumsum` to the **input** `A_cumsum` (do not treat the input
  as already being the prefix `cA`).
- bf16 cast only at final store and at matmul-2 operands (`M→bf16`, `X→bf16`, fp32 accum);
  bf16 relative error ≈2⁻⁸≈0.4% ≪ `rtol=0.05`.
- Inf/NaN watch: random `a` gives `cA[i]-cA[j]` up to ±(30–60); `exp` stays finite in fp32
  (max ≈3.4e38) but large. If a workload fails with NaN/Inf, keep exp-arg ordering
  bit-faithful to the reference and consider clamping only if the reference also produces the
  same Inf (must match, not diverge).

### 4.3 Pass criteria (from the evaluator)
- Correctness: `|out-ref| ≤ atol + rtol·|ref|` per element on every one of the 5 workloads.
- A candidate is **valid** only if **all 5** pass. Any single fail ⇒ invalid, cannot be selected
  regardless of speed.

---

## 5. Stopping criteria

Stop and prepare `SEARCH_COMPLETE` when any holds:
1. **Convergence:** best geomean speedup improves <~2% across 2–3 consecutive accepted
   candidates (plateau).
2. **Optimization space exhausted:** the §2 spine (group fusion, tiling, config, layout,
   epilogue) is explored and no untried lever has a credible hypothesis for >~2% gain.
3. **Budget:** cumulative evaluations approach 100, or token usage approaches the 1.0M soft /
   1.5M normal limit (leave margin; do not cross 1.65M absolute).
4. **Regression safety:** the current best is a valid, all-pass candidate — never end on an
   invalid best.

On stop, write `SEARCH_COMPLETE` naming the best candidate ID, its geomean, and the stop reason.
Do **not** run `final` — that is operator-gated.

---

## 6. Decision rule per candidate

- **Accept & promote to parent** iff: all 5 workloads pass **and** geomean speedup > current best
  geomean (strictly; ties broken toward simpler/robuster source).
- **Reject (keep parent)** iff: any workload fails, or geomean ≤ current best. Record why.
- **Fix branch** iff: a correctness failure with an identified minimal cause — the next ID is the
  targeted fix, not a perf probe.
- The **anchor exception:** c001 has no "current best"; it is accepted iff all 5 pass, and it
  seeds the baseline geomean (speedups are reported relative to the evaluator's reference).

---

## 7. Evidence format (`candidates.jsonl`, append-only)

One JSON object per evaluated candidate, appended in order, never rewritten. Fields:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "faithful per-(b,c,h) mapping; establish correctness + baseline geomean",
  "change_from_parent": "initial anchor",
  "validation": {
    "static_audit": "pass — index/layout/numeric checklist §4 re-verified",
    "skill_usage": "none (KernelWiki/profiling out of scope for sm_80)"
  },
  "results": {
    "per_workload": [
      {"uuid": "314bcaf9", "B": 2, "C": 4, "pass": true,  "speedup": 0.0},
      {"uuid": "019db6f8", "B": 1, "C": 3, "pass": true,  "speedup": 0.0},
      {"uuid": "6c45c19e", "B": 4, "C": 4, "pass": true,  "speedup": 0.0},
      {"uuid": "46445623", "B": 4, "C": 7, "pass": true,  "speedup": 0.0},
      {"uuid": "cccd1cb3", "B": 1, "C": 4, "pass": true,  "speedup": 0.0}
    ],
    "all_pass": true,
    "geomean_speedup": 0.0
  },
  "decision": "accept|reject|fix",
  "decision_reason": "<why>",
  "cumulative_evaluations": 1
}
```

Notes:
- `speedup`/`geomean_speedup` values are transcribed from the evaluator output (placeholders
  above are illustrative). If the evaluator reports latency instead of speedup, record raw
  per-workload times and the derived geomean, and state which was reported.
- If a workload fails, set `pass:false`, record the error class (NaN/Inf, tolerance, compile,
  launch), and keep any reported partial metrics.
- `skill_usage` is "none" for every record on this `sm_80` task.

---

## 8. Risk register & mitigations

| Risk | Signal | Mitigation |
|---|---|---|
| `A_cumsum` axis-order (`[b,h,c,s]` vs `X`'s `[b,c,s,h,d]`) | wrong-value fail on all workloads | §4.1 audit before c001; smallest workload debugs fastest |
| exp overflow → Inf/NaN | NaN/Inf correctness fail | exp-of-difference only; match reference ordering; investigate before clamping |
| bf16 matmul-2 too coarse | marginal tolerance fail | branch fp32/TF32 matmul-2 (c003) |
| group-fusion grid too small on (1,3) | c002 slower than c001 | keep c001 as parent if c002 regresses; consider hybrid grid |
| register spill from `[128,128]` fp32 accum | slow / low occupancy | `num_warps=8`; KV tiling (c004) to shrink accumulator |
| materializing G/L/M in HBM | matches naive PyTorch cost | keep all intermediates on-chip; store only `Y` |

---

## 9. What is explicitly out of scope

- No autotuner spanning multiple immutable configs (ambiguates "one kernel across 5 workloads").
- No softmax/max-subtraction, no factored exp, no HBM materialization of G/L/M.
- No non-Triton compute, no direct CUDA/profiler/evaluator invocation, no external agents/tools.
- No `final` run without explicit operator approval.

---

**Next action (subsequent turn):** implement **c001** per §2/§4, hash the source, run
`./scripts/evaluate_candidate.sh feedback c001`, then append its record per §7 and decide per §6.

---

## 10. Decision log

- **c001 (evaluated, ACCEPTED).** Per-`(b,c,h)` anchor, grid `B·C·32`, `num_warps=8`,
  `num_stages=2`. All 5 feedback workloads **PASS**. Per-workload speedup:
  189.88 / 220.58 / 269.73 / 286.96 / 171.12; **geomean 223.26×** (arith 227.65×). No
  NaN/Inf; the `A_cumsum` axis-swap and exp-of-difference handling validated by the pass.
  Baseline established; c001 is the current best/parent. Cumulative evaluations: **1**.
  - Observation: `sol_ms` scales roughly with `B·C` (0.014→0.095 ms); smallest workload
    (1,3) has the lowest speedup (kernel launch/occupancy floor), largest (4,7) the highest —
    consistent with a memory/occupancy-bound regime.
  - **Next probe → c002 (per-group `G` fusion):** grid `B·C·8`, compute `G=Cmat@Bmatᵀ` once and
    loop the 4 heads (`M_h=G∘L_h`, `Y_h=M_h@X_h`). Hypothesis: cut matmul-1 + B/C loads 4×;
    risk = reduced grid hurting occupancy on (1,3)=24 group-tiles. Gate on all-5 pass and
    geomean > 223.26×.

- **c002 (evaluated, REJECTED).** Per-group `G` fusion, grid `B·C·8`, `static_range` loop over 4
  heads, `num_warps=8`, `num_stages=2`. All 5 **PASS** (math exact), but **geomean 140.29×**
  (arith 150.71×) — a **regression** vs c001's 223.26×. Per-workload: 155.30 / 80.00 / 188.95 /
  227.55 / 101.73×. The two small workloads collapsed most: (1,3) 220.58→80.00×, (1,4)
  171.12→101.73×. Diagnosis: the op is **occupancy/memory-bound, not matmul-1-bound**. Fusing 4
  heads per program 4×-es per-program latency (serial head loop, larger footprint) while
  shrinking the grid 4×; on small `B·C` the 24/32 group-tiles under-fill 108 SMs. The redundant
  matmul-1 that fusion removes was essentially free. **c001 remains the best/parent.**
  Cumulative evaluations: **2**.
  - Takeaway: grid-shrinking optimizations are the wrong direction here — parallelism dominates.
    Retire the group-fusion branch. Redirect the spine toward levers that keep the `B·C·32` grid
    (or grow it) and cut per-program cost / memory traffic.
  - **Next probe → c003 (per-program config / KV-tiling on the winning per-`(b,c,h)` layout):**
    the accumulator is a `[128,128]` fp32 (16384 elts) plus a second `[128,128]` — register
    pressure likely caps occupancy at `num_warps=8`. Options (each its own ID, from c001 parent):
    (a) **`num_warps`/`num_stages` sweep** — try `num_warps=4` (higher occupancy if regs allow)
    and `num_stages=3`; (b) **`BLOCK_M=64` query-row split with causal block-skip** — halves the
    live accumulator and skips ~half the score work, raising the grid to `B·C·32·2`. Start with
    the cheapest lever (config sweep) since it is one-line and directly targets the
    occupancy-bound diagnosis. Gate on all-5 pass and geomean > 223.26×.

Note: after c002's rejection, `solution/solution.py` was restored to the c001 kernel (the current
best) so the next candidate branches from the winning baseline. c001's immutable source is
preserved by the controller under `runs/candidates/c001/`.
```
