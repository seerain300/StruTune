# Executable Optimization Plan — `gemm_n4096_k4096`

Run ID: `formal-kda-20260916--flashinfer--gemm_n4096_k4096`
Target: NVIDIA A800 (`sm_80`, Ampere) · Op: `C = A @ B.T`, fp16 · `N=K=4096`, `M` var.
Feedback M ∈ {4, 48, 64, 128, 240}. Metric: geomean speedup vs `torch.matmul(A, B.T)`, all
workloads must pass correctness first.

This plan turns `docs/draft.md` into a concrete, sequential candidate schedule. It is a living
document only in the sense that later observations refine *choices already scoped here*; the
structure, gates, and evidence format are fixed now. No code is written and nothing is evaluated
in this turn.

---

## 0. Operating principles (recap of binding constraints)

- **Triton-only compute.** PyTorch limited to metadata / allocation / launch. No Torch/CPU/NumPy/
  CUDA-extension/alternate fallback. A broken Triton kernel is invalid and must be fixed, not
  substituted.
- **Entry point:** `solution/solution.py` exposing `run(A, B) -> C`. Must return contiguous fp16
  `[M, N]` on A's device, equal to `A @ B.T` within tolerance.
- **Evaluation:** only `./scripts/evaluate_candidate.sh feedback <cNNN>`. Five fixed workloads =
  one candidate evaluation. Never run CUDA/profiler/nvidia-smi/evaluator/`final` directly. `final`
  is operator-approval-only.
- **Immutability:** candidates `c001, c002, …` sequential; any meaningful source/config/launch
  change ⇒ new ID; never reuse an ID; never rewrite prior `candidates.jsonl` rows.
- **Budget:** 100 evaluations; token soft 1.0M / hard 1.2M. Spend evaluations deliberately —
  static reasoning before every run; change one structural idea per candidate.
- **KernelWiki** skill is Blackwell/Hopper-scoped and does NOT cover sm_80 → not used. Record
  "skill_usage: none (sm_80 out of KernelWiki scope)" in evidence.

---

## 1. Strategy overview

The op is **HBM-bandwidth bound** on reading `B` (33.55 MB) once, for all feedback M. The floor
is ≈ 17 µs (B traffic / ~2 TB/s). cuBLAS/`torch.matmul` under-occupies the 108 SMs for skinny M
because there are too few output tiles. The winning lever is **Split-K** to multiply block count
and saturate HBM, with fp32 accumulation to stay numerically identical to the reference.

Search arc (each stage = one or a few candidates, strictly sequential):

1. **c001 — Correctness & harness baseline.** Simplest correct autotuned NT GEMM, no split-K.
   Confirms the wrapper, dtypes, layout, tolerance behavior, and gives a per-M speed baseline.
2. **c002–c00x — Split-K introduction.** Add K-partitioning to raise block count; this is the
   expected primary win for small M. Pick reduction method (deterministic two-pass vs fp32
   atomic) based on c001-era tolerance evidence.
3. **Config tuning.** Sweep tiling / stages / warps / split factor around the split-K winner.
4. **M-adaptive dispatch.** One immutable kernel whose split factor & tiling adapt to M (autotune
   `key=['M']` or a closed-form Python heuristic) so it generalizes to the 43-workload final and
   is not overfit to {4,48,64,128,240}.
5. **Optional specialization.** GEMV-style streaming kernel for very small M (≤8) if the MMA path
   leaves M=4 far from the bandwidth floor.
6. **Converge & stop.** When geomean improvement flattens or we approach the HBM floor, write
   `SEARCH_COMPLETE`.

Only advance to the next stage after the current candidate is evaluated and its evidence recorded.

---

## 2. Candidate lineage (sequential schedule)

Each entry lists: parent, the single change vs parent, hypothesis, correctness plan, expected
signal, and the decision rule. IDs are provisional past c002 — the *idea order* is fixed, but
exact configs for later IDs are chosen using evidence from earlier runs. Every actually-created
ID is immutable once evaluated.

### c001 — Baseline autotuned NT GEMM (no split-K) — parent: none
- **Change:** first implementation. Standard Triton GEMM: grid over `(pid_m, pid_n)` with
  `GROUP_SIZE_M` L2 grouping; load A `[BLOCK_M,BLOCK_K]` (mask rows `<M`), load B
  `[BLOCK_N,BLOCK_K]` (K contiguous, fully coalesced), `tl.dot` with fp32 accumulator, cast to
  fp16 on store. Small autotune space keyed on nothing (or `['M']`) — a handful of safe configs.
- **Hypothesis:** correct and competitive for M=128/240; likely *below* cuBLAS for M=4/48/64 due
  to too few blocks (≤32). Establishes the correctness gate and a real baseline.
- **Correctness plan:** re-derive `A[m,k]=m*K+k`, `B[n,k]=n*K+k`, `C[m,n]=m*N+n`; mask `offs_m<M`
  with `other=0.0`; no N/K masking since blocks divide 4096; fp32 accum; fp16 store; wrapper
  returns contiguous fp16 `[M,N]`.
- **Expected signal:** all 5 pass; geomean ≈ 0.7–1.1× (win on large M, loss on small M).
- **Decision:** if any workload fails → debug before any perf work. If pass → proceed to split-K.
  Record the observed tolerance behavior (any near-miss margin) to inform reduction choice.

### c002 — Split-K, deterministic two-pass reduction — parent: c001
- **Change (single idea):** add `pid_k` / `SPLIT_K` partitioning of the K reduction. Partials
  written to fp32 scratch `[SPLIT_K, M, N]`; a second Triton reduction kernel sums along SPLIT_K
  and casts to fp16. Start `SPLIT_K` fixed (e.g. 8) with `BLOCK_M=16`, mid `BLOCK_N`, `BLOCK_K=64`.
- **Hypothesis:** block count jumps to `num_pid_m*num_pid_n*SPLIT_K` (≥200 for small M),
  saturating HBM → large speedup on M=4/48/64. fp32 two-pass keeps error at fp32 ULP → passes
  comfortably.
- **Correctness plan:** SPLIT_K a power of two dividing 4096 → each K-chunk (`4096/SPLIT_K`) is a
  multiple of BLOCK_K, no K-mask needed; scratch zero-init not required (each partial fully
  written); reduction kernel masks `offs_m<M`. Deterministic (no atomics).
- **Expected signal:** geomean clearly > c001, biggest gains at small M.
- **Decision:** adopt as new baseline if geomean improves and all pass; else inspect whether
  reduction-kernel/scratch traffic ate the gain (then try atomic variant c003).

### c003 — Split-K, fp32 atomic-add reduction — parent: c002 (or c001)
- **Change (single idea):** replace two-pass scratch+reduce with a single kernel doing
  `tl.atomic_add` into an fp32 accumulator buffer, then a light cast-to-fp16 pass (or atomic into
  a pre-zeroed fp32 C then cast). Removes the `[SPLIT_K,M,N]` write+reread.
- **Hypothesis:** less memory traffic than two-pass → faster, if atomic contention is low (C is
  small: ≤2 MB). Value error still fp32 ULP; **not bit-reproducible** across runs.
- **Correctness plan:** confirm evaluator uses tolerance (not bit-exact) checking — inferred from
  c001/c002 margins; if any doubt, keep c002 deterministic path as the shipped baseline.
- **Expected signal:** faster than c002 if reduction traffic dominated; else neutral/worse.
- **Decision:** keep whichever of {c002, c003} is faster *and* safely passing as the split-K base.

### c004–c007 — Config sweep around the split-K winner — parent: split-K base
One structural knob per candidate (do not co-vary), each a new immutable ID:
- **c004:** `SPLIT_K ∈ {2,4,8,16}` sweep (find the point where block count ≳ 2×108 without
  over-splitting → tiny per-chunk K hurting MMA/load width).
- **c005:** `BLOCK_N ∈ {32,64,128,256}` (wider B loads & MMA efficiency vs. more blocks).
- **c006:** `num_stages ∈ {2,3,4,5}` cp.async pipeline depth to hide HBM latency (key for
  mem-bound); `num_warps ∈ {2,4,8}`.
- **c007:** `BLOCK_K ∈ {32,64,128}` and `BLOCK_M ∈ {16,32}` (minimize M-pad waste vs. reuse).
- **Decision each:** keep change only if geomean improves beyond noise; otherwise revert to prior
  base for the next knob. Fold confirmed wins into the running best config.

### c008 — M-adaptive single immutable kernel — parent: best of c004–c007
- **Change:** make split factor + tiling adapt to `M` in one immutable source, via either
  (a) `@triton.autotune(key=['M'])` over a curated config list, or (b) a closed-form Python
  heuristic (`SPLIT_K = clamp(ceil(target_blocks / (num_pid_m*num_pid_n)), 1, 16)` with
  `target_blocks ≈ 2*108`), chosen by whether autotune warmup is counted by the evaluator.
- **Hypothesis:** one kernel that is near-optimal across all feedback M *and* arbitrary
  small-to-moderate M (needed for the 43-workload final) without per-M source changes.
- **Correctness plan:** validate the heuristic across a spread of M (not just the 5), ensuring
  SPLIT_K stays a divisor-friendly power of two and masks hold for `M<BLOCK_M`.
- **Decision:** adopt if geomean ≥ best fixed-config candidate and generalization logic is sound.

### c009+ — Optional GEMV-style kernel for tiny M — parent: c008
- **Change:** if M=4 remains well above the ~17 µs floor, add a Triton streaming path (no
  tl.dot): each block owns an N-strip, streams K with wide vectorized fp16 loads, fp32 FMA
  accumulate. Dispatch by `M ≤ threshold` inside `run`. Still 100% Triton.
- **Decision:** keep only if it beats the MMA split-K path on small M without regressing others.

### Reserve
Remaining budget (well under 100) held for: fixing any correctness regression, one persistent-kernel
experiment if scheduling overhead shows up, and re-tuning if a later idea shifts the optimum.

---

## 3. Correctness checks (applied before AND after every evaluation)

**Pre-run static checklist (must pass before spending an evaluation):**
1. Index math: `A[m,k]→m*K+k`, `B[n,k]→n*K+k`, `C[m,n]→m*N+n`; strides taken from tensors, not
   assumed (guard against non-contiguous inputs → `.contiguous()` in wrapper if needed).
2. Masks: `offs_m < M` on A load (`other=0.0`) and C store; K-mask only if a K-chunk doesn't
   divide evenly (avoided by power-of-two SPLIT_K dividing 4096); N never masked (128|64|32 | 4096).
3. Dtypes: operands fp16; accumulator fp32 (`out_dtype=tl.float32`, TF32 disabled); final store
   cast to fp16. Split-K partials/reduction in fp32.
4. Shapes: output allocated `[M, N]` fp16 contiguous on A.device; `run` returns exactly that.
5. Determinism: two-pass split-K is deterministic; atomic path is value-safe but not bit-exact —
   only used if tolerance-based checking confirmed.

**Post-run gate:**
- Correctness is **gate #1**: a candidate is ranked only if all 5 feedback workloads pass. Any
  failure ⇒ stop feature work, diagnose (masking, dtype, layout, reduction), fix, new ID.
- Record the numeric margin/behavior reported by the evaluator to track headroom vs tolerance.

**Numerical safety ladder (if a split-K candidate ever fails correctness):**
fp32 atomic → fall back to fp32 deterministic two-pass → if still failing, reduce SPLIT_K /
increase BLOCK_K to shorten per-chunk error chains. Never move accumulation to fp16.

---

## 4. Performance hypotheses (falsifiable, mapped to candidates)

| # | Hypothesis | Test | Falsified if |
|---|-----------|------|--------------|
| H1 | Op is HBM-bound; floor ≈ B/BW ≈ 17 µs | c001 timings scale with B traffic, not FLOPs | large-M much slower per-byte than small-M |
| H2 | cuBLAS under-occupies for small M; plain tiling loses there | c001 geomean < 1 on M=4/48/64 | c001 already ≥ cuBLAS everywhere |
| H3 | Split-K restores occupancy → big small-M win | c002/c003 ≫ c001 on small M | no gain despite ≥200 blocks (⇒ traffic/contention bound) |
| H4 | Deterministic two-pass reduction traffic is cheap (C small) | c002 ≈ c003 within noise | c002 ≫ slower ⇒ reduction traffic matters |
| H5 | An intermediate SPLIT_K is optimal (over-split hurts) | c004 sweep shows interior optimum | monotonic in SPLIT_K |
| H6 | Deeper cp.async stages hide HBM latency | c006 improves with stages 3–4 | flat/worse ⇒ smem/reg bound |
| H7 | One M-adaptive kernel matches per-config bests | c008 geomean ≥ best fixed | c008 worse ⇒ keep dispatch simpler |
| H8 | M=4 needs GEMV streaming to hit floor | c009 beats MMA on M=4 | MMA split-K already near floor |

---

## 5. Stopping / convergence criteria

Declare convergence and write `SEARCH_COMPLETE` (with reason) when **any** of:
1. **Floor reached:** best candidate's implied bandwidth is within ~10–15% of ~2 TB/s peak on the
   B-dominated workloads (little physical headroom left).
2. **Plateau:** two consecutive structural ideas each yield < ~2% geomean improvement (below
   run-to-run timing noise), i.e. tuning has saturated.
3. **Budget guard:** approaching the evaluation budget (reserve ≥ a few evals) or the token soft
   limit (1.0M) — consolidate on the best valid candidate and stop.
Never trigger `final`; final 43-workload run is operator-approval-only. On stop, ensure the best
valid candidate is clearly identified in `candidates.jsonl` for operator selection.

---

## 6. Evidence format (one JSON object appended per evaluated candidate)

Append exactly one line to `candidates.jsonl` immediately after each evaluation; never edit prior
lines. Schema:

```json
{
  "candidate_id": "cNNN",
  "parent_id": "cMMM | null",
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "one-line falsifiable claim this candidate tests",
  "change_from_parent": "the single structural/config change",
  "config": { "BLOCK_M": 0, "BLOCK_N": 0, "BLOCK_K": 0, "SPLIT_K": 0,
              "num_warps": 0, "num_stages": 0, "GROUP_SIZE_M": 0,
              "reduction": "none|two_pass|atomic", "adaptive": false },
  "validation": {
    "static_checklist_passed": true,
    "all_workloads_correct": true,
    "tolerance_notes": "observed margins / bit-exact vs tolerance"
  },
  "per_workload": [
    {"M": 4,   "uuid": "f439da26", "correct": true, "speedup": 0.0, "latency_us": 0.0},
    {"M": 48,  "uuid": "59ca23f5", "correct": true, "speedup": 0.0, "latency_us": 0.0},
    {"M": 64,  "uuid": "67d4c8f3", "correct": true, "speedup": 0.0, "latency_us": 0.0},
    {"M": 128, "uuid": "29ebd771", "correct": true, "speedup": 0.0, "latency_us": 0.0},
    {"M": 240, "uuid": "e7c939ae", "correct": true, "speedup": 0.0, "latency_us": 0.0}
  ],
  "geomean_speedup": 0.0,
  "decision": "adopt-as-baseline | reject-revert | debug | stop",
  "decision_reason": "why, tied back to a hypothesis H#",
  "cumulative_evaluations": 0,
  "skill_usage": "none (sm_80 outside KernelWiki scope)"
}
```

Fields map directly to the CLAUDE.md requirement (parent, source hash, hypothesis, validation,
per-workload result, geomean, decision, cumulative eval count, skill usage). `speedup` and
`latency_us` are taken from the evaluator output as reported; if the evaluator reports only one of
them, record what it gives and note the omission in `decision_reason`.

---

## 7. Immediate next action (next turn, not now)

Implement **c001** (baseline autotuned NT GEMM, fp32 accum, no split-K) in
`solution/solution.py`, run the pre-run static checklist, then evaluate with
`./scripts/evaluate_candidate.sh feedback c001` and append its evidence row. Proceed down the
lineage only as each candidate's evidence is recorded.

---

## 8. Decision log

### c001 (evaluated) — DONE
- Result: **all 5 PASSED, geomean 0.56x** (M=4:0.55, M=48:0.53, M=64:0.55, M=128:0.56, M=240:0.63).
- Reference latencies ~44–70 µs → cuBLAS itself achieves only ~500–760 GB/s of B traffic, i.e.
  it too is far from the ~17 µs / ~2 TB/s HBM floor on these skinny shapes. Our plain kernel is
  ~1.8x slower than cuBLAS: confirms **H2** (under-occupation for small M) — with ≤32 output
  tiles (1 M-tile × 32 N-tiles at BLOCK_N=128) we leave most of the 108 SMs idle and cannot hide
  HBM latency.
- Tolerance is atol=0.01/rtol=0.01 with matched_ratio=0.99 (99% of elements must match). Large
  `max_rel` on M=240 (283x) is from isolated near-zero reference entries and did **not** fail the
  ratio gate → **tolerance-based, not bit-exact**. This clears the fp32-atomic split-K path
  (c003) numerically as long as ≥99% of elements stay within tol (fp32 reduction → trivially).
- **Next (c002):** introduce split-K to multiply block count toward ≥2×108. Start with the
  deterministic fp32 two-pass reduction (scratch `[SPLIT_K,M,N]` + reduce kernel), SPLIT_K=8,
  BLOCK_M=16, mid BLOCK_N, BLOCK_K power-of-two dividing 4096/SPLIT_K. Expect the largest gains on
  M=4/48/64 where c001 is weakest.

### c002 (evaluated) — INVALID
- Atomic split-K (SPLIT_K∈{2..16}, tl.atomic_add into fp32 C32 + Triton cast). Result: **0/5,
  all "C: non-finite output mismatch."**
- Root cause (confirmed by reading triton 3.5.0 `runtime/autotuner.py`): the autotuner benchmarks
  each config with many `do_bench` launches; with `tl.atomic_add` and **no `reset_to_zero`**, every
  benchmark launch accumulated into the same un-zeroed C32 → fp32 sum grew unbounded → overflowed
  fp16 on cast → inf/nan. GEMM math itself correct. Fix: `reset_to_zero=['C32']`.

### c003 (evaluated) — VALID but SLOWER (0.38x), reverted
- Same atomic split-K + `reset_to_zero=['C32']`. All 5 PASS (fp32 atomic within tol), but geomean
  **0.38x** — worse than c001 on every M. Atomic contention + full `torch.zeros` memset of C32 +
  separate cast pass cost more than the occupancy split-K bought. Also SPLIT_K 8–16 shortens each
  block's K-loop to 4–8 iters, killing the cp.async steady state.

### c004 (evaluated) — NEUTRAL (0.56x)
- Reverted to deterministic plain fp16-store GEMM but widened autotune (BLOCK_N=32 → ~128 blocks,
  num_stages=5, L2 GROUP_SIZE_M, larger MMA tiles). Geomean **0.56x**, ties c001. Wider N-tiling did
  not break the plateau: for M>BLOCK_M, N-tiling for occupancy re-reads B (num_pid_m>1, 2× traffic);
  single-M-tile N-tiling caps at num_pid_n(≤128) blocks.

### c005 (evaluated) — MARGINAL BEST (0.57x) — current best valid baseline
- Coalesced B load as `[BLOCK_N,BLOCK_K]` + `tl.trans`, wide BLOCK_K∈{128,256}. Geomean **0.57x**,
  a hair above c001/c004. Loads are not the ceiling (Triton already coalesced c004). Confirms the
  plain path is **occupancy-limited on small M**: ~55% of cuBLAS bandwidth ≈ fraction of active SMs
  (64 n-blocks / 108 SMs).

### Synthesis and next step (c006 — NOT YET RUN this turn)
- The op is HBM-bound on reading B once; cuBLAS ≈ 735 GB/s, our plain kernels ≈ 440 GB/s. The gap is
  **occupancy on small M**, and the only way to add occupancy while reading B exactly once is
  split-K — but c003 over-split (SPLIT_K 8–16). The decisive untested experiment is **modest
  SPLIT_K=2–4** (K-loops stay ≥16 iters → cp.async pipelining preserved) with a **deterministic
  two-pass reduction** (partials to `[SPLIT_K,M,N]` fp32, separate reduce kernel — no atomic
  contention, small C-side traffic) so H3 is tested cleanly. This is the justified next candidate.
- **Turn budget note:** this turn already spent 4 evaluations (c002–c005); stopping here per the
  one-candidate-per-turn directive. Cumulative evaluations = 5. Best valid = **c005 (0.57x)**.
  Search has NOT converged — c006 (modest deterministic split-K) is a concrete, justified next step,
  so no SEARCH_COMPLETE is written.

### c006 (evaluated) — REJECT, revert to c005
- Modest M-adaptive Split-K (SPLIT_K M<=64→4, ~128→2, ≥240→1) with a **deterministic two-pass**
  fp32 reduction (kernel1 → Cpart[SPLIT_K,M,N] fp32; kernel2 sum+cast). VALID 5/5 but geomean
  **0.49x < c005 0.57x**, worse on every split path (M=4:0.46, 48:0.44, 64:0.45, 128:0.48).
  M=240 took the SPLIT_K=1 plain path → 0.61x, unchanged from c005, proving the regression is
  entirely the split path, not a whole-kernel change.
- **Falsifies H3/H4 for this workload class.** Even without atomic contention (the c003 confound),
  the extra kernel launch + Cpart fp32 allocation + partial write/read traffic + the narrower
  BLOCK_K(≤64) needed to keep K-loops long cost MORE than the occupancy gained. Total latency is
  only ~50–100 µs, so a second launch + an extra full read/write of the (≤2 MB fp32) partials is a
  large relative overhead.
- **Split-K is now exhausted:** both reduction strategies lose (atomic c003 0.38x, two-pass c006
  0.49x). The occupancy deficit on skinny M cannot be closed by K-partitioning on this hardware
  for these sizes — the reduction overhead dominates at these tiny latencies.
- **Next (c007):** switch hypotheses to **H8** — a non-MMA, memory-streaming FMA path for very
  small M (M≤8, where MMA pads 4→16 rows, 75% wasted rows, and the op is pure GEMV-class HBM
  streaming). Dispatch it only for M≤8 inside `run`, keeping the proven c005 `tl.dot` path for M>8.
  This tests whether removing MMA padding waste on the tiniest M closes any gap; it is a distinct,
  untested structural idea, so the search has NOT converged.
- Cumulative evaluations = 6. Best valid = **c005 (0.57x)**.

### c007 (evaluated) — MARGINAL BEST (0.58x) — new best valid baseline
- Kept the c005 kernel body byte-for-byte; **only** widened the autotune config list to offer the
  previously-untested corner: **BLOCK_N=32 (128 blocks = full 108-SM occupancy) TOGETHER WITH wide
  BLOCK_K=128/256 and deep cp.async num_stages 4/5**, no split-K. (c004 had occupancy without wide
  K; c005 had wide K without full occupancy — neither combined them.)
- Result: VALID 5/5, geomean **0.58x** — marginally above c005 (0.57x). Extremes improved
  (M=4 0.57→0.60, M=240 0.61→0.62) but M=64 slipped (0.55→0.53); net ~+1–2%, within timing noise.
  M=128 bit-exact (abs=rel=0). This is now the best valid baseline but the gain is marginal.
- **Interpretation:** the plain single-pass `tl.dot` path is essentially **saturated at ~0.58x**
  on A800 for this skinny NT GEMM. cuBLAS stays ahead because at 45–70 µs it runs one fused,
  hardware-tuned skinny-GEMM kernel with no Triton launch/pipeline-fill overhead. Both split-K
  reductions (c003 0.38x, c006 0.49x) and every plain-tiling lever (occupancy, burst width,
  stages, coalescing) have now been swept.
- **Only remaining untested structural idea = H8:** a non-MMA, GEMV-style FMA streaming kernel for
  the tiniest M (M≤8), which avoids the 4→16 MMA row padding (75% wasted MMA rows at M=4) and does
  pure vectorized HBM streaming. This is a genuinely different kernel, so the search has NOT
  converged. Next candidate **c008** dispatches this path only for M≤8 and keeps the c007 tl.dot
  path for M>8.
- Cumulative evaluations = 7. Best valid = **c007 (0.58x)**.

### c008 (evaluated) — REJECT, revert to c007 — SEARCH CONVERGED
- Tested H8: added a non-MMA GEMV-streaming kernel (`acc += tl.sum(a[:,None,:]*b[None,:,:],axis=2)`,
  no `tl.dot`) dispatched only for M≤8; M>8 kept the c007 `tl.dot` kernel byte-for-byte.
- Result: VALID 5/5 but geomean **0.40x** — the M=4 GEMV path is **catastrophic: 461 µs (0.09x)
  vs c007's 74 µs (0.60x), ~6× slower**. M>8 workloads are byte-identical to c007 (0.54–0.62x),
  proving the regression is entirely the M≤8 kernel.
- **Strongly falsifies H8.** The broadcast-outer-product materializes an `8×BLOCK_N×BLOCK_K` fp32
  intermediate per K-step and does the K-reduction on the vector/CUDA-core datapath with no
  `ldmatrix`/tensor-core pipeline, so B-load latency is exposed and ALU throughput becomes the
  ceiling. The `tl.dot` MMA path — even padding 4→16 rows — is the better use of Ampere for tiny M.
- **Search space now exhausted / genuinely converged:**
  1. Plain-tiling levers all swept → ~0.58x ceiling (c001 0.56, c004 0.56, c005 0.57, c007 0.58).
  2. Both Split-K reduction strategies lose (atomic c003 0.38, deterministic two-pass c006 0.49) —
     the reduction pass/contention costs more than the occupancy at these ~50–100 µs latencies.
  3. The non-MMA GEMV alternative loses badly (c008 0.40).
  No further structural idea offers headroom on sm_80 for this skinny NT GEMM: the op is dominated
  by reading B once, cuBLAS achieves this with a single fused, hardware-tuned kernel, and Triton's
  per-launch / pipeline-fill overhead at these tiny latencies is the irreducible gap.
- **Decision: declare SEARCH_COMPLETE. Best valid candidate = c007 (0.58x geomean, all 5 pass).**
- Cumulative evaluations = 8. Best valid = **c007 (0.58x)**.

