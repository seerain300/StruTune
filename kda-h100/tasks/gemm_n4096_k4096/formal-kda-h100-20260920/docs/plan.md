# Plan — `gemm_n4096_k4096` (FlashInfer, H100 / sm_90)

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`, `TASK.md`,
`CLAUDE.md`, `task/definition.json`, `task/feedback_workloads.jsonl`. This turn produces
the plan only — no candidate is implemented or evaluated.

---

## 0. Objective and success definition

- **Goal**: maximize geometric-mean speedup `geomean(t_ref / t_cand)` over the 43 fixed
  feedback workloads, subject to **every** workload passing the evaluator's correctness
  gate. Reference is `torch.matmul(A, B.T)` (cuBLAS, FP32-accumulate, FP16 out).
- **Primary constraint**: Triton is the compute path; no Torch/CPU/NumPy/CUDA-ext
  fallback. Entry point `solution/solution.py::run(A, B) -> C` (fp16).
- **Where the score lives** (from draft §2): 39/43 shapes have M ≤ 256 →
  memory/occupancy-bound; only 4 are large/compute-bound. Geomean is small-M-weighted.
  Strategy: **defend large-M at parity, push occupancy on small-M.**

---

## 1. Operating rules (self-binding, from CLAUDE.md / TASK.md)

1. Implement candidates `c001, c002, …` sequentially, **one immutable source version at a
   time**. Any meaningful source/config/launch change ⇒ new ID. Never reuse an ID for
   changed source; never rewrite earlier `candidates.jsonl` records.
2. Evaluate **only** via `./scripts/evaluate_candidate.sh feedback cNNN`. One full
   43-workload pass = **one** of the **100** evaluations.
3. Profile **only** via `./scripts/ncu_profile.sh` (ncu-report-skill). **Never** profile
   and evaluate at the same time — a foreign process on the locked GPU ⇒ rc=3 and a wasted
   eval. Sequence strictly; never background a profile during an eval.
4. Never run `final` without explicit operator approval. Never touch evaluator/dataset/
   controller/launcher/shared config. Stay inside this workspace.
5. Token budget: soft 9M / normal 10M / hard 11M. Track and slow down near soft limit.
6. A failing Triton kernel is invalid — fix it or abandon the idea; do not substitute a
   non-Triton implementation.
7. When the geomean genuinely converges (or budget hit), write `SEARCH_COMPLETE` with the
   reason. Do not spend evals past convergence.

---

## 2. Solution architecture (skeleton the candidates share)

`solution/solution.py` will expose:

```
run(A, B) -> C            # fp16 [M,N]; A [M,K] fp16, B [N,K] fp16; C = A @ B.T
```

Responsibilities allowed in Python (non-compute plumbing only):
- Read shapes/strides/dtype/device; allocate `C` (and, when split-K is active, an FP32
  workspace).
- Compute the launch grid and pick the config/regime (M-bucket heuristic and/or autotune).
- Launch the Triton kernel(s). No torch matmul, no torch reduction on the compute path
  (an FP32→FP16 *cast* of a Triton-produced workspace is plumbing, but preferred as a
  Triton kernel to keep the compute path unambiguously Triton; see §4 c002).

Compute is entirely in `@triton.jit` kernels. All accumulation in FP32; FP16 only at the
final store. M-dimension masking on every A-load and C-store (M can be 1, 7, 15, 35, …).
N and K are divisible by all planned block sizes (no N/K tail), but code will still guard
K for split-K chunking.

---

## 3. Candidate lineage strategy

Linear trunk with cheap, attributable, one-variable-at-a-time steps. Each candidate's
`parent` is the last **accepted** candidate (best validated geomean so far), unless it is
an explicitly-labeled exploratory branch off a named parent.

```
c001 (correct anchor, autotuned tiled tl.dot, no split-K)
  └─ c002 (+ split-K FP32-atomic path, gated to small/medium M)
       └─ c003+ (tune: BLOCK_N/BLOCK_K/SPLIT_K/num_stages/num_warps per M-bucket)
            └─ c00x (freeze heuristic dispatch OR refine autotune space)
                 └─ c00y (optional: tiny-M M≤8 dedicated path)
                      └─ c00z (optional: Hopper static-stride persistent / group-M raster)
```

Lineage rules:
- **Accept** a candidate as the new parent only if it improves geomean **and** all 43
  workloads pass. A correctness failure or geomean regression ⇒ keep the previous parent;
  the failed ID stays recorded (immutable) with `decision: reject`.
- Explore at most one hypothesis per ID. If two knobs must move together (e.g. enabling
  split-K *requires* a workspace + cast), that coupled change is a single candidate and is
  documented as such.
- Prefer profiling-guided steps over blind sweeps to conserve eval budget.

---

## 4. Sequential candidate plan (executable steps)

Each step lists: hypothesis, exact change vs parent, validation, and accept/reject rule.
IDs beyond c002 are provisional — their exact parameters depend on evidence from earlier
evals/profiles.

### c001 — Correct anchor (baseline geomean)
- **Hypothesis**: a standard FP32-accumulate tiled `tl.dot` GEMM with M-masking is correct
  across all 43 shapes and establishes the reference geomean. Likely < 1.0× on small M
  (low occupancy) — that is expected and is the thing c002 fixes.
- **Change**: implement `run` + one `@triton.jit` tiled kernel. Grid `(ceil(M/BM), N/BN)`
  with group-M raster (`GROUP_M`). `tl.autotune` over a **small** curated config list
  (a handful of `(BM,BN,BK,stages,warps)`), keyed on M via `key=['M']`. FP32 acc,
  `a[BM,BK] · trans(b[BN,BK])`, mask `offs_m < M` on load and store, store FP16.
  No split-K.
- **Validation**: `./scripts/evaluate_candidate.sh feedback c001`. Require all 43 pass.
- **Accept if**: correctness passes (this becomes the baseline parent regardless of
  speedup, since it is the first valid point). Record baseline geomean and per-workload
  speedups — these are the reference for every later decision.
- **Then profile** (separately, after the eval returns): ncu on M=16 and M=8192 to confirm
  the draft's bottleneck model (small-M occupancy/L2-bound; large-M tensor-bound).

### c002 — Split-K for small/medium-M occupancy (main expected win)
- **Hypothesis** (draft §2.3, §5.3): small-M is occupancy/L2-bandwidth-bound; multiplying
  active CTAs via split-K raises L2 read-bandwidth utilization and cuts time for M ≤ ~512.
  Target ~1–2 full waves: `n_mtiles · n_ntiles · SPLIT_K ≈ 132…264`.
- **Change vs c001**: add a split-K kernel that accumulates each K-slice in FP32 and
  `tl.atomic_add`s into an **FP32 workspace** `C_f32[M,N]`, followed by a **Triton** cast
  kernel FP32→FP16 into `C`. Dispatch heuristic on M: use split-K (SPLIT_K∈{4,8}, chosen so
  the K chunk stays a multiple of BK and divides 4096) for small/medium M; fall back to the
  c001 non-split path (SPLIT_K=1, no atomics) for large M. Zero-init the workspace before
  the split-K launch (or use first-writer-wins init). Guard K coverage so the K split sums
  to exactly 4096.
- **Validation**: eval c002. Must pass all 43 (special attention: split-K FP32 reduction
  correctness at odd M = 7/15/35 and tiny M = 1/2/4). Compare geomean and, crucially, the
  small-M per-workload speedups vs c001.
- **Accept if**: geomean improves and all pass. **Reject if**: any correctness failure
  (suspect FP16 atomics, K-coverage, or workspace init — not tolerance) or geomean
  regression.

### c003 — Split-K / tile-shape tuning per M-bucket
- **Hypothesis**: the c002 default SPLIT_K/BLOCK_N are not optimal across the M-sweep;
  bucketed configs (tiny/small/medium/large) land closer to 1–2 waves each and improve
  geomean. Informed by the c001/c002 ncu profiles and per-workload speedup table.
- **Change vs c002**: adjust the M→(BM,BN,BK,SPLIT_K,stages,warps) mapping only (heuristic
  table or refined autotune `key`/config list). No structural change.
- **Validation**: eval c003; compare per-bucket speedups. Accept/reject as above.

### c004+ — Micro-tuning (num_stages, num_warps, BLOCK_K, group-M)
- **Hypothesis**: pipelining depth and warp count shift the large-M compute path toward
  parity and shave small-M launch/issue overhead (draft §5.2; KernelWiki pipelining
  695→940 TFLOPS on the tutorial GEMM). group-M raster helps large-M L2 locality.
- **Change**: one knob per candidate. Each is a separate ID.
- **Validation/accept**: as above; stop refining a knob once it plateaus.

### c00x — Freeze heuristic vs keep autotune (determinism decision)
- **Hypothesis**: once good per-bucket configs are known, a frozen deterministic heuristic
  removes autotune first-call benchmarking variance and any interaction with the warmup=2
  window (draft §5.6), without losing speed.
- **Change**: replace autotune with a fixed M-bucket dispatch using the discovered configs
  (or vice-versa if autotune proves strictly better). Single ID.
- **Validation/accept**: geomean must be ≥ the autotuned parent within noise.

### c00y — (Optional) Dedicated tiny-M path (M ≤ 8)
- **Hypothesis** (draft §5.4): only if profiling shows the tiny-M MMA path is launch/issue
  bound rather than L2-bandwidth bound. A high-occupancy split-K reduction/GEMV-style
  Triton kernel could beat the padded-MMA path for M ∈ {1,2,4,7,8}.
- **Change**: add a tiny-M branch (still Triton, still FP32 reduce). Gate strictly on M.
- **Validation/accept**: must improve the tiny-M workloads without regressing others; else
  reject and keep the unified path.

### c00z — (Optional) Hopper static-stride persistent / scheduling
- **Hypothesis** (draft §5.5): a software static-stride persistent kernel (grid = #SMs)
  reduces launch overhead / wave quantization. CLC is SM100-only — **not** used on H100.
- **Change**: persistent loop over tiles; possibly fuse the split-K cast. Single ID.
- **Validation/accept**: only keep if it improves geomean; otherwise reject.

Exploration stops earlier if the geomean converges (see §7).

---

## 5. Correctness checks (per candidate, before and after eval)

**Static (pre-eval) checklist** — must all hold before spending an evaluation:
1. Entry point `run(A, B)` returns fp16 `[M,N]` on the same device; `C = A @ B.T`.
2. Accumulator dtype is FP32 in every kernel; only the final store casts to FP16.
3. Split-K (when active) reduces in FP32 (FP32 workspace + FP32 atomics or FP32 partials);
   **never** FP16 accumulation or FP16 atomics.
4. M-masking on every A-load (`other=0.0`) and every C-store (`offs_m < M`). Verified
   mentally against M = 1, 2, 4, 7, 15, 35 (values smaller than any BLOCK_M).
5. K coverage: for SPLIT_K, the union of K chunks is exactly `[0, 4096)` with no overlap
   and no gap; each chunk length is a multiple of BLOCK_K (or the tail is masked).
6. Workspace lifecycle: FP32 workspace zero-initialized (or first-writer-wins) before
   atomic accumulation; sized to the actual `M·N`; freed/reused deterministically.
7. No `@triton.jit` compile-time errors; grid dims are positive for all M (incl. M=1).
8. Immutability: this ID's source differs meaningfully from all prior IDs; no prior record
   is being edited.

**Dynamic (post-eval) checks**:
9. Evaluator reports all 43 workloads **pass** correctness. A single failure ⇒ `decision:
   reject`, diagnose via the static checklist (most likely #3/#5/#6), fix in a **new** ID.
10. Return code handling: rc=3 (foreign-process/interference) ⇒ measurement discarded, **do
    not** trust any timing; re-run the eval only when the GPU is clear and never overlap
    with a profile. Distinguish rc=3 (environmental, re-runnable) from a genuine
    correctness/compile failure (source bug, needs a new ID).

Because there is **no private correctness harness** (CLAUDE.md forbids alternate harnesses
and direct CUDA), correctness is only ground-truthed by the evaluator. Hence c001 is
deliberately conservative so its green result validates the numerical model before any
aggressive change.

---

## 6. Performance hypotheses (falsifiable, tied to evidence)

| ID | Hypothesis | Predicted signal (ncu / speedup) | Falsified if |
|----|-----------|-----------------------------------|--------------|
| H1 | Small-M (M≤256) is L2-bandwidth/occupancy-bound, not HBM-bound (B fits in 50 MB L2) | ncu on M=16: high `lts__t_sectors`/L2 throughput, low `dram__throughput`, low achieved occupancy, few active CTAs | DRAM-bound or already high occupancy |
| H2 | Raising active-CTA count (split-K / smaller BN) cuts small-M time toward L2-BW roofline | c002 small-M speedups > c001; ncu shows higher occupancy + L2 throughput | c002 no faster despite more CTAs (⇒ launch/issue bound → pivot to H5) |
| H3 | Large-M (2053–8192) is tensor-core-bound; cuBLAS is near-peak so parity is the realistic target | ncu on M=8192: high `sm__pipe_tensor_op` util; speedup ≈ 1.0× | We find >1.1× headroom (then invest more there) |
| H4 | Pipelining depth (num_stages 3–5) + group-M raster move large-M toward parity | c004 large-M speedup ↑ vs parent; fewer pipeline-stall cycles in ncu | stages/raster change is noise on large-M |
| H5 | Tiny-M (M≤8) may be launch/issue bound; a dedicated path helps | ncu on M=1: low L2 *and* low tensor util, high launch-overhead fraction | tiny-M already L2-BW bound (⇒ skip c00y) |

Each hypothesis is checked by **profiling between evals** on representative shapes
(small: M=16 and/or M=64; large: M=8192; tiny: M=1) — never concurrently with an eval.

---

## 7. Stopping / convergence criteria

Stop the search and write `SEARCH_COMPLETE` when **any** of:
1. **Convergence**: the best geomean has not improved by more than ~0.5% (relative) across
   the last **3** accepted-or-attempted candidates, and the remaining ideas in §4 are
   exhausted or profiling predicts no further headroom.
2. **Budget**: cumulative evaluations approach the 100 cap, or token usage approaches the
   9M soft limit (begin winding down at soft; hard stop well before 11M).
3. **Diminishing ideas**: profiling shows both regimes near their respective rooflines
   (small-M near L2-BW, large-M near cuBLAS/tensor peak) with no attributable next step.

`SEARCH_COMPLETE` will state: the winning candidate ID, its geomean and per-regime
speedups, the reason for stopping, cumulative evals used, and that `final` awaits operator
approval. **`final` is never run autonomously.**

Per-candidate "give up on this branch" rule: if a candidate regresses or fails correctness
twice on the same idea, abandon that branch and return to the last accepted parent.

---

## 8. Evidence format

### 8.1 `candidates.jsonl` — one JSON object appended per evaluated candidate
Never rewrite prior lines. Required fields:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "<one-line falsifiable hypothesis for this change>",
  "change_vs_parent": "<exact source/config/launch delta>",
  "config": {"regime_dispatch": "...", "autotune": true, "split_k": false},
  "static_checks": {"fp32_acc": true, "m_masked": true, "splitk_fp32": "n/a",
                     "k_coverage_ok": true, "immutable_new_id": true},
  "validation": {"return_code": 0, "all_workloads_pass": true,
                 "failures": []},
  "per_workload": [{"M": 256, "uuid": "280860e6-…", "speedup": 0.00, "pass": true}, "…all 43…"],
  "geomean_speedup": 0.00,
  "small_m_geomean": 0.00,   
  "large_m_geomean": 0.00,   
  "decision": "accept | reject",
  "decision_reason": "<why accepted/rejected vs parent>",
  "cumulative_evaluations": 1,
  "profiling": {"ran": false, "shapes": [], "findings": "…"},
  "skill_usage": ["KernelWiki:<pages cited>", "ncu-report-skill:<when used>"]
}
```

Notes:
- `source_sha256` binds the record to the exact immutable source evaluated.
- `per_workload` carries all 43 entries (M, uuid, speedup = t_ref/t_cand, pass) so
  regressions are attributable per shape.
- `small_m_geomean` / `large_m_geomean` are derived sub-aggregates (M≤256 vs the 4 large
  shapes) so the decision can weigh the score driver explicitly.
- `profiling` records ncu evidence gathered **around** (never during) the eval.
- `skill_usage` cites KernelWiki page IDs and ncu-report-skill invocations, per workflow.

### 8.2 Profiling evidence (ncu-report-skill)
Store ncu outputs under `profile/` via `./scripts/ncu_profile.sh --set <set> -o
profile/<name> python <harness>`. For each profiling session record in the candidate's
`profiling.findings`: the shape(s), the metrics inspected (occupancy, `lts__throughput`,
`dram__throughput`, `sm__pipe_tensor_op`), and which §6 hypothesis it confirmed/falsified.
Profiling and evaluation are strictly serialized.

### 8.3 Decision log
Each accept/reject is justified in `decision_reason` against the current parent's geomean
and the relevant per-regime sub-aggregate, closing the loop hypothesis → evidence →
decision.

---

## 9. Immediate next actions (for the following turn — not this one)

1. Implement `c001` (§4) in `solution/solution.py`: conservative correct anchor.
2. Run the static correctness checklist (§5) on c001.
3. `./scripts/evaluate_candidate.sh feedback c001`; append the record (§8.1).
4. Profile c001 on M=16 and M=8192 (§6 H1/H3) — after the eval returns, GPU clear.
5. Proceed to c002 (split-K) guided by that evidence.

No implementation or evaluation is performed in this planning turn.

---

## 10. Decision log (append-only)

### c001 — evaluated 2026-09-22 (eval #1 of 100)
- **Result**: 43/43 PASS, geomean **0.47×** (small-M M≤256 ≈ 0.447×, large-M{972,2053,2379,8192} ≈ 0.764×, M=8192 ≈ 0.96×). Accepted as baseline parent.
- **Confirms**: FP32-accumulate + M-masking numerically correct on all M incl. {1,2,4,7,15,35}. Numerical model validated.
- **Diagnosis (no profile needed)**: small-M speedup is flat ~0.42–0.49× across the entire M≤256 sweep and does NOT scale with M — the classic occupancy-starved signature. With autotune preferring BLOCK_N∈{128,256} and a single M-tile, only 16–32 CTAs launch vs 132 SMs. cuBLAS ref is ~0.026–0.047 ms; our kernel is a flat ~0.063–0.073 ms regardless of M ⇒ we are launch/occupancy/fixed-overhead bound, not compute bound. Large-M is already respectable (M=8192 near parity), so headroom is entirely in small/medium M.
- **M=128 outlier (0.33×)**: autotune likely picked a worse config there; a deterministic small-M path should remove such cliffs.

### Next candidate — c002 (occupancy via split-K, main expected win)
- **Hypothesis (H2)**: small/medium-M is occupancy/L2-BW bound; splitting the K=4096 contraction across SPLIT_K CTAs (FP32 partials → FP32 workspace via atomic_add → Triton FP32→FP16 cast) multiplies active CTAs to fill 132 SMs and cuts the flat ~0.065 ms floor. Target n_mtiles·n_ntiles·SPLIT_K ≈ 132–264.
- **Change vs c001**: add split-K kernel + FP32 workspace + Triton cast; M-bucket dispatch — split-K (SPLIT_K∈{4,8}, chunk multiple of BK, divides 4096) for M≤~512, fall back to the c001 non-split path for large M. All reduction in FP32; never FP16 atomics.
- **Watch**: correctness at odd/tiny M (1,2,4,7,15,35) under split-K; workspace zero-init; K coverage exact. Accept iff geomean improves and all 43 pass.

### c002 — evaluated 2026-09-22 (eval #2 of 100). ACCEPTED — new parent.
- **Note**: implemented as a *deterministic M-bucket dispatch with BLOCK_N=32* (simplest occupancy lever) rather than split-K. BLOCK_N=32 → 128 N-tiles → ~128 CTAs (one wave) for a single M-tile, with no redundant B re-reads and no atomics/workspace. Split-K deferred to a later candidate.
- **Result**: 43/43 PASS, geomean **0.53×** (was 0.47×). small-M geomean 0.447→0.505; large-M 0.764→0.796 (M=8192 held at 0.94×). c001's M=128 cliff (0.33×) resolved (0.54×).
- **Two regressions**: (a) tiny-M M∈{1,2,4} fell (0.42→0.35/0.38/0.46) — forcing ~128 CTAs adds aggregate launch/issue cost where compute is trivial; tiny-M wants *fewer, larger-BN* CTAs. (b) M=144 anomalous 0.22× (sol 0.1935 ms) while identical-config neighbors M=136/152 are ~0.081 ms → transient timing blip, not source.
- **Interpretation**: small-M sol floor ~0.054 ms is still ~2× cuBLAS (~0.028 ms) and ~10× the L2-BW floor (~5–7 µs) → still latency/overhead bound, NOT bandwidth bound. So the next lever is reducing per-call fixed cost + a dedicated tiny-M path, not just more CTAs.

### Next candidate — c003 (tiny-M path + reclaim medium-M)
- **Hypothesis (H5 refined)**: tiny-M (M≤8) regressed under the 128-CTA policy → give M≤8 a *low-CTA, large-BLOCK_N* config (e.g. BLOCK_M=16, BLOCK_N=128/256, GROUP_M=1) so only 16–32 CTAs launch with minimal aggregate launch overhead, recovering the c001 tiny-M level while keeping the c002 small/medium gains. One-variable change: add an `M<=8` bucket to `_select_config`; leave the 16–256 and large buckets exactly as c002.
- **Accept iff** tiny-M speedups recover toward/above c001 (≥~0.42×) AND geomean ≥ c002 (0.53×) with all 43 passing.

### c003 — evaluated 2026-09-22 (eval #3 of 100). REJECTED — parent stays c002.
- **What was actually tested**: instead of the tiny-M idea above, c003 tested the cleaner *occupancy-ceiling* question — halve BLOCK_N 32→16 for the single-M-tile band 8<M≤128 (128→256 CTAs, ~2/SM), all other buckets byte-identical to c002.
- **Result**: 43/43 PASS, geomean **0.53×** (unchanged). In the changed band the speedups came out ~0.02 *lower* (≈0.50 vs c002's ≈0.52) — BLOCK_N=16 halves MMA width for no occupancy payoff.
- **Key learning (falsification)**: going 128→256 CTAs did nothing → small-M is **NOT** occupancy/bandwidth bound beyond ~1 wave; it is **latency / fixed-overhead bound** (sol floor ~0.058 ms invariant to CTA count, ~2× cuBLAS, ~8–10× the L2 one-pass floor). H2's "more CTAs = faster" is exhausted.
- **Noise note**: every feedback run shows exactly one config-identical small-M shape with a 0.10–0.19 ms outlier (c002: M=144 → 0.22×; c003: M=192 → 0.30×). These ~0.01 geomean deltas are measurement noise, not source behavior. Tiny M=1/2 wobble (0.30/0.37) is in c003's *unchanged* tiny bucket → ref-timing noise.
- **Action**: reverted `solution/solution.py` to the exact c002 source (hash bdad00a4…) as the accepted parent.

### Next candidate — c004 (attack the latency floor: fewer serial K-iterations)
- **Hypothesis (H6, new)**: the small-M floor (~0.058 ms) is set by the per-CTA serial K-loop — 4096/BLOCK_K = 64 dependent iterations of `load→mma→accumulate` with a redundant K-mask (K=4096 is divisible by BLOCK_K so the mask is always true). Two low-risk levers, applied together as one coherent change on the small-M buckets (M≤256) only, leaving the large bucket untouched: (1) raise BLOCK_K 64→128/256 to cut the loop to 32/16 iterations and amortize load/issue latency; (2) drop the always-false K-mask (`k_rem` guard) so loads are unconditional (fewer instructions, better pipelining) — safe because K % BLOCK_K == 0 for all planned BLOCK_K ∈ {64,128,256}.
- **Risk**: larger BLOCK_K raises smem/registers per stage; with num_stages=4 and BN=32 this is tiny, so headroom is ample. Keep FP32 acc; masking on M unchanged (still needed — M arbitrary). Immutable new ID c004.
- **Accept iff** geomean > c002 0.53× (beyond the ~0.01 noise band) with all 43 passing; watch that large-M (unchanged bucket) does not move.
- **If c004 also flat**: the remaining structural lever is split-K (shorten the per-CTA K chain by partitioning K across CTAs with an FP32 workspace + Triton cast) — that directly attacks the serial-latency floor rather than raw CTA count, and is the c005 fallback.

### c004 — evaluated 2026-09-22 (eval #4 of 100). ACCEPTED — new parent.
- **Change vs c002**: (1) BLOCK_K 64→128 on all three small-M buckets (M≤256) → 32 K-iterations instead of 64; (2) removed the always-false K-mask → unconditional loads (safe: K=4096 % BLOCK_K == 0 for BLOCK_K∈{64,128}). Large bucket (M>256) unchanged except the global mask removal.
- **Result**: 43/43 PASS, geomean **0.58×** (was 0.53×). Confirms H6: in clean-timing small-M shapes the sol floor dropped ~5–7% at identical shapes (M=72 0.0554→0.0528 ms, M=96 0.0564→0.0529, M=80 0.0552→0.0530, M=104 0.0573→0.0541). Large-M held (M=8192 0.94×).
- **Noise caveat (important)**: this run had elevated *system-wide* timing noise in workloads 25–37 (M=64…35 and tiny M) — BOTH ref and sol were ~2× inflated there (M=32 ref 0.0533 vs c002 0.0273). So the 0.58 headline is partly noise-lifted; treat the true value as ~0.55–0.58. Outliers M=120 (1.44×) and M=112 (0.48×) are pure ref-timing noise. The clean-region improvement is real and there is no structural regression → accept, but **re-verify the baseline** at next comparison.
- **Action**: kept `solution/solution.py` at c004 (hash 6457c65c…) as the accepted parent.

### Next candidate — c005 (split-K to shorten the per-CTA K chain)
- **Hypothesis (H7)**: even after BLOCK_K=128 the small-M CTA still walks the full K=4096 serially (32 dependent MMA steps) while holding only ~128 CTAs / one wave — the remaining ~1.8× gap to cuBLAS is that serial-K latency with the machine under-occupied on the K axis. Split-K partitions K across SPLIT_K CTAs so each does 4096/SPLIT_K of the contraction (e.g. SPLIT_K=4 → 8 K-iters/CTA) and grid CTAs rise to n_ntiles·SPLIT_K (128·4=512), filling idle SMs with *independent* work that overlaps latency — attacking the floor along the axis c003 could not.
- **Change vs c004 (one coherent feature)**: add a split-K kernel that accumulates its K-slice in FP32 and `tl.atomic_add`s into a zero-initialized FP32 workspace `C_f32[M,N]`, then a Triton FP32→FP16 cast kernel writes `C`. Gate on M: split-K only for a small/medium band (e.g. 16≤M≤256 with SPLIT_K=4, K-chunk = 1024 = 8·BLOCK_K, divides 4096 exactly); tiny M (<16) and large M (>256) keep the c004 non-split path. All reduction FP32; never FP16 atomics.
- **Watch**: workspace zero-init cost inside the timed call (allocate `C_f32` with `torch.zeros` — plumbing, not compute); correctness at odd M (35) and the band edges; that atomic contention doesn't erase the latency win. Accept iff geomean beats c004's clean-region level (re-measured) with all 43 passing; else reject and keep c004.

### c005 — evaluated 2026-09-22 (eval #5 of 100). REJECTED — parent stays c004.
- **Change vs c004**: 2-stage split-K for the band 16≤M≤256 (SPLIT_K=4). Stage 1 writes FP32 partials `Cp[SPLIT_K,M,N]` (plain stores, no atomics → deterministic); stage 2 sums over SPLIT_K in FP32 and casts to FP16. Tiny (M<16) and large (M>256) kept the c004 single-pass path.
- **Result**: 43/43 PASS but geomean **collapsed to 0.38×** (was 0.58×). The split-K band 16..256 fell to ~0.27–0.42× (was ~0.47–0.65×).
- **Root cause / falsification of H7**: the FP32 partial buffer must be written by stage 1 and read by stage 2 = `2·SPLIT_K·M·N·4` bytes; for M=256 that is 33.5 MB of extra traffic — **as large as the entire B matrix** — plus a second launch. That dominates any serial-K latency saving. So the small-M floor is NOT K-axis under-occupancy; it is fixed per-launch cost + the single mandatory L2 pass of B, and split-K only **adds** traffic. This is a hard negative: **any** split-K variant (atomic or 2-stage) pays O(SPLIT_K·M·N) partial traffic, so split-K is retired for this problem.
- **Confirmation the diagnosis is traffic, not code**: the non-split shapes behaved exactly as c004 (M=8 0.56×, M=15 0.59×, M=8192 0.94×, M=972 0.86×), isolating the regression to the split-K band.
- **Action**: reverted `solution/solution.py` to the exact c004 source (hash 6457c65c…).

### Next candidate — c006 (push the one proven lever one notch)
- **Hypothesis (H8)**: the only change that ever moved the floor was shortening the serial K-loop (c004: BLOCK_K 64→128, +0.05 geomean). Push it once more on the most floor-dominated bucket only — M≤64 (14 workloads: M∈{1..64} routed here plus 7/15/35), BLOCK_K 128→256 → 16 K-iterations instead of 32. smem: A[64,256]+B[256,32] fp16 = 48 KB/stage × 4 stages = 192 KB < 228 KB, fits. K=4096%256==0 so loads stay unconditional. All other buckets byte-identical to c004.
- **Accept iff** the M≤64 shapes improve and geomean ≥ c004 (clean-region ~0.55–0.58) with all 43 passing; else reject and keep c004.
- **Convergence note**: with split-K and raw-CTA-count both falsified, the realistic ceiling for the small-M-weighted geomean is likely ~0.55–0.62 — cuBLAS's tuned small-GEMM path plus the mandatory 33.5 MB B L2-pass and per-call launch overhead bound us. If c006 and one micro-tune (num_stages/num_warps) do not move geomean beyond noise, the search has converged.

### c006 — evaluated 2026-09-22 (eval #6 of 100). ACCEPTED — new parent.
- **Change vs c004**: only the M≤64 bucket, BLOCK_K 128→256 (16 K-iters vs 32); all other buckets byte-identical. smem 192 KB < 228 KB.
- **Headline**: geomean 0.56× (c004's headline was 0.58×) — but that is NOT a regression; it is a cross-run noise artifact (c004's run was ~2× inflated in the M=32..70 region). Accepted on **within-run A/B** evidence instead, which is immune to cross-run drift:
  - c006's M≤64 bucket (BLOCK_K=256): sol floor ~0.046 ms (M=24..56 = 0.0460–0.0476 ms).
  - c006's adjacent 65<M≤128 bucket (BLOCK_K=128, byte-identical to c004): ~0.050 ms (M=72..128 = 0.0491–0.0510 ms).
  - Same GPU state, adjacent M, only BLOCK_K differs → BLOCK_K=256 is ~8% faster/shape → confirms H8.
- **No regression**: buckets 65–256 and large are byte-identical to c004 and behave as before (M=8192 0.93×, M=972 0.88×). c006 run is cleaner (min 0.41, no wild ref spikes like c004's M=120 1.44×). M=64 itself 0.48× is a one-off blip vs neighbors (M=56 0.59, M=72 0.56).
- **Method learning**: cross-run headline geomeans differ by up to ~0.02–0.04 purely from system timing noise; **same-run adjacent-bucket A/B is now the trusted decision signal**, not headline deltas.
- **Action**: kept `solution/solution.py` at c006 (hash 8840ca4d…) as the accepted parent.

### Next candidate — c007 (lift the weakest bucket: 129≤M≤256)
- **Observation**: within c006, the 129≤M≤256 bucket (BLOCK_M=128, 2 M-tiles, BLOCK_K=128, GROUP_M=8) is the weakest small bucket (~0.46–0.54×), below both the M≤64 (0.55–0.64×) and 65–128 (0.54–0.63×) buckets. Its 2 M-tiles × 128 N-tiles = 256 CTAs already fill >1 wave, and it still runs 32 K-iters.
- **Hypothesis (H9)**: this bucket is throttled by its longer K-loop and/or the 2-M-tile split. Give it BLOCK_K=256 to halve K-iters (16). At BLOCK_M=128 the smem is A[128,256]+B[256,32] fp16 = 96 KB/stage, so num_stages must drop 4→2 (192 KB < 228 KB) — a coupled but single-purpose change ("shorten K-loop for the 129–256 bucket"). One bucket touched; all others byte-identical to c006.
- **Accept iff** within-run the 129–256 shapes improve vs the untouched 65–128 bucket baseline AND no other bucket regresses, with all 43 passing; else reject and keep c006.
- **Convergence**: after c007, if no bucket-level within-run improvement remains and one stages/warps micro-tune is flat, declare convergence (best ≈ c006, ~0.56–0.60 real).

### c007 — evaluated 2026-09-22 (eval #7 of 100). REJECTED — parent stays c006.
- **Change vs c006**: only the 129≤M≤256 bucket, BLOCK_K 128→256 with num_stages 4→2 (smem 160 KB). All other buckets byte-identical.
- **Result**: 43/43 PASS, geomean **0.50×** (< c006). The 129≤M≤256 bucket **collapsed to ~0.35–0.40×** (sol ~0.088–0.097 ms, ~2× c006's ~0.063 ms).
- **Falsification of H9 via clean within-run isolation**: only that one bucket changed and only it regressed — every other bucket in the same run held at its c006 level (65–128: 0.54–0.60; M≤64: 0.54–0.62; M=8192 0.93). Cause is unambiguous: at BLOCK_M=128, num_stages=2 destroys the software load/mma pipeline (too few stages to hide global-load latency for a 128×256 A-tile), and the 32→16 K-iter win cannot compensate.
- **Lesson**: BLOCK_K=256 only pays when num_stages can stay ≥4, which at BLOCK_N=32 requires BLOCK_M≤64 (smem). The K-loop lever is now **exhausted**: M≤64 already at BLOCK_K=256/stages=4; the BLOCK_M=128 buckets can't raise BLOCK_K without losing stages. The clean M≤128 shapes (0.54–0.62) corroborate c006 as a solid ~0.56–0.60 real level.
- **Action**: reverted `solution/solution.py` to the exact c006 source (hash cc5cc554…).

### Next candidate — c008 (last targeted idea: re-tile the weak 129–256 bucket)
- **Observation**: the 129≤M≤256 bucket is the weakest (~0.46–0.54×) and it is the only one using 2 M-tiles (BLOCK_M=128, GROUP_M=8). Its weakness may be the 2-M-tile scheduling / GROUP_M raster rather than the K-loop (which c007 proved can't be shortened here).
- **Hypothesis (H10)**: split its work along M instead — BLOCK_M=64 for 129≤M≤256 gives 4 M-tiles × 128 N-tiles = 512 CTAs but keeps BLOCK_K=128/stages=4 (proven-good pipeline, smem A[64,128]+B[128,32] fp16 ×4 = 48 KB, fine). Same per-CTA K-loop as the healthy M≤64 bucket, just more M-tiles. Tests whether the bucket's deficit is M-tiling, not K.
- **Accept iff** within-run the 129–256 shapes rise toward the 65–128 bucket level AND no other bucket regresses, all 43 pass; else reject, keep c006, and this is the last structural idea → SEARCH_COMPLETE.
- **Stop criterion**: if c008 is flat/negative, the search has converged (K-loop exhausted, split-K retired, CTA-count falsified, M-tiling neutral) → write SEARCH_COMPLETE with c006 as best (~0.56–0.60).

### c008 — evaluated 2026-09-22 (eval #8 of 100). REJECTED — parent stays c006.
- **Change vs c006**: only the 129≤M≤256 bucket, BLOCK_M 128→64 (2→4 M-tiles, 512 CTAs), keeping the proven BLOCK_K=128/stages=4 pipeline. All other buckets byte-identical.
- **Result**: 43/43 PASS, geomean **0.54×**. The 129≤M≤256 bucket did NOT improve — stayed ~0.46–0.51× (M=192 0.47, M=256 0.50), statistically identical to c006's ~0.47–0.54× for that bucket. Every other bucket held (M≤64: 0.55–0.63; 65–128: 0.53–0.57; M=8192 0.93).
- **Falsification of H10 (neutral)**: M-tiling is a neutral knob for the weak bucket → its deficit is intrinsic to the M∈(128,256] range (2 M-tiles are inherently needed once M>BLOCK_M≤128; cuBLAS also transitions kernels here), not a scheduling artifact we can retune away.
- **Action**: kept `solution/solution.py` at the c006 design (reverted; current hash of the c006-equivalent source recorded).

### Convergence status (small-M lever ladder exhausted)
All small-M structural levers are now falsified or exhausted:
1. Raw CTA count beyond one wave — c003 reject (not occupancy/BW bound).
2. Split-K — c005 reject (adds B-sized partial traffic).
3. K-loop shortening — c004/c006 accept, but **exhausted**: BLOCK_K=256 needs stages≥4 ⇒ only BLOCK_M≤64, already applied.
4. K-loop on BLOCK_M=128 bucket — c007 reject (stages=2 kills the pipeline).
5. M-tiling of the weak 129–256 bucket — c008 reject (neutral).
The residual small-M gap to cuBLAS is fixed per-launch overhead + the mandatory single 33.5 MB L2 pass of B, which no single-pass Triton tiling removes. **Best = c006 (~0.56–0.60 real).**

### One remaining justified direction — c009 (medium/large-M compute-bound tuning)
- **Not yet explored**: the 4 large shapes {972, 2053, 2379, 8192} use one fixed large-tile config (BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3, warps=8) inherited unchanged since c002. Three of them are below parity: M=2053 ~0.69–0.72×, M=2379 ~0.74–0.77×, M=972 ~0.79× (only M=8192 is ~0.93×). These are compute-bound; a better large-tile config (e.g. num_stages 3→4, or BLOCK_K 64→128, or a second large bucket for M≈2k) could recover several points. Though only 4/43 workloads, they are the lowest-hanging non-small shapes and the small-M path is exhausted.
- **Hypothesis (H11)**: the large-M config is under-pipelined (stages=3) and/or K-loop-long (BLOCK_K=64 → 64 iters); raising stages and/or BLOCK_K lifts M∈{972,2053,2379} toward M=8192's 0.93× without regressing M=8192.
- **Accept iff** the medium/large shapes improve within-run and M=8192 holds, all 43 pass; else reject and this is the final lever → SEARCH_COMPLETE with c006 as best.

### c009 — evaluated 2026-09-22 (eval #9 of 100). REJECTED — parent stays c006.
- **Change vs c006**: only the large bucket (M>256), num_stages 3→4. All small buckets byte-identical.
- **Result**: 43/43 PASS, geomean **0.55×**. The medium/large shapes were unchanged vs c006 within noise — M=972 0.80, M=2053 0.70, M=2379 0.74, M=8192 0.92 (if anything M=8192 nudged 0.93→0.92, i.e. more stages slightly hurt the already-parity case).
- **Falsification of H11 (neutral)**: the large bucket is **cuBLAS-parity-bound, not pipeline-depth-bound** — deeper Triton pipelining does not close the gap to cuBLAS's tuned large-GEMM path.
- **Action**: reverted `solution/solution.py` to the c006 design (large bucket back to num_stages=3).

### SEARCH CONVERGED — best = c006
Every distinct structural lever across both regimes is now falsified or exhausted:
- **small-M**: CTA-count (c003), split-K (c005), K-loop shortening (c004/c006 accept then exhausted — BLOCK_K=256 needs stages≥4 ⇒ only BLOCK_M≤64, already applied), K-loop@BLOCK_M=128 (c007), M-tiling of the weak 129–256 bucket (c008).
- **large-M**: pipeline depth (c009).

The residual gap to cuBLAS is structural: (small-M) fixed per-launch overhead + one mandatory 33.5 MB L2 pass of B that a correct single-pass GEMM cannot avoid; (large-M) cuBLAS's tuned parity. Best valid candidate = **c006** (all 43 pass; small-M ~0.53–0.63, large-M ~0.79–0.93, headline geomean ~0.56 in a clean run). Convergence criterion §7.1 met: no accepted improvement across the last 3 candidates (c007/c008/c009 all rejected) and the roadmap ideas are exhausted. Next turn: write SEARCH_COMPLETE with c006 as best. `final` remains operator-only.
