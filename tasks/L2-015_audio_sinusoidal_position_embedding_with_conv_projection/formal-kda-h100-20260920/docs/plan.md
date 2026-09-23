# Plan — L2/015 Audio Sinusoidal Position Embedding with Conv Projection

Executable, sequential KDA optimization plan. Builds on `docs/draft.md`. Target H100 `sm_90`,
bf16. Submission `solution/solution.py::run(...)`. Primary compute **must be Triton** (no
`F.conv2d/linear/gelu`, no Torch/CPU/NumPy/CUDA-extension fallback). Only validation channel is
`./scripts/evaluate_candidate.sh feedback cNNN` (full 16-workload set = 1 eval). Profiling only via
`./scripts/ncu_profile.sh`, strictly serialized with evals.

---

## 0. Reference contract to reproduce exactly

Pipeline (see draft §1). Fixed constants: `IC1=1, C=384, F0=80→40→20→10, K=3, stride=2, pad=1`,
`conv_out_dim=3840`, `d_model=1024`, `embed_scale=32.0`, `pos` shape `(1500,1024)`.

Index math (single shared helper reused by all conv kernels):
- output length `O = (N + 2·1 − 3)//2 + 1 = ceil(N/2)` (⇒ `t1=ceil(T/2)`, `t2=ceil(t1/2)`, `t3=ceil(t2/2)`).
- input tap for output `o`, kernel `k∈{0,1,2}`: `i = 2·o − 1 + k`; **mask** taps with `i<0` or `i≥N` to 0.
- freq chain `80→40→20→10` uses the identical formula on the H axis.
- flatten column `k∈[0,3840)` ⇒ `c=k//10, f=k%10` (channel-major, freq-minor).
- linear: `out[m,n]=Σ_k x[m,k]·W[n,k]` (W is `(1024,3840)`, used transposed in `tl.dot`).
- epilogue: `x*embed_scale + pos[t3_index]`; output bf16 `(B,t3,1024)`.

Numerics rules (all candidates): fp32 accumulators for every reduction; **exact erf** GELU
`0.5·x·(1+erf(x/√2))` via `libdevice.erf` in fp32; round to bf16 when storing each conv
intermediate (mirrors reference storing bf16 between stages); conv bias added in fp32 then rounded.

---

## 1. Lineage strategy

Sequential, immutable candidates. Each `cNNN` = one source version = at most one feedback eval.
Parent is the best *valid* ancestor unless a branch is explicitly exploratory. Any meaningful
source/config/launch change ⇒ new ID; never reuse an ID for changed source; never rewrite prior
`candidates.jsonl` records.

Change **one dominant variable per candidate** so each eval attributes a clear delta:
- c001: establish correct fully-Triton baseline (correctness gate; layout = simplest that works).
- c002: layout for heavy convs (channels-last intermediates) — the expected biggest lever.
- c003: epilogue fusion (bias+GELU into conv; flatten+linear+scale+pos-add already fused in c001).
- c004+: autotune / shape-bucketed configs for conv2·conv3 (cost center) and large-B GEMM.
- c00x: targeted micro-opts guided by ncu (K-loop order, num_stages/warps, conv1 handling,
  optional conv3→linear fusion) — each isolated.

Branch rule: if a candidate regresses geomean or fails correctness, revert to the last valid
parent and try a *different* single change; do not stack an unproven change on a broken one.

---

## 2. Candidate specifications (executable, in order)

### c001 — correctness-first, fully Triton (baseline)
- **Goal**: pass all 16 workloads; produce first valid geomean.
- **Kernels**:
  1. `conv1`: `IC=1, K=9`. Direct/small-GEMM: per output pixel load ≤9 masked input scalars,
     accumulate against `W1 (384×9)`; fuse bias + erf GELU; write bf16. (K<16 ⇒ avoid `tl.dot`
     initially; manual FMA or pad-K to 16.) Choose intermediate layout here (see below).
  2. `conv2`, `conv3`: implicit GEMM. `M=B·OF·OT`, `N=OC=384`, `K=IC·9=3456`. Weight reshaped
     `(384,3456)` (contiguous, free). K-loop over ic-blocks × 9 taps with padding mask; fp32 acc;
     fuse bias + erf GELU; write bf16.
  3. `fused_linear`: `M=B·t3`, `K=3840`, `N=1024`. Read conv3 output with `c=k//10, f=k%10`
     (no materialized permute). Epilogue `acc*embed_scale + pos[t]`; write bf16.
- **Layout for c001**: pick the *simplest correct* layout (NCHW intermediates acceptable) — do not
  optimize layout yet; correctness is the only c001 success criterion beyond "runs".
- **Block sizes**: conservative fixed (e.g. BLOCK_M=64/128, BLOCK_N=128, BLOCK_K=64); no autotune.
- **Success gate**: 16/16 correctness pass. Perf may be ≤1.0×; that's fine for a baseline.
- **If fail**: fix index/mask/flatten/GELU derivation (do NOT add a Torch fallback); the fix is a
  new ID only if source changed after an eval was spent.

### c002 — channels-last intermediates for conv2/conv3
- **Hypothesis**: making IC contiguous (NHWC `(B,F,T,C)`) gives coalesced, MMA-friendly K for the
  cost-center convs ⇒ large speedup on B∈{16,32,64} and long-T rows. conv1 writes NHWC directly;
  fused_linear reads NHWC conv3 (`k→(c,f)`) unchanged.
- **Change vs c001**: intermediate layout only. Keep block sizes fixed.
- **Success**: geomean > c001 with 16/16 pass and no tiny-shape (B≤2, B=64/T=128) regression >~5%.

### c003 — epilogue fusion confirm / tighten
- **Hypothesis**: ensure bias+GELU fully fused in conv epilogues and no stray eager copies
  (`.contiguous()` doing real work). Remove any residual Torch materialization.
- **Change vs c002**: fusion/launch-count only (target ≈4 launches: conv1, conv2, conv3, linear).
- **Success**: geomean ≥ c002, 16/16 pass.

### c004 — autotune conv2/conv3
- **Hypothesis**: shape-bucketed `@triton.autotune` (keys on B-bucket, t2/t3, choose
  BLOCK_M/N/K, num_warps∈{4,8}, num_stages∈{2,3,4}) closes MMA-utilization gaps on large workloads
  without hurting small ones.
- **Change vs c003**: autotune configs for conv2/conv3 only.
- **Success**: geomean > c003; verify tiny shapes unaffected (they hit small-M configs).

### c005 — autotune fused_linear + grid/occupancy for tiny shapes
- **Hypothesis**: linear GEMM (M as low as 64) and the very small conv workloads are
  launch/occupancy bound; tune configs + ensure enough grid tiles for 132 SMs on mid shapes.
- **Change vs c004**: linear autotune + grid tweaks.

### c006+ — ncu-guided micro-opts (each isolated)
Candidate ideas, pick by evidence: K-loop order (`taps×ic` vs `ic×taps`) for spatial-window reuse;
conv1 as padded `tl.dot` vs FMA; `num_stages` sweep on the dominant kernel; optional conv3→linear
fusion (only if linear or conv3 shows as a bottleneck and the fused shape stays MMA-efficient).
One variable per candidate; re-check full 16-workload correctness each time.

---

## 3. Correctness checks (every candidate)

Hard gates (candidate invalid if any fails):
1. **All 16 feedback workloads pass** the evaluator's per-workload tolerance
   (`max_atol∈[0.92,1.3]`, `max_rtol=0.05`, `required_match_ratio=0.98`).
2. Output dtype **bf16**, shape exactly `(B, t3, 1024)`.
3. Fully Triton compute — no `F.conv2d/linear/gelu`, no Torch/NumPy/CPU/CUDA-ext fallback.

Derivation self-review before coding/altering each kernel (no external harness allowed):
- conv output length `= ceil(N/2)`; tap `i=2o−1+k`; OOB taps masked to 0 (both H and W axes).
- freq chain `80→40→20→10`; flatten `k=c·10+f`; last odd output column masked correctly.
- linear reduction fp32; `×embed_scale`; `pos[:t3]` add broadcast over batch; `t3≤541<1500` so
  pos slice always in-bounds.
- GELU = exact erf, computed in fp32 on the bf16-rounded conv output (order matches reference).

Regression guard: compare per-workload pass/fail and per-workload speedup vs the current best;
flag any workload that flips to fail or drops >~5% in speedup, especially tiny shapes
(B=1/2, B=64·T=128) which weigh equally in geomean.

---

## 4. Performance hypotheses (ranked, falsifiable)

H1. **Conv2 & conv3 dominate** (≈80–90% of FLOPs); optimizing them drives geomean. *Test*: ncu
   kernel-time breakdown on B=32/T=4328. *Falsified if* linear/conv1 comparable.
H2. **Channels-last** yields the single biggest speedup by making K contiguous for `tl.dot`.
   *Test*: c002 vs c001 geomean.
H3. **Fusion to ~4 launches** removes the eager permute-copy + separate gelu/scale/add overhead,
   helping small/latency-bound shapes most. *Test*: c003 vs c002 on tiny rows.
H4. **Shape-bucketed autotune** recovers MMA utilization on large-B and long-T rows without hurting
   small ones. *Test*: c004/c005 per-workload deltas.
H5. Tiny shapes (B≤2, t3≤131; B=64/T=128) are launch/occupancy bound, not compute bound. *Test*:
   ncu occupancy/achieved-warps on those shapes; low compute utilization ⇒ favor fewer launches +
   lighter kernels over bigger tiles.

Profiling protocol: run `./scripts/ncu_profile.sh --set basic -o profile/<tag> python <harness>`
only when no eval is running; never launch profiling in the background during timing (foreign
process ⇒ controller return code 3, wasted eval). Use ncu-report-skill to read reports. Prefer
one representative large shape + one tiny shape per profiling session.

---

## 5. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- Geomean improvement over the best valid candidate is <~2% across 2 consecutive *distinct*
  optimization attempts (genuine convergence), OR
- ncu shows the dominant kernel(s) near a hardware roofline (compute- or memory-bound) with no
  clear remaining lever, OR
- Approaching budget: token soft limit 9M (begin winding down; prefer only high-confidence
  candidates), normal 10M, hard 11M; or nearing 100 evals (leave margin — never overrun).
Never run `final` without explicit operator approval; keep ≥1 eval of margin for the operator-run
final on the best valid candidate.

Between candidates, if two ideas are close, spend an ncu session (not an eval) to choose, to
conserve the eval budget.

---

## 6. Evidence format

After each `./scripts/evaluate_candidate.sh feedback cNNN`, append **one** complete JSON object
(one line) to `candidates.jsonl` — never rewrite earlier records. Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "…what this candidate changes and why…",
  "change_vs_parent": "single dominant variable changed",
  "validation": {
    "all_pass": true,
    "num_workloads": 16,
    "num_pass": 16,
    "per_workload": [
      {"uuid": "67119fc1-…", "axes": {"batch_size":2,"time_dim":1688,"time_after_conv":211},
       "pass": true, "speedup": 1.00}
    ]
  },
  "geomean_speedup": 1.00,
  "decision": "keep|revert|branch — rationale",
  "cumulative_evals": 1,
  "skills_used": ["ncu-report-skill?"],
  "notes": "profiling findings, next step"
}
```

Rules: record parent, source hash, hypothesis, validation (per-workload pass + speedup), geomean,
decision, cumulative evaluation count, and skill usage for every evaluated candidate. Keep
`solution/solution.py` immutable once its ID is evaluated; a changed source needs a new ID. Store
ncu reports under `profile/<tag>` and reference them from the `notes`/`skills_used` fields.

---

## 7. Immediate next step (next turn)

Implement **c001** (fully-Triton, correctness-first) in `solution/solution.py` per §2, self-review
correctness per §3, then run `./scripts/evaluate_candidate.sh feedback c001` and append its record
to `candidates.jsonl`. Do not evaluate anything in this planning turn.
