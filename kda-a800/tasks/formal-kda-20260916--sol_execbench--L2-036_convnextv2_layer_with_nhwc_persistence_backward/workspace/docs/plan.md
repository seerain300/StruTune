# Plan — L2/036 ConvNextV2 Layer (NHWC persistence) Backward

Executable, sequential KDA optimization plan. Derived from `TASK.md`, `task/definition.json`,
`task/feedback_workloads.jsonl`, and `docs/draft.md`. **This turn produces the plan only — no code,
no evaluation.**

Target HW: **NVIDIA A800, `sm_80` (Ampere)**, dtype **float32**. Submission: `solution/solution.py`
exposing `run(...)` with the exact signature from `definition.json`. Compute must be **Triton**;
PyTorch only for allocation/metadata/launch. **No Torch/CPU/NumPy/cuDNN compute fallback** — a failing
Triton path is invalid, never substituted.

> **Skill usage note.** `KernelWiki` targets Blackwell (sm_100)/Hopper (sm_90); this task is Ampere
> (sm_80), so it is **out of scope** and will not be invoked unless a genuinely portable idea (e.g. a
> generic Triton GEMM/reduction pattern) is needed — in which case its use will be recorded in the
> candidate record's `skill_usage` field. No profiler / `nvidia-smi` / direct CUDA / alternate harness
> is permitted; the only trusted signal is `./scripts/evaluate_candidate.sh feedback cNNN`.

---

## 0. Objective & metric

- **Primary metric:** geometric-mean speedup over the reference across the **5 fixed feedback workloads**,
  gated on **every** workload passing correctness. A workload that fails correctness invalidates the
  candidate for ranking.
- **Correctness rule (assumed, confirm on c001):** per output tensor, elementwise
  `|a − ref| ≤ max_atol + max_rtol·|ref|` must hold for `≥ required_match_ratio` (0.98) of elements,
  for **all 11** outputs. Per-workload tolerances:

  | id | B | H | W | M=B·H·W | HW | max_atol | max_rtol | match |
  |----|---|---|---|---------|------|----------|----------|-------|
  | W1 | 2 | 28 | 28 | 1,568  | 784  | 0.68 | 1e-3 | 0.98 |
  | W2 | 1 | 56 | 56 | 3,136  | 3136 | 0.75 | 1e-3 | 0.98 |
  | W3 | 16| 56 | 56 | 50,176 | 3136 | 5.2  | 1e-3 | 0.98 |
  | W4 | 16| 14 | 14 | 3,136  | 196  | 0.96 | 1e-3 | 0.98 |
  | W5 | 4 | 56 | 56 | 12,544 | 3136 | 1.3  | 1e-3 | 0.98 |

- W3 dominates the geomean cost (compute/BW bound); W1/W4 are small (launch/latency bound). One
  immutable launch config must serve all 5 → autotune keyed on shape.

---

## 1. Budget & discipline

- **Evaluation budget:** 100 feedback evals. Target using **≤ ~12** (one per candidate; do not use evals
  as an interactive debugger). Every eval = all 5 workloads = 1 count.
- **Token budget:** soft 1,000,000 / normal 1,500,000 / absolute 1,650,000 (includes all input/cache/output).
  Prefer surgical edits and reading only needed file regions.
- **Immutability:** one source version per candidate id. Any meaningful source/config/launch change ⇒ new
  id. Never reuse an id for changed source; never rewrite earlier `candidates.jsonl` records.
- **`final`** (14-workload) is operator-approval-only; never auto-run.

### 1.1 Candidate source & immutability mechanism (executable)
For each candidate `cNNN`:
1. Author/replace `solution/solution.py` with the immutable source for `cNNN`.
2. Archive a verbatim copy at `runs/candidates/cNNN/solution.py` **before** evaluating, and record its
   `sha256` in the candidate record (`git hash-object`-style / `sha256sum`), so the exact source behind
   every record is recoverable and later edits to `solution/solution.py` cannot silently mutate history.
3. Evaluate with `./scripts/evaluate_candidate.sh feedback cNNN`.
4. Append exactly one JSON object to `candidates.jsonl` (schema in §7).
5. Only then begin the next candidate. Sequential, one source version at a time.

> The trusted controller locks the GPU, rejects foreign processes, and checks immutable id/hash; we do
> not inspect or modify it, the evaluator, dataset, launcher, or shared config.

---

## 2. Reference math to replicate (authoritative summary)

Reverse order; must be bit-for-bit in intent, including two **non-standard** steps. Uses the **provided**
forward intermediates (never recompute forward). `M = B·H·W`, LN reduces over `C=128`, GRN spatial norm
reduces over `HW` per `(b, c4)`, `C4=512`, dwconv 7×7 pad 3 groups C.

1. **Residual/drop-path split:** `grad_residual = grad_output` (raw); `grad_x_nchw = grad_output·drop_mask/keep_prob`, `keep_prob=0.9`.
2. **Permute NCHW→NHWC:** `grad_x_projected[b,h,w,c] = grad_x_nchw[b,c,h,w]` → `(M,C)`.
3. **pwconv2 bwd:** `grad_x_grn = grad_x_projected @ pwconv2_weight` `(M,C4)`; `grad_pwconv2_weight = grad_x_projectedᵀ @ x_grn` `(C,C4)`; `grad_pwconv2_bias = grad_x_projected.sum(M)` `(C,)`.
4. **GRN bwd:** `grad_x_grn_scaled = grad_x_grn·grn_weight`; `grad_grn_weight = (grad_x_grn·x_grn_scaled).sum(0,1,2,keepdim)` `(1,1,1,C4)`; `grad_grn_bias = grad_x_grn.sum(0,1,2,keepdim)` `(1,1,1,C4)`; `grad_norm_features = (grad_x_grn_scaled·x_gelu).sum(1,2,keepdim)` `(B,1,1,C4)`; `grad_x_gelu = grad_x_grn + grad_x_grn_scaled·norm_features`.
5. **norm_features = global_features/(gf_mean+eps)** — ⚠️ **non-standard, do NOT sum over channels:**
   `grad_global_features = grad_norm_features/(gf_mean+eps)`;
   `grad_gf_mean = −grad_norm_features·global_features/(gf_mean+eps)²` (kept elementwise `(B,1,1,C4)`);
   `grad_global_features += grad_gf_mean / C4` (per-`(b,c4)`, each channel gets only its own term).
6. **global_features = ‖x_gelu‖₂ over (H,W):** `grad_x_gelu += x_gelu·grad_global_features/(global_features+eps)` (broadcast over H,W).
7. **GELU (tanh) bwd:** `k0=0.7978845608028654`, `k1=0.044715`; `inner=k0·(x_expanded+k1·x_expanded³)`; `t=tanh(inner)`; `cdf=0.5(1+t)`; `pdf=0.5(1−t²)·k0·(1+3k1·x_expanded²)`; `gelu_grad=cdf+x_expanded·pdf`; `grad_x_expanded=grad_x_gelu·gelu_grad`. Single `tanh` reused in cdf & pdf.
8. **pwconv1 bwd:** `grad_x_ln = grad_x_expanded @ pwconv1_weight` `(M,C)`; `grad_pwconv1_weight = grad_x_expandedᵀ @ x_ln` `(C4,C)`; `grad_pwconv1_bias = grad_x_expanded.sum(M)` `(C4,)`.
9. **LN affine bwd:** `grad_x_normalized = grad_x_ln·layernorm_weight`; `grad_layernorm_weight = (grad_x_ln·x_normalized).sum(M)` `(C,)`; `grad_layernorm_bias = grad_x_ln.sum(M)` `(C,)`.
10. **LN normalization bwd** (`N=C=128`, per-token over channel): `std=sqrt(var+eps)`;
    `grad_x_nhwc = grad_x_normalized/std`;
    `grad_var = −(grad_x_normalized·(x_nhwc−mean)).sum(−1,keepdim)/(2·(var+eps)·std)`;
    `grad_mean = −(grad_x_normalized/std).sum(−1,keepdim)`;
    `grad_mean += grad_var·(−2·(x_nhwc−mean).sum(−1,keepdim)/N)` (**keep** this ≈0 term);
    `grad_x_nhwc += grad_var·(2·(x_nhwc−mean)/N)`; `grad_x_nhwc += grad_mean/N`.
11. **Permute NHWC→NCHW:** `grad_x_dwconv[b,c,h,w] = grad_x_nhwc[b,h,w,c]`.
12. **Depthwise conv bwd** (7×7, pad 3, stride 1, groups C):
    - **input grad:** `grad_x[b,c,iy,ix] = grad_output[b,c,iy,ix] + Σ_{i,j=0..6} grad_x_dwconv[b,c,iy−i+3,ix−j+3]·W[c,0,i,j]` (valid taps only; **kernel flipped** vs forward).
    - **weight grad:** `grad_dwconv_weight[c,0,i,j] = Σ_{b,oy,ox} grad_x_dwconv[b,c,oy,ox]·residual[b,c,oy+i−3,ox+j−3]` (valid positions; residual zero-padded by 3).
    - **bias grad:** `grad_dwconv_bias[c] = grad_x_dwconv.sum(0,2,3)` `(C,)`.

**Top functional risk:** the two conv directions' flip/offset (§12) and the non-standard `gf_mean` step (§5).

Output order returned by `run`: `(grad_x, grad_dwconv_weight, grad_dwconv_bias, grad_layernorm_weight,
grad_layernorm_bias, grad_pwconv1_weight, grad_pwconv1_bias, grad_grn_weight, grad_grn_bias,
grad_pwconv2_weight, grad_pwconv2_bias)`. Shapes/dtypes per §1.1 of the draft, including keepdim
`grad_grn_weight/bias (1,1,1,C4)`.

---

## 3. Kernel decomposition (baseline for c001)

Temporaries in NHWC-flat contiguous `(M, ·)` buffers so GEMMs are row-major. Allocate with
`torch.empty`/`zeros`; all compute in Triton.

- **K1 — permute + drop-path + pwconv2 bias:** produce `grad_x_projected (M,C)` from `grad_output` (NCHW)
  with `·drop_mask[b]/keep_prob`; accumulate `grad_pwconv2_bias[c]` over M.
- **K2 — GEMM** `grad_x_grn = grad_x_projected @ pwconv2_weight` `(M,C4)`, fp32 accumulate.
- **K3 — GEMM** `grad_pwconv2_weight = grad_x_projectedᵀ @ x_grn` `(C,C4)`, K=M.
- **K4 — GRN reduce (pass 1):** per-`(b,c4)` spatial sums → `grad_norm_features (B,C4)`; global sums →
  `grad_grn_weight (C4)`, `grad_grn_bias (C4)`; then compute `grad_global_features (B,C4)` (§2.5 non-standard).
- **K5 — GRN apply + GELU (pass 2):** `grad_x_gelu` (residual + scaled + L2-norm paths) then GELU-grad →
  `grad_x_expanded (M,C4)`; optionally accumulate `grad_pwconv1_bias (C4)` here.
- **K6 — GEMM** `grad_x_ln = grad_x_expanded @ pwconv1_weight` `(M,C)`.
- **K7 — GEMM** `grad_pwconv1_weight = grad_x_expandedᵀ @ x_ln` `(C4,C)`, K=M.
- **K8 — LN affine + normalization bwd:** per-token load C=128, accumulate `grad_layernorm_weight/bias`
  over M, emit `grad_x_nhwc (M,C)` via §2.9–2.10. C=128 fits one row-block → single-pass row kernel.
- **K9 — depthwise input-grad stencil:** `grad_x = stencil(grad_x_nhwc→NCHW, W_flip) + grad_output`.
- **K10 — depthwise weight-grad + bias:** `grad_dwconv_weight (C,49)`, `grad_dwconv_bias (C)`.

All `tl.dot` start with `input_precision="ieee"` (true fp32) in c001. Reductions accumulate in fp32,
deterministic where cheap. Channel axes (C=128, C4=512) are clean multiples of 32/64/128 → simple masking;
mask the M tail and conv boundaries.

---

## 4. Sequential candidate roadmap

Change **one lever per candidate**. Accept a candidate as the new lineage parent only if it (a) passes all
5 workloads and (b) does not regress geomean vs the current best (for speed candidates: improves geomean).
Record parent, hypothesis, result, decision for each.

| id | parent | lever / change | hypothesis | accept rule |
|----|--------|----------------|-----------|-------------|
| **c001** | — | Correct modular port (K1–K10), `ieee` dot, deterministic reductions, conservative blocks | Passes all 5; already > reference by killing the 128-iter Python unfold loop | all 5 pass ⇒ baseline; record geomean |
| **c002** | best | Depthwise weight-grad → split over (channel, spatial-block) + atomic/scratch reduce | W3 weight-grad is serialized in c001; parallel split raises W3 throughput | all 5 pass AND geomean ↑ |
| **c003** | best | GEMM autotune: block sizes, `num_warps∈{4,8}`, `num_stages∈{2,3,4}`, grouped-M ordering | Better L2 reuse / occupancy on the 4 GEMMs, esp. W3/W5 | all 5 pass AND geomean ↑ |
| **c004** | best | Fuse elementwise chains into GEMM epilogues/prologues (drop-path+permute into K2 A-load; bias into GEMM epilogues; GELU/GRN into K6 A-load; LN-affine into K8) | Memory-bound elementwise passes cut DRAM traffic | all 5 pass AND geomean ↑ |
| **c005** | best | Depthwise input-grad tuning (spatial tile size, cache 49 weights in regs/shared, coalesced W) | K9 stencil BW on large HW (W2/W3/W5) | all 5 pass AND geomean ↑ |
| **c006** | best | Try `input_precision="tf32"` on GEMMs (guarded) | Faster TensorCore GEMM; risk: `rtol=1e-3` on small W1/W2/W4 | keep ONLY if all 5 still pass AND geomean ↑; else revert to parent |
| **c007** | best | Reduce kernel count / fuse small kernels for W1/W4 launch-overhead path | Small workloads are launch-bound; fewer launches lowers latency | all 5 pass AND geomean ↑ |
| **c008+** | best | Remaining autotune/fusion refinements as evidence directs (e.g. split-K for K3/K7 on W3, single-config tuning per shape via autotune keys) | Close remaining gap on the dominant/critical workload | all 5 pass AND geomean ↑ |

Notes:
- The roadmap is a priority order, not a fixed count. Re-prioritize after each eval based on which
  workload/tensor the evidence flags. Skip a planned lever if a prior eval shows it is unlikely to help.
- If a speed candidate fails correctness, it is invalid; revert lineage to the last accepted parent and
  either fix (new id) or move to the next lever.
- Never introduce a Torch/CPU compute fallback to "rescue" a failing Triton kernel.

---

## 5. Correctness checks

### 5.1 Pre-eval self-checks (by construction, before spending any eval)
1. **Signature & outputs:** `run(...)` matches the exact arg list/order; returns the 11 outputs in the
   order of §2, each with the exact shape/dtype (incl. keepdim `(1,1,1,C4)` for grad_grn_*, `(C,)` biases).
2. **Shape/stride table:** every kernel's I/O pointer, stride, and mask matches the NHWC-flat/NCHW layout;
   M-tail and channel masks present; conv boundary masks present.
3. **Math line-by-line vs §2:** including the two non-standard steps (§2.5 unsummed `gf_mean`; §2.10 kept
   `(x−mean).sum` term) and the single reused `tanh` in GELU with exact `k0`,`k1`.
4. **Provided intermediates used directly** (`mean`,`var`,`global_features`,`gf_mean`,`norm_features`,
   `x_grn_scaled`,`x_normalized`,`x_ln`,`x_expanded`,`x_gelu`) — no forward recomputation.
5. **Depthwise flip/offset cross-check:** input-grad uses flipped kernel `iy−i+3`; weight-grad correlates
   with residual at `oy+i−3`; forward is `out[oy]=Σ in_pad[oy+i]·W[i]`. Verified by construction.
6. **fp32 accumulation** in all `tl.dot` and all reduction accumulators.
7. **Triton-only compute:** no `F.linear`/`F.conv*`/`torch.tanh`/CPU/NumPy in the compute path;
   PyTorch limited to alloc/metadata/launch.

### 5.2 Post-eval triage (map failure → suspect)
- `grad_x` and/or `grad_dwconv_weight` fail only → depthwise indexing/flip/offset (§2.12).
- `grad_grn_*` / `grad_x`-chain fail → non-standard `gf_mean` step (§2.5) done the textbook way.
- Broad small-magnitude misses across many tensors (esp. W1/W2/W4, i.e. tight relative error) → TF32 in a
  `tl.dot`; switch that dot to `ieee`.
- A single tensor fails on **all** elements → wrong constant / wrong transpose / wrong reduction axis
  (0.98 match cannot save a systematic error).
- LN-derived (`grad_layernorm_*`, `grad_x_nhwc`) fail → dropped/altered LN term (§2.10) or wrong `N`.
- Only large workloads (W3) fail on reductions → accumulation precision; use fp32 partials / split-K.

### 5.3 Confirm-once assumptions (on c001)
- The exact evaluator match rule (`|a−ref| ≤ atol + rtol·|ref|`, ≥98%). Adjust interpretation if c001's
  reported diagnostics differ.
- Whether `ieee` is required or `tf32` is admissible per workload (drives c006 decision).

---

## 6. Performance hypotheses & measurement

- **H1 (primary):** Eliminating the reference's Python `for g in range(C)` unfold weight-grad loop with one
  fused Triton kernel yields a large speedup on all workloads, dominant on W3. → c001 should already beat
  the reference geomean.
- **H2:** Parallelizing depthwise weight-grad across spatial blocks (c002) improves W3 (M≈50k) most.
- **H3:** GEMM autotune + grouped-M (c003) improves the four GEMMs on compute-bound W3/W5.
- **H4:** Fusing elementwise/reduction chains into GEMM epilogues/prologues (c004) cuts DRAM traffic on the
  memory-bound elementwise stages (helps all, esp. large HW).
- **H5:** TF32 (c006) speeds GEMMs but may break `rtol=1e-3` on small workloads — accept only if all pass.
- **H6:** For W1/W4 (launch-bound), fewer/fused kernels reduce latency (c007).
- **Measurement:** the evaluator's per-workload speedup and geomean are the sole signal. Compare each
  candidate's geomean to the current best; keep only monotone improvements as the lineage parent. No
  profiler/ncu available, so reason about BW/compute bounds analytically to choose the next lever.

---

## 7. Evidence format (`candidates.jsonl`)

Append exactly one JSON object per evaluated candidate (never rewrite prior lines). Schema:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "timestamp": "2026-09-17T00:00:00Z",
  "source_path": "runs/candidates/c001/solution.py",
  "source_sha256": "<sha256 of the archived immutable source>",
  "hypothesis": "Correct modular Triton port (K1-K10), ieee dot; beats reference by removing the 128-iter Python unfold weight-grad loop.",
  "lever": "baseline correctness port",
  "validation": {
    "pre_eval_selfcheck": "passed (signature/shapes/math/flip/fp32/triton-only)",
    "assumptions_confirmed": ["match_rule", "ieee_needed_or_tf32_ok"]
  },
  "per_workload": [
    {"id": "W1", "uuid": "ad073c89-...", "B": 2,  "H": 28, "W": 28, "pass": true, "speedup": 0.0, "failing_outputs": []},
    {"id": "W2", "uuid": "0fcdcad9-...", "B": 1,  "H": 56, "W": 56, "pass": true, "speedup": 0.0, "failing_outputs": []},
    {"id": "W3", "uuid": "812896b7-...", "B": 16, "H": 56, "W": 56, "pass": true, "speedup": 0.0, "failing_outputs": []},
    {"id": "W4", "uuid": "817ddfe6-...", "B": 16, "H": 14, "W": 14, "pass": true, "speedup": 0.0, "failing_outputs": []},
    {"id": "W5", "uuid": "5395f6da-...", "B": 4,  "H": 56, "W": 56, "pass": true, "speedup": 0.0, "failing_outputs": []}
  ],
  "all_pass": true,
  "geomean_speedup": 0.0,
  "decision": "accept-as-parent | reject-regression | reject-incorrect | revert",
  "decision_reason": "all 5 pass; new best geomean",
  "cumulative_evals": 1,
  "skill_usage": "none (Ampere sm_80; KernelWiki out of scope)",
  "notes": "raw evaluator fields captured verbatim where reported"
}
```

- Fill `speedup`/`geomean_speedup`/`failing_outputs` from the evaluator's actual report (values above are
  placeholders). Capture any raw evaluator diagnostics verbatim in `notes`.
- `cumulative_evals` is the running count of feedback evaluations used (each candidate = +1).
- `decision` records whether the candidate becomes the new lineage parent.

---

## 8. Stopping / convergence criteria

Stop and write `SEARCH_COMPLETE` (with the reason) when any holds:
1. **Convergence:** best geomean does not improve by a meaningful margin (e.g. < ~2%) across 2–3
   consecutive accepted-or-attempted speed candidates, and remaining levers are judged low-value.
2. **Budget:** approaching the token soft/normal limit or the 100-eval cap (leave margin; never exceed).
3. **Ceiling reached:** candidates are bounded by cuBLAS/cuDNN-competitive GEMM/stencil throughput with no
   correctness headroom for further precision/fusion tricks.

On stop: ensure the best valid candidate is the current `solution/solution.py` and its record is the
accepted parent; write `SEARCH_COMPLETE` naming the best candidate id, its geomean, and the stop reason.
**Never run `final` without explicit operator approval.**

---

## 9. Immediate next actions (next turn, after this plan)

1. Implement **c001** per §3 (all Triton), run the §5.1 self-checks by construction.
2. Archive `runs/candidates/c001/solution.py`, record its sha256.
3. Evaluate: `./scripts/evaluate_candidate.sh feedback c001`.
4. Append the c001 record to `candidates.jsonl` (§7); triage per §5.2 if any workload fails.
5. Proceed to c002+ per the §4 roadmap, one lever and one id at a time.
