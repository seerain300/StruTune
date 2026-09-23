# Executable Optimization Plan — L1/020 Vision Patch Merger (LN + Spatial Shuffle + 2-layer GELU MLP)

Target: **NVIDIA A800 (sm_80, Ampere)**. Compute in **Triton** only; PyTorch used solely for tensor
metadata and launch plumbing. No Torch/CPU/NumPy/CUDA-extension/alternate computational fallback.
This plan operationalizes `docs/draft.md`. It defines the workflow mechanics, the sequential candidate
lineage, correctness checks, performance hypotheses, stopping criteria, and the evidence format.
**No candidate is implemented or evaluated in this turn.**

---

## 0. Ground truth recap (authoritative constants)

Signature (see `task/definition.json`):
```
run(hidden[P,1536] bf16, grid_thw[G,3] int64,
    ln_weight[1536] bf16, ln_bias[1536] bf16,
    fc1_weight[6144,6144] bf16, fc1_bias[6144] bf16,
    fc2_weight[3584,6144] bf16, fc2_bias[3584] bf16,
    eps: float32) -> output[M,3584] bf16
```
Constants: `C=1536`, `merge=2`, `E=6144=4*C`, `O=3584`, `M = P/4 = num_merged`.

Semantic pipeline (must be reproduced exactly):
1. **LayerNorm** per input patch over C=1536, computed in **fp32**, `var` with `unbiased=False` (÷1536),
   affine `xhat*ln_weight + ln_bias` in fp32, cast result to **bf16**. (Generator passes
   `ln_weight=ones`, `ln_bias=zeros`, but implement the **general** affine using the passed tensors.)
2. **Spatial 2×2 shuffle** per grid `(t,h,w)` (h,w even), grids concatenated in input order.
3. **FC1**: `A[M,E] @ fc1_weight[E,E]^T + fc1_bias` → bf16.
4. **GELU exact (erf)**: `0.5*x*(1+erf(x/√2))`.
5. **FC2**: `gelu[M,E] @ fc2_weight[O,E]^T + fc2_bias` → bf16 = output.

### 0.1 Exact shuffle index mapping (implementation contract)
Per grid with dims `(t,h,w)`, `hm=h/2`, `wm=w/2`. For an output row local index
`m_local ∈ [0, t*hm*wm)` decode `(tt, hh_m, ww_m)`:
```
ww_m =  m_local % wm
tmp  =  m_local // wm
hh_m =  tmp % hm
tt   =  tmp // hm
```
For block `β ∈ {0,1,2,3}` set `a = β // 2`, `b = β % 2` (a outer, b inner → β = a*2 + b), then:
```
h_idx    = hh_m*2 + a
w_idx    = ww_m*2 + b
src_local = (tt*h + h_idx)*w + w_idx
src_global = grid_row_offset  + src_local        # input patch row in `hidden`
dst_global = grid_merged_offset + m_local        # output row in A_shuffled
```
Column placement: block β occupies `A_shuffled[dst_global, β*C:(β+1)*C]`.
Grid offsets: `grid_row_offset` = prefix sum of `t*h*w`; `grid_merged_offset` = prefix sum of
`t*hm*wm`. Σ over grids of `t*hm*wm = P/4 = M` by construction. This is a pure **row permutation**
of LN'd patches; because LN is per-patch over C it commutes with the gather.

### 0.2 Feedback workloads (fixed; do not modify)
| # | uuid | P | M | G | atol | rtol |
|--:|------|--:|--:|--:|-----:|-----:|
| 1 | 31eaac46 | 2048 | 512  | 2 | 0.0022 | 0.05 |
| 2 | 24f153b1 | 6400 | 1600 | 1 | 0.0020 | 0.05 |
| 3 | 9376d72b | 1024 | 256  | 4 | 0.0014 | 0.05 |
| 4 | 38e61f30 | 512  | 128  | 2 | 0.0021 | 0.05 |
| 5 | 3b69084d | 4096 | 1024 | 4 | 0.0026 | 0.05 |

M ∈ {512, 1600, 256, 128, 1024}. All divisible by 64; **1600 is NOT divisible by 128** → GEMM must
mask the M tail (never assume M % BLOCK_M == 0). N dims: E=6144 (=48·128), O=3584 (=28·128) both clean.

Roofline (A800 ≈ 312 TFLOPS bf16, ≈ 1.9 TB/s HBM; weights fc1=75.5 MB + fc2=44.0 MB ≈ 119.5 MB):
- **M=128, 256**: memory-bound (weight streaming ≈ 63 µs dominates); baseline **host overhead**
  (per-grid `.item()` syncs, `permute/reshape` copies, `torch.cat`, fp32 LN materialization, many
  kernel launches) is the fat target → biggest expected wins.
- **M=1600, 1024**: compute-bound; the two GEMMs are irreducible. Win here = matching cuBLAS
  throughput with Triton + removing overhead. This is the hardest, highest-risk case for geomean.

---

## 1. Workflow mechanics (how a candidate is built & scored)

1. Source lives at **`solution/solution.py`**, exposing `run(...)` with the exact signature.
2. To evaluate candidate `cNNN`: write/overwrite `solution/solution.py` with that version's source,
   then run **`./scripts/evaluate_candidate.sh feedback cNNN`** (the only permitted correctness+timing
   signal). Five fixed workloads = **one** candidate evaluation.
3. The controller locks a GPU, checks the immutable candidate ID/hash, and runs the official evaluator.
   Once `cNNN` is evaluated, its source hash is locked → **any** source/config/launch change requires a
   **new** ID (`cNNN+1`). Never reuse an ID for changed source; never rewrite earlier `candidates.jsonl`
   records.
4. After each eval, append exactly one JSON record (schema in §6) to `candidates.jsonl`.
5. Constraints honored throughout: no direct CUDA/profiler/`nvidia-smi`/external-evaluator runs; a
   failing Triton kernel is **invalid** (no fallback); stop at budget or convergence; write
   `SEARCH_COMPLETE` on genuine convergence; never run `final` without explicit operator approval.

Budgets: **100** candidate evals; tokens soft **1.0M** / normal **1.5M** / absolute **1.65M**
(includes all input, cache, and output tokens). Token budget is the binding constraint → keep eval
logs and iteration count lean; plan ~10–18 candidates, not 100.

---

## 2. Kernel architecture (shared across candidates)

Three Triton kernels; PyTorch only allocates outputs and computes launch grids/strides.

- **K1 — LN + shuffle → `A_shuffled[M,E]` bf16.** One program per (output row, block β) or per output
  row looping β=0..3. Gather source patch row `src_global`, load 1536 bf16, cast fp32, compute
  mean & var (fp32, ÷1536), `xhat = (x-mean)*rsqrt(var+eps)`, `y = xhat*ln_w + ln_b` (fp32), store
  1536 bf16 contiguously at columns `[β*C:(β+1)*C]`. Preferred layout: **one program per output row**,
  contiguous 6144-wide store, 4 gathered strided loads; per-program working set ≤ ~6 KB (loop β).
- **K2 — FC1 GEMM + bias + GELU.** `A[M,E] @ fc1_weight[E,E]^T + fc1_bias`, **fp32 accumulator**,
  bias epilogue, **erf** GELU, output bf16. Tiled `tl.dot`; M-tail masked.
- **K3 — FC2 GEMM + bias.** `gelu[M,E] @ fc2_weight[O,E]^T + fc2_bias`, fp32 accumulator, bias
  epilogue, output bf16. M-tail masked.

Index construction (choose per candidate, both are "plumbing"):
- **Option A (default, c001):** one tiny `grid_thw.to('cpu')` transfer (G×3 int64, G≤4), compute
  prefix sums + a compact `src_base`/index form on host, upload once. Single small sync — orders of
  magnitude cheaper than the reference's `3·G` per-grid `.item()` syncs.
- **Option B (later refinement):** on-GPU index kernel scanning `grid_thw` (G≤4) → zero host sync.
  Adopt only if evidence shows the Option-A transfer is non-negligible.

Fusion boundaries: FC1 and FC2 cannot fuse (nonlinearity + different K/N). Keep K2, K3 separate.
Materializing `A_shuffled` in K1 then feeding K2 is preferred over gathering inside the FC1 prologue
(LN needs the full 1536 row; 1536 is not a power of two). The extra A write/read (~19.6 MB at M=1600)
is negligible vs 191 GFLOP. A prologue-fused K1+K2 is an **optional late experiment** only.

---

## 3. Numerical correctness contract (must hold for every candidate)

1. LN mean/var in **fp32**, `unbiased=False` (÷1536), affine in fp32, output cast to bf16. bf16
   accumulation is forbidden (would break atol).
2. GEMMs use **fp32 accumulator**, bf16 inputs/outputs (matches cuBLAS accumulation).
3. **GELU = exact erf** (`tl.math.erf`/libdevice). Do **not** use tanh-approx unless a candidate
   proves it stays within the tightest atol (0.0014); erf is the reference and the safe default.
4. **Rounding order FC1→GELU:** reference rounds FC1 to bf16 (cuBLAS bf16 output) *before* GELU.
   Default replicates this (cast acc+bias to bf16, then GELU). A pure-fp32-epilogue variant is an
   A/B experiment, not the default.
5. Output dtype bf16, round-to-nearest (Triton default).
6. Fully **general** grid handling: `G ∈ {1,2,4}`, arbitrary even `(h,w)`, `t≥1`; correct M-tail
   masking in both GEMMs (esp. M=1600).
7. Determinism: the correctness-first path uses **no** split-K atomics. If split-K (fp32 atomic add)
   is later adopted for speed, re-verify all five tolerances because summation reordering perturbs bits.

**Static pre-eval checklist (run mentally before every `evaluate_candidate.sh`):** signature & dtypes
exact; shuffle indices match §0.1; LN in fp32/÷1536; erf GELU; both biases applied; M-tail masked;
general G/(t,h,w); no Torch compute path; kernel launches sized from shapes (no needless sync).

---

## 4. Candidate lineage (sequential, adaptive)

Each candidate is one immutable source version with one primary hypothesis. Advance the tree along
whichever branch the evidence favors; abandon regressions. IDs are assigned in evaluation order;
the exact set past c003 adapts to results (planned nodes below are the intended search order).

```
c001 (correctness anchor)
 └─ c002 (broaden GEMM autotune space)         ← main throughput lever
     ├─ c003 (small-M path: split-K / small blocks for M=128,256)
     ├─ c004 (K1 LN+shuffle vectorization / layout)
     ├─ c005 (epilogue A/B: rounding order; erf vs tanh within tol)
     ├─ c006 (GROUP_M / L2 swizzle & stream-k / persistent GEMM)
     ├─ c007 (on-GPU index kernel, Option B)   ← only if A-transfer shows cost
     └─ c008 (optional K1→FC1 prologue fusion)
 └─ … refinements combining the winning knobs …
```

### c001 — Correctness-first anchor (parent: none)
- **Build:** K1 (per-output-row, fp32 LN, erf), K2 (FC1: BM128/BN128/BK64, warps8, stages3, fp32
  accum, bias, bf16-before-GELU), K3 (FC2: BM128/BN128/BK64, warps8, stages3, bias). Index Option A.
  A small `@triton.autotune` set (2–4 safe configs) keyed on M is acceptable but keep it minimal to
  bound compile/eval time.
- **Hypothesis:** correct on all 5 workloads; establishes anchor per-workload speedups + geomean.
  Expect ≥1× on small M (overhead removal), unknown on M=1600.
- **Pass/fail gate:** all 5 correct → anchor accepted. Any failure → treat as diagnostic, fix
  semantics in **c002** (new ID), never patch c001 in place.

### c002 — Broaden GEMM autotune space (parent: c001)
- **Change:** expand `@triton.autotune` configs for K2/K3: `BLOCK_M∈{64,128}`, `BLOCK_N∈{64,128,256}`,
  `BLOCK_K∈{32,64}`, `num_warps∈{4,8}`, `num_stages∈{3,4,5}`, `GROUP_M∈{4,8}`; key = `(M-bucket)`.
- **Hypothesis:** autotune closes most of the cuBLAS gap on compute-bound M=1024/1600 and improves
  small M. Primary geomean lever.
- **Validation:** all 5 pass; geomean ≥ c001. Keep the winning config family for the branch.

### c003 — Small-M throughput (parent: best of c001/c002)
- **Change:** dedicated small-M strategy for M=128/256 — either finer blocks (raise tile count toward
  108 SMs) or **split-K** (2–4 way) to fill SMs; if split-K uses fp32 atomics, verify tolerance.
- **Hypothesis:** M=128/256 are memory-bound & under-occupied (M=128, BN=128 → ~48 programs); more
  concurrency raises their speedup, which dominate the geomean.
- **Validation:** all 5 pass (re-check atol after any atomic split-K); geomean ≥ parent.

### c004 — K1 LN+shuffle tuning (parent: best so far)
- **Change:** vectorized 1536-wide load/store, reduction tiling (`1536=3·512`), one-pass sum/sum(x²)
  vs two-pass, block-β layout tweaks; consider one-program-per-(row,β).
- **Hypothesis:** K1 is memory-bound and cheap vs GEMMs; only matters at small M — small but free win.
- **Validation:** all 5 pass; geomean ≥ parent (accept even flat if it simplifies later fusion).

### c005 — Epilogue numerics A/B (parent: best so far)
- **Change:** test pure-fp32 FC1 epilogue (GELU on fp32, round after) vs default; optionally tanh-GELU.
- **Hypothesis:** rounding-order/GELU-form is perf-neutral but may free a faster epilogue; keep only
  if **within all tolerances** and not slower.
- **Validation:** all 5 pass with margin; adopt only if strictly non-harmful.

### c006 — L2 swizzle / persistent / stream-k GEMM (parent: best so far)
- **Change:** GROUP_M swizzle tuning; optionally a persistent or stream-k GEMM for better SM residency
  on large M.
- **Hypothesis:** improves L2 reuse of the shared weight matrices and tail efficiency on M=1600.
- **Validation:** all 5 pass; geomean ≥ parent.

### c007 — On-GPU index kernel, Option B (parent: best so far)
- **Change:** replace host `grid_thw.to(cpu)` with a device index kernel (scan G≤4).
- **Hypothesis:** removes the last host sync; likely negligible but confirm.
- **Validation:** all 5 pass; adopt only if geomean improves or ties.

### c008 — Optional K1→FC1 prologue fusion (parent: best so far)
- **Change:** gather+LN normalized rows inside the FC1 prologue, skipping A materialization (handle the
  1536/non-power-of-two reduction carefully).
- **Hypothesis:** saves one A round-trip; matters only at small M; higher complexity/risk.
- **Validation:** all 5 pass; adopt only on clear net win.

**Combination pass:** after individual knobs are characterized, produce a final candidate merging the
best-performing settings; re-verify all 5.

---

## 5. Performance hypotheses & decision logic

- **H1 (overhead removal):** fused Triton pipeline removes host syncs, extra copies (`cat`, permute),
  fp32 LN materialization, and launch overhead → large speedups on M=128/256 (memory/overhead-bound).
- **H2 (GEMM parity):** autotuned Triton GEMM reaches ~0.85–1.0× cuBLAS on compute-bound M=1024/1600;
  the deficit (if any) is the main risk to geomean.
- **H3 (geomean composition):** with 3 small/medium workloads (128,256,512) and 2 large (1024,1600),
  strong small-M wins can carry the geomean even if large-M sits slightly under cuBLAS.
- **Decision rule per candidate:** *accept & set as new branch parent* iff all 5 pass AND geomean ≥
  parent (ties broken toward simpler/lower-risk source). *Reject* (keep parent) otherwise. A candidate
  that fails any correctness check is **invalid** regardless of speed.
- **Attribution:** always read per-workload speedups, not just geomean, to know which axis moved and
  where headroom remains (esp. compare M=1600 vs cuBLAS gap against small-M gains).

---

## 6. Evidence format (one JSON object per evaluated candidate → `candidates.jsonl`)

Append-only; never edit prior lines. Required fields (satisfies CLAUDE.md §7):
```json
{
  "candidate_id": "c001",
  "parent_id": null,
  "source_sha256": "<sha256 of solution/solution.py at eval time>",
  "timestamp": "2026-09-17T..Z",
  "hypothesis": "correctness-first anchor: fused LN+shuffle, Triton FC1(GELU)+FC2",
  "changes_vs_parent": "initial implementation",
  "validation": {
    "all_correct": true,
    "static_checklist_passed": true,
    "notes": "erf GELU, fp32 LN, M-tail masked, index option A"
  },
  "per_workload": [
    {"uuid": "31eaac46", "M": 512,  "correct": true, "speedup": 0.0, "baseline_ms": 0.0, "candidate_ms": 0.0},
    {"uuid": "24f153b1", "M": 1600, "correct": true, "speedup": 0.0, "baseline_ms": 0.0, "candidate_ms": 0.0},
    {"uuid": "9376d72b", "M": 256,  "correct": true, "speedup": 0.0, "baseline_ms": 0.0, "candidate_ms": 0.0},
    {"uuid": "38e61f30", "M": 128,  "correct": true, "speedup": 0.0, "baseline_ms": 0.0, "candidate_ms": 0.0},
    {"uuid": "3b69084d", "M": 1024, "correct": true, "speedup": 0.0, "baseline_ms": 0.0, "candidate_ms": 0.0}
  ],
  "geomean_speedup": 0.0,
  "decision": "accept|reject|invalid",
  "cumulative_evals": 1,
  "skill_usage": "none (KernelWiki N/A: Ampere/sm_80, not Blackwell/Hopper)"
}
```
- `speedup`/`ms` values are filled from the evaluator output for that run (0.0 placeholders above are
  schema illustration only). If the evaluator does not print per-workload ms, record whatever it does
  report (speedup and correctness at minimum) and note the source.
- `decision` reflects §5 rule; `cumulative_evals` increments by 1 each candidate (5 workloads = 1 eval).

---

## 7. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
1. **Convergence:** best geomean improves < ~2% across 3 consecutive accepted candidates, and remaining
   planned axes are exhausted or expected to be within noise.
2. **Budget:** approaching the token soft limit (1.0M) with no promising untested axis, or nearing the
   100-eval cap (unlikely given token budget binds first).
3. **Ceiling reached:** large-M GEMM is at cuBLAS parity and small-M is occupancy-bound with no further
   structural lever — diminishing returns confirmed by evidence.

On stop: ensure the best valid candidate's source is the current `solution/solution.py` state
described in its record, document the chosen candidate and rationale, and **do not** run `final`
(operator-only; requires explicit approval).

---

## 8. Skill usage

- **KernelWiki:** not applicable — it covers Blackwell (SM100/B200) and Hopper (SM90/H100); target is
  **A800 / sm_80 (Ampere)**. Standard Ampere Triton GEMM/reduction practice applies instead.
- No other external knowledge sources are permitted. Optimization relies on the evaluator feedback loop
  and the reasoning in this plan / `docs/draft.md`.

---

## 9. Immediate next action (next turn, not now)

Implement **c001** exactly per §2–§3 (correctness-first anchor), run the static checklist (§3), then
`./scripts/evaluate_candidate.sh feedback c001`, and append its record (§6). Do not implement or
evaluate anything this turn.
