# Plan: L1/002 — VAE fused residual block (Conv3x3 → GroupNorm → SiLU, ×2, + residual)

Task id: `sol_execbench / L1 / 002_vae_conv3x3_groupnorm_silu_residual_fused`
Target HW: **NVIDIA A800, `sm_80` (Ampere)**. Framework: **Triton** for all compute; PyTorch only for metadata/allocation/launch.
Basis: this plan operationalizes `docs/draft.md` (§5 design space, §7 validation) into an ordered, immutable candidate roadmap.

This document is the executable roadmap only. **No solution code and no evaluation is produced in this turn.** Implementation of `c001` begins in the next turn.

---

## 0. Objective, metric, and hard rules (restated for execution)

- **Deliverable:** `solution/solution.py` exposing `run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)` returning `(B,C,H,W)` float32, computed by Triton kernels only.
- **Ranking metric:** geometric-mean speedup over the 5 feedback workloads vs the evaluator's PyTorch reference. **Every selected workload must pass correctness** (`|a-b| ≤ max_atol + max_rtol·|b|`, per-workload `max_atol ∈ [0.0028, 0.0034]`, `max_rtol = 1e-5`). A single correctness failure ⇒ candidate invalid regardless of speed.
- **Evaluation:** only `./scripts/evaluate_candidate.sh feedback <id>`; the 5 workloads together = **one evaluation**. Budget **100 evals**; token soft **1.0M** / normal **1.5M** / absolute **1.65M**. `final` (20 workloads) is **operator-approval-only** — never run without explicit approval.
- **Immutability:** candidates `c001, c002, …` are sequential and immutable; any meaningful source/config/launch change ⇒ new id; never reuse an id for changed source; `candidates.jsonl` is append-only.
- **No fallback of any kind** (Torch/CPU/NumPy/CUDA-extension/alternate-impl compute). A failing Triton kernel is an invalid candidate — diagnose and fix in Triton, never patch with a Torch path.
- **No local execution** of CUDA/Triton/torch/profilers/`nvidia-smi`/alternate harness. The evaluator is the **sole oracle** for correctness and speed.
- **Skill usage:** `KernelWiki` (Blackwell/Hopper) does not apply to Ampere `sm_80`; profilers are prohibited. Every candidate record will list `skill_usage: "none"` with that justification.

---

## 1. Kernel decomposition (reference → Triton)

Reference pipeline (7 math stages): `conv1 → gn1 → silu1 → conv2 → gn2 → silu2 → +residual`.

Triton kernels to build (all FP32 storage for intermediates; see draft §4.3):

- **`K_conv`** — generic implicit-GEMM 3×3 conv, `stride=1, padding=1`, NCHW in/out, FP32 accumulate. Reused for conv1 and conv2. Per draft §5.1: output tile `[BLOCK_M pixels, BLOCK_N channels]`, accumulate over 9 taps, each an FP32-accumulating `tl.dot` over `K_cin = 256` (optionally 2×128). Padding via masked input loads. `input_precision` is a build-time knob (`"ieee"` vs `"tf32"`).
- **`K_gn_stats`** — per-`(b,g)` (b∈[0,B), g∈[0,32)) mean and `rstd = rsqrt(var+eps)`. Reduces over `C/G=8` channels × H × W = `8·H·W` elements. **Biased (population) variance** to match PyTorch. Two-pass or Welford in FP32 (draft §4.2). Writes `mean[B,32]`, `rstd[B,32]`.
- **`K_gn_apply_silu`** — elementwise `y = silu((v - mean_g)·rstd_g·weight[c] + bias[c])`, fully parallel; a variant `K_gn_apply_silu_add` also adds the residual `x`.

**c001 pipeline (6 launches):**
1. `t1 = K_conv(x, conv1_weight)`
2. `(mean1,rstd1) = K_gn_stats(t1)`
3. `t2 = K_gn_apply_silu(t1, mean1, rstd1, norm1_weight, norm1_bias)`
4. `t3 = K_conv(t2, conv2_weight)`
5. `(mean2,rstd2) = K_gn_stats(t3)`
6. `out = K_gn_apply_silu_add(t3, mean2, rstd2, norm2_weight, norm2_bias, residual=x)`

Later candidates fuse/reorganize these (see §3) without changing the math contract of §2.

---

## 2. Correctness contract (must hold for every candidate)

Verified only through the evaluator, but every implementation must be reasoned against this checklist before evaluation:

1. **Conv indexing/padding:** output `(b,co,oh,ow)` sums `w[co,ci,kh,kw]·x[b,ci,oh+kh-1,ow+kw-1]` over `ci∈[0,256), kh,kw∈{0,1,2}`; out-of-range `ih/iw` **masked to 0**; batch index per pixel — no cross-batch/cross-row bleed.
2. **GroupNorm grouping:** channels split into 32 groups of 8; statistics reduce over `(8 channels, H, W)` per `(b,g)`; **biased variance** (divide by `N=8·H·W`); affine applied **per channel** with `norm_weight[c]`, `norm_bias[c]`; `eps` inside the sqrt exactly.
3. **SiLU:** `v·sigmoid(v)`, FP32 sigmoid, applied after each GroupNorm.
4. **Residual:** add the **original `x`** (pre-conv1) only after the second SiLU.
5. **Dtypes:** inputs/intermediates/output all float32; no bf16/fp16 intermediate storage (draft §4.3).
6. **Shape/stride generality:** kernels must handle all 5 shapes (B∈{1,4,32,64}, H=W∈{64,128,256,768}); no hard-coded spatial dims; `C=256`, `G=32`, `k=3` may be treated as constants.

**Numerical guardrails** (draft §4): FP32 accumulate in `tl.dot`; hierarchical/tiled reduction for large groups (WL5 group = 8·768·768 ≈ 4.7M elems) to bound variance error ~1e-4; `tl.math.rsqrt` (no fast-math reciprocal); TF32 only where §4.1 argues normalization absorbs the ~5e-4 relative conv error.

---

## 3. Candidate lineage roadmap (one lever per id)

Discipline: **exactly one meaningful change per candidate**; keep as new best-parent only if it is valid AND improves geomean; on regression or invalidity, **branch from the best-known-good parent** (do not stack speculative changes). The roadmap is ordered by expected value/risk; **later ids are chosen adaptively** from the observed evaluator results per the decision rules in §4.

| id | parent | single lever (change vs parent) | hypothesis (expected effect) | primary risk |
|----|--------|----------------------------------|------------------------------|--------------|
| **c001** | — | **Correctness anchor.** 6-kernel pipeline (§1), implicit-GEMM conv NCHW, `input_precision="ieee"`, two-pass/Welford biased-variance GN, single-program-per-`(b,g)` stats, conservative fixed blocks (`BLOCK_M=64, BLOCK_N=64`, `num_warps=4`, `num_stages=2/3`). | Passes all 5 correctness; establishes baseline geomean (likely <1.0 — IEEE conv is ~8× slower than TF32 cuDNN). | conv masking / GN grouping / biased-var bugs. |
| **c002** | c001 | **TF32 conv** (`input_precision="tf32"` in both `K_conv` dots; everything else identical). | ~5–8× conv speedup; big geomean jump. Key open question: does atol survive vs a possibly-TF32 cuDNN reference (draft §4.1, §8)? | tightest atol (WL3 0.0028) fails. |
| **c003** | best(c001,c002) | **Autotune conv** over `BLOCK_M∈{32,64,128}`, `BLOCK_N∈{64,128,256}`, `num_warps∈{4,8}`, `num_stages∈{2,3,4}`, shape-keyed on `(B,H,W)`. | Better tensor-core utilization across both shape regimes (draft §2, §5.6); geomean gain, esp. large workloads. | autotune compile time / config that fails a shape. |
| **c004** | best so far | **Weight prepermute** (one-time Triton permute of `conv{1,2}_weight` to tap-contiguous `[kh,kw,cin,cout]`) so tap tiles are contiguous. | Coalesced weight loads → faster dots; weights tiny (2.4 MB) so permute cost negligible. | permute indexing bug (correctness). |
| **c005** | best so far | **NHWC layout** for x/intermediates via a small Triton transpose kernel; conv reads contiguous `[pixel,cin]` tiles. | Coalesced input gather → conv throughput up on big workloads (draft §5.2). May not pay if L2 already absorbs strided reuse. | transpose overhead > gain; layout bug. |
| **c006** | best so far | **Fuse GN2-apply + SiLU + residual** into one epilogue kernel (already partly in c001 step 6; ensure single pass, no extra temporaries). | Removes 1–2 full-tensor memory passes; low-risk bandwidth win, esp. small workloads (draft §5.5.2, §6). | — (low risk). |
| **c007** | best so far | **Fuse GN1-apply + SiLU into conv2 input load** (read raw `t1`, apply normalize·affine·silu on-the-fly using precomputed stats; drop `t2` materialization). | Saves a full write+read of a 256-ch map. Trade-off: activation recomputed ~9× (overlapping taps) in a compute-bound kernel (draft §5.5.1). | recompute cost > bandwidth saved → regression. |
| **c008** | best so far | **Split-reduction GN stats** (multiple programs per `(b,g)` + two-stage/atomic combine) to raise occupancy on low-`B·G` big-spatial workloads (WL5=32 groups, WL3=32 groups). | Speeds stats on WL3/WL5 without hurting high-`B·G` (WL2/WL4); FP32 partials keep error ~1e-4. | atomic nondeterminism / combine bug. |
| **c009** | best so far | **Fuse conv1 epilogue → GN1 stats** via atomic group-sum accumulation during conv1 (saves a read of `t1` for stats). | Removes a stats read pass; optional bandwidth win. | atomic complexity/nondeterminism; low expected payoff — deprioritize if c006–c008 converge. |

**Contingency branches** (triggered by §4 decision rules, allocated new sequential ids as needed):
- **B1 — TF32 fails atol (c002 or later):** try mixed precision (IEEE conv1 + TF32 conv2, since GN after conv1 must absorb error before the tighter conv2), or **3xTF32 / error-corrected split** dot (draft §4.1, §8). Anchor remains the IEEE best.
- **B2 — one workload fails, others pass:** treat as masking/edge bug (draft §4.5) or shape-specific config; fix from same parent, re-issue new id; do not add unrelated levers.
- **B3 — Triton conv cannot beat cuDNN after c003–c005:** stop chasing conv; extract all remaining value from fusion (c006–c009), accept geomean may be near/below 1.0, document honestly, converge (draft §5.7, §7.6).
- **B4 — stats kernel dominates WL5 runtime in c001:** promote c008 (split-reduction) earlier in the order.

---

## 4. Decision rules (per evaluated candidate)

After each `feedback` evaluation, classify and act:

1. **INVALID** (any workload fails correctness): record failure + suspected cause (numerical §4.1–4.4 vs masking §4.5 vs shape). **Next id fixes exactly that, branching from the same parent.** Never add a Torch fallback; never widen scope.
2. **VALID + geomean > best_geomean × 1.00:** promote to **new best-parent**; next id applies the next roadmap lever.
3. **VALID but geomean ≤ best_geomean:** record; **do not promote**; **revert to best-parent** and try the next lever (avoid stacking regressions).
4. **Attribution:** because every id changes exactly one lever, geomean delta is causally attributable to that lever. Log the delta and the keep/revert decision explicitly.

`best_geomean` starts undefined; the first VALID candidate sets it. Track `best_candidate_id` alongside.

---

## 5. Performance hypotheses (quantified expectations)

From draft §2/§6 (A800: FP32 ~19.5 TFLOP/s, TF32 ~156 TFLOP/s, HBM ~1.5–2.0 TB/s; per-call 2× conv of M×256×2304):

- **Conv dominates compute.** IEEE conv is ~8× slower than TF32; **c001 (IEEE) geomean expected <1.0**; **c002 (TF32)** is the largest single lever and the gate for a competitive geomean.
- Order-of-magnitude TF32 conv time (≈50–70% util): WL5≈13–18 ms, WL1/WL4≈6–8 ms, WL2≈3–4 ms, WL3 sub-ms (launch/occupancy-bound).
- **Elementwise/reduction traffic** ≈ several full-tensor passes/call (each `4·B·C·H·W` bytes; WL5 tensor=604 MB). Fusing GN/SiLU/residual (c006–c009) saves ~1–3 passes ⇒ concrete low-risk win vs the reference's separate kernels, most visible on WL3 (launch-bound) and bandwidth-heavy WL5.
- **We compete against cuDNN conv + separate GN/SiLU/add.** Our edge = matching TF32 tensor-core throughput (c002–c005) **plus** removing the reference's extra memory passes/launches (c006–c009). Net geomean >1.0 is plausible but **not guaranteed** and is the main project risk (draft §5.7) — measured, not assumed.
- **Expected trajectory:** c001 <1.0 → c002 large jump → c003–c005 incremental conv gains → c006–c008 fusion gains → convergence.

Each candidate's record states its **specific numeric hypothesis** (which workloads should move and roughly how much) so the observed per-workload speedups confirm or refute the mechanism.

---

## 6. Stopping / convergence criteria

Stop and write `SEARCH_COMPLETE` (with reason) when the **first** of these holds:

1. **Budget:** cumulative evaluations reach **100**, OR token usage approaches the **1.0M soft limit** (wind down: finish the in-flight candidate, then stop and record). Hard stop before 1.5M/1.65M.
2. **Convergence:** best geomean improves by **< 2% across 3 consecutive VALID candidates**, and the remaining roadmap levers are exhausted or expected < 2% (per §5).
3. **Lever exhaustion:** all roadmap levers (c001–c009) and any triggered contingency branches evaluated, with no untried change expected to beat `best_geomean`.
4. **Dead-end honesty (B3):** if Triton conv provably cannot beat cuDNN and fusion gains are captured, converge and document rather than overfit block configs.

`SEARCH_COMPLETE` records: reason, `best_candidate_id`, best geomean, per-workload speedups of the best, total evaluations used, and the levers that did/didn't help. **`final` is never run without explicit operator approval.**

---

## 7. Evidence format (append-only `candidates.jsonl`)

One complete JSON object appended per evaluated candidate (never rewrite earlier records). Schema:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_hash": "<sha256 of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "IEEE Triton pipeline passes all 5; baseline geomean (expected <1.0).",
  "lever": "correctness anchor: implicit-GEMM conv NCHW, ieee, biased-var GN, 6 kernels",
  "validation": {
    "correctness_reasoning": "conv masking + GN grouping + biased var + residual checked vs §2",
    "all_workloads_pass": true
  },
  "per_workload": [
    {"uuid": "952fec71-f323-5dab-9340-dad59ad7a3f1", "B": 4,  "H": 256, "W": 256, "pass": true, "speedup": 0.00},
    {"uuid": "6284bbd2-bc2e-5002-856a-c4f2c7582254", "B": 32, "H": 64,  "W": 64,  "pass": true, "speedup": 0.00},
    {"uuid": "a4c930bb-4694-570b-8d51-97ba053f74ac", "B": 1,  "H": 128, "W": 128, "pass": true, "speedup": 0.00},
    {"uuid": "144fdf2b-0296-5d64-819b-eab16d18c752", "B": 64, "H": 64,  "W": 64,  "pass": true, "speedup": 0.00},
    {"uuid": "8de1dc63-b34a-5618-ba36-abaa79de54cf", "B": 1,  "H": 768, "W": 768, "pass": true, "speedup": 0.00}
  ],
  "geomean": 0.00,
  "decision": "keep|revert|invalid — rationale referencing §4",
  "cumulative_eval_count": 1,
  "skill_usage": "none (KernelWiki is Blackwell/Hopper-only; target is Ampere sm_80; profilers prohibited)"
}
```

Rules: `speedup`/`geomean`/`pass` copied verbatim from the evaluator output; if the evaluator reports observed atol/rtol, add it under `per_workload[*].observed_atol`. `source_hash` computed via `sha256sum solution/solution.py` at the moment of evaluation. `cumulative_eval_count` is monotonic across all rows. On INVALID, set failing `pass:false`, omit/zero speedups it didn't produce, and record the diagnosis in `decision`.

---

## 8. Execution checklist (next turns, in order)

1. **c001:** implement `solution/solution.py` (kernels per §1), reason through §2 checklist, `sha256sum`, evaluate `feedback c001`, append record (§7), apply §4 decision.
2. **c002:** flip `input_precision` to `"tf32"` only; evaluate; decide (watch atol on WL3).
3. **c003–c009:** proceed lever-by-lever per §3, branching on §4 outcomes and §3 contingencies.
4. Monitor budget/tokens continuously (§6); wind down cleanly at soft limit.
5. On convergence, write `SEARCH_COMPLETE` with the §6 reason set. Do **not** run `final` without operator approval.

---

## 9. Decision log (append-only; observed evaluator results)

### c001 (IEEE anchor) — INVALID
- Result: all 5 workloads `RUNTIME_ERROR`, `max_abs = max_rel = 0.0` (kernels produced no output → compile/launch fault, not a numeric mismatch). `cumulative_evaluations = 1`.
- Read: a **uniform** failure across every shape points to a construct shared by all workloads, not shape/masking edge cases.

### c002 (TF32) — INVALID
- Single lever vs c001: conv `input_precision` `"ieee" → "tf32"`; kernels otherwise byte-identical.
- Result: **identical** to c001 — all 5 `RUNTIME_ERROR`, `max_abs = max_rel = 0.0`. `cumulative_evaluations = 2`.
- **Refutes** the "fp32 ieee dot has no tensor core on sm_80" hypothesis: the failure is **precision-independent**. Because flipping only the *value* of `input_precision` changed nothing, the fault is either (a) the **presence** of the `tl.dot(..., input_precision=...)` kwarg (possible API mismatch with the installed Triton — older builds use `allow_tf32=` instead), or (b) another compile-time construct shared by all kernels.
- Constraint: local execution of Triton/torch and filesystem introspection outside the workspace are blocked, so the Triton version cannot be inspected directly; the evaluator is the only oracle. Diagnosis must proceed by single-lever source edits.

### c003 (remove input_precision kwarg + host inv_group_size + plain range) — INVALID
- Single lever vs c002: dropped the `tl.dot(input_precision=)` kwarg (plain `tl.dot`), passed a host-side `inv_group_size` instead of `group_size.to(tl.float32)`, kept plain `range()` loops.
- Result: **identical** uniform `RUNTIME_ERROR` on all 5, `max_abs = max_rel = 0.0`. `cumulative_evaluations = 3`.
- **Refutes** the input_precision-kwarg hypothesis. Key deduction: the fault poisons the output of the **very first kernel launched** (the conv) — a later kernel's fault would still leave nonzero conv output (`max_abs > 0`), never observed. So the trace-time fault is a construct **unique to the conv kernel**. Remaining suspect: capture of the module-global Python int `_KS` inside the `@triton.jit` conv body.

### c004 (kernel-size as tl.constexpr param; remove all module-global capture from conv) — VALID ✅ (first passing candidate)
- Single lever vs c003: made kernel-size a proper `tl.constexpr` kernel parameter `KS`, removing every reference to the module-global `_KS` from the jitted conv body.
- Result: **5/5 PASS**. Geomean **0.1610x**, arithmetic mean 0.1628x. `cumulative_evaluations = 4`.
  - Per-workload speedups: WL1(B4,256²)=0.149x, WL2(B32,64²)=0.141x, WL3(B1,128²)=0.194x, WL4(B64,64²)=0.139x, WL5(B1,768²)=0.192x.
- **Root cause CONFIRMED:** the c001–c003 uniform faults were module-global `_KS` capture inside the jitted conv (Triton on this build raises at trace time when a jit body reads a module-global Python int). Now eliminated → correct output, and fp32 `tl.dot` (native TF32 on sm_80) passes the tight atol as the draft predicted.
- **Decision:** keep as best-parent (`best_candidate_id=c004`, `best_geomean=0.1610`). Naive un-tuned implicit-GEMM conv is ~5–7× slower than cuDNN. Large-batch/small-spatial (WL2/WL4) is worst → conv under-utilizes tensor cores. Next lever = conv autotune (see §3 c003-row lever, re-slotted).

### c005 plan (next turn) — one lever
- **Lever:** add `@triton.autotune` to the conv kernel over `BLOCK_M∈{32,64,128}`, `BLOCK_N∈{64,128,256}`, `BLOCK_K∈{64,128,256}`, `num_warps∈{4,8}`, `num_stages∈{2,3,4}`, keyed on `(M, C)` (equivalently shape). Everything else identical to c004. Hypothesis: bigger tiles + more stages raise tensor-core utilization, most on WL2/WL4 (currently ~0.14x). Watch autotune compile time and that every config remains correct on all shapes.

### c005 (conv autotune) — VALID ✅ new best
- Single lever vs c004: wrapped `_conv3x3_kernel` in `@triton.autotune` (11 configs over BLOCK_M/N/K, num_warps, num_stages; key `[M,C,H,W]`); grid uses `meta[BLOCK_M/N]`. Conv/GN math byte-identical.
- Result: **5/5 PASS**. Geomean **0.4183x** (up from c004 0.1610x, **+160%**), arithmetic 0.4199x. `cumulative_evaluations = 5`.
  - Per-workload: WL1 0.149→0.406, WL2 0.141→0.382, WL3 0.194→0.449, WL4 0.139→0.387, WL5 0.192→0.476. Uniform lift confirms c004's conv was tensor-core-throughput-bound.
- **Decision:** keep as best-parent (`best_candidate_id=c005`, `best_geomean=0.4183`). Still <1.0 vs cuDNN — the conv remains the bottleneck.

### c006 plan (next turn) — one lever
- **Lever:** **weight prepermute**. The conv currently reads weights with the strided access `w[co, ci, kh, kw]` (stride `KK=9` in `ci`, non-contiguous W tiles). Add a one-time Triton kernel that permutes each `conv{1,2}_weight` from `(Cout,Cin,3,3)` to tap-contiguous `(kh,kw,Cin,Cout)`, so each tap's W tile `[cin, cout]` is fully contiguous → coalesced weight loads and a cleaner `tl.dot` operand. Weights are tiny (2.4 MB) so the permute cost is negligible and reused across all M-tiles. The conv kernel's weight-pointer math changes to index the permuted layout; conv output math is identical. Hypothesis: coalesced weight loads improve conv throughput, most on the large workloads. Risk: permute indexing bug (correctness) — re-verify the (kh,kw,cin,cout) mapping against §2.
- **Alternative if c006 disappoints:** the conv re-loads the full input for all 9 taps; consider caching input tiles in SRAM across taps (restructure the tap loop inside the K loop), or an NHWC layout transform (draft §5.2) so input `[pixel,cin]` tiles are contiguous.

### c006 (weight prepermute) — VALID ✅ new best
- Single lever vs c005: one-time Triton permute of each conv weight `(Cout,Cin,3,3)`→`[KK,Cin,Cout]` (cout stride-1); conv reads permuted weight with coalesced W-tile. Autotune configs + all other kernels byte-identical.
- Result: **5/5 PASS**. Geomean **0.7857x** (up from c005 0.4183x, **+88%**), arithmetic 0.7869x. `cumulative_evaluations = 6`.
  - Per-workload: WL1 0.406→0.810, WL2 0.382→0.795, WL3 0.449→0.707, WL4 0.387→0.829, WL5 0.476→0.793. Uniform lift → the c005 conv was heavily L2/weight-load-bound.
- **Decision:** keep as best-parent (`best_candidate_id=c006`, `best_geomean=0.7857`). Now approaching cuDNN parity (best WL4 0.829x). Conv weight traffic largely solved; remaining conv inefficiency is redundant **input** re-loads (input reloaded for all 9 taps).

### c007 plan (next turn) — one lever
- **Lever:** **cut redundant conv input traffic / elementwise passes.** Two candidate mechanisms, pick the one with the larger expected remaining win:
  1. **Input reuse across taps in the conv.** Currently the 3×3 tap loop reloads overlapping input rows 9× from L2. Restructure so the input is loaded once per (M-tile, cin-tile) and reused for the applicable taps (shift within SRAM), or move the K(cin) loop outermost with the 9 taps sharing a cached input panel. Reduces conv input L2 traffic ~9×→~1× on the reused region.
  2. **Fuse GN-apply+SiLU into conv2's input load** (read raw `t1`, apply normalize·affine·silu on-the-fly using precomputed stats; drop `t2` materialization) — saves a full write+read of a 256-ch map. Trade-off: activation recomputed ~9× (draft §5.5.1); may not pay in a compute-bound conv.
- Start with mechanism (1) if the conv is still the dominant cost (it is — conv >> elementwise). Keep exactly one mechanism per candidate id; branch from c006. Watch that restructuring preserves masking/correctness.
- **Note on convergence:** we have gone 0.1610→0.4183→0.7857. Each conv lever has paid off strongly; continue while conv gains remain, then harvest fusion (GN/SiLU/residual pass reduction). Converge when consecutive VALID candidates improve geomean <2% and remaining levers are exhausted (plan §6).


