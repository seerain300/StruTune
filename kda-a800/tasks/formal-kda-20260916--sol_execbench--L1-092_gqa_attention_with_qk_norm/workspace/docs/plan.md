# Plan — L1/092 GQA Attention with QK Norm (GLM-4.5-Air), A800 / sm_80

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`, `task/definition.json`,
`task/feedback_workloads.jsonl`, `TASK.md`, `CLAUDE.md`. Triton-only compute; PyTorch for metadata /
launch plumbing only. This turn writes the plan only — **no candidate is implemented or evaluated.**

---

## 0. Ground rules (binding for every candidate)

- **Entry point:** `solution/solution.py` exposes `run(hidden_states, q_proj_weight, q_proj_bias,
  k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight,
  k_norm_weight, cos, sin, rms_norm_eps)` and returns `output [B,S,4096]` bf16.
- **Compute = Triton only.** No `F.linear`, `torch.matmul`, `torch.bmm`, `scaled_dot_product_
  attention`, `torch.softmax`, `torch.rsqrt`-on-tensors-in-the-math-path, einsum, or CUDA-extension.
  PyTorch is allowed **only** for: `.shape`/stride reads, `torch.empty`/`empty_like` allocation,
  `.view`/`.reshape`/`.transpose` of metadata (no compute), dtype/device queries, grid-lambda
  construction, and passing pointers to kernels. Elementwise ops (RMSNorm, RoPE, bias, softmax,
  scaling) must live inside Triton kernels.
- **No fallback of any kind.** If a Triton kernel fails to compile/run, the candidate is invalid;
  fix the kernel — never route around it with Torch/NumPy/CPU.
- **Immutability & lineage.** Candidate source for `cNNN` is frozen once evaluated. Any change to
  source, autotune configs that alter the compiled artifact, launch grid, or dtype path ⇒ new ID.
  IDs are sequential (`c001, c002, …`), never reused.
- **Evaluation channel.** Only `./scripts/evaluate_candidate.sh feedback cNNN`. Five fixed workloads
  = one evaluation. Never run `final` without explicit operator approval. Never call the evaluator,
  a profiler, `nvidia-smi`, CUDA directly, or any alternate correctness harness.
- **Budgets.** ≤100 candidate evaluations; token soft 1.0M / hard 1.2M. Spend evals on hypotheses
  with a clear expected delta; do not burn evals on speculative micro-tweaks.
- **Skills.** `KernelWiki` (Blackwell/Hopper) and `ncu-report-skill` (B200 profiling) are both
  out-of-scope for Ampere sm_80 and for the no-profiler rule; **not invoked**. Recorded as
  `skills_used: []` in every record with a one-line justification field.

### 0.1 Fixed problem constants (hard-code in the kernels)
```
HIDDEN = 4096   NH = 96   NKV = 8   HEAD_DIM = 128   GROUPS = 12   HALF = 64
Q_OUT = 12288   KV_OUT = 1024   SCALE = 128 ** -0.5   # 0.08838834764831845
```
`B`, `S`, `M=B*S`, `rms_norm_eps` come from inputs at launch time.

---

## 1. Candidate lineage strategy

Tree-structured search. Each node = one immutable candidate; edges = single-hypothesis deltas from a
chosen parent (usually the current best-correct). **Correctness is a gate**: a candidate that fails
any workload cannot become a parent for perf work — only for a correctness-fix child.

```
c001  correctness-first baseline (decomposition A, static tiles) ── GATE: must PASS all 5
  │
  ├─ c00x  GEMM tiling/autotune for projections + output GEMM (biggest FLOP share)
  │     │
  │     └─ c00x  flash-attention tile tuning + causal block early-exit
  │           │
  │           ├─ c00x  fuse RMSNorm+RoPE into attention prologue (option B)
  │           │
  │           └─ c00x  KV-load reuse across the 12-head group (memory-traffic cut)
  │
  └─ (correctness-fix branch only if c001 fails a workload)
```

Lineage rules:
1. **Advance the frontier** = best candidate that PASSes all five workloads, ranked by geomean.
2. Each new candidate changes **exactly one** design variable vs its parent (kernel fused/split,
   tiling, layout, precision path). This keeps attribution clean given feedback-only evidence.
3. If a change **regresses geomean or breaks correctness**, its parent remains the frontier; do not
   stack further changes on the regressor. Record the negative result and branch again from frontier.
4. Keep a small number of live hypotheses; prefer depth on the winning line over breadth.

### 1.1 Concrete candidate ladder (initial; may adapt to feedback)
- **c001 — Baseline, correctness-first (decomposition A).**
  Kernels: (K1) Q proj GEMM +bias, (K2) K proj GEMM +bias, (K3) V proj GEMM +bias, (K4)
  RMSNorm+RoPE for Q, (K5) RMSNorm+RoPE for K, (K6) V repack into `[B,NKV,S,D]` (or read V in-place),
  (K7) flash-attention (GQA, causal, online softmax, fp32), (K8) output GEMM. Static, safe tile
  sizes (e.g. BLOCK_M=64, BLOCK_N=64/128, BLOCK_K=32, num_warps=4, num_stages=2/3). No autotune yet
  (deterministic artifact, easy to reason about). **Goal: PASS ×5, establish baseline geomean.**
- **c002 — Projection + output GEMM performance.** Add `@triton.autotune` (or hand-picked shape-
  specialized configs) to K1–K3 and K8. Larger BLOCK_N for Q proj (N=12288) and output (N=4096,
  K=12288). Only-change: GEMM tiling. Hypothesis: recover cuBLAS-competitive GEMM throughput.
- **c003 — Flash-attention tiling + causal early-exit.** Tune BLOCK_M/BLOCK_N, num_warps/stages,
  skip fully-masked key blocks (`key_block_start > query_block_end`), tighten diagonal masking.
  Only-change: attention kernel.
- **c004 — Fuse RMSNorm+RoPE into attention prologue (option B).** Attention loads raw projected
  Q/K tiles, applies norm+RoPE on the fly; drop K4/K5 and their Q/K round-trip. Only-change:
  norm/RoPE fusion. (Guard: K-tile recompute cost vs saved memory traffic.)
- **c005 — KV-load reuse across the group of 12.** One program services a block of query heads that
  share a KV head, so K/V tiles are loaded once per group. Only-change: attention grid/tiling.
- **c006+ — conditional:** K/V fused projection (option C) if K/V proj shows up as non-trivial;
  split-K for the M=128 workload if it under-utilizes SMs; precision-margin variants if any
  workload is correct-but-marginal.

Exact IDs beyond c001–c005 depend on measured deltas; the ladder is re-planned after each eval.

---

## 2. Correctness checks (static, pre-evaluation, per candidate)

Because local execution is prohibited, every candidate passes a **static review checklist** before
consuming an evaluation. Reject/rewrite before eval if any item is uncertain.

### 2.1 Math-equivalence checklist (against draft §1.2)
- [ ] **Projections** compute `x @ W^T + bias` (W stored `[N,K]`; reduce over K=4096). Bias added in
      fp32, result cast bf16. Q→`[M,12288]`, K/V→`[M,1024]`.
- [ ] **Head reshape** consistent: Q `[B,S,96,128]`, K/V `[B,S,8,128]`; strides correct.
- [ ] **RMSNorm** over last dim 128, in **fp32**: `mean(x²)`, `rsqrt(var+eps)`, `× weight`, cast bf16.
      Uses `q_norm_weight` for Q, `k_norm_weight` for K. `eps` is the fp32 scalar `rms_norm_eps`.
- [ ] **RoPE** applied after norm, elementwise over full 128 width:
      `out = x*cos + rotate_half(x)*sin`, `rotate_half(x) = cat(-x[64:128], x[0:64])`.
      Sign: low half (0..63) gets `-x[64:128]·sin`; high half (64..127) gets `+x[0:64]·sin`.
      `cos/sin` indexed at `[b, s, 0:128]`, broadcast across heads. **Do not assume half-duplication.**
- [ ] **GQA mapping** `kv_head = q_head // 12` (integer). No physical 12× KV expansion.
- [ ] **Scores** `= (q·k) * SCALE`, reduce over 128 in fp32.
- [ ] **Causal mask** keeps `key_j ≤ query_i` (diagonal included), masks `j > i` with `-inf`.
      No fully-masked row (diagonal always present) ⇒ no NaN.
- [ ] **Softmax** in fp32 (online max/sum); accumulate `p·v` in fp32; cast attn output bf16.
- [ ] **Head merge** → `[B,S,12288]` row-major matching reference `transpose(1,2).reshape`.
- [ ] **Output proj** `attn @ o_proj_weight^T`, reduce over K=12288 in fp32, **no bias**, →`[B,S,4096]`.
- [ ] **Boundary masking** for non-pow2 S (293, 373) and small M (128): OOB rows load 0 / store
      masked; OOB keys get `-inf` score; OOB reductions contribute 0.
- [ ] **Dtypes:** all DRAM tensors bf16; all accumulators/reductions fp32; output bf16.

### 2.2 Structural checks
- [ ] No Torch compute op in `run` (grep the source for `F.linear`, `matmul`, `bmm`, `softmax`,
      `scaled_dot_product_attention`, `einsum`, `@` on tensors) before evaluating.
- [ ] Grid dimensions cover all (B, heads, M-blocks, N-blocks); no partial coverage.
- [ ] Strides passed explicitly; no accidental reliance on contiguity that the harness may not give.
- [ ] Output tensor allocated once, correct shape/dtype/device; returned directly.

### 2.3 Correctness evidence from evaluation
- The feedback evaluator reports per-workload pass/fail against `(atol, rtol)` from
  `feedback_workloads.jsonl`. Treat **any** failing workload as a correctness bug to localize:
  - Only non-pow2 S fail → boundary masking bug.
  - Only M=128 fail → grid/coverage or degenerate-tile bug.
  - All fail with small margin → a systematic precision path issue (accumulation/rounding).
  - All fail large → structural bug (GQA map, RoPE sign, transpose/stride, causal direction).

---

## 3. Performance hypotheses (each maps to a candidate)

| # | Hypothesis | Change | Expected effect | Risk |
|---|---|---|---|---|
| H1 | A correct flash-style attention (no S² materialization, no 12× KV expand) beats the reference's materialized path. | c001 attention kernel. | Attention memory traffic ↓ ~10× vs reference; net geomean > 1 even with untuned GEMMs. | GEMMs may be slow enough to offset; measured in c001. |
| H2 | Projection + output GEMMs are the FLOP bottleneck; tiling/autotune brings them near cuBLAS. | c002 GEMM configs. | Largest absolute time reduction (GEMMs ≈ 650 GFLOP vs 55 attention). | Triton GEMM may still trail cuBLAS on some shapes ⇒ net-neutral. |
| H3 | Attention tile tuning + causal block-skip halves attention work (triangular). | c003. | ~2× on attention portion for large S (1024, 512). | Small for tiny S (128). |
| H4 | Fusing RMSNorm+RoPE into attention removes a Q/K round-trip. | c004. | Q is `[M,12288]` bf16 = large; removing its write+read is a real DRAM saving. | K-tile norm/RoPE recompute across M-blocks; net depends on S. |
| H5 | KV-load reuse across the 12-head group cuts K/V DRAM reads ~12×. | c005. | Meaningful for larger B·heads; improves occupancy-bound cases. | Register pressure / occupancy drop. |
| H6 | M=128 workload under-utilizes SMs; split-K or smaller M-tiles help. | c006 (conditional). | Better small-M throughput. | Split-K adds a reduction pass. |

Performance evidence = **geomean speedup** across the five workloads reported by the evaluator, plus
per-workload speedups (to see which shape a change helps/hurts). A change is **kept** only if it
improves geomean without breaking any workload's correctness; otherwise reverted (branch stays at
parent).

---

## 4. Stopping criteria

Stop the search and write `SEARCH_COMPLETE` (with the reason) when **any** holds:
1. **Convergence:** geomean improvement < ~1–2% across ≥3 consecutive candidates on the frontier
   line, and no untried hypothesis has a plausible >5% expected delta.
2. **Budget:** approaching 100 evaluations or the 1.0M-token soft limit (leave margin below the 1.2M
   hard limit; stop implementing new candidates once soft limit is near).
3. **Ceiling reached:** GEMMs are cuBLAS-competitive and attention is flash-optimal with KV reuse —
   remaining ops are memory-bound at hardware roofline; no structural win left.

At stop: the frontier (best PASS-all-five candidate by geomean) is the recommended candidate for the
operator-approved `final` run. **Do not run `final` autonomously.**

Roll-back / branch policy during search:
- Regressor (worse geomean, still correct): discard branch; note in record; branch from frontier.
- Correctness break: mandatory fix-child before any perf work continues on that line.
- Marginal-correct workload (passes but near tolerance): prefer the more reference-faithful precision
  variant (bf16-intermediate RoPE; match reference softmax bf16-rounding of `p`) even at small cost.

---

## 5. Evidence format (append-only `candidates.jsonl`)

One JSON object appended per evaluated candidate. Never rewrite earlier records. Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "Correct flash-style GQA attention + Triton GEMMs beats materialized reference.",
  "design_delta": "baseline: decomposition A, static tiles, no autotune",
  "static_validation": {
    "math_checklist": "pass",
    "structural_checklist": "pass",
    "notes": "GQA kv_head=h//12; causal j<=i; RoPE full-128 rotate_half; fp32 accum"
  },
  "per_workload": [
    {"uuid": "ccea3f00", "B": 1, "S": 128,  "correct": true, "speedup": 0.00},
    {"uuid": "1e96cc88", "B": 4, "S": 293,  "correct": true, "speedup": 0.00},
    {"uuid": "d0ce8c45", "B": 1, "S": 1024, "correct": true, "speedup": 0.00},
    {"uuid": "c92bb0a5", "B": 1, "S": 512,  "correct": true, "speedup": 0.00},
    {"uuid": "65a1864d", "B": 8, "S": 373,  "correct": true, "speedup": 0.00}
  ],
  "all_correct": true,
  "geomean_speedup": 0.00,
  "decision": "keep|revert|fix",
  "reason": "<why kept/reverted; what the per-workload pattern shows>",
  "cumulative_evaluations": 1,
  "skills_used": [],
  "skills_note": "KernelWiki/ncu-report-skill out-of-scope for sm_80 + no-profiler rule",
  "next": "c002: autotune projection + output GEMMs"
}
```

Rules:
- `speedup` and `geomean_speedup` are copied from the evaluator output verbatim (no re-derivation of
  correctness; the evaluator is the sole correctness/perf authority).
- `cumulative_evaluations` increments by 1 per candidate (5 workloads = 1 eval).
- If a workload fails, set `correct: false`, `all_correct: false`, `decision: "fix"`, and record the
  localized bug class in `reason`.
- `source_sha256` recorded so immutability is auditable (compute with `sha256sum solution/solution.py`
  at eval time; do not edit a candidate's source after its record is written).

---

## 6. Execution checklist (per candidate, in order)

1. Choose parent = current frontier (or fix-target). State the single design delta.
2. Implement `solution/solution.py` for the new candidate ID (write the Triton kernels + `run`).
3. Run the static correctness checklist (§2.1/§2.2). Fix until all items are confident.
4. Snapshot immutability: record `sha256sum solution/solution.py`.
5. Evaluate once: `./scripts/evaluate_candidate.sh feedback cNNN`.
6. Append the evidence record (§5) to `candidates.jsonl`.
7. Decide keep/revert/fix; update the frontier; plan the next single delta.
8. Check stopping criteria (§4). If converged → write `SEARCH_COMPLETE` with reason. Else loop.

---

## 7. Immediate next action (next turn, not this one)
Implement **c001** (decomposition A, correctness-first) exactly per §1.1 and §2.1, then evaluate once.
No `plan.md`/`draft.md` edits required to proceed.

---

## 8. Decision log (updated after each eval)

### c001 — evaluated (eval #1). DECISION: keep. **Frontier = c001.**
- Result: **PASS 5/5, geomean 1.7135x**, avg 1.7324x (source_sha256 `81c9240…`).
- Per-workload speedup: S=128 →1.4649, S=293(B4) →1.7034, S=1024 →2.2476, S=512 →1.6510,
  S=373(B8) →1.5951.
- Reading: H1 confirmed — flash-style attention already beats the materialized reference across the
  board with no correctness issue (loose `rtol=0.05` tolerated fp32 online softmax fine). The
  smallest gains are the compute-heavy cases (tiny M=128 launch overhead; largest M=2984 where the
  untuned 64×64×32 GEMMs cap throughput), which points at **H2 (GEMM tiling)** as the next lever.
- Next: **c002** — shape-specialized tiling/autotune for the projection + output GEMMs (biggest FLOP
  share), single design delta vs c001, everything else identical.

### c002 — evaluated (eval #2). DECISION: keep. **Frontier = c002.**
- Result: **PASS 5/5, geomean 2.3308x**, avg 2.3437x (source_sha256 `b6397e0…`).
- Per-workload speedup: S=128 →2.5543, S=293(B4) →2.1654, S=1024 →2.6682, S=512 →2.0015,
  S=373(B8) →2.3290.
- Reading: H2 confirmed — swapping the untuned 64×64×32 unswizzled GEMM for an autotuned,
  L2-swizzled (GROUP_M) matmul with fp32 `tl.dot(a,b,acc)` and an EVEN_K fast path lifted geomean
  1.7135→2.3308 (+36%). Biggest gains where c001 was GEMM-capped: tiny M=128 (1.46→2.55) and largest
  M=2984 B=8 (1.60→2.33). Attention is now the next-largest remaining share.
- Next: **c003** — single delta on the flash-attention kernel (BLOCK_M/BLOCK_N + num_warps/stages
  autotune, keep GEMMs from c002 unchanged). Watch large-S (S=1024) and small-S launch overhead.
