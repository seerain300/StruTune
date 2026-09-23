# Draft — `gdn_prefill_qk4_v8_d128_k_last`

Gated Delta Net (GDN) prefill, GVA head configuration, **k‑last** state layout, captured from
Qwen3‑Next linear‑attention layers (TP=4). Target hardware: **NVIDIA A800 (`sm_80`, Ampere)**.
Primary implementation must be **Triton**; PyTorch only for metadata/launch plumbing.

This draft analyzes the operation, constraints, numerical risks, the Triton design space, and the
validation strategy. No plan and no code are produced in this step.

---

## 1. Operation analysis

### 1.1 Shapes, dtypes, and constants

Constant axes (from `task/definition.json`):

- `num_q_heads (Hq) = 4`, `num_k_heads (Hk) = 4`, `num_v_heads (Hv) = 8`, `head_size (D) = 128`.
- `num_sab_heads = max(Hq, Hv) = 8` — outputs and state are laid out over the **8 value heads**.
- GVA expansion factor `Hv // Hq = 2` (each q/k head is shared by two v heads).

Variable axes: `total_seq_len (T)`, `num_seqs (N)`, `len_cu_seqlens = N + 1`.

Inputs:

| tensor | shape | dtype | notes |
|---|---|---|---|
| `q` | `[T, 4, 128]` | bf16 | query, 4 heads |
| `k` | `[T, 4, 128]` | bf16 | key, 4 heads |
| `v` | `[T, 8, 128]` | bf16 | value, 8 heads |
| `state` | `[N, 8, 128, 128]` | f32 | recurrent state, **k‑last `[N, H, V, K]`**, optional |
| `A_log` | `[8]` | f32 | log decay parameter (per v‑head) |
| `a` | `[T, 8]` | bf16 | input‑dependent decay (per v‑head) |
| `dt_bias` | `[8]` | f32 | decay bias (per v‑head) |
| `b` | `[T, 8]` | bf16 | update‑gate input (per v‑head) |
| `cu_seqlens` | `[N+1]` | int64 | variable‑length batching offsets |
| `scale` | scalar f32 | | fixed `0.08838834764831843 = 1/sqrt(128)` in all feedback workloads |

Outputs:

| tensor | shape | dtype | notes |
|---|---|---|---|
| `output` | `[T, 8, 128]` | bf16 | attention output over the 8 v‑heads |
| `new_state` | `[N, 8, 128, 128]` | f32 | updated recurrent state, **k‑last `[N, H, V, K]`** |

Constraints: `len_cu_seqlens == N + 1`, `T == cu_seqlens[-1]`. Sequences are contiguous slices of the
token axis given by `cu_seqlens`.

### 1.2 Gate / gating math (per token `t`, per v‑head `h`)

```
x    = a.float() + dt_bias.float()                      # [T, 8]
g    = exp( -exp(A_log.float()) * softplus(x) )          # [T, 8]  decay in (0, 1]
beta = sigmoid(b.float())                                # [T, 8]  in (0, 1)
```

`softplus(x) = log(1 + exp(x)) ≥ 0`, `exp(A_log) > 0`, so the exponent is `≤ 0` and `g ∈ (0, 1]`.
`g` is a **scalar** per (token, head), so it scales the whole `[K,V]` state uniformly (not a
per‑channel diagonal). This matters: the cumulative within‑chunk decay is a scalar cumulative product.

### 1.3 The recurrence (reference semantics)

Internally the reference transposes the k‑last input state `[H, V, K]` → `state_HKV = [H, K, V]`
(K first, V last). For each token `t` in a sequence, with `q_t, k_t ∈ R^K` (K=128), `v_t ∈ R^V`
(V=128), scalars `g_t, beta_t`:

```
old   = g_t * S_{t-1}                 # [K, V]   apply decay
old_v = k_tᵀ @ old                    # [V]      = k_t · columns of old
new_v = beta_t * v_t + (1-beta_t) * old_v
S_t   = old + k_t ⊗ (new_v - old_v)   # [K, V]   rank‑1 update
      = old + beta_t * k_t ⊗ (v_t - old_v)          (algebraically identical)
o_t   = scale * q_tᵀ @ S_t            # [V]      output uses the UPDATED (inclusive) state
```

Key observations:

- This is the **gated delta rule**: `S_t = g_t S_{t-1} + k_t ⊗ u_t` with pseudo‑value
  `u_t = beta_t (v_t − k_tᵀ (g_t S_{t-1}))`.
- The output at token `t` uses `S_t` (**inclusive** of the current token) — the intra‑chunk
  attention mask is lower‑triangular **including the diagonal**.
- GVA: for v‑head `h ∈ [0,8)`, `q` and `k` come from head `h // 2` (this reproduces
  `repeat_interleave(2)` without materializing an expanded tensor).
- `q`, `k`, `v` are bf16 in memory; `state` is f32. The reference performs all matmuls in **float32**
  (`a.float() @ b.float()`), and the state accumulates in f32.
- Final store: `new_state[seq] = S_last.transpose(-1,-2)` → back to k‑last `[H, V, K]`.

### 1.4 Chunk‑parallel reformulation (target for high performance)

The reference is an O(T) Python loop over tokens — catastrophically slow. The standard fast form is
the **chunked gated delta rule** (WY / UT‑transform), which is what
`flashinfer.gdn.chunk_gated_delta_rule` implements. Split each sequence into chunks of size `C`
(e.g. 32/64). Within a chunk with entry state `S₀`, define the scalar cumulative decay
`γ_i = Π_{j≤i} g_j` (γ₀ across the chunk boundary handled in log space):

```
S_i = γ_i S₀ + Σ_{l≤i} (γ_i/γ_l) k_l ⊗ u_l
```

Pseudo‑value `u_l = beta_l (v_l − k_lᵀ (g_l S_{l-1}))` depends on earlier `u_m (m<l)`, giving a
lower‑triangular linear system solved once per chunk (the **UT / WY transform**, i.e. inverting
`(I + tril(M,-1))` where `M_{lm} = beta_l (γ_l/γ_m)(k_l·k_m)`). Building blocks per chunk:

1. **Preprocess**: `log g`, `beta`, within‑chunk cumulative `Γ = cumsum(log g)` (log space, f32).
2. **Intra matrices**: decayed `A = tril((q k^T) or (k k^T) weighted by γ_i/γ_l)`.
3. **UT transform**: solve the strictly‑lower‑triangular system to get `U` (and `W` for the state
   contribution). Sizes are `[C, C]` and `[C, V]`, `[C, K]`.
4. **Inter‑chunk scan** (sequential over chunks per (seq,head)): `S_new = γ_C S₀ + (K_scaled)ᵀ U`.
5. **Output**: `o = scale * ( intra: tril‑masked decayed (q k^T) @ U  +  inter: γ ⊙ (q @ S₀) )`.

Parallelism: independent across `N × Hv` (= up to `34×8 = 272` programs for the largest workload)
and across chunks *within* the intra‑chunk matmuls; the inter‑chunk scan is sequential per
(seq, head). Compute is tiny (≈100k FLOP/token/head ⇒ ≈6.5 GFLOP at T=8192) — the op is
**latency/dependency‑bound, not compute‑bound**, so kernel‑launch count and occupancy matter more
than raw MMA throughput.

---

## 2. Constraints (hard rules that shape the design)

From `CLAUDE.md` / `TASK.md`:

- **Triton‑only compute.** No Torch/CPU/NumPy/CUDA‑extension/alternate‑impl fallback. A failing
  Triton kernel is *invalid* — it may not be silently backed by a Torch path. PyTorch usage is
  limited to tensor metadata, allocation, and launch plumbing (e.g. computing grids, `cu_seqlens`
  handling, output allocation).
- **Submission surface**: `solution/solution.py` exposing `run(q, k, v, state, A_log, a, dt_bias, b,
  cu_seqlens, scale)` returning `(output, new_state)` with the exact shapes/dtypes above.
- **Immutable candidates.** Implement `c001`, `c002`, … sequentially, one source version at a time;
  never reuse an ID for changed source; append one JSON record per evaluation to `candidates.jsonl`
  without rewriting prior records.
- **Evaluation.** Only via `./scripts/evaluate_candidate.sh feedback <id>`; the five fixed feedback
  workloads together count as one evaluation. Budget: **100 evaluations**. Token soft/hard limits:
  1.0M / 1.2M. `final` is operator‑only.
- **Isolation.** Work only in this workspace; do not inspect parent dirs, baselines, evaluator,
  controller, or other tasks. Permitted external knowledge sources are the `KernelWiki` and
  `ncu-report-skill` skills only — but see §6 on their applicability here. No web/subagents/MCP.
- **Do not run** CUDA, a profiler, `nvidia-smi`, the external evaluator directly, or any alternate
  correctness harness. (In this environment the general `Bash` tool is also disabled, so the *only*
  sanctioned execution channel is the `evaluate_candidate.sh` launcher.)
- **Metric.** Geometric‑mean speedup across selected workloads; **every** workload must pass
  correctness or the candidate is invalid.

Layout / semantics constraints to respect exactly:

- State I/O is **k‑last `[N, H, V, K]`**; internally the recurrence is on `[K, V]`. Load with a
  V↔K transpose relative to storage; store new_state transposed back. Getting the V/K axes swapped
  is the single most likely functional bug — call it out in validation.
- Output is over **8 v‑heads**; `q`/`k` are indexed with `head // 2` (GVA), never expanded in DRAM
  unless deliberately chosen.
- Output uses the **inclusive** (post‑update) state ⇒ diagonal‑inclusive causal mask.
- `scale` multiplies only the final output (`q @ S_t`), not the state update.
- Variable‑length batching: iterate sequences via `cu_seqlens`; sequences may be as short as a few
  tokens (workloads have seqs down to ~10–35 tokens) and empty sequences (`seq_len ≤ 0`) must be
  skipped (contribute nothing, but `new_state` still written — check reference: empty seq leaves
  `new_state[seq]` as the zero‑init from `torch.zeros`, since the inner loop never runs and the
  `state_HKV`... actually it `continue`s before writing → `new_state[seq]` stays **zeros**, not the
  input state. Must replicate: for `seq_len ≤ 0`, `new_state[seq] = 0`, not the passed‑in state).

### 2.1 Feedback workload profile

| uuid (short) | T | N | avg tok/seq | regime |
|---|---|---|---|---|
| 1efaf2a9 | 42 | 2 | 21 | tiny — launch/latency bound |
| d3dc3577 | 35 | 1 | 35 | tiny, single seq (only 8 programs) |
| 1d0cc342 | 294 | 3 | 98 | small |
| 15856e8c | 3028 | 5 | 606 | medium |
| 5d3fc66a | 8192 | 34 | 241 | large (best parallelism: 272 seq·head programs) |

Implications:
- Two of five workloads are *tiny* (T=35, 42). There, fixed overheads (kernel launches, gate
  precompute, `.item()` host syncs on `cu_seqlens`) dominate. Minimizing launch count and avoiding
  host↔device syncs is as important as the main kernel. Prefer passing `cu_seqlens` to the kernel and
  computing grids without per‑sequence `.item()` loops on the host.
- The large workload rewards the chunked/MMA path and good occupancy; small ones reward low overhead.
- A single immutable kernel must handle all five ⇒ favor a design whose chunk size / block sizes are
  robust across `seq_len` from ~10 to ~600 (mask/pad partial chunks correctly).

---

## 3. Numerical risks

The reference computes **everything in float32** (bf16 operands cast up, f32 matmuls, f32 state).
Output is bf16, `new_state` is f32. Both are checked. Risks, ordered by importance:

1. **State matmul precision (TF32 vs f32).** On Ampere, `tl.dot` with f32 operands defaults to TF32
   (~10‑bit mantissa). The matmuls that touch the f32 state (`k @ old_state`, `q @ S₀`, and the
   `Kᵀ @ U` state update) would then round f32 operands to TF32. Over up to `T/C ≈ 128` sequential
   chunks the error in `new_state` (a checked f32 output) can accumulate. Mitigations to keep in the
   toolbox: use `tl.dot(..., input_precision="ieee")` (true f32, slower) or `"tf32x3"` (3‑pass, near
   f32) for state‑touching matmuls; keep the state accumulator itself in f32 registers/SMEM and add
   (not MMA) rank‑1 updates in the naive path. The bf16→f32 output has ~8‑bit mantissa, so TF32 is
   *likely* fine for `output`; `new_state` is the tighter target.

2. **bf16 operand matmuls are essentially exact vs reference.** `q,k,v` are already bf16 in DRAM, so
   a bf16×bf16→f32 MMA computes the same products as the reference's f32 matmul of bf16‑origin
   operands, only differing in accumulation order. So `q k^T`, `k k^T` intra‑chunk matmuls carry
   negligible extra error — the precision worry is specifically the *f32* state operands (point 1).

3. **Gate stability / log‑space accumulation.** Compute `log g = -exp(A_log)*softplus(x)` and do the
   within‑chunk cumulative decay as a **cumsum in log space (f32)**, exponentiating ratios
   `exp(Γ_i − Γ_l)` (always `≤ 0` exponent ⇒ `≤ 1`, no overflow; underflow to 0 is the physically
   correct vanishing of old contributions). Use a numerically safe `softplus` (`log1p(exp(x))` with
   the `x` large branch → `x`) to avoid `exp` overflow for large `x`.

4. **UT transform conditioning.** The `(I + tril(M,-1))⁻¹` solve within a chunk is well‑conditioned
   for reasonable `beta` and decay (strictly lower‑triangular nilpotent ⇒ the inverse is a finite
   Neumann series of length `C`), but large `beta·(k·k)` products with slow decay can amplify error.
   Solve in f32. Smaller chunk `C` (e.g. 32) reduces conditioning risk at some throughput cost.

5. **Sigmoid/softplus in bf16 inputs.** `a`, `b` are bf16 — cast to f32 *before* `+dt_bias`,
   `softplus`, `sigmoid` (reference does `a.float()`, `b.float()`). Doing gate math in bf16 would
   diverge.

6. **Empty / partial chunks & masking.** Partial last chunks (seq_len not divisible by `C`) must be
   masked so out‑of‑range tokens contribute 0 to matmuls, the UT solve, and the state update.
   Off‑by‑one in the inclusive‑diagonal mask flips the semantics.

7. **Accumulation order across chunks.** The reference is a pure left‑to‑right scalar recurrence.
   Chunking changes summation order; combined with TF32 this is the main source of `new_state`
   drift. The **naive per‑token Triton path reproduces the exact recurrence order** and is the safest
   correctness anchor.

8. **Tolerances are not specified** in `definition.json`. Treat them as unknown; the official
   evaluator is the source of truth (pass/fail per workload). Because I cannot run a local harness,
   the first candidate must be the highest‑confidence (order‑preserving, f32) implementation to
   establish that correctness passes at all, before trading precision for speed.

---

## 4. Triton design space

Primitives available on `sm_80`: bf16/f16 tensor cores (bf16×bf16→f32), TF32 for f32 MMA, up to
~164 KB shared memory per SM (opt‑in), `tl.dot`, `tl.cumsum`, `tl.associative_scan`,
`input_precision` control. No TMA/tcgen05/warp‑spec (those are Hopper/Blackwell).

State footprint per (seq, head): `[K=128, V=128]` f32 = **64 KB**. This is the central resource
constraint: it fits in registers distributed across a warp‑group accumulator, or in SMEM, but limits
occupancy. Splitting `V` (or `K`) into blocks (e.g. `BV=64`) halves the tile to 32 KB and raises the
program count — a key tuning axis.

### Axis A — algorithmic strategy

- **A0. Naive per‑token recurrence (correctness anchor / first candidate).**
  One program per `(seq, v_head)` (grid `N×8`). Keep `S = [128,128]` f32 in registers/SMEM; loop
  over the sequence's tokens performing exact rank‑1 updates and the `q@S` output with f32 (no
  tensor cores, or bf16 MMA only where exact). Reproduces the reference's order → highest correctness
  confidence. Far faster than the Python reference (no host loop, no per‑token kernel launch), but
  low arithmetic intensity and serial per head. Good baseline to *pass first*, then beat.

- **A1. Chunked gated delta rule (UT/WY transform) — main performance target.**
  Chunk size `C ∈ {16,32,64}`. Per (seq,head): preprocess gates, build decayed intra matrices, solve
  the `[C,C]` UT transform, sequential inter‑chunk state scan, then intra+inter output. Uses tensor
  cores for the `[C,·]` matmuls. Best throughput on the medium/large workloads. Higher
  implementation and numerical risk (§3.1, §3.4). Typically several fused sub‑kernels.

- **A2. Hybrid.** Naive per‑token for tiny sequences (T≤ threshold), chunked for the rest — but a
  *single immutable kernel/config* must serve all five workloads, so any branch must be inside the
  kernel or chosen by host‑side shape logic that is still one candidate. Keep as a later refinement.

### Axis B — parallelization & tiling

- Grid over `(seq, head)`; optionally add a `V`‑block (`BV`) axis for occupancy. The inter‑chunk scan
  is sequential per (seq,head,V‑block); the `K` reduction is internal.
- `BK`, `BV` block sizes (64 or 128); `C` chunk size; `num_warps ∈ {4,8}`; `num_stages ∈ {2,3,4}`.
- For tiny workloads, favor fewer/larger programs to cut launch overhead; for the large workload,
  favor more programs (smaller `BV`) for occupancy. This tension is the main autotune target.

### Axis C — precision knobs

- `input_precision`/`allow_tf32` per matmul: bf16 operands → default; f32/state operands →
  `"ieee"` or `"tf32x3"` (§3.1). Expose as a compile‑time constant to A/B against the evaluator.
- Keep gate math and the log‑cumulative decay strictly in f32.

### Axis D — layout & memory

- Load `state` `[V,K]`→ transpose to `[K,V]` on load; store `new_state` transposed back. Use masked
  loads for partial chunks and to zero empty sequences' `new_state`.
- Avoid materializing GVA‑expanded `q_exp`/`k_exp` in DRAM: index `head//2` inside the kernel.
- Compute `g`/`beta` either in a small fused preprocessing kernel over `[T,8]` or inline per chunk.
  Inline avoids an extra pass but recomputes; a tiny separate pass is cheap and simplifies the main
  kernel — decide by measurement.
- Minimize host↔device syncs: pass `cu_seqlens` to the kernel and derive per‑program `(start, len)`
  on device; avoid `.item()` per sequence (hurts the tiny workloads most).

### Axis E — launch structure

- Single fused main kernel vs. a small pipeline (gate precompute → chunk prep → scan → output).
  Fewer launches help tiny workloads; a pipeline can raise throughput on the large one. Start fused
  and simple; split only if profiling‑by‑evaluation (indirect) suggests it.

### Candidate ladder (sequencing intent, not a plan)

1. `c001` = A0 (order‑preserving f32 recurrence) — establish correctness passes on all five.
2. Introduce chunking / tensor cores (A1) with conservative `C` and high‑precision state matmuls;
   confirm correctness still passes, measure speedup.
3. Tune precision knobs (Axis C), block sizes and chunk size (Axis B), and launch structure (Axis E)
   for geomean, re‑validating correctness at each immutable step.
4. Optionally hybridize (A2) for the tiny workloads if they dominate the geomean.

---

## 5. Validation strategy

- **Source of truth = official evaluator only.** I cannot run CUDA, a profiler, or any local
  correctness harness (both by rule and because `Bash` is disabled here). Each
  `./scripts/evaluate_candidate.sh feedback cNNN` runs all five workloads and consumes one of the
  100 evaluations. Therefore: reason precisely on paper first; do not spend evaluations on
  speculative variants.
- **Correctness‑first ordering.** `c001` is deliberately the order‑preserving f32 recurrence so the
  first evaluation answers "does the pipeline wire up and pass tolerance at all?" before any
  precision/perf trade. Only after a green baseline do I introduce chunking and tensor cores.
- **Guardrails I will self‑check before every evaluation** (cheap, catches the likely bugs):
  1. State V↔K transpose on both load and store (k‑last `[N,H,V,K]` ⇄ internal `[K,V]`).
  2. GVA indexing `q_head = k_head = h // 2` for `h ∈ [0,8)`.
  3. Inclusive (diagonal) causal mask; output uses post‑update state; `scale` on output only.
  4. Gate math in f32: `x=a.f32()+dt_bias`, `g=exp(-exp(A_log)*softplus(x))`, `beta=sigmoid(b)`.
  5. Empty sequences (`seq_len ≤ 0`) ⇒ `new_state[seq] = 0` (matches reference `continue` before
     writing), `output` untouched (zeros) for those tokens (there are none).
  6. Output dtype bf16, `new_state` dtype f32; exact output/new_state shapes.
  7. Partial‑chunk masking so padding contributes 0 to matmuls / UT solve / state update.
  8. Handle `state is None` (all feedback workloads *do* pass a state, but keep the signature honest).
- **Metric interpretation.** Rank by geometric‑mean speedup across the five workloads; a candidate is
  only valid if *all five* pass. Because two workloads are tiny and two are medium/large, the geomean
  balances launch overhead vs. throughput — I will watch that a throughput win on the large workload
  doesn't regress the tiny ones (and vice versa).
- **Evidence recording.** For every evaluated candidate append one complete JSON record to
  `candidates.jsonl` with: parent, source hash, hypothesis, validation status, per‑workload result
  (pass/fail + speedup), geomean, decision (keep/reject/branch), cumulative evaluation count, and
  skill usage. Never rewrite earlier records.
- **Convergence / stop.** Stop at the evaluation or token budget, or when successive immutable
  candidates no longer improve geomean meaningfully; then write `SEARCH_COMPLETE` with the reason.
  Never run `final` without explicit operator approval.

---

## 6. Skill applicability note

- **KernelWiki** targets NVIDIA **Blackwell (SM100/B200)** and **Hopper (SM90/H100)** techniques
  (tcgen05/TMEM/CLC/NVFP4/2‑SM, warp specialization, TMA, FA‑4, etc.). This task runs on **A800
  (`sm_80`, Ampere)**, which has none of those features, so KernelWiki is out of scope and I do not
  expect to invoke it; the relevant primitives here are Ampere bf16/TF32 tensor cores, SMEM, and
  Triton scans. If a genuinely Hopper/Blackwell‑specific question arises it would not apply to this
  hardware anyway.
- **ncu-report-skill** requires running Nsight Compute on a B200 (`sm_100`) and profiling, which is
  both hardware‑mismatched and forbidden here ("do not run a profiler"). I will therefore not use it;
  performance signal comes indirectly from the official evaluator's measured speedups.

---

## 7. Open questions to resolve empirically (via the evaluator, minimizing spend)

1. Exact correctness tolerance — inferred from whether `c001` (f32, order‑preserving) passes.
2. What baseline the speedup is measured against (naive Python reference vs. an optimized kernel).
   The Python‑loop reference is extremely slow, so even A0 should show large speedups; the geomean
   scale will reveal the baseline's nature.
3. Whether TF32 state matmuls stay within `new_state` tolerance, or `"ieee"`/`"tf32x3"` is required.
4. Best `C`/`BV`/`num_warps`/`num_stages` per the workload mix, and whether the tiny workloads need a
   low‑overhead path (A2) to protect the geomean.

**Next step:** write `docs/plan.md` (executable plan), then implement `c001` as the order‑preserving
f32 correctness anchor. Not done in this turn.
