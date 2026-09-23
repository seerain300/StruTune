# Draft — `gdn_prefill_qk4_v8_d128_k_last` (Gated Delta Net prefill, k-last, H100/sm_90)

Status: draft only. No plan, no solution code created this turn.

---

## 1. What the operation computes

This is the **prefill** pass of a **Gated Delta Net** (GDN) linear-attention layer captured
from Qwen3-Next linear-attention blocks (TP=4), in **GVA** mode with a **k-last** recurrent
state layout. It is a per-sequence, per-head causal linear recurrence with an error-correcting
("delta rule") state update and a scalar per-token forget gate.

### 1.1 Signature (from `task/definition.json`)

`run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)` → `(output, new_state)`

Fixed axes (constants): `num_q_heads = 4`, `num_k_heads = 4`, `num_v_heads = 8`,
`head_size = 128`. Variable axes: `total_seq_len` (T), `num_seqs` (N), `len_cu_seqlens = N+1`.

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `q` | `[T, 4, 128]` | bf16 | query, K-space (head_size = K = 128) |
| `k` | `[T, 4, 128]` | bf16 | key, K-space |
| `v` | `[T, 8, 128]` | bf16 | value, V-space (V = 128) |
| `state` | `[N, 8, 128, 128]` | fp32 | recurrent state, **k-last** `[N, H, V, K]`; **optional** (may be `None`) |
| `A_log` | `[8]` | fp32 | learnable log-decay, per v-head |
| `a` | `[T, 8]` | bf16 | input-dependent decay, per (token, v-head) |
| `dt_bias` | `[8]` | fp32 | decay bias, per v-head |
| `b` | `[T, 8]` | bf16 | update-gate input, per (token, v-head) |
| `cu_seqlens` | `[N+1]` | int64 | variable-length batch offsets |
| `scale` | scalar | fp32 | output scale; here always `0.08838834764831843 = 1/sqrt(128)` |
| **`output`** | `[T, 8, 128]` | bf16 | attention output (V-space, follows v-heads) |
| **`new_state`** | `[N, 8, 128, 128]` | fp32 | updated state, k-last `[N, H, V, K]` |

Constraints: `len_cu_seqlens == num_seqs + 1`, `total_seq_len == cu_seqlens[-1]`.

### 1.2 Gate derivation (elementwise, per token × v-head)

```
x    = a.float() + dt_bias.float()          # [T, 8]
g    = exp( -exp(A_log.float()) * softplus(x) )   # [T, 8]  in (0, 1]
beta = sigmoid(b.float())                    # [T, 8]  in (0, 1)
```
`g` is a **scalar** decay per (token, head): it multiplies the whole `[K,V]` state matrix.
`beta` is the delta-rule update strength.

### 1.3 GVA head expansion

`num_v_heads / num_q_heads = 8/4 = 2`. The reference does
`q_exp = q.repeat_interleave(2, dim=1)` and `k_exp = k.repeat_interleave(2, dim=1)`, so
q-head `h` and k-head `h` each feed **two** consecutive v-heads `2h` and `2h+1`.
`num_sab_heads = max(4,8) = 8`; both `output` and `new_state` carry 8 heads.

Practical consequence: iterate over the **8 v-heads**; for v-head `hv` read q/k from
head `hv // 2`. No physical `repeat_interleave` materialization is needed — just index q/k
with `hv >> 1`.

### 1.4 Per-token recurrence (reference, exact)

Internally the reference works in `[K, V]` layout (it transposes the k-last `[V,K]` state to
`[K,V]` at entry, and transposes back at exit). For each token `t` in a sequence, per head:

```
old   = g_t * S            # decay first          [K,V]
old_v = k_t @ old          # k_t is [1,K] row     [1,V]
new_v = beta_t * v_t + (1 - beta_t) * old_v       [1,V]
S     = old - k_t^T @ old_v + k_t^T @ new_v        [K,V]
o_t   = scale * (q_t @ S)                          [1,V]   ← uses POST-update S
```

I verified on CPU (float64) that this collapses **exactly** to the standard scalar-gated
delta rule:

```
u_t = beta_t * ( v_t - k_t @ (g_t * S_{t-1}) )        # pseudo-value row [1,V]
S_t = g_t * S_{t-1} + k_t^T @ u_t                      # rank-1 update
o_t = scale * (q_t @ S_t)                              # output from post-update state
```
(`max|Δ| ≈ 2.2e-16`, output match True.) This compact form is the target for a fast kernel.

Two structural facts that matter for kernelization:
- **Output uses the post-update state** `S_t`, i.e. the current token's own (k,v) contribution
  is included in its output. Any chunked formulation must include the diagonal/self term.
- The gate `g_t` is a **scalar per head** (not per K/V dimension), so within a chunk the
  cumulative decay is a scalar product `γ_i = Π_{j≤i} g_j` — this is what makes the standard
  chunked (matmul) reformulation tractable.

---

## 2. Constraints, edge cases, and hazards

1. **`state` may be `None`.** When absent, initial `S = 0`. The evaluator's feedback set uses
   `state` present for every workload (all list a `state` tensor), but the signature marks it
   optional and the reference branches on `state is None`. The kernel/launcher must handle both.
2. **k-last I/O layout.** Input `state[n]` is `[V, K]` per head; the math wants `[K, V]`.
   Output `new_state[n]` must be written back as `[V, K]`. This is a transpose at the
   boundary — either transpose in the launcher (torch, allowed for plumbing) or index with
   swapped strides inside the kernel. Getting this transpose wrong is the most likely
   correctness bug and is invisible for symmetric shapes (K=V=128), so it must be tested
   against a non-symmetric probe or by matching the reference bit-layout carefully.
3. **Variable-length batching via `cu_seqlens`.** Sequences are concatenated along T; each has
   its own independent state. `seq_len` ranges from a handful to a few thousand. The reference
   explicitly `continue`s on `seq_len <= 0`; a zero-length sequence must still leave a valid
   (unchanged / initial) `new_state[n]`. Boundaries are **not** chunk-aligned, so any chunked
   kernel must mask/handle the ragged final chunk of each sequence.
4. **Tiny sequences dominate the count.** Workload census (100 workloads):
   `total_seq_len` min 6 / median 139 / max 8192; `num_seqs` 1–57.
   Buckets: `T≤64`: **28**, `65–512`: **32**, `513–4096`: **22**, `>4096`: **18**
   (16 of them exactly 8192, with 20–57 sequences → avg 140–410 tokens/seq).
   Because the metric is a **geometric mean over all 100 workloads**, the many small shapes
   (launch-overhead / fixed-cost bound) weigh as heavily as the 8192-token throughput cases.
   Minimizing kernel-launch count and per-launch overhead is as important as tensor-core MFU.
5. **q/k are only 4 heads, v/output/state are 8 heads.** Do not accidentally produce a
   4-head output. `output` and `new_state` are 8-head.
6. **`scale` applies only to `output`, not to `new_state`.** New state is the raw recurrent
   matrix; only the read-out `q @ S` is scaled.
7. **Grid / occupancy holder.** The controller locks one H100 and holds occupancy; profiling
   releases it temporarily. Do not run profiling and evaluation concurrently (return code 3
   wastes a budget slot).

---

## 3. Numerical risks

The reference is deliberately **float32** throughout (`matmul` casts to `.float()`), accumulates
the recurrence in fp32, and casts only the final `output` to bf16. `new_state` stays fp32.

- **Accumulation precision.** The kernel must accumulate the state and all matmuls in **fp32**.
  bf16 inputs are fine (they match the reference's fp32-upcast-of-bf16 values exactly, since
  upcast is lossless), but partial sums must be fp32. A pure-bf16 tensor-core path (bf16
  accumulate) would drift far outside tolerance over thousands of tokens.
- **Chunked vs sequential order.** A chunked/WY reformulation is *algebraically* equal to the
  sequential recurrence but **not bitwise** — summation order differs. Over up to ~2854
  tokens/seq the fp32 `new_state` can diverge from the reference at the ~1e-4…1e-3 relative
  level. Since `new_state` is a checked fp32 output, this is the **single biggest correctness
  risk**. The evaluator's tolerance is **not stated** in `definition.json`; c001 must probe it
  empirically. Mitigations if too tight: smaller chunk size (less reordering), Kahan-style or
  higher-precision intra-chunk accumulation, or a within-chunk sequential inner loop.
- **Gate transcendentals.** `g = exp(-exp(A_log) * softplus(a+dt_bias))`. `exp(A_log)` and
  `softplus` are both non-negative, product ≥ 0, so `g ∈ (0,1]` — no overflow of `g` itself.
  But `softplus(x) = log(1+exp(x))` overflows for large `x`; use the numerically stable form
  `softplus(x) = max(x,0) + log1p(exp(-|x|))` (Triton: build from `tl.log`/`tl.exp` or use the
  stable identity). `a` is bf16 so `x` magnitude is bounded, but stability is cheap insurance.
  Compute gates in **fp32** to match the reference (`a.float()`, `dt_bias.float()`).
- **Cumulative decay underflow.** Within a long chunk, `γ_i = Π g_j` can get very small; if
  the chunked math divides by `γ` (as in `k̃ = k / γ` style WY formulations) this risks
  catastrophic cancellation / overflow. Prefer formulations that multiply by decays rather
  than divide, or bound chunk size so `Π g` stays representable. This is a known GDN chunk
  pitfall (small chunk sizes, e.g. 64, are standard for exactly this reason).
- **bf16 output rounding.** Final `output` cast to bf16 gives ~2^-8 relative granularity;
  small absolute errors below that are invisible, which gives the output check natural slack.
  The state check has no such cushion.
- **Empty / length-1 sequences.** `seq_len==1` still runs one update; `seq_len==0` writes the
  initial state unchanged. `total_seq_len==6` (workload 1) exercises the extreme small path.

---

## 4. Triton design space

Fixed problem geometry per (sequence, head): state `S` is `[K=128, V=128]` fp32 = 64 KiB.
There are 8 heads. Work is intrinsically sequential **across chunks/tokens within a sequence**,
fully parallel **across sequences and heads**.

### 4.1 Parallelization axes
- **Embarrassingly parallel:** `num_seqs × num_v_heads` (N×8). For the 8192/57-seq case that is
  up to 456 independent recurrences — ample for H100's 132 SMs. For the `N=1` small cases only
  8 programs exist, so those are latency-bound and want a low-overhead kernel.
- **Sequential:** chunk/token progression within each sequence's state.
- A `[128,128]` state tile maps naturally onto a Triton block; K and V both = 128 fit a single
  program's registers/SMEM comfortably at bf16 inputs + fp32 accumulator.

### 4.2 Candidate algorithms

**(A) Token-recurrent kernel (simplest, correctness-first).**
One program per (seq, head); keep `S[128,128]` in fp32; loop tokens doing the rank-1 compact
update of §1.4. Pros: trivially matches the reference order → tightest numerical agreement on
`new_state`; simple; handles ragged lengths naturally. Cons: rank-1 outer products give poor
tensor-core utilization; the 8192-token workloads become long serial loops → weak throughput.
Good as **c001 correctness anchor / tolerance probe**, likely not the final performer.

**(B) Chunked (matmul) delta rule — FLA `chunk_gated_delta_rule` style.**
Split each sequence into chunks of size `C` (e.g. 32/64/128). Within a chunk, form the
intra-chunk interaction as matmuls; because the delta term makes token `i`'s pseudo-value
depend on earlier tokens *in the same chunk*, resolve the lower-triangular dependence with the
**UT / WY transform** (invert `(I − tril(diag(β) K̃K̃ᵀ, −1))`), then:
  - intra-chunk output = causal-masked `(Q̃ K̃ᵀ) U` plus decay-weighted terms,
  - inter-chunk output = `Q̃ @ S_chunk_start`,
  - state carry `S_{c+1} = γ_C·S_c + K̃ᵀ U` (decay-scaled).
Pros: turns the O(T) recurrence into `T/C` chunk steps of `[C,128]×[128,128]` matmuls → strong
MFU on the large batches. Cons: complex; WY inversion and decay scaling are the numerical
hot-spots (§3); ragged tails need masking. This is the standard high-throughput path and the
realistic route to a large geomean win on the 8192 cases.

**(C) Chunk-with-sequential-inner (hybrid).**
Chunk across sequences for parallelism but run a short sequential inner loop inside each chunk
(no WY inversion). Balances numerical fidelity (closer to reference order) against some
tensor-core use. Useful fallback if (B)'s state error exceeds tolerance.

### 4.3 Tuning knobs
- **Chunk size `C`** — trades MFU vs decay stability vs tail waste. Start `C=64`.
- **Head batching** — process all 8 heads in one program (share q/k reads for the 2 v-heads
  per q/k head) vs one head per program. GVA lets each q/k tile serve two v-heads, halving q/k
  traffic.
- **`num_warps` / `num_stages`** — pipeline the chunk loop; `[128,128]` fp32 accumulator limits
  register budget, so `num_warps=4` and modest staging are the starting point.
- **Grid granularity for tiny shapes** — a persistent / single-launch kernel that internally
  loops over sequences avoids per-sequence launch overhead that would dominate the 28 `T≤64`
  workloads and the `N=1` cases.
- **State-transpose placement** — do the k-last↔[K,V] transpose via strided indexing in-kernel
  (no extra kernel launch) rather than a separate torch transpose, to keep launch count minimal.

### 4.4 Hopper-specific notes (from KernelWiki)
`kernel-gated-delta-net`, `technique-chunk-parallelism`, `contest-flashinfer-track-c`
(FlashInfer MLSys'26 Track C) confirm: prefill = chunk-based parallel, decode = streaming; the
Hopper prefill path is the established/"done" one (the contest's open frontier is *Blackwell*,
not Hopper). For sm_90 the practical toolset is **Triton wgmma-backed `tl.dot`** on fp32-accum;
small chunk sizes are recommended precisely to control the gating/decay numerics. TFLA-style
two-level tiling is the throughput ceiling but is well beyond a first candidate. These pages are
architecture/context references (bf16 inputs, fp32 accumulate, chunk C≈64), not drop-in code.

---

## 5. Environment facts (observed)

- Benchmark interpreter (via `ncu_profile.sh` / evaluator): torch driven; the conda python at
  `/home/ziming/miniconda3/bin/python` reports **torch 2.8.0+cu128, triton 3.4.0** (CUDA 12.8).
  `fla`/`flashinfer` are **not** importable → the solution must be **self-contained Triton**,
  not a wrapper around an installed GDN library. `einops` is present.
- GPU: single **H100 80GB HBM3** (`sm_90`), controller-locked, occupancy held.
- No tolerance field is present in `definition.json`; tolerance is enforced by the hidden
  evaluator and must be discovered empirically via c001.
- Baseline for the speedup metric is the reference `run(...)` in `definition.json` — a pure
  sequential fp32 Python double-loop, i.e. very slow; correctness (every selected workload must
  pass) gates the ranking.

## 6. Validation strategy

- **CPU algebra pre-checks (host-only, no GPU/kernel):** already confirmed the compact
  delta-rule reduction (§1.4). Reuse the same style to sanity-check the chunked math offline in
  float64 against the reference recurrence *before* writing Triton, especially the WY transform
  and the k-last transpose (use an asymmetric K≠V toy to catch transpose bugs that K=V=128
  hides).
- **Candidate cadence (immutable, sequential):**
  - **c001** = algorithm (A) token-recurrent, correctness anchor + **tolerance probe**. Its per
    -workload pass/fail reveals the evaluator's rtol/atol and whether fp32 state agreement is
    the binding constraint. Record parent=none, source hash, hypothesis, full per-workload
    results, geomean, decision, cumulative eval count, skill usage in `candidates.jsonl`.
  - **c002+** = introduce chunking (B/C) once tolerance is known; tune `C`, head batching,
    warps/stages; consider a persistent single-launch kernel for the small-shape tail.
  - Any source/config/launch change ⇒ new candidate ID; never mutate an evaluated version.
- **Evaluate only** with `./scripts/evaluate_candidate.sh feedback cNNN` (full 100-workload set
  = one evaluation). Budget: 100 evaluations; token soft/normal/absolute 9M/10M/11M.
- **Profiling** only via `./scripts/ncu_profile.sh` using the `ncu-report-skill` workflow, and
  **never** concurrent with an evaluation (foreign-process interference → rc 3, wasted slot).
  Profile the 8192-token workloads (throughput frontier) to guide chunk/warp tuning; the tiny
  shapes are launch-overhead bound and are better addressed by design (single-launch) than by
  profiling.
- **Correctness first, then speed:** a failing Triton kernel is invalid — no Torch/CPU/NumPy
  fallback is permitted. Keep an fp32 accumulation invariant in every candidate.
- **Convergence / stop:** stop at budget, token limit, or when geomean improvement genuinely
  converges; then write `SEARCH_COMPLETE` with the reason. `final` only on explicit operator
  approval.

## 7. Open questions to resolve empirically (via c001, not assumption)

1. Evaluator tolerance for bf16 `output` and fp32 `new_state` — is chunked-order state drift
   acceptable, or must c001-style sequential order be preserved?
2. Is `state` ever `None` in the hidden feedback set, or always provided? (Handle both; feedback
   file always lists it, but the signature/reference allow `None`.)
3. Relative cost split between the small-shape tail (60 workloads ≤512 tokens) and the 8192
   cases — determines whether launch-overhead reduction or MFU dominates the geomean.
