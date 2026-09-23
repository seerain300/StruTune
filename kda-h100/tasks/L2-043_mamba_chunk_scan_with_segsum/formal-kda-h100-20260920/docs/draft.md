# Draft — L2/043 Mamba-2 Chunk Scan with Segment Sum (H100 / sm_90)

Status: analysis only. No code, no plan yet. This document establishes a correct
mathematical model of the reference, enumerates constraints and numerical risks,
maps the Triton design space, and defines the validation strategy that the later
`docs/plan.md` and candidates must satisfy.

---

## 1. Operation summary

This is the core Mamba-2 **Structured State-Space Duality (SSD)** chunk-scan
forward pass (the `torch_forward` reference from HF `Mamba2Mixer`, specialized to
one call). It maps a discretized SSM over a sequence, split into fixed chunks, and
produces both the per-token output and the final recurrent state.

### 1.1 Fixed problem constants (from `task/definition.json`)

| Symbol | Meaning | Value |
|---|---|---|
| `H` (num_heads) | heads | 16 |
| `P` (head_dim) | value/head dim | 64 |
| `N` (state_size / dstate) | SSM state dim | 256 |
| `G` (n_groups) | B/C groups | 1 |
| `Q` (chunk_size) | scan chunk length | 256 |
| `hidden_out` | output width | `H*P = 1024` |

Variable axes: `batch_size` (B) and `seq_len` (S). Feedback set covers
`B ∈ {1,2,4,8,16,32}`, `S ∈ {256, 293, 512, 541, 997, 1024, 1321, 1879, 2048, 4096}`.

### 1.2 Inputs / outputs (all bf16 on device)

- `hidden_states x`: `[B, S, H, P]`
- `A`: `[B, H, S]` — discretized `A` (negative log decay), **note the transposed
  layout** vs `x` (head before seq).
- `B_mat`: `[B, S, G=1, N]`
- `C_mat`: `[B, S, G=1, N]`
- `D`: `[H]` — per-head skip scalar
- `initial_states`: `[B, H, P, N]`
- Output `output`: `[B, S, H*P=1024]`
- Output `final_state`: `[B, H, P, N]`

Because `G=1`, **B and C are shared across all 16 heads** (broadcast). This is the
single most important structural fact for optimization.

---

## 2. Exact math model of the reference

Let `Q=256` be chunk length, `pad = (Q - S%Q) % Q`, `S_pad = S+pad`,
`NC = S_pad/Q` chunks. Everything below is done in **float32** internally, then the
final results are cast to bf16.

Per chunk `c`, per head `h`, let `a` be the length-`Q` slice of `A` (padded with 0),
and define the within-chunk cumulative sum:

```
acs[t] = sum_{u=0..t} a[u]            (t = 0..Q-1)     # torch.cumsum
```

### 2.1 Intra-chunk (diagonal) term

The `segment_sum` helper builds the lower-triangular decay matrix
```
L[i,j] = exp( acs[i] - acs[j] )      for i >= j     (i,j in 0..Q-1)
       = 0                            for i <  j     (masked to -inf then exp)
L[i,i] = 1
```
(Verified: `segment_sum` computes `sum_{u=j+1..i} a[u] = acs[i]-acs[j]` with a
strict-lower `tril(diag=-1)` mask then `tril(diag=0)` fill with `-inf`.)

B/C contraction over state (head-independent because G=1):
```
CB[i,j] = sum_{s=0..N-1} C[i,s] * B[j,s]          # [Q, Q], SAME for all heads
M[h,i,j] = CB[i,j] * L[h,i,j]                      # per head (L depends on A[h])
Y_diag[h,i,d] = sum_{j<=i} M[h,i,j] * x[h,j,d]     # causal, [Q, P]
```
So the diagonal term is a **decayed causal attention**: scores `CB` (contraction of
C·B over the 256 state dim), decay mask `L`, values `x`.

### 2.2 Per-chunk state contribution

```
decay_states[h,t] = exp( acs[Q-1] - acs[h,t] )          # decay to chunk end
chunk_state[h,d,s] = sum_{t=0..Q-1} x[h,t,d] * B[t,s] * decay_states[h,t]   # [P, N]
```
Equivalently `chunk_state = x^T @ (B ⊙ decay_states)`.

### 2.3 Inter-chunk recurrence (state passing)

Let `dchunk[h,c] = acs_c[Q-1]` be the total log-decay of chunk `c` for head `h`.
With `S_0 = initial_states`, the reference `decay_chunk = exp(segment_sum(...))`
einsum is exactly the linear recurrence (proven by expanding the segment sum):
```
S_c = exp(dchunk[h,c-1]) * S_{c-1} + chunk_state_{c-1}      # [P, N], c = 1..NC
```
- `states_out[c] = S_c` is the state **entering** chunk `c` (used by Y_off).
- `final_state = S_{NC}` (i.e. `new_states[:, -1]`).

### 2.4 Off-diagonal (state→output) term

```
state_decay_out[h,t] = exp( acs[h,t] )
Y_off[h,t,d] = ( sum_{s} C[t,s] * S_c[d,s] ) * state_decay_out[h,t]    # [Q, P]
```

### 2.5 Combine + residual + reshape

```
y[h,t,d] = Y_diag[h,t,d] + Y_off[h,t,d] + D[h] * x[h,t,d]       # fp32
```
Slice off padding (`t < S`), reshape `[B,S,H,P] -> [B,S,H*P]` (head-major, dim-minor),
cast to bf16. `final_state` cast to bf16.

---

## 3. Why the reference is slow (optimization target)

The reference materializes several very large fp32 intermediates in HBM and runs
many separate kernels/einsums:

- `L = exp(segment_sum(A))`: `[B, H, NC, Q, Q]` fp32. For `B=4, S=2048` this is
  `4·16·8·256·256·4B = 537 MB`. For `B=8,S=1024`: 537 MB. This dominates.
- `G/CB`: `[B, NC, Q, Q, H]` fp32 — another ~half-GB class tensor, and it even
  carries a redundant `H` axis despite being head-independent.
- `M = G*L`, `Y_diag` einsum, `states` einsum, `decay_chunk` einsum — each a full
  pass over these giant tensors.

So the reference is **memory-bandwidth bound on giant `Q×Q` intermediates**, with a
redundant head axis on `CB`, plus many launch/allocation overheads. A fused Triton
implementation that keeps the `Q×Q` blocks in registers/SMEM and never writes them
to HBM should win large, especially on the big-`Q×Q` shapes.

Key exploitable redundancies:
1. `CB` is identical for all 16 heads → compute once per `(b, chunk)`.
2. `L`, decays are rank-structured (`exp(acs[i]-acs[j])`) → never materialize the
   full `[Q,Q]`; derive from a length-`Q` cumsum on the fly.
3. Padding is zero-fill → handled by masked loads, no separate pad kernel.

---

## 4. Triton design space

The inter-chunk recurrence (§2.3) is the only sequential dependency; the diagonal
term and per-chunk states are fully parallel. This yields the canonical 3-stage
decomposition (as in `mamba_ssm` / FLA `mamba_chunk_scan_combined`). KernelWiki
`technique-chunk-parallelism` and `kernel-gated-delta-net` confirm this
intra-parallel / inter-sequential split as the standard pattern; the H100 target
means Triton `tl.dot` (bf16/tf32 → fp32 accumulate) tensor-core matmuls, not
tcgen05/TMEM (those are Blackwell-only and out of scope here).

### 4.1 Option A — three-kernel decomposition (robust, proven)

1. **`chunk_cumsum`**: per `(b, h, chunk)` compute `acs` (length-`Q` cumsum of A,
   padded 0) and chunk totals `dchunk`. Cheap; can be fused into the consumers via
   `tl.cumsum` instead of a standalone kernel.
2. **`chunk_state`**: parallel over `(b, chunk, h)` (+ tiling of `P`,`N`):
   `chunk_state = x^T @ (B ⊙ decay_states)` → `[B,NC,H,P,N]`.
3. **`state_passing`**: sequential over `c`, parallel over `(b,h)` and state
   elements (`P·N = 16384`, tiled): apply `S_c = exp(dchunk)·S_{c-1}+chunk_state`;
   emit `states_out` (entering-state per chunk) and `final_state`.
4. **`chunk_scan`**: parallel over `(b, chunk, h)` (+ row tiling of `Q`):
   compute `CB` (shared across heads within a `(b,chunk)` if we loop heads inside),
   apply `L`, causal mask → `Y_diag`; add `Y_off` from `states_out`; add `D·x`;
   write output with de-pad + reshape.

Pros: matches a battle-tested structure, easy to get correct, each kernel simple.
Cons: writes `chunk_state`/`states_out` to HBM (small: `[B,NC,H,P,N]`; e.g. B=8,
S=1024, NC=4 → `8·4·16·64·256·4B = 512 MB`… actually non-trivial). Mitigate by
storing these intermediates in bf16 or fp16, or fusing state-passing.

### 4.2 Option B — per-(b,h) sequential-over-chunks fused kernel

One program per `(b, h)`; loop chunks in order carrying state `S [P,N]=[64,256]` in
registers/SMEM. Within each chunk do diagonal term, Y_off from current `S`, then
update `S`. No HBM state intermediates; `final_state` from last iteration.

Pros: minimal HBM traffic; naturally correct recurrence; `final_state` free.
Cons: parallelism only `B·H` (as low as 16 for `B=1`), underutilizing 132 SMs on
small-batch shapes; `S [64,256]` fp32 = 64 KB register/SMEM pressure per program;
recomputes `CB` per head (loses the cross-head sharing) unless heads are looped
inside a `(b,chunk)`-centric layout (which conflicts with sequential-per-head).

### 4.3 Option C — hybrid (likely best)

- Parallel `chunk_state` + parallel `chunk_scan` (Option A kernels 2 & 4) to
  maximize SM occupancy and exploit `CB` head-sharing inside `chunk_scan`.
- A lightweight `state_passing` kernel for the short sequential scan (NC ≤ 16).
- Fuse `chunk_cumsum` into both consumers via `tl.cumsum` on a `[Q]` vector to
  avoid an extra HBM round-trip and extra kernel launch.

Decision for the plan will start from Option A/C (correctness first), measure, then
consider fusing state-passing to cut the `chunk_state` HBM intermediate.

### 4.4 Tiling and launch parameters to sweep

- `Q×Q` block: tile output rows `Q` into `BLOCK_M ∈ {32,64,128}`; contraction over
  `j` (also `Q`) into `BLOCK_N`; state contraction `N=256` into `BLOCK_K ∈ {64,128,256}`.
- Causal structure: skip `j`-blocks fully above the diagonal (block `j_start > i_end`).
- `P=64`, `N=256` are small/fixed — good for keeping `x` tiles resident.
- `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}` autotune candidates.
- Grid choices: `chunk_scan` grid `(B*NC, H, num_row_blocks)` or fold `H` into a
  loop to reuse `CB`. Evaluate both.

### 4.5 Triton feature availability (to confirm empirically in c001)

- `tl.cumsum(x, axis=0)` for the length-`Q` prefix sum of A.
- `tl.dot` with bf16 inputs / fp32 accumulator; optionally `allow_tf32`.
- If `tl.cumsum` is unavailable/slow, fall back to `tl.associative_scan` or an
  explicit log-step scan, or precompute `acs` in a tiny standalone kernel.

---

## 5. Numerical risks and mitigations

Tolerances (per workload): `atol ∈ [0.014, 0.046]`, `rtol = 0.05`,
`required_match_ratio = 0.98` (2% of elements may fail). These are moderately loose,
which gives headroom — but the exponential decays create sharp risks:

1. **Exponential overflow / dynamic range.** `A` is "random" bf16; if it carries a
   positive component, `acs` over 256 steps can reach ±O(16). `L[i,j]=exp(acs[i]-acs[j])`
   and `state_decay_out=exp(acs[t])` can then be very large. fp32 max ≈ 3.4e38
   (exp≈88); differences up to ~±32 give `exp≈1e14`, safe in fp32 but risky if any
   intermediate is done in bf16 (max ≈ 3.4e38 but 8-bit mantissa). **Mitigation:**
   keep all cumsum/exp/decay math and matmul accumulation in **fp32**, mirroring the
   reference exactly; never store decays in bf16.
2. **Match the reference's factorization, not an algebraically-equal one.** The
   reference computes `L` via `exp(segment_sum)` (an independent cumsum), and
   `Y_off` via `exp(acs[t])`. Computing `L` as a ratio `state_decay_out[i]/[j]` or
   otherwise reassociating can drift. **Mitigation:** compute `L[i,j]=exp(acs[i]-acs[j])`
   from the same `acs` used everywhere; keep the diagonal `L[i,i]=1` exact via the
   causal mask (`i>=j`), `0` above diagonal.
3. **bf16 vs fp32 matmul products.** Inputs are bf16 (exactly representable); the
   reference upcasts to fp32 then multiplies (fp32-precision products) accumulated
   in fp32. A `tl.dot` on bf16 operands rounds each product to bf16 mantissa before
   accumulate — over `N=256` (CB) and `Q=256` (Y_diag) sums, and then amplified by
   large `L`, this could push some elements past atol. **Mitigation ladder:**
   (a) try bf16 `tl.dot` first (fastest) and measure match-ratio margin;
   (b) if marginal, use tf32 (`allow_tf32=True`, 10-bit mantissa) for the
   accuracy-critical contractions; (c) fp32 dot only as last resort (slow).
4. **Cumsum accumulation order.** `tl.cumsum` vs torch `cumsum` differ in rounding;
   over 256 fp32 adds the drift is tiny (≪ atol). Acceptable.
5. **Padding correctness.** Padded seq positions must load `x=0`, `B=0`, `C=0`, and
   `A=0` (so `exp(0)=1` decay but zero contribution because `x=0`, and chunk-end
   cumsum unchanged). Masked loads with `other=0.0` reproduce the reference exactly.
   Must verify on the non-multiple-of-256 shapes (293, 541, 997, 1321, 1879).
6. **Single-chunk / initial-state path.** For `S=256` (NC=1) the only inter-chunk
   term is from `initial_states` (S_0). `final_state = exp(dchunk_0)·S_0 + chunk_state_0`.
   Must not special-case away the initial state.
7. **Output layout.** `[B,S,H,P] → [B,S,H*P]` is head-major (`out[...,h*64+d]`).
   A transposed write (`d*16+h`) would silently fail correctness — pin this in the
   first candidate and verify.
8. **`D` residual in fp32 before cast.** `+ D[h]*x` added in fp32 pre-bf16-cast.

---

## 6. Validation strategy

The only sanctioned correctness/timing path is
`./scripts/evaluate_candidate.sh feedback <cid>` over the full 16-workload feedback
set (one evaluation per immutable candidate). Guardrails:

1. **Self-check before spending an evaluation.** Since I cannot run the official
   evaluator ad hoc, each candidate will embed an internal, disabled-by-default
   torch reference (copied semantics, used only in a local harness — never as a
   runtime fallback) so I can eyeball logic; but the *authoritative* check is the
   evaluator. The shipped `run(...)` must be Triton-only with no torch compute
   fallback (contract §"No Torch computational fallback").
2. **c001 = correctness-first, unoptimized-but-fused.** Simplest tiling that is
   obviously correct (e.g. `BLOCK_M=64`, full `N`/`Q` contraction, fp32 or tf32
   dots). Goal: pass all 16 workloads' correctness before chasing speed. A failing
   Triton kernel is invalid — do not paper over with torch.
3. **Shape coverage.** The feedback set already spans: exact multiples (256, 512,
   1024, 2048, 4096), non-multiples needing pad (293, 541, 997, 1321, 1879), single
   chunk (256), and batch scaling (1→32). Every candidate is judged on all of them,
   so boundary/pad bugs surface immediately.
4. **Numerical margin tracking.** Record per-workload pass/fail and, where the
   evaluator exposes it, the match ratio / max error, to see how much tolerance
   headroom each precision choice (bf16 vs tf32 dot) leaves before committing to the
   fastest safe variant.
5. **Profiling discipline.** Use `./scripts/ncu_profile.sh` (never raw `ncu`) only
   *between* evaluations, never concurrently (concurrent GPU use → controller
   discards the timing, return code 3, wastes an eval). Profile the largest shapes
   (`B=1,S=4096`; `B=4,S=2048`; `B=8,S=1024`) to find the dominant kernel and guide
   tiling/occupancy sweeps.
6. **Candidate hygiene.** Immutable IDs `c001, c002, …`; any source/config/launch
   change ⇒ new ID; append one JSON record per evaluated candidate to
   `candidates.jsonl` (parent, source hash, hypothesis, validation, per-workload
   result, geomean, decision, cumulative eval count, skill usage). Stop / write
   `SEARCH_COMPLETE` when converged; never run `final` without operator approval.

### 6.1 Success criteria

- **Correctness:** all 16 feedback workloads pass (atol/rtol/match-ratio) — mandatory.
- **Performance:** primary metric is geometric-mean speedup vs reference; target a
  large win driven by eliminating the giant `L`/`CB` HBM intermediates and the
  redundant head axis.

---

## 7. Open questions to resolve in the plan / early candidates

1. Installed Triton version and whether `tl.cumsum`/`tl.associative_scan` and
   `tl.dot(..., allow_tf32=...)` behave as expected on this sm_90 toolchain (probe in c001).
2. Whether fusing `state_passing` into `chunk_scan` (Option B/C) beats the clean
   3-kernel Option A once HBM traffic of `chunk_state` is measured.
3. Best precision for the two contractions (`CB` over `N=256`, `Y_diag`/`Y_off` over
   `Q`/`N`) that keeps ≥98% match with maximal speed (bf16 vs tf32).
4. Grid layout for `chunk_scan`: loop heads inside a `(b,chunk)` program to reuse
   `CB` vs one program per `(b,chunk,h)` for more parallelism on small batches.

---

## 8. Skill usage log (this draft)

- **KernelWiki**: `technique-chunk-parallelism` (intra-parallel / inter-sequential
  chunk scan pattern, chunk-size tradeoff), `kernel-gated-delta-net` (chunked
  linear-attention prefill structure, Triton `tl.dot` usage, precision caveats),
  `technique-pipeline-stages` (num_stages/occupancy guidance), and Mamba SSD PR
  references (`pr-flashinfer-2709` "Mamba2 SSD Combined Forward", plus vLLM/FLA
  chunk-scan kernels) confirming the standard `chunk_cumsum → chunk_state →
  state_passing → chunk_scan` decomposition. tcgen05/TMEM material noted as
  Blackwell-only and therefore not applicable to this sm_90 target.
- **ncu-report-skill**: not yet used; reserved for post-c001 profiling of the
  dominant kernel on the largest shapes.
