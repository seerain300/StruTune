# Draft — `gdn_decode_qk4_v8_d128_k_last` (H100 / sm_90)

Gated Delta Net single-token **decode** with GVA (grouped value attention) and a
**k-last** recurrent-state layout. Captured from Qwen3-Next linear-attention layers (TP=4).
Target: a Triton kernel exposed as `solution/solution.py::run(...)` that maximises geomean
speedup over the reference while every selected workload passes correctness.

---

## 1. Operation analysis

### 1.1 Signature (from `task/definition.json`)

Inputs:
- `q` : `[B, 1, Hq=4, D=128]` bf16
- `k` : `[B, 1, Hk=4, D=128]` bf16
- `v` : `[B, 1, Hv=8, D=128]` bf16
- `state` : `[B, Hv=8, V=128, K=128]` **f32**, k-last (`K` is the contiguous last dim). Optional → zeros.
- `A_log` : `[Hv=8]` f32  (log decay parameter, learnable)
- `a` : `[B, 1, Hv=8]` bf16  (input-dependent decay)
- `dt_bias` : `[Hv=8]` f32  (decay bias)
- `b` : `[B, 1, Hv=8]` bf16  (update-gate input)
- `scale` : scalar f32 (all feedback workloads use `0.08838834764831843 = 1/sqrt(128)`)

Outputs:
- `output` : `[B, 1, Hv=8, V=128]` bf16
- `new_state` : `[B, Hv=8, V=128, K=128]` f32, k-last

Constants for this definition: `Hq=Hk=4`, `Hv=8`, `K=V=D=128`, `T=1`. Only `batch_size` varies.

### 1.2 GVA head mapping

`num_v_heads // num_q_heads = 2`. The reference expands q,k via `repeat_interleave(2, dim=1)`,
so **v-head `h` uses q/k-head `h // 2`**:
```
h_v : 0 1 2 3 4 5 6 7
h_qk: 0 0 1 1 2 2 3 3
```
No data expansion is needed in the kernel — just index `q[b, h//2, :]`, `k[b, h//2, :]`.

### 1.3 Gate computation (per (b, h_v), scalar)
```
x    = a.float() + dt_bias.float()          # [B,1,Hv]
g    = exp(-exp(A_log) * softplus(x))        # decay gate  ∈ (0, 1]
beta = sigmoid(b.float())                    # update gate ∈ (0, 1)
```
`A_log`, `dt_bias` are per-v-head; `a`, `b` are per-(b, v-head). One `g`, one `beta` per head.

### 1.4 Delta-rule state update (reference, per (b,h))

The reference works in a transposed `[K,V]` frame: `h_state[k,v] = state[b,h,v,k]`
(`.transpose(-1,-2)` of the k-last `[V,K]` slice). With scalars `g`, `beta`:
```
old_state[k,v]   = g * h_state[k,v]                       # = g * state[b,h,v,k]
old_v[v]         = Σ_k k_h[k] * old_state[k,v]            # k · (g·state row)
new_v[v]         = beta*v_h[v] + (1-beta)*old_v[v]
state_remove     = outer(k_h, old_v)
state_update     = outer(k_h, new_v)
h_state_new[k,v] = old_state[k,v] - state_remove + state_update
output[v]        = scale * Σ_k q_h[k] * h_state_new[k,v]
new_state[b,h,v,k] = h_state_new[k,v]
```

### 1.5 Key algebraic simplification — **the whole op is independent per v-row**

Substituting `new_v - old_v = beta*(v_h - old_v)`:
```
h_state_new[k,v] = g*state[b,h,v,k] + k_h[k] * beta*(v_h[v] - old_v[v])
```
Define the k-last **row** `S_row = state[b,h,v,:]` (contiguous, length K=128). Then for a single v:
```
sk        = Σ_k k_h[k] * S_row[k]           # dot(k_h, S_row)
old_v     = g * sk
delta     = beta * (v_h[v] - old_v)         # scalar for this v
new_row[k]= g * S_row[k] + k_h[k] * delta   # length-K row → new_state[b,h,v,:]
out[v]    = scale * Σ_k q_h[k] * new_row[k]
```
**Every v-row (v = 0..127) is fully independent**, reads one contiguous 128-f32 row, and writes
one contiguous 128-f32 row. This is the central structural insight and it matches the "k-last"
layout perfectly: the reduction axis `k` is the contiguous axis, so a per-row reduction is a
coalesced load. Total independent work items = `B · Hv · V = B · 1024`.

Optional further fusion (algebraically exact, reorders f32 ops):
```
out[v] = scale * ( g*(q_h·S_row) + delta*(q_h·k_h) )
```
where `q_h·k_h` is one scalar per head. This avoids materialising `new_row` just to dot with q,
but still requires writing `new_row` to `new_state`. Treat as an optimisation to A/B test against
the faithful "materialise-then-dot" form (see §4, numerical risk).

### 1.6 Roofline / bottleneck

Per v-row: read 512 B (`S_row` f32×128) + write 512 B (`new_row`) + write 4 B (`out`) +
tiny per-head loads (`q,k,v` bf16×128, scalars). Compute ≈ 2 length-128 dot products + a
length-128 scale-and-FMA ≈ ~500 flops for ~1 KB traffic → arithmetic intensity ≪ roofline knee.
**This is a memory-bandwidth-bound kernel** dominated by streaming the 64 KB/head state in and out.

State traffic (read+write): `B · 8 · 64 KB · 2`.
- B=64 → 64 MB → ~19 µs at ~3.35 TB/s HBM3 (speed-of-light floor).
- B=1  → 1 MB → sub-µs of traffic, so the small-batch regime is **occupancy/launch-overhead
  bound**, not bandwidth bound.

Tensor cores are irrelevant (matvecs on a memory-bound op); plain f32 FMA in-register is correct
and sufficient. Confirmed against `pattern-memory-bound` in KernelWiki: optimise wide/coalesced
loads, cache policy, occupancy — *not* compute.

---

## 2. Workload set

`task/feedback_workloads.jsonl` = **54 workloads** (this is the full/final set too). Only
`batch_size` varies; scale is constant `1/sqrt(128)`.

| batch_size | # workloads | independent v-rows (B·1024) | regime |
|-----------:|------------:|----------------------------:|--------|
| 1  | 10 | 1,024   | occupancy / launch-overhead bound |
| 4  | 8  | 4,096   | ramp-up |
| 8  | 7  | 8,192   | ramp-up |
| 16 | 7  | 16,384  | approaching BW-bound |
| 32 | 7  | 32,768  | BW-bound |
| 48 | 7  | 49,152  | BW-bound |
| 64 | 8  | 65,536  | BW-bound |

Geomean weights every workload equally, so the **small-batch cases (B=1..8, 25 of 54 workloads)
matter as much as the large ones**. A design that only saturates bandwidth at B=64 but launches
too few blocks / has high per-launch overhead at B=1 will drag the geomean down. H100 has 132 SMs;
B=1 gives only 1,024 independent rows — the grid/tiling must still produce enough blocks and
enough warps to hide latency there.

---

## 3. Constraints & isolation rules (operational)

- Primary implementation **must be Triton**; PyTorch only for metadata/launch plumbing. No Torch /
  CPU / NumPy / CUDA-extension / alternate-impl fallback — a failed Triton path is invalid.
- Immutable candidates `c001`, `c002`, … one source version at a time; any meaningful source /
  config / launch change ⇒ new candidate ID; never reuse an ID for changed source.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback <id>` (full 54-set = 1 evaluation).
  Budget: 100 evaluations. Token soft/normal/absolute = 9M / 10M / 11M.
- Do **not** run CUDA / `nvidia-smi` / the evaluator / any alternate correctness harness directly.
- Profiling **only** via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow). **Never** profile
  and evaluate at the same time or on the same GPU — a foreign process during timing ⇒ return code 3,
  discarded measurement, one wasted evaluation. Serialise strictly.
- `final` only with explicit operator approval.
- Append one JSON record per evaluated candidate to `candidates.jsonl`; never rewrite earlier records.

### Correctness contract
Output `output` bf16, `new_state` f32, exact shapes above. Reference computes in f32 from bf16
inputs cast via `.float()`; we must cast identically (load bf16 → f32) so we incur *no extra*
quantisation beyond the reference. Tolerance is enforced by the official evaluator (not disclosed);
assume bf16-appropriate rtol/atol and keep all accumulation in f32.

---

## 4. Numerical risks & mitigations

1. **softplus overflow.** `softplus(x)=log(1+e^x)` overflows in f32 for `x ≳ 88`. `x = a+dt_bias`
   (a is bf16). Use the numerically-stable form `softplus(x) = max(x,0) + log1p(exp(-|x|))`
   (equivalently `relu(x)+log(1+exp(-|x|))`). PyTorch `F.softplus` (default `beta=1, threshold=20`)
   returns `x` linearly for `x>20`; the stable form matches this to f32 precision, so no tolerance
   gap. If `tl.log1p` is unavailable, use `tl.log(1+e^{-|x|})` (safe: argument ∈ (1,2]).
2. **`exp(A_log)` / gate underflow.** `g = exp(-exp(A_log)*softplus(...))`. The outer exp has a
   non-positive argument ⇒ `g ∈ (0,1]`, no overflow; large magnitude simply underflows `g→0`
   (benign, matches reference).
3. **sigmoid stability.** `beta = sigmoid(b)`. Use `tl.sigmoid` (stable) or the branch-stable
   manual form; avoid naive `1/(1+exp(-b))` which overflows for very negative `b`.
4. **Reduction order.** f32 tree-reduction in Triton over K=128 vs PyTorch f32 `@` accumulation may
   differ in the last ~1 ULP. With K=128 the error is far below any bf16 tolerance.
5. **Output reformulation (§1.5 optional fusion).** `out = scale*(g*(q·S_row) + delta*(q·k))`
   reorders the summation relative to the reference "materialise `new_row` then `q·new_row`". Exact
   in real arithmetic; in f32 it can differ slightly. **Mitigation:** c001 uses the *faithful*
   materialise-then-dot form to de-risk correctness; only introduce the fused form as a later
   candidate and confirm it still passes.
6. **Zero-state path.** `state` is present in every feedback workload (optional=true handled by
   metadata), but the kernel should still be correct if `state` is absent (treat as zeros). Handle in
   the Python wrapper (allocate zeros) rather than branching in-kernel.
7. **Head indexing.** v-head→q/k-head is `h//2`; an off-by-one here silently corrupts half the heads.
   Assert/verify against the `repeat_interleave` semantics.
8. **Layout / transpose.** Both `state` and `new_state` are k-last `[B,H,V,K]`; the reference's
   `.transpose` is conceptual only. Reading row `state[b,h,v,:]` and writing `new_state[b,h,v,:]`
   requires **no** physical transpose — do not accidentally reintroduce one.
9. **Contiguity assumptions.** Must handle the actual strides the evaluator provides; prefer
   stride-parameterised addressing over assuming C-contiguity, or `.contiguous()` in the wrapper if
   cheap. Verify safetensors tensors are contiguous before relying on it.

---

## 5. Triton design space

### 5.1 Parallelisation (the main lever)
The op is embarrassingly parallel over `(b, h_v, v)` — `B·1024` independent rows. Candidate grids:
- **(A) per-(b,h) × v-tile:** grid `(B*Hv, ceil(V/BLOCK_V))`, each program fixes `(b,h)`, handles
  `BLOCK_V` rows. Loads `q,k` (128 each) once, amortised over `BLOCK_V` rows. Simple, faithful.
- **(B) flat row grid:** 1D grid over `B·Hv·(V/BLOCK_V)` tiles; derive `(b,h)` from the tile id.
  Same work, easier to size for occupancy.
- **(C) split-K within a row (unlikely needed):** K=128 fits one program; splitting the reduction
  adds atomics/extra passes for no benefit on a BW-bound op. Deprioritise.

Because small batch is occupancy-bound, `BLOCK_V` should likely be **small for small B** (more
blocks: B=1 with BLOCK_V=8 → 128 blocks; BLOCK_V=4 → 256 blocks) and **larger for big B** (better
amortisation of the q/k load, fewer launches). This argues for `@triton.autotune` keyed on batch
size, or a wrapper that picks `BLOCK_V`/grid from `B`.

### 5.2 Per-program compute (vectorised, register-resident)
Load `S_tile` as `[BLOCK_V, K]` (each row contiguous in K ⇒ coalesced). Then:
```
sk   = tl.sum(S_tile * k[None,:], axis=1)          # [BLOCK_V]
old_v= g * sk
delta= beta * (v_tile - old_v)                     # [BLOCK_V]
new_tile = g*S_tile + k[None,:]*delta[:,None]      # [BLOCK_V, K]
out  = scale * tl.sum(q[None,:]*new_tile, axis=1)  # [BLOCK_V]  (faithful form)
store new_tile → new_state; store out → output
```
`q,k` are length-128 bf16 loaded once → f32. Scalars `g,beta` computed once per program from
`a,b,A_log,dt_bias` (tiny loads). State stays in f32 registers throughout.

### 5.3 Tuning knobs (each a candidate axis)
- `BLOCK_V` ∈ {1,2,4,8,16,32} and its interaction with `B`.
- `num_warps` ∈ {1,2,4,8} — with `BLOCK_V*128` f32 elements per program, pick so each thread owns a
  few contiguous elements; too many warps starves work at small `BLOCK_V`.
- `num_stages` — limited benefit (single state load, no deep pipeline), but worth a sweep.
- Vectorised/wide loads of the contiguous K row (128×f32 = 512 B ⇒ natural 128-bit vectorisation).
- Cache policy: state is streamed once (read once, written once) → `evict_first` / streaming hints on
  the state load may help large-B bandwidth (per `pattern-memory-bound` / `technique-vectorized-loads`).
- Fusing multiple heads or the whole `(h,v)` of one batch element per program to raise per-block work
  at small B (reduces launch overhead), vs. more blocks for occupancy — A/B.
- Faithful vs. fused output formula (§1.5 / §4.5).
- Optionally handle `output` as `[B,1,Hv,V]` write with the singleton T dim folded in the wrapper.

### 5.4 Launch-overhead considerations (small batch)
KernelWiki (`kernel-gated-delta-net`, caveats) notes Triton decode kernels suffer CPU launch
overhead at small batch. We cannot use CUDA graphs (evaluator-controlled), so we must (a) keep the
Python `run()` wrapper lean — no per-call graph building, minimal host work, precompute strides once;
(b) prefer a **single kernel launch** over the whole batch (grid covers all `(b,h,v-tile)`), never a
Python loop over batch/heads; (c) avoid redundant `.contiguous()`/allocations. A single fused launch
also matters for the B=1..8 workloads that dominate 25/54 of the geomean.

### 5.5 What *not* to do
- No tensor-core matvec attempts (memory-bound; wasted effort).
- No split-K / atomics.
- No Torch fallback for any shape (including B that is small) — invalid submission.

---

## 6. Validation strategy

Because we may not run CUDA, `nvidia-smi`, the evaluator internals, or any *alternate* correctness
harness directly, validation is layered:

1. **Paper derivation (done above).** The per-v-row reformulation (§1.5) is proven algebraically
   equal to the reference; head mapping, layout, and dtype casts are matched exactly. c001 uses the
   faithful materialise-then-dot form to minimise f32 reordering.
2. **Dtype/shape audit before each candidate.** Confirm bf16→f32 casts, f32 accumulation, bf16
   `output` / f32 `new_state`, exact output shapes, and stride-correct addressing.
3. **Official feedback evaluation** (`./scripts/evaluate_candidate.sh feedback cNNN`) is the sole
   correctness+performance oracle: it runs the full 54-workload set (warmup 2 / 10 iters) and reports
   per-workload pass + geomean. This is the authoritative gate for every candidate.
4. **Profiling for optimisation guidance only**, strictly serialised with evaluation, via
   `./scripts/ncu_profile.sh --set … -o profile/rN python harness.py` (ncu-report-skill workflow).
   Use it to confirm the memory-bound hypothesis (DRAM throughput vs. SoL, achieved occupancy,
   launched-block count especially at B=1), *not* as a correctness check. Never launch it while an
   evaluation is running (foreign process ⇒ rc 3 ⇒ wasted evaluation).
5. **Candidate ledger discipline.** Each evaluated candidate appends one JSON record to
   `candidates.jsonl` (parent, source hash, hypothesis, validation, per-workload result, geomean,
   decision, cumulative eval count, skill usage). Stop when converged; then write `SEARCH_COMPLETE`.

### Candidate roadmap (tentative, to be detailed in `docs/plan.md`)
- **c001** — correctness baseline: single fused launch, grid (B·Hv, V/BLOCK_V), faithful math,
  modest `BLOCK_V`, f32 accumulation, stable softplus/sigmoid. Goal: all 54 pass; record baseline geomean.
- **c002+** — occupancy tuning for small B (BLOCK_V/grid vs. batch, autotune), then large-B bandwidth
  (vectorised/streamed state loads, cache policy), then optional fused output formula, `num_warps`/
  `num_stages` sweeps. One immutable source change per candidate; keep the best valid parent.

---

## 7. Key references (KernelWiki)
- `wiki/kernels/gated-delta-net.md` — GDN mechanism, streaming decode kernel sketch, 128×128 state,
  Triton launch-overhead caveat, FLA as the fast reference.
- `sources/contests/flashinfer-mlsys26/track-c-gated-delta-net.md` — this exact contest track
  (`qk4_v8_d128`), decode = recurrent single-token state update, state-management emphasis.
- `wiki/patterns/memory-bound.md` — optimise coalesced/wide loads, cache policy, occupancy; do not
  optimise compute; **profile first** to confirm the bottleneck.
- `sources/blogs/gated-delta-net.md` — decode-step delta-rule reference (decay then `k⊗v`, `q·S`).
