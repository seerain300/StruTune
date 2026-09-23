# Draft — `gdn_decode_qk4_v8_d128_k_last`

Gated Delta Net (GDN) decode, single-token, GVA head config, **k-last** state layout.
Captured from Qwen3-Next linear-attention layers (TP=4). Target GPU: **NVIDIA A800 (sm_80, Ampere)**.

This document is analysis only. No code, no `docs/plan.md` yet.

---

## 1. Operation summary

Single-token (`T = 1`) recurrent update of a per-head linear-attention state, followed by a
read-out. Every `(batch, v-head)` pair is a fully independent problem; the op is a batched
collection of tiny state updates.

### Fixed problem dimensions (from `definition.json`)
- `num_q_heads = num_k_heads = 4`, `num_v_heads = 8` (GVA, 2 v-heads per q/k-head).
- `head_size K = V = 128`.
- `seq_len T = 1`.
- `batch_size B` is the only variable axis. Feedback set: `B ∈ {1, 4, 8, 16, 64}`.
- Constraints: `num_v_heads >= num_q_heads`, `num_v_heads % num_q_heads == 0`,
  `num_k_heads == num_q_heads`. Here the expansion factor is `G = num_v_heads / num_q_heads = 2`.

### Inputs / outputs / dtypes
| tensor | shape | dtype | notes |
|---|---|---|---|
| `q` | `[B,1,4,128]` | bf16 | query, K-dim last |
| `k` | `[B,1,4,128]` | bf16 | key, K-dim last |
| `v` | `[B,1,8,128]` | bf16 | value, V-dim last |
| `state` | `[B,8,128,128]` | f32 | recurrent state, **k-last** layout `[B,H,V,K]`; optional (None → zeros) |
| `A_log` | `[8]` | f32 | per-v-head log-decay |
| `a` | `[B,1,8]` | bf16 | input-dependent decay |
| `dt_bias` | `[8]` | f32 | per-v-head decay bias |
| `b` | `[B,1,8]` | bf16 | update-gate input |
| `scale` | scalar f32 | | fixed `0.08838834764831843 = 1/sqrt(128)` in every feedback workload |
| **`output`** | `[B,1,8,128]` | bf16 | read-out (V-dim last) |
| **`new_state`** | `[B,8,128,128]` | f32 | updated state, k-last `[B,H,V,K]` |

Head mapping (from `repeat_interleave(G=2, dim=1)`): for v-head `h ∈ [0,8)`,
`q_head = k_head = h // 2`.

### Gates (per `(b,h)`, scalars)
```
x    = a[b,0,h] (f32) + dt_bias[h]                 # a cast bf16->f32
g    = exp( -exp(A_log[h]) * softplus(x) )          # in (0,1]
beta = sigmoid( b[b,0,h] (f32) )                    # in (0,1)
```

---

## 2. Math derivation and the crucial simplification

The reference does everything per `(b,h)` in a transposed `[K,V]` working layout with three
explicit outer/mat-vec products. Working the algebra out **directly in the stored `[V,K]`
layout** collapses it to one scalar-scaled state plus a single rank-1 update. Let
`S = state[b,h]` with shape `[V,K]` (`S[v,k]`), and let `q_h, k_h ∈ R^K`, `v_h ∈ R^V`.

Reference steps (renamed): `H = S^T` so `H[k,v] = S[v,k]`; `OS = g*H`;
`old_v = k_h @ OS`; `new_v = beta*v_h + (1-beta)*old_v`;
`H_new = OS - outer(k_h, old_v) + outer(k_h, new_v)`;
`output = scale * (q_h @ H_new)`; `new_state = H_new^T`.

Simplify:
```
old_v[v]        = sum_k k_h[k]*OS[k,v] = g * sum_k k_h[k]*S[v,k] = g * (S @ k_h)[v]
new_v - old_v   = beta*(v_h - old_v)
H_new[k,v]      = g*S[v,k] + k_h[k]*(new_v[v]-old_v[v])
new_state[v,k]  = H_new[k,v] = g*S[v,k] + delta_v[v]*k_h[k]
output[v]       = scale * sum_k q_h[k]*H_new[k,v] = scale * (new_state @ q_h)[v]
```

### Final compute kernel (all per `(b,h)`, everything f32)
```
old_v[v]        = g * dot_k( S[v,:], k_h )              # V mat-vecs of length K
delta_v[v]      = beta * ( v_h[v] - old_v[v] )
new_state[v,k]  = g * S[v,k] + delta_v[v] * k_h[k]      # scale + rank-1 outer
output[v]       = scale * dot_k( new_state[v,:], q_h )  # V mat-vecs of length K
```

**Everything is independent per output row `v`.** A single row `v` only needs: that row of
`S` (`128` f32), the head-shared vectors `q_h, k_h` (`128` each), the scalars `g, beta, scale`,
and `v_h[v]`. This makes the natural parallel unit a `(b, h, v)` triple — an embarrassingly
parallel, per-row streaming computation over the state. This is the central insight the Triton
design will exploit.

Re-derivation verified by hand against the reference (both mat-vec and outer-product terms);
correctness will be reconfirmed by the evaluator on `c001`.

---

## 3. Performance characterization (why this is bandwidth-bound)

Per `(b,h)` the arithmetic is: `S@k_h` (`128*128` MACs) + `g*S + outer` (`128*128` FMAs) +
`new_state@q_h` (`128*128` MACs) ≈ `3 * 16384 ≈ 4.9e4` flops. Times `B*H = 8B` heads.

Memory movement per `(b,h)`: read `S` = `128*128*4 = 64 KiB`, write `new_state` = `64 KiB`,
plus negligible `q,k,v,a,b` (a few hundred bytes) and a `128*2 B` output row. So per head
≈ `128 KiB` of unavoidable state traffic; arithmetic intensity ≈ `4.9e4 / 1.3e5 ≈ 0.37
flop/byte` — deep in the **memory-bound** regime.

Total state traffic (read+write): `B * 8 * 128 KiB = B * 1 MiB`.
- `B=1` → 1 MiB, `B=4` → 4, `B=8` → 8, `B=16` → 16, `B=64` → 64 MiB.

A800 HBM2e ≈ 2.0 TB/s. Ideal times: `B=64` ≈ 64 MiB / 2 TB/s ≈ **32 µs**; `B=1` ≈ **0.5 µs**
(latency/launch-bound, not bandwidth-bound at this size). Consequences for the design:
- The only path to speed is **one streaming pass** over `state`/`new_state` with fully
  coalesced 128-wide f32 accesses, and **enough occupancy** to saturate HBM.
- FLOPs are trivial → **tensor cores give nothing**; and TF32 tensor cores would *hurt*
  accuracy. Keep all reductions as plain f32 FMA (`tl.sum`), no `tl.dot`.
- Small `B` (esp. `B=1`, only 8 heads) is the hard case: it is launch/occupancy-bound, so
  we must split work finely (per `v`-block) to expose enough programs to fill 108 SMs.

The k-last `[B,H,V,K]` layout is favorable: `state[b,h,v,:]` is contiguous over `K`, so a
`[BLOCK_V, K]` tile is a contiguous, naturally coalesced block for both load and store.

---

## 4. Constraints & assumptions to honor

- Triton must do the computation. PyTorch only for metadata/launch (allocating outputs,
  reading shapes/strides, computing grid). **Gate math (`g`, `beta`) is computation → must be
  in Triton**, not torch. Cheapest: recompute the two scalars inside the main kernel per
  program (trivial cost), avoiding a separate launch.
- No Torch/CPU/NumPy/CUDA-extension fallback. A failing Triton kernel is invalid; do not
  substitute a torch path.
- `state` is optional. Feedback workloads always provide it (safetensors), but `run(...)`
  should still handle `state is None` by treating it as zeros (allocate-and-zero in the launch
  wrapper, which is permitted plumbing; the zero-state math then reduces to
  `new_state = outer(beta*v_h, k_h)`, `output = scale*new_state@q_h`).
- Do not change the fixed feedback workloads; do not touch evaluator/controller/config.
- Only evaluate via `./scripts/evaluate_candidate.sh feedback <id>`; five workloads = one eval.
- Immutable candidates: any source/config/launch change ⇒ new candidate ID appended to
  `candidates.jsonl`.

---

## 5. Numerical risks and mitigations

1. **softplus overflow.** `softplus(x)=log(1+exp(x))` overflows for large `x`. Match
   `F.softplus` (default `beta=1, threshold=20`): use
   `sp = where(x > 20, x, log(1 + exp(x)))`. For `x ≤ 20`, `exp(x) ≤ 4.85e8` is safe in f32;
   for `x < -20`, `exp(x)→0`, `log1p→0`. (`tl` has no `log1p`; `log(1+exp(x))` is fine inside
   the guarded range.)
2. **`exp(A_log)` large ⇒ `g` underflows to 0.** This matches the reference (decay → full
   forget). No special handling needed; f32 underflow to 0 is correct behavior.
3. **bf16 → f32 casts are exact** (bf16 ⊂ f32). Reference upcasts `q,k,v,a,b` to f32 before
   compute; do the identical upcast in-kernel. No precision lost relative to reference here.
4. **Reduction/summation order.** Reference uses torch f32 matmul; we use f32 `tl.sum` over
   `K=128`. Different accumulation order gives ~1e-6 relative drift — negligible vs bf16 output
   and f32 state tolerances. Keep the accumulator f32.
5. **Output rounding f32 → bf16.** Round-to-nearest-even in both torch `.to(bfloat16)` and
   Triton `.to(tl.bfloat16)`; should match bit-for-bit for the same f32 input.
6. **`new_state` is f32 and load-bearing for the next decode step** — do NOT store it as bf16;
   keep the full f32 rank-1 update. The `g*S` term and the outer product must both be f32.
7. **`scale`** applied only to the read-out `output`, not to `new_state`. Keep them separate.
8. **Tolerance.** `definition.json` exposes no explicit `tolerance`; assume the evaluator's
   default (typical: loose for bf16 `output`, tighter for f32 `new_state`). Staying in f32
   throughout is the conservative choice and should pass comfortably. Confirm empirically on
   `c001`.

---

## 6. Triton design space

### 6.1 Chosen kernel shape (baseline direction)
One fused kernel. Parallel unit = `(bh, v_block)` where `bh = b*8 + h`, and `v_block` covers
`BLOCK_V` consecutive rows of `V`. Each program:
1. Decode `b, h` from `bh`; `qk_head = h // 2`.
2. Recompute scalars `g, beta` (load `a,b` bf16→f32, `A_log,dt_bias` f32; softplus/sigmoid/exp).
3. Load head-shared vectors `q_h[K]`, `k_h[K]` (bf16→f32), and `v_tile[BLOCK_V]` (bf16→f32).
4. Load `S_tile = state[b,h, v0:v0+BLOCK_V, :]` → `[BLOCK_V, K]` f32 (contiguous, coalesced).
5. `old_v = g * sum_K(S_tile * k_h[None,:])`            # `[BLOCK_V]`
6. `delta_v = beta * (v_tile - old_v)`                   # `[BLOCK_V]`
7. `new_state_tile = g*S_tile + delta_v[:,None]*k_h[None,:]`  # `[BLOCK_V, K]`
8. store `new_state_tile` → `new_state[b,h,v0:.., :]`
9. `out = scale * sum_K(new_state_tile * q_h[None,:])`   # `[BLOCK_V]`
10. store `out` (→ bf16) → `output[b,0,h,v0:..]`

Reductions are over the last axis (`K=128`) with `tl.sum` in f32. No `tl.dot`, no tensor cores,
no TF32.

### 6.2 Parallelization / occupancy knobs
- **`BLOCK_V`** ∈ {8, 16, 32, 64, 128}: rows per program. Program count = `8B * (128/BLOCK_V)`.
  - `B=64`: even `BLOCK_V=128` → 512 programs (saturates SMs; maximizes `q/k` reuse & fewest
    gate recomputes).
  - `B=1`: need small `BLOCK_V` (8→1024, 16→512, 32→256 programs) to fill 108 SMs; `BLOCK_V=128`
    would give only 8 programs (severe under-utilization).
  - ⇒ **autotune `BLOCK_V` keyed on `batch_size`** so small `B` splits finely and large `B`
    uses fat tiles.
- **`num_warps`** ∈ {1,2,4,8}: with `[BLOCK_V,128]` tiles, tune vs `BLOCK_V` (e.g. small tile →
  1–2 warps to keep many resident blocks; fat tile → 4 warps).
- **`num_stages`**: limited benefit (single load of `S`, no K-loop pipeline); keep 1–2.
- **Grid**: 1-D `(8B * ceil(128/BLOCK_V),)` or 2-D `(8B, ceil(128/BLOCK_V))`. Masking on the
  `V` tail only needed if `BLOCK_V ∤ 128`; all candidate `BLOCK_V` divide 128, so no mask.

### 6.3 Alternatives considered (and why deferred/rejected)
- **One program per full head, `[128,128]` in-register**: great `q/k` reuse and one state pass,
  but only `8B` programs — collapses for small `B`. Good *config point* for large `B`, reachable
  as the `BLOCK_V=128` autotune choice; not a separate kernel.
- **Split-K partial reductions + atomics**: unnecessary — `V`-splitting already provides ample
  parallelism, and K=128 is a cheap in-thread reduction. Adds atomics/precision risk. Reject.
- **`tl.dot` / tensor cores (TF32 or bf16)**: FLOPs are trivial (memory-bound) so no speedup,
  and TF32 loses precision vs the f32 reference. Reject.
- **Separate gate prologue kernel** writing `g,beta` to scratch: avoids per-`v_block`
  recompute, but the recompute is a few transcendental ops per program — cheaper than an extra
  launch + extra global round-trip. Keep gates in-kernel; revisit only if profiling (evaluator
  timing) suggests it matters.
- **Fusing the two v-heads that share `q_head/k_head`**: halves the (already negligible) `q/k`
  loads but complicates indexing and state tiling. Not worth it.
- **bf16 state storage**: forbidden by correctness (f32 `new_state` output). Reject.

---

## 7. Validation strategy

- **No private harness**: rules forbid running CUDA / profiler / the external evaluator's
  internals / any alternate correctness harness directly. The *only* feedback channel is
  `./scripts/evaluate_candidate.sh feedback <id>` (correctness + geomean speedup over the 5
  fixed workloads = one evaluation).
- **Pre-eval correctness discipline** (since evals are the sole oracle and budget is finite):
  1. Trust the hand-derivation in §2 (mat-vec + rank-1 form), matched op-by-op to the reference,
     including the `q/k` head expansion `h//2`, k-last indexing, `scale` on read-out only,
     and f32 accumulation.
  2. Keep `c001` deliberately simple and obviously-correct (modest fixed `BLOCK_V`, no autotune,
     no fancy tiling) so the first eval isolates *correctness*, not tuning bugs.
  3. Only after `c001` passes correctness do we layer on autotune/occupancy changes, one
     immutable candidate at a time, each re-checking correctness before reading speedup.
- **Numerical margin**: everything in f32; guarded softplus; exact bf16→f32 upcasts; RNE
  output cast. Expect to pass default tolerance with margin.
- **Interpretation of results**: record per-workload pass/fail + speedup and geomean in
  `candidates.jsonl`; watch small-`B` workloads (`B=1,4`) which are latency/occupancy-bound and
  will dominate the geomean's downside, vs `B=64` which is bandwidth-bound.

---

## 8. Candidate roadmap (for `docs/plan.md`, not implemented here)

- **c001 — correctness baseline**: fused row-parallel kernel, fixed `BLOCK_V` (e.g. 16),
  gates in-kernel, f32 everywhere, no autotune. Goal: pass all 5, establish baseline speedup.
- **c002+ — occupancy/tiling**: introduce `@triton.autotune` over `BLOCK_V ∈ {8,16,32,64,128}`
  and `num_warps ∈ {1,2,4,8}` keyed on `batch_size`; verify small-`B` splits fine, large-`B`
  uses fat tiles.
- **later**: micro-tuning (`num_stages`, vector widths / `.to` placement), optional gate
  prologue only if evidence shows recompute matters, `state=None` fast path. Stop when the
  geomean converges; then write `SEARCH_COMPLETE`. Never run `final` without operator approval.

---

## 9. Skill usage

- **KernelWiki**: not applicable — it targets Blackwell (SM100/B200) and Hopper (SM90/H100);
  this task is Ampere **A800/sm_80**. No tcgen05/TMEM/CLC/2-SM features exist here.
- **ncu-report-skill**: not applicable — it profiles on B200/sm_100, and the task forbids
  running a profiler directly. Timing feedback comes solely from the evaluator.
- Net: no external skill is used; design rests on the reference algebra and A800 roofline.
