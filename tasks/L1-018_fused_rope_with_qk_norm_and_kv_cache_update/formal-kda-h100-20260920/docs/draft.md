# Draft — L1/018 Fused RoPE + QK-Norm + KV-Cache Update (H100 / sm_90)

## 0. Purpose of this document
Analysis-only draft. No code, no `plan.md` yet. Establishes the exact operation
semantics, the constraints of the task, numerical risks, the Triton design space,
and how correctness/performance will be validated. The executable plan and the
first candidate (`c001`) follow in later turns.

---

## 1. Operation semantics (from `task/definition.json` reference)

The reference (`@torch.no_grad() def run(...)`) fuses three logically separate stages
that a stock HF model would launch as 5+ CUDA kernels plus several intermediate
allocations. The fused result must reproduce, per element:

### 1.1 Inputs
| name | shape | dtype | notes |
|---|---|---|---|
| `query` | `[B, Hq=96, S, D=128]` | bf16 | current-step queries |
| `key` | `[B, Hkv=8, S, D=128]` | bf16 | GQA keys |
| `value` | `[B, Hkv=8, S, D=128]` | bf16 | GQA values (NOT normed/roped) |
| `position_ids` | `[B, S]` | int64 | `= arange(cache_len, cache_len+S)` broadcast over B |
| `key_cache` | `[B, Hkv=8, max_pos=262144, D=128]` | bf16 | updated in place |
| `value_cache` | `[B, Hkv=8, max_pos, D=128]` | bf16 | updated in place |
| `cache_position` | `[S]` | int64 | `= arange(cache_len, cache_len+S)` |
| `q_norm_weight` | `[D=128]` | bf16 | RMS weight (all-ones in `get_inputs`, treat as general) |
| `k_norm_weight` | `[D=128]` | bf16 | RMS weight |
| `inv_freq` | `[half_D=64]` | fp32 | `1/(theta^(2i/D))`, `theta=1e7` |
| `rms_norm_eps` | scalar | fp32 | `1e-6` |

### 1.2 Math (exactly as reference)
For each row `x` of length `D=128` (one `(b, head, s)` vector):

1. **RMS norm (per head, per position), computed in fp32:**
   - `xf = x.float()`
   - `var = mean(xf^2)` over the 128-element last dim
   - `xn = xf * rsqrt(var + eps)`
   - `xn = (weight.float() * xn)` → later cast to bf16 as part of RoPE result

2. **RoPE** with `freqs[j] = position * inv_freq[j]`, `j∈[0,64)`:
   - `emb = cat([freqs, freqs])` so `cos[0:64]==cos[64:128]==cos(freqs)` and same for sin.
   - `cos, sin` are computed in fp32 then **cast to bf16** (`.to(query.dtype)`) in the reference.
   - `rotate_half(x) = cat([-x[64:128], x[0:64]])`.
   - `out = x*cos + rotate_half(x)*sin`. With the cat structure this reduces per row to:
     - `o[0:64]   = xn[0:64]*c  - xn[64:128]*s`
     - `o[64:128] = xn[64:128]*c + xn[0:64]*s`
     where `c=cos(freqs)`, `s=sin(freqs)`, both length-64.
   - Result cast to bf16.

3. **Cache update:**
   - `key_cache[b, kv, cache_position[s], :] = key_rotated[b, kv, s, :]`
   - `value_cache[b, kv, cache_position[s], :] = value[b, kv, s, :]` (raw value, no norm/rope)
   - Because `cache_position == arange(cache_len, cache_len+S)`, the destination is a
     **contiguous slice** `cache[:, :, cache_len:cache_len+S, :]`. The reference expresses
     it as advanced-index scatter; we can write it as a direct contiguous store, but should
     still read `cache_position`/`position_ids` from the tensors to stay correct for any inputs.

### 1.3 Outputs
- `query_rotated [B,Hq,S,D]` bf16 (new tensor)
- `key_rotated  [B,Hkv,S,D]` bf16 (new tensor)
- `key_cache`   (same tensor, updated in place, returned)
- `value_cache` (same tensor, updated in place, returned)

Key observations:
- `value` is copied verbatim into `value_cache` — no compute, pure scatter/copy.
- `cos/sin` depend only on `(b, s)` (not on head), and are identical for Q and K.
- Q dominates traffic (96 heads vs 8).

---

## 2. Workload set (13 feedback shapes)

| # | uuid-prefix | B | S | cache_len | Q rows (B·96·S) | max_atol |
|---|---|---|---|---|---|---|
| 1 | 2ff5eaf3 | 1 | 1 | 0 | 96 | 1e-5 |
| 2 | 3c3b2b9b | 1 | 256 | 128 | 24 576 | 4.9e-3 |
| 3 | 8f5402ae | 4 | 1 | 2048 | 384 | 1e-5 |
| 4 | 520daee2 | 1 | 128 | 0 | 12 288 | 1e-5 |
| 5 | 052e17c9 | 2 | 512 | 256 | 98 304 | 4.4e-3 |
| 6 | d1dbaf22 | 2 | 128 | 0 | 24 576 | 1.3e-3 |
| 7 | 6a214883 | 1 | 1024 | 512 | 98 304 | 4.5e-3 |
| 8 | 8603b54e | 1 | 4096 | 2048 | 393 216 | 9.8e-3 |
| 9 | 39eb0040 | 1 | 2048 | 0 | 196 608 | 3.4e-3 |
| 10 | f71286e4 | 1 | 293 | 131 | 28 128 | 2.9e-3 |
| 11 | 3c5b9152 | 8 | 128 | 0 | 98 304 | 2e-3 |
| 12 | d90ee047 | 8 | 256 | 1024 | 196 608 | 5.6e-3 |
| 13 | 13f5b933 | 1 | 541 | 0 | 51 936 | 4.6e-3 |

Categories that matter for design:
- **Decode / tiny** (`S=1`): #1, #3. Only 96·B rows → latency- and launch-overhead-bound.
  Our win here is 1–2 fused kernels vs the reference's many launches on tiny tensors.
- **Prefill / large** (`S≥512`): #5,#7,#8,#9,#11,#12,#13. Bandwidth-bound; the win is
  eliminating redundant Q traffic + intermediate materialization.
- **Odd / non-power-of-two S**: #10 (293), #13 (541), and cache offsets #10 (131). Force
  masking on the sequence tiling; no assumption that S or cache_len is aligned.
- **All-ones norm weights** in `get_inputs`, but code must not special-case that (weights
  are real inputs; treat generally).
- **Largest cache index touched** = `cache_len+S-1` = `2048+4096-1 = 6143` (#8) and
  `1024+256-1` (#12) — all well below `max_pos=262144`, so no cache-bounds masking needed,
  but the destination row index can be large (int32 index into a 262144-row axis: offset
  `(b*Hkv+kv)*262144 + pos` can exceed 2^31? `8*262144 = 2,097,152` rows × 128 =
  268M elements × up to (b*Hkv+kv)… for B≤8: max linear element offset ≈
  `(8*8-1)*262144*128 ≈ 2.11e9` > 2^31 (2.147e9)? It is ~2.11e9 < 2.147e9, marginal.
  **Risk flagged**: use int64 offset arithmetic for cache addressing to be safe.

---

## 3. Task constraints (from CLAUDE.md / TASK.md)

- Primary implementation **must be Triton**; PyTorch only for metadata/launch plumbing and
  output-tensor allocation. **No Torch/CPU/NumPy/CUDA-extension computational fallback** —
  a failed Triton kernel is invalid, not something to paper over with `torch`.
- Entry point: `solution/solution.py` exposing `run(...)` with the reference signature.
- Evaluate ONLY via `./scripts/evaluate_candidate.sh feedback <cid>`; full 13-workload set =
  one candidate evaluation. Budget: 100 evals; token soft 9M / normal 10M / hard 11M.
- Candidates are immutable & sequential (`c001`, `c002`, …); never reuse an ID for changed source;
  append one JSON record per eval to `candidates.jsonl`, never rewrite.
- Profiling only through `./scripts/ncu_profile.sh` (ncu-report-skill workflow). **Never profile
  and evaluate at the same time** — a foreign process on the locked GPU makes the controller
  discard the measurement (rc=3) and wastes a budget slot.
- Do not run CUDA / nvidia-smi / the evaluator / alternate harnesses directly.
- `final` only with explicit operator approval.
- Isolation: work only in this workspace; only permitted external knowledge = `KernelWiki`
  and `ncu-report-skill` skills.

---

## 4. Numerical analysis & risks

### 4.1 Tolerance semantics (assumption to confirm)
Workload #1 (`S=1, cache_len=0`) has `max_atol=1e-5` yet the reference output is bf16 and
our fp32 reduction order will differ from PyTorch's (`mean` uses a different summation tree),
which can move a normalized value by ~1 bf16 ulp (~8e-3 for O(1) magnitudes). A **strict**
`max(|actual-expected|) ≤ 1e-5` would be impossible for a bf16 reference. Therefore the
evaluator almost certainly applies the standard combined rule
`|actual - expected| ≤ max_atol + max_rtol·|expected|` (torch.allclose semantics) with
`max_rtol = 0.05`. Under that rule the 5% relative term dominates everywhere except extremely
near zero, where bf16 granularity is itself tiny — so accuracy is essentially a non-issue.
**Action**: design for the combined rule, but still compute cleanly in fp32 to keep error minimal;
if `c001` ever fails correctness on the tight-atol shapes, revisit this assumption first.

### 4.2 Internal precision plan
- Load bf16 inputs → cast to **fp32**; do RMS (sum of squares, `rsqrt`), weight multiply,
  and RoPE all in fp32; cast final result to bf16 once at store. This is at least as accurate
  as the reference (reference casts `cos/sin` to bf16 before the RoPE multiply; doing the
  multiply in fp32 is strictly closer to true value and well within `rtol=0.05`).
- `inv_freq` is fp32 — load as fp32, no precision loss. `position` int64 → cast to fp32
  (max ~6143, exactly representable in fp32).
- `rsqrt`: use `tl.rsqrt`/`1/tl.sqrt` in fp32 with `+eps` before sqrt, matching
  `rsqrt(var+eps)`.

### 4.3 cos/sin of large arguments
`freqs = pos * inv_freq`. For low-frequency channels `inv_freq≈1`, and `pos` up to ~6143, so
arguments reach ~6143 radians. `tl.sin/tl.cos` operate in fp32 with hardware range reduction,
matching torch's fp32 `emb.cos()`. No extra range-reduction logic needed; recompute per row
rather than materialize a `[B,S,128]` cos/sin buffer.

### 4.4 Reduction correctness
RMS reduction is over the full `D=128` in one tile (no cross-tile reduction), so no split-reduction
numerical subtlety. `mean = sum(x^2)/128`.

---

## 5. Memory & performance characterization

This is a **memory-bandwidth-bound streaming** op (elementwise + a length-128 reduction).
Traffic per element-vector unit `U = B·S·D·2 bytes`:

| tensor | reads | writes | units |
|---|---|---|---|
| Q (Hq=96) | 1 | 1 (query_rotated) | 96 read + 96 write = **192·U/Hscale** |
| K (Hkv=8) | 1 | 2 (key_rotated + key_cache) | 8+16 = **24** |
| V (Hkv=8) | 1 | 1 (value_cache) | 8+8 = **16** |

(Using per-head unit `u = B·S·D·2`.) Total ≈ `192u(Q) + 24u(K) + 16u(V)` → **Q ≈ 83%** of all
HBM traffic. Optimization must make the **Q RMS+RoPE pass single-read/single-write and
bandwidth-saturating**; K/V is a small tail.

Rough SOL (largest, #8 B1S4096): total bytes ≈ 243 MB → at H100 HBM3 ~3.35 TB/s ⇒ ~**72 µs**
lower bound. The reference is far above this because it:
- materializes `query_norm`, `key_norm` (extra full read+write of Q,K),
- materializes `cos`, `sin`, `emb` (`[B,S,128]` ×2), `freqs`,
- materializes `query_rotated`/`key_rotated` from a second Q/K pass,
- does an advanced-index scatter for both caches,
- launches 5+ kernels.

So the fused kernel should recover roughly the redundant-traffic + launch-overhead factor.
On tiny decode shapes (#1,#3) the reference is dominated by launch overhead of many kernels
on ~96–384 rows; a single fused launch should give the largest relative speedup there.

---

## 6. Triton design space

### 6.1 Kernel decomposition (leading candidate)
Two kernels (both memory-optimal, simplest to reason about):
- **Q kernel**: grid over Q rows; per row → RMS-norm(q_norm_weight) → RoPE → store `query_rotated`.
- **KV kernel**: grid over K/V rows (Hkv=8); per row →
  - K: RMS-norm(k_norm_weight) → RoPE → store `key_rotated` and store into `key_cache` slice;
  - V: copy raw value → store into `value_cache` slice.
  Fusing K and V in one launch is natural (same `[B,Hkv,S,D]` iteration space) and keeps
  launches at 2 total.

Alternative decompositions to keep in reserve:
- 3 kernels (Q, K, V) — marginally simpler V copy, one extra launch.
- 1 mega-kernel over Q∪K with a head-count branch — messier, no clear BW benefit.
- V copy via a dedicated vectorized copy kernel (still Triton) if fusing hurts occupancy.

### 6.2 Row/tile mapping
Each "row" is a contiguous `D=128` bf16 vector (256 B). Two grid strategies:

**(A) 3-D structured grid** `(B, H, ceil(S/BLOCK_S))`: program owns one `(b,head)` and a
`BLOCK_S`-block of positions → tile `[BLOCK_S, 128]`. `position_ids[b, s0:s0+BLOCK_S]` loads
cleanly; `cos/sin` become a `[BLOCK_S, 64]` outer product. Clean, no boundary crossing across
head/batch, straightforward masking on the S dimension. Preferred default.

**(B) Flattened-row grid** over `total_rows = B·H·S`, program owns `BLOCK_ROWS` consecutive rows
and derives `(b,head,s)` by integer division. Better load balance / occupancy control,
especially for `S=1` decode (can pack many heads per program), but needs per-row index math and
masked `position_ids` gather. Keep as an optimization for the low-parallelism decode shapes.

For `S=1` (#1: grid would be only 96 programs under (A)), consider strategy (B) or packing all
`Hq` rows of a batch into one tile so few, fat programs still cover the SMs; but the work is so
small that even a single launch already beats the reference — micro-optimize only if profiling says so.

### 6.3 In-kernel cos/sin (recompute, don't materialize)
Compute `freqs = pos * inv_freq[0:64]`, `c=cos(freqs)`, `s=sin(freqs)` inside the kernel.
Reused for the two halves via the cat structure (`o1=x1·c−x2·s`, `o2=x2·c+x1·s`). Avoids the
`[B,S,128]` cos/sin buffers and their traffic/launch. `inv_freq` (64 fp32) and both norm weights
(128 bf16) are tiny and effectively cached.

### 6.4 Cache write addressing
Destination element offset for `(b,kv,s)` into `key_cache`/`value_cache`:
`base = ((b*Hkv + kv)*max_pos + cache_position[s]) * D`. Use tensor strides from PyTorch and
**int64 offsets** (see §2 marginal-2^31 risk). `cache_position` is contiguous, so the store is a
coalesced contiguous block; still read `cache_position[s]` to stay general and mask tail rows.

### 6.5 Autotuning knobs
- `BLOCK_S` (strategy A) or `BLOCK_ROWS` (strategy B) ∈ {16, 32, 64, 128}.
- `num_warps` ∈ {2, 4, 8}; `num_stages` ∈ {1, 2, 3}.
- `BLOCK_D = 128` fixed (whole head), power of two, no D masking.
- Autotune keyed on shape buckets; keep configs small to avoid compile blow-up. Because the op
  is BW-bound, the goal is enough resident warps to hide latency and saturate HBM, not compute.

### 6.6 Layout assumptions
`get_inputs` builds all tensors with `torch.randn`/`arange` → standard contiguous strides.
Use `.stride()` explicitly rather than hard-coding, and allocate outputs contiguous. Do not
assume `cache_len` alignment.

---

## 7. Correctness edge cases to cover in `c001`
- `S=1` reduction and RoPE at `position=0` (`cos=1, sin=0` → RoPE is identity; output = RMS-norm only).
- `S` not a multiple of `BLOCK_S` (#10=293, #13=541) → S-dimension masking on loads/stores.
- `cache_len` non-aligned (#10=131) → destination offset via read `cache_position`, no alignment assumption.
- `value` written **unmodified** (a frequent bug: accidentally norming/roping V).
- Return the *same* cache tensors (in-place update), plus freshly allocated `query_rotated`/`key_rotated`.
- General (non-unit) norm weights, even though `get_inputs` passes ones.
- int64 cache offsets to avoid 32-bit overflow on the 262144-length axis.

---

## 8. Validation strategy
1. **Local reasoning only** for correctness before spending an eval — no torch/CUDA locally
   (forbidden). Re-derive the per-row math (§1.2) and check indexing by hand.
2. **`c001` = simplest fully-correct fused implementation** (strategy A, modest fixed block,
   `num_warps=4`), whose sole job is to pass all 13 workloads and establish a baseline geomean.
   Do not chase performance before correctness is proven by the evaluator.
3. Evaluate with `./scripts/evaluate_candidate.sh feedback c001`; record per-workload pass/fail,
   speedups, geomean, cumulative eval count, and skill usage in `candidates.jsonl`.
4. If a shape fails correctness, first re-check the tolerance assumption (§4.1) and the
   value-copy / cache-index logic before touching precision.
5. **Then** optimize sequentially (`c002`…): profile the winning candidate with
   `ncu-report-skill` via `./scripts/ncu_profile.sh` (never concurrent with an eval), target
   achieved HBM bandwidth on the Q kernel, tune block/warps/stages, and reconsider strategy B
   for decode shapes. One meaningful change per candidate ID.
6. Stop when geomean converges or budget nears; write `SEARCH_COMPLETE` with the reason.
   Run `final` only after operator approval.

## 9. Skill usage plan
- **KernelWiki**: consult for H100 (sm_90) memory-bound streaming / elementwise-fusion best
  practices (vectorized bf16 loads, warp/occupancy targets, `num_stages` for BW saturation)
  during `plan.md` and before performance candidates.
- **ncu-report-skill**: use for every performance diagnosis (achieved BW %, memory throughput,
  occupancy, launch overhead on decode shapes) via the workspace `ncu_profile.sh` launcher only.

## 10. Open questions / risks to track
- Exact evaluator tolerance rule (combined vs strict) — assumed combined torch.allclose (§4.1).
- Whether fusing V-copy into the K kernel or a separate copy kernel gives better BW/occupancy
  (decide via profiling).
- Decode (`S=1`) parallelism: structured grid may under-fill the GPU; measure before switching
  to the flattened/head-packed grid.
- 2^31 offset margin on the cache axis — use int64 indexing defensively.
- Autotune compile-time vs coverage trade-off across the 13 shapes.
