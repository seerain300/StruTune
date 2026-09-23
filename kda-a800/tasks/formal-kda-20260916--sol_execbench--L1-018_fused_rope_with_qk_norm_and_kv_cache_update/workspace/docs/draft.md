# Draft — L1/018 Fused RoPE + QK-Norm + KV-Cache Update

Target: NVIDIA **A800 (`sm_80`, Ampere)**. Primary implementation must be **Triton**; PyTorch only
for metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.

> Skill note: the installed `KernelWiki` and `ncu-report-skill` skills are scoped to Blackwell
> (SM100/B200) and Hopper (SM90/H100). This task is Ampere `sm_80`, so neither skill is applicable;
> no skill is consulted for this draft. If profiling on the target were permitted, `ncu` is disallowed
> by task rules anyway (no direct profiler/CUDA/nvidia-smi runs).

---

## 1. What the operation computes

The reference (`run`) fuses five logically separate GPU operations into one op:

1. **Per-head RMS normalization** of `query` and `key` over the last axis (`head_dim = 128`).
2. **RoPE** (rotary position embedding, GPT-NeoX / `rotate_half` variant) applied to the normalized
   Q and K, using angles derived from `position_ids` and `inv_freq`.
3. **KV-cache scatter update**: write the rotated K and the (unmodified) V into `key_cache` /
   `value_cache` at rows given by `cache_position`.

### 1.1 Shapes and dtypes (constants for this task)

| tensor | shape | dtype |
|---|---|---|
| `query` | `[B, 96, S, 128]` | bf16 |
| `key`, `value` | `[B, 8, S, 128]` | bf16 |
| `position_ids` | `[B, S]` | int64 |
| `key_cache`, `value_cache` | `[B, 8, 262144, 128]` | bf16 |
| `cache_position` | `[S]` | int64 |
| `q_norm_weight`, `k_norm_weight` | `[128]` | bf16 |
| `inv_freq` | `[64]` | **float32** |
| `rms_norm_eps` | scalar | float32 (=1e-6) |

Constants: `num_attention_heads = 96`, `num_key_value_heads = 8` (GQA ratio 12),
`head_dim = 128`, `half_head_dim = 64`, `max_position_embeddings = 262144`,
`rope_theta = 1e7`, `rms_norm_eps = 1e-6`.

Outputs: `query_rotated [B,96,S,128] bf16`, `key_rotated [B,8,S,128] bf16`, and the **in-place
updated** `key_cache`, `value_cache` (returned as-is).

### 1.2 Exact reference math (to replicate bit-closely enough)

RMS norm (note fp32 accumulation, bf16 output):
```
x32   = x.float()
var   = mean(x32^2, axis=head_dim)          # fp32, divide by 128
xn    = x32 * rsqrt(var + eps)              # fp32
out   = (weight.float() * xn).to(bf16)      # bf16  <-- rounded here
```

RoPE angles:
```
freqs = position_ids[b,s].float() * inv_freq[j]    # j in [0,64), fp32
emb   = concat([freqs, freqs])  -> length 128
cos   = emb.cos().to(bf16)                          # rounded to bf16
sin   = emb.sin().to(bf16)                          # rounded to bf16
```
`rotate_half(x): x1=x[:64], x2=x[64:] -> concat([-x2, x1])`.

apply_rope (broadcast cos/sin across heads):
```
out = x * cos + rotate_half(x) * sin
```
Written per pair `(j, j+64)`, with angle `theta_j = pos * inv_freq[j]`:
```
out[j]    = xn[j]    * cos(theta_j) - xn[j+64] * sin(theta_j)
out[j+64] = xn[j+64] * cos(theta_j) + xn[j]    * sin(theta_j)
```
So dims `j` and `j+64` form a 2-D rotation pair (halves paired, **not** adjacent pairs).

Cache update (in place, only the `S` selected rows change):
```
key_cache[b, :, cache_position[s], :]   = key_rotated[b, :, s, :]
value_cache[b, :, cache_position[s], :] = value[b, :, s, :]     # V is copied, NOT normed/roped
```
`position_ids` drives the **angle**; `cache_position` drives the **cache row index**. They are equal
in every feedback workload (`arange(cache_len, cache_len+S)`), but the kernel must read them
independently for correctness/generality.

---

## 2. Feedback workloads

| # | B | S | cache_len | atol | rtol | character |
|---|---|---|---|---|---|---|
| 1 | 1 | 1024 | 512 | 4.5e-3 | 0.05 | prefill-ish, positions 512..1535 |
| 2 | 1 | 541  | 0   | 4.6e-3 | 0.05 | non-power-of-2 S, positions 0..540 |
| 3 | 1 | 1    | 0   | **1e-5** | 0.05 | **decode, single token, position 0** |
| 4 | 2 | 512  | 256 | 4.4e-3 | 0.05 | batch 2, positions 256..767 |
| 5 | 1 | 2048 | 0   | 3.4e-3 | 0.05 | largest prefill, positions 0..2047 |

Observations that shape design:
- **Workload 3 (S=1, pos=0)**: `theta = 0` → `cos=1, sin=0` **exactly** → RoPE is the identity, so the
  output is *purely* the RMS-normed Q/K. The tight `atol=1e-5` is not really binding because `rtol=0.05`
  on O(1) magnitudes dominates (effective tol ≈ `max(atol, rtol·|ref|)`), but it means the RMS-norm
  path must be faithful (fp32 variance, correct weight application). No cos/sin rounding error can
  appear here since the angles are exactly 0.
- **Mixed regimes**: decode (S=1) vs prefill (S up to 2048). The kernel/launch config must be robust
  across tiny and large `S` and across non-power-of-2 `S` (541).
- Positions stay small (≤ 2047), so `pos·inv_freq` is a comfortable fp32 range for `tl.sin/tl.cos`.

---

## 3. Constraints and correctness-critical details

1. **fp32 for the RMS variance.** Never accumulate `x^2` in bf16. Load bf16, cast to fp32, reduce.
2. **Weight application order.** `(weight.float() * xn)` then round to bf16 — matches reference. Weights
   are `ones` in these workloads, but read them anyway for generality.
3. **Halves-paired RoPE**, not adjacent-pairs. Dim `j` pairs with `j+64`; both share `theta_j`.
4. **V is not transformed** — it is copied verbatim into `value_cache`. Do not norm/rope V.
5. **In-place cache update; do not disturb other rows.** Only `S` rows per (b, kv_head) are written.
   The huge cache tensors must be mutated in place and returned as the same objects so unchanged rows
   stay bit-identical to the input.
6. **Pointer-offset overflow risk.** Cache is `[B,8,262144,128]`. The base offset for a (b,kv_head)
   block is `(b*8+h)*262144*128`; for B=2 that reaches `15*262144*128 ≈ 5.03e8`, under int32 max
   (2.147e9) but uncomfortably close. Compute cache offsets in **int64** to be safe and future-proof.
   The Q/K/V input tensors are small enough for int32, but I will use int64 for cache indexing.
7. **Independent index tensors.** Use `position_ids[b,s]` for angle, `cache_position[s]` for the write
   row. Load both; do not assume equality.
8. **int64 index loads.** `position_ids` and `cache_position` are int64; load as int, cast to fp32 for
   the angle and keep int64 for addressing.
9. **rsqrt semantics.** Reference uses `torch.rsqrt(var+eps)`. Use `tl.rsqrt` (or `1/tl.sqrt`); the
   difference is negligible vs tolerance.
10. **Determinism / no aliasing.** `query_rotated` and `key_rotated` are fresh output tensors (allocate
    with `torch.empty_like`); cache tensors are the passed-in ones.

---

## 4. Numerical-risk analysis

- **cos/sin rounding.** Reference rounds cos/sin to bf16 *before* multiplying; a fused kernel keeps
  fp32 cos/sin and rounds only the final products. Worst-case per-term deviation from the reference is
  ≈ `2^-8 · |xn|` ≈ `0.004·|xn|`. With `|xn| = O(1)` and `rtol=0.05` binding on O(1) outputs, this is
  comfortably inside tolerance. Being *more* precise than the reference is safe because tolerance is
  measured against the (rounded) reference, and `rtol·|ref|` (~0.05) dwarfs the rounding gap.
- **xn rounding before RoPE.** Reference feeds bf16 `xn` into RoPE; a fused kernel can keep fp32 `xn`.
  Same ~0.004 relative envelope, within tolerance. *Optional fidelity knob:* round `xn` to bf16 and
  back to fp32 before RoPE to emulate the reference more exactly if any workload is borderline. Cheap;
  keep as a fallback lever, not needed for the first candidate.
- **Variance precision.** Must be fp32; bf16 accumulation over 128 squared terms would lose too much.
- **Workload 3 tight atol.** Safe: angles are exactly 0, so the op reduces to faithful RMS norm; the
  binding constraint is `rtol·|ref|`.
- **`tl.sin/tl.cos` accuracy.** Triton uses fp32 libdevice; positions ≤ 2047, `inv_freq ≤ 1`, so
  arguments are small and accuracy is far better than the bf16 target. No range-reduction concerns.
- **Overflow / NaN.** RMS `var ≥ 0`, `+eps` guards div-by-zero; inputs are `randn` (well-scaled). No
  overflow expected in bf16 outputs.

---

## 5. Triton design space

The op is **memory-bound** elementwise + a 128-wide reduction. Q dominates traffic (96 heads vs 8).
Lower-bound traffic per workload ≈ read+write Q (`2·B·96·S·128·2 B`) + read+write K + K-cache write +
read V + V-cache write. For workload 1 that is ~50 MB for Q alone → tens of µs at ~2 TB/s. Kernels are
tiny; **occupancy, coalescing, and launch/config overhead dominate**. The big win over the reference is
eliminating the materialization of `cos/sin/emb` (`[B,S,128]` bf16 each) and collapsing 5+ launches.

### 5.1 Kernel decomposition options

- **A — three kernels:** Q (norm+rope), K (norm+rope+cache write), V (copy to cache). Simplest, most
  coalesced per-kernel; V-copy is a trivial scatter.
- **B — two kernels:** Q kernel; combined **K+V** kernel where each program handles one
  `(b, kv_head, seq-block)`: norm+rope K → `key_rotated` + `key_cache`, and copy V → `value_cache`.
  Fewer launches, shares position/cache-index math. **Preferred baseline.**
- **C — one kernel:** single launch covering both the 96 Q-heads and 8 K-heads (+V) via a fused grid
  with per-program role branching. Minimizes launches but complicates indexing (different head counts,
  cache only for K/V). Consider only if launch overhead proves material for tiny-S workloads.

Chosen starting point: **Option B (2 kernels)** — Q kernel and K+V kernel.

### 5.2 Per-program work mapping

Within a fixed `(b, head)`, the `S` tokens are contiguous rows of 128 (`query` stride: head-dim
innermost, then seq). Natural mapping:

- **Program = one block of `R` consecutive seq tokens of one `(b, head)`.**
  Grid ≈ `(B·H, ceil(S/R))` (or flattened 1-D). Each program:
  - loads a tile `a = x[:, 0:64]` and `b = x[:, 64:128]` (shape `[R, 64]` each, bf16→fp32),
  - reduces `sum(a^2)+sum(b^2)` over the 64-axis → per-row variance `[R,1]` (fp32),
  - normalizes: `an = a*inv*w0`, `bn = b*inv*w1`,
  - computes `theta[r,j] = pos[r]*inv_freq[j]` → `cos,sin` `[R,64]`,
  - `out_lo = an*cos - bn*sin`, `out_hi = bn*cos + an*sin`,
  - stores `out_lo`→cols 0:64, `out_hi`→cols 64:128 as bf16,
  - (K+V kernel only) scatters `out` to `key_cache[b,h,cache_position[r],:]` and `value` to
    `value_cache[b,h,cache_position[r],:]`.

The two-half load makes `rotate_half` free (just reuse `an`/`bn`) and gives a clean `[R,64]` tile with
`BLOCK_D = 64` matching `half_head_dim`. RMS reduction sums both halves.

### 5.3 Tunable parameters (design space, not committed yet)

- `R` = seq tokens per program (`BLOCK_S`): trade parallelism vs per-program reuse. Candidates 1, 2, 4,
  8, 16. For S=1 (decode) `R=1` degenerates to one program per (b,head).
- `num_warps` (1–8) and `num_stages` for the tiny 128-wide inner dim.
- Grid layout: `(b,head)` outer × seq-block inner vs fully flattened 1-D grid with modular decode.
- Load width: load full 128 in one `[R,128]` tile then slice halves, vs two `[R,64]` loads. Two loads
  are conceptually cleaner; a single 128-wide load may coalesce better — compare.
- Whether to fuse Q+K into one launch (Option C) if launch overhead shows up on S=1 / S=541.
- Optional `xn→bf16→fp32` fidelity rounding (guarded by a `constexpr` flag).

### 5.4 Baseline first, then optimize

Candidate `c001`: correct, simple **Option B**, `R` modest (e.g. 8), fp32 internal, no autotune — get a
green, measured baseline. Subsequent candidates: autotune `R`/`num_warps`, try single-128-load layout,
try Option C fusion, try loading `inv_freq`/weights once, reduce redundant recompute of cos/sin.

---

## 6. Launch / plumbing plan (PyTorch allowed only here)

- Allocate outputs with `torch.empty_like(query)` / `torch.empty_like(key)`.
- Cache tensors: mutate in place, return the same objects.
- Derive strides/shapes from tensors; pass scalars (`eps`, `S`, head counts) as kernel args.
- Compute grid sizes on host from shapes. No torch math on the actual data (no torch cos/sin/rmsnorm) —
  all computation stays in Triton to honor the "no Torch computational fallback" rule and to truly fuse.
- Handle `S=1` and non-power-of-2 `S` (541) via masked tail loads (`mask = seq_offs < S`).

---

## 7. Validation strategy

1. **Correctness via the official evaluator only** — `./scripts/evaluate_candidate.sh feedback cNNN`
   over the five fixed workloads. Every selected workload must pass (`atol/rtol` per row above).
2. **Reasoned pre-checks before each evaluation:**
   - RoPE pairing is halves-paired (`j`↔`j+64`), matching `rotate_half`.
   - Variance divides by 128 and is fp32; `+eps` inside the rsqrt.
   - V copied unchanged; K normed+roped before cache write.
   - `position_ids` used for angle, `cache_position` for cache row.
   - Cache offsets computed in int64; only `S` rows touched; other rows untouched (in-place).
   - Masking correct for `S=541` and `S=1`.
3. **Regime coverage** is already spanned by the five workloads: decode (3), non-pow2 (2), batch>1 (4),
   large prefill (5), nonzero cache_len (1,4). No extra harness is permitted or needed.
4. **Numerical-fidelity fallback:** if a borderline miss appears, enable the `xn→bf16` (and optionally
   `cos/sin→bf16`) emulation flag to track the reference rounding more exactly — as a *new* candidate ID.
5. **Performance metric:** geometric-mean speedup vs reference across passing workloads. Expect large
   gains from removing cos/sin/emb materialization and collapsing 5+ launches into 1–2 fused kernels.

---

## 8. Risks & mitigations (summary)

| risk | mitigation |
|---|---|
| bf16 variance loss | fp32 accumulation always |
| wrong RoPE pairing | halves-paired `(j, j+64)`, unit-check against reference formula |
| cos/sin rounding drift | rtol=0.05 covers it; optional bf16 emulation flag as fallback candidate |
| int32 pointer overflow on cache | int64 cache offsets |
| clobbering untouched cache rows | in-place scatter of only `S` rows; return same tensors |
| V accidentally transformed | copy V verbatim |
| tiny S=1 / non-pow2 S | masked loads, robust grid, `R` degrades gracefully |
| launch overhead on small S | fuse to 2 kernels now; consider single-launch (Option C) later |

---

## 9. Next step

Proceed to `docs/plan.md` (executable plan): concrete kernel signatures, grid math, `constexpr`
parameters, autotune space, and the ordered candidate ladder (`c001` correct baseline → tuned variants).
No solution code before the plan is complete.
