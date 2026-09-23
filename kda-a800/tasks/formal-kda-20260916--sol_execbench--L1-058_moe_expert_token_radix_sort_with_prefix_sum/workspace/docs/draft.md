# Draft — L1/058 MoE Expert Token Radix Sort with Prefix Sum

Target: NVIDIA A800 (`sm_80`, Ampere). Primary implementation must be Triton; PyTorch only
for tensor metadata / launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.

## 1. Operation specification (from `task/definition.json`)

### 1.1 Signature
```
run(topk_idx: int32[batch_size, seq_len, num_experts_per_tok]) ->
    (sorted_token_indices: int32[num_tokens],
     expert_offsets: int32[num_experts + 1])
```
Constants: `num_experts = 256`, `num_experts_per_tok = 8`.
Derived: `num_tokens = batch_size * seq_len * 8`, `num_experts_plus_one = 257`.

### 1.2 Reference semantics (must match exactly)
```python
num_experts = 256
flat = topk_idx.reshape(-1)                     # C-contiguous flatten, length N = num_tokens
_, sorted_token_indices = flat.sort(stable=True)  # STABLE argsort by expert id
expert_offsets = torch.zeros(num_experts + 1, dtype=int32, device=...)
expert_offsets[1:] = torch.bincount(flat.long(), minlength=256).cumsum(0).to(int32)
return sorted_token_indices.to(int32), expert_offsets
```

Interpretation:
- `flat[i]` is the expert id (0..255) of the i-th flattened token-expert assignment,
  `i = ((b*seq_len)+s)*8 + k`.
- `sorted_token_indices` is the **argsort permutation** of `flat` by expert id, using a
  **stable** tie-break: within one expert's bucket, the original flat indices appear in
  **strictly ascending order**.
- `expert_offsets[0] = 0`; `expert_offsets[e+1] = sum_{j<=e} count[j]` (inclusive prefix of the
  histogram). So `expert_offsets[e]` is the **exclusive** prefix = base write offset of expert `e`,
  and `expert_offsets[e]:expert_offsets[e+1]` is expert `e`'s slice in the sorted output.

This is a **counting/radix sort** on a bounded key range (0..255), with a prefix sum for offsets.
No floating-point math anywhere.

### 1.3 Feedback workload sizes (5 fixed)
| WL | batch | seq | N = batch*seq*8 |
|----|-------|-----|-----------------|
| 1  | 2     | 1120| 17920 |
| 2  | 8     | 288 | 18432 |
| 3  | 4     | 544 | 17408 |
| 4  | 8     | 256 | 16384 |
| 5  | 4     | 512 | 16384 |

All N are small (~16k–18k). Keys are uniform random over 256 experts (`torch.randint`), so the
mean bucket size is ~64–72 and buckets are well-balanced. Note keys are drawn independently per
slot, so duplicate experts within a token are possible — irrelevant to correctness because the
reference operates on the flat array. The flat array is only ~64–72 KB (int32), i.e. it comfortably
fits in A800 L2 (40 MB), which strongly favors cache-friendly re-reading strategies.

## 2. Constraints and correctness requirements

1. **Exact match, not "a valid sort".** The evaluator compares our tensors against the reference
   outputs (int tolerance is effectively exact; atol=1e-5, rtol=0.01 on integer data). A *valid*
   but non-stable permutation will differ from `torch.sort(stable=True)` and **fail**. Therefore
   our scatter must reproduce stable ordering (ascending original index within each expert bucket).
2. **No computational fallback.** Both the histogram+prefix and the scatter/argsort must be Triton
   kernels. `torch.bincount`, `torch.cumsum`, `torch.sort` are computational and disallowed as the
   implementation. `torch.empty`/`reshape`/`view`/`contiguous`/stride queries/launch grid math are
   allowed plumbing.
3. **dtypes.** Outputs int32. Index values up to N-1 (~18431) and cumulative counts up to N
   (~18432) fit int32 with huge margin. Internal accumulation can stay int32.
4. **Contiguity / flatten order.** Must treat the input as a C-contiguous 1D array of length N.
   `torch.randint` output is contiguous; use `topk_idx.reshape(-1)` (a view for contiguous input).
   Defensive `.contiguous()` if needed.
5. **Shapes.** `sorted_token_indices` length exactly N; `expert_offsets` length exactly 257 with
   `expert_offsets[0] == 0` and `expert_offsets[256] == N`.
6. **Empty experts.** Some experts may receive 0 tokens (rare at these sizes but possible). The
   design must write nothing for such experts and keep `expert_offsets[e] == expert_offsets[e+1]`.

## 3. Numerical / correctness risk register

- **Stability tie-break** (highest risk): any atomic-counter scatter where atomic ordering is
  nondeterministic across the array will *not* be stable → mismatch. Must use an order-preserving
  scatter.
- **Exclusive vs inclusive prefix off-by-one**: base offset of expert `e` is the *exclusive* prefix
  `expert_offsets[e]`; within a bucket the per-element rank is the *exclusive* running count
  (`inclusive_cumsum - self`). Getting either wrong shifts the whole bucket by one → mismatch.
- **`expert_offsets[0]` must be 0** and array length exactly 257 (histogram has 256 bins, offsets
  has 257). Off-by-one in array size or a missing leading zero fails.
- **Tail masking**: when N is not a multiple of the block size, masked lanes must (a) not be counted
  in the histogram and (b) never match any expert during scatter. Set masked keys to a sentinel
  outside [0,255] (e.g. 256 or -1) and exclude from prefix/reduction with the load mask.
- **Reshape/flatten order** must be C-order to match `reshape(-1)`.
- **Integer overflow**: none realistic (values << 2^31); keep int32.
- **Determinism across runs**: order-preserving scatter is fully deterministic; histogram via
  commutative sums (atomics or `tl.histogram`) is order-independent and safe.

## 4. Triton design space

The problem decomposes into (A) histogram + exclusive/inclusive prefix → `expert_offsets`, and
(B) stable scatter → `sorted_token_indices` using expert base offsets. Below are candidate schemes.

### 4.1 Stable scatter (the hard, decisive part)

**Scheme S1 — expert-parallel sequential scan (preferred first candidate).**
Grid = 256 programs, one per expert `e`. Program `e`:
- reads `base = expert_offsets[e]` (from kernel A);
- loops over the flat array in tiles, in ascending index order;
- for each tile: `mask = (v == e)`; exclusive within-tile prefix = `tl.cumsum(mask.to(int32)) - mask`;
  write position = `base + running + excl_prefix`; `tl.store(out + pos, idx, mask)`;
  `running += sum(mask)`.
- Because tiles are visited in increasing index order and lanes within a tile carry increasing
  index, matched indices are written in strictly ascending order → **stable by construction**.
- Each program writes only into `[base, base+count)` — disjoint ranges, so **no atomics on the
  output**, no write races.
- Work is O(E·N) ≈ 256·18k ≈ 4.7M compares, but the flat array (~64–72 KB) is reused by all 256
  programs and lives in L2 → effectively an L2-bandwidth-bound scan; extremely cheap at this size.
  Occupancy: 256 blocks over 108 SMs is a healthy 2–3 blocks/SM.
- Pros: simple, fully deterministic/stable, no intermediate buffers, no atomics. Cons: redundant
  E×N reads (fine for small N; would not scale to very large N).

**Scheme S2 — tiled counting sort (per-tile histogram + column scan + scatter).**
Classic parallel radix sort: K1 computes per-tile histograms `H[num_tiles][256]`; a scan produces
`start[t][e] = expert_offsets[e] + sum_{t'<t} H[t'][e]`; K2 scatters each tile, where element with
expert `e` at intra-tile rank `r` goes to `start[t][e] + r`. Intra-tile per-expert rank needs a
keyed prefix count — obtainable via an outer-comparison `(T×T)` reduction (feasible for modest T,
~T=256–512) or a segmented scan. Reads the array only ~2N total.
- Pros: O(N) reads, scales to large N. Cons: more kernels, an intermediate `num_tiles×256` buffer,
  and trickier intra-tile keyed-rank code (T×T register pressure). At N~18k the E·N scan of S1 is
  already trivial, so S2's benefit is marginal here; keep as a fallback/optimization.

**Scheme S3 — atomic scatter (REJECTED for correctness).** `pos = atomicAdd(write_pos[e], 1)`
gives a valid but non-stable permutation → fails exact comparison. Do not use.

**Decision:** start with S1 (correct, simple, cache-friendly). Consider S2 only if profiling shows
the scatter dominates and needs the O(N) read reduction.

### 4.2 Histogram + prefix → `expert_offsets`

**Scheme A1 — single-program histogram+cumsum (preferred, 1 kernel).**
One program loops over the flat array accumulating a 256-wide histogram (via `tl.histogram` per
tile with masked tail handling, or masked `tl.atomic_add`/manual bincount), then `tl.cumsum` over
the 256 counts to produce the inclusive prefix; store `expert_offsets[0]=0`,
`expert_offsets[1:]=inclusive_cumsum`. N is small (~9 tiles at BLOCK=2048), 256-bin accumulation is
cheap. Uses a single SM (low utilization) but negligible wall-clock at this size.

**Scheme A2 — parallel atomic histogram + tiny cumsum (2 kernels).**
K1: many blocks, `tl.atomic_add(counts + v, 1, mask)` (order-independent, safe). K2: one block does
`tl.cumsum` over 256 → `expert_offsets`. Better GPU utilization for the count; extra kernel launch.
Prefer if A1's serial histogram shows up in profiling.

**`tl.histogram` availability caveat:** confirm the installed Triton exposes `tl.histogram`,
`tl.cumsum`, `tl.associative_scan`. Fallbacks: histogram via masked `tl.atomic_add`; exclusive
prefix via `tl.cumsum(x) - x`. `tl.atomic_add` and `tl.cumsum` are broadly available; `tl.histogram`
is newer — have the atomic fallback ready. (I could not run Python in this environment to probe the
version directly; verify at implementation time via the first candidate's eval feedback.)

### 4.3 Kernel-count / fusion trade-off
At N~18k these are tiny kernels dominated by launch overhead, so fewer kernels is better. The
minimal correct pipeline is **2 kernels**: (K1 = A1 histogram+prefix → `expert_offsets`), then
(K2 = S1 stable scatter → `sorted_token_indices`, reading base offsets from `expert_offsets`).
Cross-expert prefix inherently needs a global barrier between counting and scatter, so a fully
single-kernel solution would require a persistent/cooperative grid with a global sync — higher
complexity, deferred unless needed. The baseline (torch `sort` + `bincount` + `cumsum`) is 3+
generic launches; our 2 fused, range-specialized kernels should beat it primarily by cutting
launch/generic-sort overhead and exploiting the bounded 0..255 key range.

### 4.4 Tuning knobs (for later plan/candidates)
- `BLOCK` for the scatter scan (512 / 1024 / 2048) and for the histogram loop.
- `num_warps` / `num_stages` per kernel.
- Histogram A1 vs A2 (utilization vs launch count).
- Optionally have each scatter program handle a small contiguous *range* of experts to trade grid
  size vs redundant reads (keep 256 first for max parallelism + simplicity).
- S1 vs S2 if scatter dominates.

## 5. Proposed candidate roadmap (sketched; full plan in docs/plan.md later)
- **c001**: 2-kernel baseline = A1 (histogram+prefix) + S1 (expert-parallel stable scatter),
  conservative BLOCK/num_warps. Goal: correctness on all 5 workloads first.
- Subsequent candidates: tune BLOCK/num_warps/num_stages; try A2 (parallel atomic histogram) if the
  count kernel matters; try S2 tiled counting sort if the scatter is the bottleneck; consider fusing
  the leading zero / output writes.
- Immutable IDs, one source version per candidate, evaluate only via
  `./scripts/evaluate_candidate.sh feedback cNNN`.

## 6. Validation strategy
- **Correctness reasoning (primary, pre-eval):** the stability argument (S1 writes ascending indices
  per bucket → matches `sort(stable=True)`); exclusive/inclusive prefix bookkeeping; length/leading-
  zero checks; tail-mask sentinel handling; empty-expert handling.
- **Evaluator feedback (only permitted execution):** `evaluate_candidate.sh feedback cNNN` runs the
  5 fixed workloads = one candidate evaluation, reporting per-workload correctness + speedup and
  geomean. I cannot run CUDA, a profiler, `nvidia-smi`, python, or any alternate harness directly
  (Bash is disabled here and the rules forbid it), so all empirical validation goes through the
  trusted evaluator.
- **Manual invariants to eye-check per candidate:** `expert_offsets[0]==0`, `expert_offsets[256]==N`,
  monotonic non-decreasing offsets, `sorted_token_indices` is a permutation of `0..N-1`, and each
  bucket's indices are strictly ascending.
- **Convergence:** stop when geomean speedup plateaus across candidate variants or the
  token/eval budget nears; then write `SEARCH_COMPLETE`. Never run `final` without operator approval.

## 7. Skill usage note
Target is A800 / `sm_80` (Ampere). The `KernelWiki` skill (Blackwell SM100 / Hopper SM90) and
`ncu-report-skill` (B200 / sm_100 profiling) are **not applicable** to this Ampere, integer,
counting-sort task, so no skill is invoked. Techniques used here (counting sort, prefix scan,
order-preserving scatter, L2-resident re-reads) are generic Ampere-appropriate patterns.
