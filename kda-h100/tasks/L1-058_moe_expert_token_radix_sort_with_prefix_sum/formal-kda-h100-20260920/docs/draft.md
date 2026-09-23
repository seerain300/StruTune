# Draft — L1/058 MoE Expert Token Radix Sort with Prefix Sum

Task: optimize `058_moe_expert_token_radix_sort_with_prefix_sum` on H100 (`sm_90`).
Primary implementation must be Triton; PyTorch only for metadata/launch plumbing; no
Torch/CPU/NumPy/CUDA-extension fallback. Submission is `solution/solution.py` exposing `run(topk_idx)`.

---

## 1. Exact operation semantics

### 1.1 Signature and shapes

- Input `topk_idx`: shape `(batch_size, seq_len, num_experts_per_tok)`, **int32**, values in `[0, 255]`.
  - `num_experts = 256` (const), `num_experts_per_tok = 8` (const).
- `N = batch_size * seq_len * num_experts_per_tok` = number of token→expert assignments (the flattened length).
- Outputs:
  - `sorted_token_indices`: shape `(N,)`, **int32**.
  - `expert_offsets`: shape `(num_experts + 1,) = (257,)`, **int32**.

### 1.2 Reference algorithm (authoritative)

```python
num_experts = 256
flat = topk_idx.reshape(-1)                     # length N, values 0..255
_, sorted_token_indices = flat.sort(stable=True)  # STABLE argsort -> permutation
expert_offsets = torch.zeros(257, int32)
expert_offsets[1:] = torch.bincount(flat.long(), minlength=256).cumsum(0).to(int32)
return sorted_token_indices.to(int32), expert_offsets
```

Decoded meaning:

1. **`expert_offsets`** is the histogram of expert IDs turned into an **inclusive** cumulative sum,
   prepended with 0:
   - `expert_offsets[0] = 0`
   - `expert_offsets[e+1] = #{ i : flat[i] <= e }` for `e = 0..255`
   - Equivalently `expert_offsets[e] = #{ i : flat[i] < e }` = **exclusive** prefix sum of the
     histogram = the starting write offset of expert `e` in the sorted array.
   - `expert_offsets[256] == N` always.
   - `expert_offsets[e] : expert_offsets[e+1]` is exactly the slice of `sorted_token_indices`
     belonging to expert `e`.

2. **`sorted_token_indices`** is the **stable argsort** of `flat` by expert ID. Because the sort is
   stable, this is *exactly* the stable counting-sort permutation: for `e = 0,1,...,255` in
   increasing order, list the original positions `i` where `flat[i] == e`, in **increasing `i`
   order**. There is a unique correct permutation, and we must reproduce it bit-for-bit.

So the whole op is a **stable counting/radix sort on a bounded key (0..255)** plus the **prefix sum**
of its histogram. No floating point anywhere.

### 1.3 Concrete sizes in the feedback set (16 workloads = also the final set)

| bs | seq | N = bs*seq*8 | N/256 |
|----|-----|--------------|-------|
| 8  | 256 | 16384 | 64 |
| 64 | 128 | 65536 | 256 |
| 2  | 1024| 16384 | 64 |
| 32 | 128 | 32768 | 128 |
| 4  | 512 | 16384 | 64 |
| 1  | 2048| 16384 | 64 |
| 4  | 544 | 17408 | 68 |
| 2  | 1056| 16896 | 66 |
| 2  | 1088| 17408 | 68 |
| 2  | 2048| 32768 | 128 |
| 1  | 2080| 16640 | 65 |
| 1  | 2112| 16896 | 66 |
| 2  | 1120| 17920 | 70 |
| 1  | 4096| 32768 | 128 |
| 4  | 1024| 32768 | 128 |
| 8  | 288 | 18432 | 72 |

Observations:
- **N is small**: `16384 ≤ N ≤ 65536`. Average tokens per expert ≈ `N/256` ≈ 64–256.
- **Every seq_len is a multiple of 32** (256,128,1024,…,544=17·32, 288=9·32, 1056=33·32, 2080=65·32,
  2112=66·32, 1120=35·32). Since `N = bs*seq*8`, **N is always a multiple of 256** in this set (in
  fact of `8*32 = 256`). I will still implement general masking so correctness never depends on this,
  but it means a `BLOCK` that divides 256 (e.g. 256, 512, 1024) tiles the data with no ragged tail on
  most shapes.
- These are **micro-kernels**: the total data (`≤ 64K` int32 = 256KB) fits in L2 (50MB). Wall-clock
  time is dominated by **kernel-launch overhead, allocation, and fixed per-launch costs**, not by
  bandwidth or compute. This is the single most important performance fact for this task.

### 1.4 Where the reference spends time (the thing to beat)

The reference issues several CUDA kernels + allocations:
1. `flat.sort(stable=True)` → CUB radix sort (multiple internal kernels + workspace alloc).
2. `torch.zeros(257)` allocation + memset.
3. `flat.long()` (int32→int64 copy/alloc).
4. `torch.bincount(...)` (another kernel + alloc).
5. `.cumsum(0)` (another kernel + alloc).
6. `.to(int32)` casts and slice-assignment `expert_offsets[1:] = ...`.

At `N ≈ 16K–64K` this is **launch/alloc-bound**. Our win comes from **collapsing this into 1–2
Triton launches with ≤3 allocations (2 outputs + optional small scratch)** and avoiding the int64
detour. We do not need to be bandwidth-optimal; we need to be launch-count-optimal and allocation-lean.

---

## 2. Constraints

- **Triton-only** compute. Torch allowed for: `reshape`/`view` of the input (free, metadata only),
  output tensor allocation (`torch.empty`), grid computation, and dtype/shape bookkeeping. No
  `torch.sort`, `bincount`, `cumsum`, `argsort` in the compute path — those would be a Torch
  fallback and are forbidden.
- Outputs must be exactly int32, exact shapes `(N,)` and `(257,)`.
- No global barrier exists inside a single Triton launch → any design that needs *all* per-block
  histograms before scattering requires **≥2 launches** (or a single-block design).
- Must run correctly across all 16 shapes (feedback == final workload set).
- Tolerance `atol=1e-5, rtol=0.01` is on **integer** tensors → effectively **exact match required**.
  A "valid but different" permutation (e.g. from unordered atomics) will FAIL because indices differ
  by large integer amounts.

---

## 3. Numerical / correctness risks

Even though there is no floating-point arithmetic, the correctness surface is delicate:

1. **Stability is mandatory.** Within each expert bucket the token indices must be in ascending
   original order. Any approach that assigns write slots with **`atomic_add` on a per-expert counter
   is NON-deterministic in order** and will (almost always) produce a differently-ordered bucket →
   FAIL. We must use an *ordered* scatter (deterministic rank) or an *ordered* gather.
   - Careful subtlety: atomics *do* produce a valid counting sort of the histogram (the offsets/counts
     are still correct), but the per-bucket *ordering* of token indices will not match torch's stable
     sort. So atomics are acceptable ONLY for computing counts/offsets, never for placing indices.

2. **Off-by-one in the prefix sum.** `expert_offsets` is the *exclusive* prefix (start offsets), but
   the reference stores the *inclusive* cumsum into `expert_offsets[1:]` with `expert_offsets[0]=0`.
   Concretely: `expert_offsets[1:257] = inclusive_cumsum(counts)`, `expert_offsets[0] = 0`. The
   *scatter base* for expert `e` is `expert_offsets[e]` = exclusive prefix = `inclusive[e]-counts[e]`.
   Must not confuse inclusive vs exclusive; must produce all 257 entries; must have
   `expert_offsets[256] == N`.

3. **Empty experts.** With random data and `N/256` as low as 64, some experts may receive 0 tokens
   (especially small-N shapes). Then `expert_offsets[e] == expert_offsets[e+1]` (zero-length range).
   The scatter/gather must write nothing for those and must not go out of bounds. Prefix-sum handles
   this automatically; just verify no `count>0` assumption anywhere.

4. **Ragged tail masking.** For general `N` not divisible by `BLOCK`, masked-out lanes must not
   pollute the histogram. `tl.histogram(x, num_bins=256)` semantics for out-of-range / masked lanes
   must be pinned down: safest is to load with `other = 256` (an out-of-range sentinel bin that a
   256-bin histogram ignores) OR load `other = 0` and subtract the number of masked lanes from bin 0.
   In the scatter/gather pass, masked lanes must be excluded from the compaction count and never
   written. (In this fixed set N%256==0, but I keep masking for robustness and to avoid surprises.)

5. **Index dtype / overflow.** All positions ≤ `N ≤ 65536` fit comfortably in int32; histogram counts
   fit in int32; cumulative sums ≤ N fit in int32. No int64 needed. Must still make sure Triton
   intermediate accumulators are int32 (not accidental int1/int16 from a comparison) and that stored
   outputs are cast to int32.

6. **`sorted_token_indices` holds ORIGINAL flat indices `i` (0..N-1)**, not expert IDs. Easy to
   accidentally store the wrong quantity. The value written is the source position `i`; the *location*
   written to is `base(expert) + rank`.

---

## 4. Triton design space

The op factors into three deterministic sub-steps: **(H) histogram**, **(P) exclusive prefix sum over
256 bins**, **(S) ordered placement of token indices**. The design axes are (a) how many launches,
(b) how the ordered placement avoids atomics, (c) parallelization granularity.

### Option A — Single-block fully fused (1 launch, 1 SM)
One program does H over all N (loop of tiles, accumulate 256-bin histogram via `tl.histogram`),
computes P (exclusive scan of 256 counts in-register), writes `expert_offsets`, then a second tile
loop performs S: maintain a running position vector `pos[256]` initialized to the exclusive offsets;
for each tile, compute the per-tile keyed rank and scatter, then `pos += tile_histogram`.
- **Pros:** minimum launch overhead (1 launch), no scratch buffer, naturally stable (tiles processed
  in ascending order, ascending within tile).
- **Cons:** runs on a single SM → serializes ~64–1024 tiles; may be latency-bound. The within-tile
  keyed rank still needs a one-hot cumsum (see S-methods). Register/SRAM pressure for a `pos[256]`
  vector carried across the loop.

### Option B — Classic 3-phase counting/radix sort (2–3 launches, multi-SM)
1. **K1 (per-block histogram):** grid = `ceil(N/BLOCK)`; each block computes a 256-bin local histogram
   via `tl.histogram`, writes row `block_hist[b, :]` into a `(num_blocks, 256)` scratch.
2. **K2 (prefix sum):** reduce `block_hist` over blocks → total `counts[256]`; exclusive scan →
   `expert_offsets`; and per-block base `block_base[b,e] = expert_offsets[e] +
   exclusive_prefix_over_blocks(block_hist[:,e])`. (This is SGLang's `moe_align_block_size` block/warp
   scan pattern, PR-7884 / PR-7437.)
3. **K3 (scatter):** each block re-reads its `BLOCK` elements, computes each element's global position
   `block_base[b, expert] + within_block_rank(expert)`, and writes the token index. Stable because
   blocks are ordered and within-block rank is ordered.
- **Pros:** multi-SM parallel; scales with N; the well-known robust design.
- **Cons:** 2–3 launches + a `(num_blocks,256)` scratch alloc; more overhead at tiny N, which may
  negate the parallelism benefit. K2 is a small scan (num_blocks ≤ 256 here).

### Option C — Expert-parallel ordered gather (2 launches, 256-way parallel)
1. **K1:** histogram + exclusive prefix → `expert_offsets` (single block, tiny).
2. **K2:** grid = 256 (one program per expert `e`). Program `e` streams all N elements in tiles, and
   for each tile builds `mask = (expert == e)`, computes the compacted offset via `cumsum(mask)`
   carried across tiles, and writes matching **token indices** consecutively starting at
   `expert_offsets[e]`. This is a per-expert stream compaction.
- **Pros:** dead-simple correctness/stability (each expert's whole output range owned by one program,
  ascending scan → stable); no cross-block prefix, no atomics, no one-hot 256-wide matrix; only 2
  launches; scratch-free (offsets is an output).
- **Cons:** input is read 256× (once per expert program) = up to `256 * 256KB = 64MB` of loads, but
  mostly L2-resident → cheap at these sizes. Some experts empty → program exits fast (load imbalance,
  irrelevant at this scale).

### Option D — Hybrid: fused histogram+prefix in K1, block-scatter in K2
Compute counts and `expert_offsets` in K1 (single block or grid+atomic-for-counts-only), then a
multi-block ordered scatter using per-block bases derived on the fly. Essentially B collapsed to 2
launches by folding P into K1's output.

### Sub-methods for the ordered placement (S) — how to get within-block/keyed rank without atomics

- **S1 one-hot cumsum:** build `onehot[j,e] = (expert[j]==e)` shape `(BLOCK, 256)`, exclusive
  `tl.cumsum` along `j`, gather column `expert[j]` → within-block rank. Cost: `BLOCK*256` ints in SRAM.
  H100 has ~228KB SRAM/SM; `BLOCK=256 → 256*256*4 = 256KB` too big; `BLOCK=128 → 128KB` feasible;
  `BLOCK=64 → 64KB` comfortable. Width fixed at 256 regardless of BLOCK.
- **S2 pairwise compare:** `cmp[j,k] = (expert[j]==expert[k]) & (k<j)`, rank = `sum_k cmp`. Cost
  `BLOCK*BLOCK`. Cheaper than S1 when `BLOCK < 256`, worse when larger.
- **S3 per-expert compaction (Option C's method):** `cumsum(mask)` along the stream for a single
  expert — width 1, cheapest, but only applies to the expert-parallel decomposition.
- **S4 sequential scalar loop:** maintain `pos[256]`, loop elements one at a time. Correct but scalar
  → slow in Triton; avoid unless everything else fails.

### Preferred exploration order (rationale)
Given the problem is **launch/alloc-bound and tiny**, start with the designs that minimize launches
and allocations while remaining trivially stable:
1. **Option C (expert-parallel gather, 2 launches, S3)** — simplest correct/stable, scratch-free,
   good parallelism (256 programs), 256× reads are free at this scale. Strong first candidate.
2. **Option A (single-block fused, 1 launch, S1 with small BLOCK)** — fewest launches; compare against
   C to see whether launch savings beat single-SM serialization.
3. **Option B/D (classic multi-block radix, S1)** — the scalable reference design; likely best only at
   the larger shapes (N=65536), profile to confirm.
Autotune `BLOCK` / `num_warps` per design. Use `tl.histogram` for H (verify it exists in the installed
Triton and its out-of-range semantics before relying on it; fall back to one-hot-sum histogram if not).

### Key primitives to verify in the installed Triton (do this in the first candidate)
- `tl.histogram(x, num_bins)` presence + masked/out-of-range behavior.
- `tl.cumsum(x, axis=...)` inclusive; derive exclusive by subtract-self or shift.
- `tl.sum(x, axis=...)`, `tl.where`, `tl.arange`, broadcasting for one-hot.
- Whether a carried loop-scalar/vector (`pos[256]`) across a `for` range compiles efficiently.

---

## 5. Performance model & expectations

- Bandwidth for the whole op is negligible (`≤ ~1MB` of traffic even with 256× reads → sub-µs at
  3 TB/s). Therefore the metric is essentially **fixed overhead**: launches + allocations + kernel
  prologue/epilogue.
- Target: **2 launches** (C) or **1 launch** (A) vs the reference's ~5–6. Expect the main speedup to
  come from launch/alloc reduction, not from clever compute.
- Allocation discipline: allocate exactly `sorted_token_indices` (`empty(N,int32)`) and
  `expert_offsets` (`empty(257,int32)`), plus at most one small scratch (`(num_blocks,256)`) only if a
  multi-block design is chosen. Avoid `torch.zeros` where `empty` + full write suffices (but ensure
  every element is written; e.g. `expert_offsets[0]` must be explicitly set to 0).
- Compile-time constants: `NUM_EXPERTS=256`, `EPT=8` are constexpr. `N` varies → keep the kernel
  N-generic (masked) and only specialize `BLOCK`.
- Watch for **autotune recompilation cost** across 16 shapes: too many autotune configs × shapes can
  add measurable warmup, but warmup (2 iters) is excluded from timing; still keep the config set lean.

---

## 6. Validation strategy

Hard constraint: **no direct CUDA / nvidia-smi / torch-run / alternate correctness harness** in this
workspace (even generic `python`/shell is sandbox-denied here). All empirical validation goes through:

```
./scripts/evaluate_candidate.sh feedback cNNN     # full 16-workload feedback set = 1 evaluation
```

and profiling (only when NOT evaluating) through:

```
./scripts/ncu_profile.sh --set basic -o profile/rN python harness.py
```

Because I cannot pre-check numerically offline, correctness must be argued **by construction** before
each evaluation, then confirmed by the evaluator:

1. **Algebraic proof per candidate.** For each design, write the invariant that guarantees the output
   equals the stable counting sort: (i) `expert_offsets[e] = #{i: flat[i]<e}`, `expert_offsets[0]=0`,
   `expert_offsets[256]=N`; (ii) placement position of token `i` = `expert_offsets[flat[i]] +
   #{j<i : flat[j]==flat[i]}`; (iii) the scan producing (ii) processes `j` in strictly ascending
   order. Verify these hold in the code (base offset source, rank direction, tie handling).
2. **Edge-case checklist before evaluating:** empty experts (zero ranges), ragged tail masking
   (masked lanes excluded from both histogram and compaction), `expert_offsets[0]` explicitly zeroed,
   int32 casts on both outputs, correct value = source index (not expert id).
3. **Cheap-first evaluation cadence.** Each feedback run costs 1 evaluation (budget 100) and tokens.
   Evaluate a candidate only once its correctness argument is complete. Start with the simplest correct
   design (Option C) to establish a **known-good, known-fast baseline candidate `c001`**, record its
   geomean, then iterate on faster designs, keeping each as an immutable `cNNN`.
4. **Regression discipline.** Record per-workload pass/fail + speedup, geomean, parent, source hash,
   hypothesis, decision, cumulative eval count, and skill usage in `candidates.jsonl` (append-only).
   A candidate that fails ANY workload's correctness is rejected regardless of speed.
5. **Profiling loop (optional, between evaluations only):** if a design is correct but not clearly
   winning, use `ncu_profile.sh` to confirm whether the bottleneck is launch overhead vs single-SM
   serialization vs the 256× reads, then choose between Options A/B/C accordingly. Never profile and
   evaluate concurrently (return-code-3 discards the measurement and burns an evaluation).
6. **Convergence / stop.** Stop when geomean improvement across successive candidates flattens or when
   token/eval budget nears; then write `SEARCH_COMPLETE` with the reason. Never run `final` without
   explicit operator approval.

---

## 7. Open questions to resolve in candidate 1

- Does the installed Triton expose `tl.histogram`, and does it ignore out-of-range bins so masking via
  `other=256` is safe? If not, use a one-hot-sum histogram.
- Does a carried 256-vector across a Python-`for` tile loop compile without huge spills (relevant to
  Option A)?
- For Option C, is 256-way grid with per-program stream compaction actually faster than the reference
  at N=16K (launch overhead of a 256-block grid vs a 1-block grid)? Measure geomean.
- Best `BLOCK` / `num_warps` per design via a small autotune set.

**Decision for first candidate:** implement **Option C (expert-parallel ordered gather, 2 launches)**
as `c001` — it is the most obviously correct + stable + allocation-lean design and gives a solid
baseline geomean. Then explore Option A (1-launch fused) and Option B (multi-block radix) as `c002+`,
guided by the evaluator geomean and, if needed, ncu profiling.
