# Draft — `gqa_paged_decode_h32_kv8_d128_ps1` (FlashInfer, H100 / sm_90)

## 1. What the operation computes

Batched **Grouped-Query-Attention decode** with a **paged KV cache**, captured from
Llama-3.1-8B. One query token per sequence (decode step); the KV cache is read through a
per-sequence page table.

Fixed constants (from `task/definition.json`):

| Symbol | Value | Meaning |
|---|---|---|
| `num_qo_heads` | 32 | query/output heads |
| `num_kv_heads` | 8 | KV heads |
| `head_dim` (`D`) | 128 | per-head dimension |
| `page_size` | 1 | **one token per page** |
| `gqa_ratio` | 32/8 = **4** | query heads sharing one KV head |

Variable axes: `batch_size` (number of query tokens), `num_pages`, `len_indptr =
batch_size+1`, `num_kv_indices = kv_indptr[-1]`.

### Inputs / outputs

- `q`: `[batch, 32, 128]` bf16
- `k_cache`, `v_cache`: `[num_pages, 1, 8, 128]` bf16 (dim-1 is the page slot, size 1)
- `kv_indptr`: `[batch+1]` int32 — per-sequence offsets into `kv_indices`
- `kv_indices`: `[num_kv_indices]` int32 — page ids (= token row ids, since `page_size=1`)
- `sm_scale`: fp32 scalar ≈ `1/sqrt(128) = 0.08838834764831843`
- `output`: `[batch, 32, 128]` bf16
- `lse`: `[batch, 32]` fp32 — **base-2** log-sum-exp of the scaled logits

### Reference semantics (fp32 math)

For each sequence `b` and query head `h`:
```
kv_head      = h // 4
token_ids    = kv_indices[kv_indptr[b] : kv_indptr[b+1]]      # length = seq_len
K            = k_cache[token_ids, 0, kv_head, :]              # [seq_len, 128]
V            = v_cache[token_ids, 0, kv_head, :]              # [seq_len, 128]
logits       = (q[b,h] · K^T)                                # [seq_len]
scaled       = logits * sm_scale
lse[b,h]     = logsumexp(scaled) / ln(2)                      # base-2 LSE
attn         = softmax(scaled)
output[b,h]  = (attn · V).to(bf16)
```
Empty sequence (`kv_indptr[b] >= kv_indptr[b+1]`): reference leaves `output[b]=0` and
`lse[b]=-inf`.

Notes that matter:
- **No causal mask** — decode attends to *all* cached tokens.
- Reference casts `k_cache`/`v_cache` to fp32, does all matmuls/softmax in fp32, and only
  rounds the final `output` to bf16. `lse` stays fp32.
- The 48 feedback workloads == the full official set (TASK.md §8/§12). The README's "five
  workloads" line is stale; `feedback` and `final` differ only in warmup/iteration counts.

## 2. Memory layout of the paged cache (critical for the gather)

`k_cache[num_pages, 1, 8, 128]` is row-major, so element `(p, 0, kh, d)` lives at flat
offset `p*1024 + kh*128 + d`. Therefore:
- The 128 `d`-values for one `(page, kv_head)` are **contiguous** (256 B in bf16).
- Loading a tile of `BLOCK_N` tokens for one kv-head is a **gather** over the page id with a
  contiguous 256 B chunk per gathered row — good coalescing granularity.
- Indices reach `num_pages ≈ 81390`; `p*1024 ≈ 8.3e7` fits int32, but I will **cast page ids
  to int64** before pointer arithmetic to remove any overflow risk with block offsets.

## 3. Workload distribution (from `feedback_workloads.jsonl`)

Three regimes, 16 workloads each:

| batch | count | total `num_kv_indices` | ≈ per-seq len | ≈ KV bytes (K+V, all 8 heads) | regime |
|---|---|---|---|---|---|
| 1 | 16 | 2 … 547 | 2 … 547 | ~8 KB … ~2.2 MB | **launch/latency-bound**, very under-occupied |
| 16 | 16 | 1034 … 20911 | ~65 … ~1300 | ~4 MB … ~85 MB | mixed |
| 64 | 16 | ~50900 … ~59000 | ~795 … ~920 | ~210 … ~240 MB | **HBM-bandwidth-bound** |

KV volume per program (one kv-head) = `seq_len * 128 * 2 B`. At ~3.35 TB/s the batch=64
cases cost ~70 µs if we read K+V exactly once. Reading them more than once (see §4) is the
main performance risk.

Parallelism if we launch one program per `(batch, kv_head)`:
- batch=1 → **8 CTAs** (of 132 SMs) — badly under-occupied, but the work is tiny.
- batch=16 → 128 CTAs — ~one wave.
- batch=64 → 512 CTAs — several waves, good bandwidth saturation.

## 4. The dominant optimization: GQA packing (reuse K/V across the group)

The 4 query heads sharing a kv-head must read the *same* `K`/`V`. A naive
"one program per `(batch, query_head)`" design (grid `batch*32`) reads each K/V tile **4×**.
Because the kernel is bandwidth-bound on the big workloads, **packing the group of 4 query
heads into one program that loads K/V once cuts HBM traffic ~4×** — the single most
important decision. This is the standard GQA flash-decode layout.

Chosen primary design (**Approach B**):
- Grid = `(batch, num_kv_heads)` = `(batch, 8)`.
- Each program owns one `(b, kv_head)`, handles its **4 query heads** together.
- Read `q_group = q[b, kv_head*4 : kv_head*4+4, :]` → `[4,128]`.
- Loop over the sequence in tiles of `BLOCK_N`:
  - gather page ids `kv_indices[start+offs_n]`, load `K_tile [BLOCK_N,128]`, `V_tile`.
  - `logits = q_group @ K_tile^T` → `[4, BLOCK_N]` (fp32 accum).
  - online softmax (base-2, see §5) updating `m[4]`, `l[4]`, `acc[4,128]`.
- Write `output[b, kv_head*4 + 0..3, :]` and `lse[b, kv_head*4 + 0..3]`.

Small-M note: the group dimension is only 4. Tensor-core MMA wants M≥16, so I will pad the
query-group tile to `BLOCK_H=16` (rows 4..15 masked, ignored on store). The wasted MMA
compute is irrelevant because the kernel is memory-bound. (Alternative: keep M=4 and let
Triton pad internally, or replace `q@kᵀ` with a broadcast-multiply + `tl.sum` reduction over
D — a valid fallback if `tl.dot` with tiny M misbehaves.)

### Alternatives considered

- **Approach A — grid `batch*32`, one head each.** Simpler indexing but 4× KV traffic →
  rejected as primary (kept only as a mental baseline).
- **grid = `batch`, loop 8 kv-heads inside.** Same KV traffic as B but only `batch`
  programs → 1 CTA for batch=1. Worse occupancy than B. Rejected.
- **Approach C — split-KV / flash-decoding.** Split each sequence into `S` chunks across
  extra programs, then a second "combine" kernel merges partials via LSE. This raises
  occupancy for the low-parallelism regime (batch=1 → 8 CTAs). But here batch=1 sequences
  are short (≤547 tokens, ≤2.2 MB), so they are launch-bound, not bandwidth-bound, and a
  second launch adds its own overhead on tiny data. **Deferred**: only add it as a later
  candidate if profiling shows the batch=1/16 cases are occupancy-limited and the combine
  cost is amortized. The big batch=64 cases already fill the machine, so split-KV would not
  help them.

## 5. Numerical design and risks

### 5.1 Base-2 LSE — exact match via a folded scale
Reference: `lse = logsumexp_e(scaled)/ln2 = log2(Σ e^{scaled_i})`.

Fold `log2e = 1/ln2` into the scale. Let `t_i = scaled_i * log2e = (q·k) * (sm_scale *
log2e)`. Because `2^{t_i} = 2^{scaled_i·log2e} = e^{scaled_i}`:
```
m = max_i t_i
l = Σ_i exp2(t_i - m)
lse_out = m + log2(l)          # == log2(Σ e^{scaled_i}), exactly the reference
p_i     = exp2(t_i - m) / l    # softmax (base-invariant)
out     = Σ_i p_i · V_i
```
So I pass `qk_scale = sm_scale * log2e` (a host-side python float) to the kernel, use
`tl.exp2`, and emit `lse = m + log2(l)`. This reproduces the base-2 LSE without a stray
`/ln2`. **Risk if I forget the fold** → LSE off by a `ln2` factor; guard with a mental check
and the first evaluation.

### 5.2 Empty sequence
`seq_len == 0` ⇒ no tiles processed ⇒ `m=-inf`, `l=0`, `acc=0`. `acc/l` would be `0/0=NaN`.
Guard on store: `out = where(l>0, acc/l, 0)`, `lse = where(l>0, m+log2(l), -inf)`. Matches
reference (`output=0`, `lse=-inf`). Any per-element zero-length case in batch=16/64 is
covered even if none actually occurs.

### 5.3 Precision of the matmuls vs reference (fp32)
- `q@kᵀ`: `q`,`k` are already bf16 (same bits the reference casts up to fp32). A bf16×bf16
  tensor-core dot with **fp32 accumulate** reproduces the reference fp32 dot essentially
  exactly (products are exact in fp32; the reduction order differs only slightly). Do **not**
  use tf32.
- `p@V`: reference does fp32×fp32. Two options:
  1. keep `p` fp32, cast `V` to fp32, `tl.dot` in fp32 (closest to reference, more regs);
  2. cast `p` to bf16 and dot with bf16 `V` (tensor-core, faster, tiny extra error).
  Output is bf16 with attention-style tolerance, so option 2 is very likely within tol; I
  will start with whichever compiles cleanly and confirm via evaluation, keeping option 1 as
  the accuracy fallback.
- Always accumulate `m`, `l`, `acc` in **fp32**.

### 5.4 Overflow / masking
- Max-subtraction in the online softmax prevents `exp2` overflow. Logit magnitude ≈
  `O(sqrt(128))·sm_scale ≈` a few — safe even without it, but keep it for correctness on all
  inputs.
- Out-of-range tokens in the last tile: mask `offs_n < seq_len`; set their logits to `-inf`
  before `exp2` (contribute 0). Gathered K/V for masked lanes loaded with `other=0` and a
  valid (clamped) pointer.

### 5.5 Determinism / host sync
Grid comes from static shapes (`batch = q.shape[0]`, `num_kv_heads = 8`). Read
`kv_indptr[b]`/`[b+1]` **inside** the kernel — no `.item()`, no host sync, no `num_kv_indices`
dependence on the host. `sm_scale` arrives as a scalar; fold `log2e` on the host into a
python float. Data-dependent loop bound `range(0, seq_len, BLOCK_N)` is fine in Triton.

## 6. Triton design space / tunables

- `BLOCK_N` ∈ {32, 64, 128}: larger ⇒ fewer iterations but more shared mem
  (`K+V = BLOCK_N*128*2B` each; at 128 that is 32 KB×2, ×`num_stages`). Start 64.
- `BLOCK_H` = 16 (padded group; actual 4). `BLOCK_D` = 128 (whole head, one tile — no D-loop).
- `num_warps` ∈ {2, 4, 8}; `num_stages` ∈ {2, 3, 4} to pipeline the gather latency.
- Second-dot dtype (fp32 vs bf16 `p`) as in §5.3.
- Optional autotune, but keep the config set small so warmup compile time (counted in the
  coarse feedback run) stays low.
- Later, if needed: split-KV `S` factor + combine kernel (Approach C).

Entry point `solution/solution.py::run(q, k_cache, v_cache, kv_indptr, kv_indices,
sm_scale)` allocates `output = torch.empty([batch,32,128], bf16)` and `lse =
torch.empty([batch,32], fp32)`, computes `qk_scale = sm_scale*log2e`, launches the kernel on
grid `(batch, 8)`, returns `(output, lse)`. Pure Triton compute; PyTorch only for
allocation/launch (no Torch/CPU/NumPy fallback, per contract).

## 7. Validation strategy

Constraints: I may **not** run CUDA/nvidia-smi directly or any alternate correctness harness;
the only sanctioned correctness+timing path is `./scripts/evaluate_candidate.sh feedback
cNNN` (one call = one of 100 evaluations over all 48 workloads). Profiling is only via
`./scripts/ncu_profile.sh` and must **never** overlap an evaluation (return code 3 wastes a
budget slot).

Because evals are the only oracle, correctness must be reasoned right *before* the first run:
1. Re-derive the base-2 LSE fold (§5.1) and the empty-seq guard (§5.2) on paper before coding.
2. Cross-check tensor shapes/strides against the layout in §2.
3. `c001` = Approach B, conservative config (`BLOCK_N=64`, `BLOCK_H=16`, `num_warps=4`,
   `num_stages=2`, fp32 accum). Evaluate once; confirm **all 48 pass correctness** and record
   the geomean baseline.
4. If any workload fails correctness, diagnose from the reported per-workload result (most
   likely LSE base, empty-seq, or masking) and fix in `c002` — never loosen math to "pass".
5. Once correct, iterate on performance: tune `BLOCK_N`/`num_warps`/`num_stages`, then the
   second-dot dtype. Each meaningful change = a new immutable candidate id.
6. Use one `ncu_profile.sh` pass (separate from any eval) to confirm the big batch=64 case is
   HBM-bound and reading K/V once; check achieved BW% and occupancy. If batch=1/16 are
   occupancy-limited, introduce split-KV (Approach C) as its own candidate and compare.
7. Record every candidate in `candidates.jsonl` (parent, source hash, hypothesis, per-workload
   result, geomean, decision, cumulative eval count, skill usage). Stop when the geomean
   plateaus or the budget is hit; write `SEARCH_COMPLETE` with the reason. Never run `final`
   without operator approval.

### Success criteria
- Every selected workload passes correctness (hard gate).
- Maximize geometric-mean speedup vs the reference. The big wins come from (a) GQA packing to
  read K/V once, (b) enough CTAs + pipelined gathers to saturate HBM on batch=64/large-16,
  and (c) low launch overhead on the tiny batch=1 cases.
