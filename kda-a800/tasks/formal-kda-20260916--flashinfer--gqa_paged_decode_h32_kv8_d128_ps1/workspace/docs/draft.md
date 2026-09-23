# Draft Analysis — `gqa_paged_decode_h32_kv8_d128_ps1`

Target hardware: **NVIDIA A800 (sm_80, Ampere)**. Primary implementation must be Triton;
PyTorch only for metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-extension fallback.

---

## 1. Operation semantics

Batched **Grouped-Query-Attention decode** with a **paged KV cache**, captured from Llama-3.1-8B.
Each batch element carries exactly **one query token** (decode step), attends over a variable-length
set of cached KV tokens, and produces one output vector plus its log-sum-exp.

### Fixed constants
- `num_qo_heads = 32`
- `num_kv_heads = 8`  → GQA ratio `G = 32 / 8 = 4`
- `head_dim = 128`
- `page_size = 1`  → **each page holds exactly one token**, so `kv_indices` are effectively *token* indices.

### Inputs
| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `q` | `[B, 32, 128]` | bf16 | one query per batch element |
| `k_cache` | `[num_pages, 1, 8, 128]` | bf16 | paged; page stride = `1*8*128 = 1024` |
| `v_cache` | `[num_pages, 1, 8, 128]` | bf16 | same layout |
| `kv_indptr` | `[B+1]` | int32 | per-sequence page offsets (prefix sum) |
| `kv_indices` | `[num_kv_indices]` | int32 | page ids = token ids (page_size=1) |
| `sm_scale` | scalar | f32 | ≈ `1/sqrt(128) = 0.0883883…` |

### Outputs
| Tensor | Shape | Dtype |
|---|---|---|
| `output` | `[B, 32, 128]` | bf16 |
| `lse` | `[B, 32]` | f32 |

### Per `(b, h)` computation (reference)
Let `kv_head = h // G`, token set `T_b = kv_indices[kv_indptr[b] : kv_indptr[b+1]]`.
```
k_head = k_cache[T_b, 0, kv_head, :]         # [num_tokens, 128], fp32
v_head = v_cache[T_b, 0, kv_head, :]         # [num_tokens, 128], fp32
logits = (q[b,h] · k_head.T) * sm_scale      # [num_tokens], fp32
lse[b,h] = logsumexp(logits) / ln(2)         # base-2 LSE
attn   = softmax(logits)
output[b,h] = (attn · v_head).to(bf16)
```
Empty sequence (`kv_indptr[b] >= kv_indptr[b+1]`): `output[b] = 0`, `lse[b] = -inf`.

**Reference numeric domain:** `q`, `k`, `v` are cast **bf16 → fp32** before all math; logits, softmax,
and the PV accumulation are fp32; only the final `output` is rounded back to bf16. Our kernel must
match this domain (fp32 accumulation everywhere, bf16 only at the store).

---

## 2. Workload characterization (the 5 fixed feedback workloads)

`page_size=1` ⇒ `num_kv_indices` = total KV tokens across the batch. Derived quantities:

| WL | B | num_pages | num_kv_indices | avg tok/seq | base CTAs (B·8) | query rows (B·32) | K+V bytes | HBM-bound floor* |
|----|----|-----------|----------------|-------------|-----------------|-------------------|-----------|------------------|
| W1 | 16 | 30163 | 20911 | ≈1307 | 128 | 512  | ≈85.7 MB | ≈43 µs |
| W2 | 16 | 1396  | 1369  | ≈86   | 128 | 512  | ≈5.6 MB  | ≈2.8 µs |
| W3 | 1  | 82    | 65    | 65    | 8   | 32   | ≈0.27 MB | launch-bound (~µs) |
| W4 | 16 | 2708  | 2681  | ≈168  | 128 | 512  | ≈11.0 MB | ≈5.5 µs |
| W5 | 64 | 60071 | 50902 | ≈795  | 512 | 2048 | ≈208.5 MB| ≈104 µs |

\* K+V bytes = `num_kv_indices · 8 heads · 128 · 2 B · 2 (K&V) = num_kv_indices · 4096 B`; floor assumes ~2.0 TB/s HBM.

### Roofline / boundedness
- FLOPs per `(token, qo_head)` = QK (128 MAC) + PV (128 MAC) = 512 FLOP; ×32 heads = `num_kv_indices·16384` FLOP total.
- Bytes (K+V) = `num_kv_indices·4096`. **Arithmetic intensity ≈ 4 FLOP/byte** (with GQA reuse).
- A800 ridge point ≈ `312 TFLOP/s ÷ 2 TB/s ≈ 156 FLOP/byte`. Since `4 ≪ 156`, the kernel is
  **strongly HBM-bandwidth-bound**. Compute-utilization of tensor cores is irrelevant to the ceiling.
- **GQA reuse is the single most important structural decision:** if K/V are reloaded once per *qo* head
  (32×) instead of once per *kv* head (8×), effective bytes rise 4× and AI drops to ~1 FLOP/byte. The kernel
  must load each `(token, kv_head)` slice **once** and reuse it across its `G=4` query heads.

### Parallelism profile
- Base parallel unit `(batch, kv_head)` gives `B·8` CTAs: W1/W2/W4 = 128, W5 = 512, **W3 = 8**.
- A800 has 108 SMs. W5 (512) saturates; W1/W2/W4 (128) marginally fill; **W3 (8) is severely
  under-occupied and additionally has only 65 total tokens → latency/launch-bound.**
- Long sequences (W1 ≈1307, W5 ≈795) make each CTA long-running; splitting the KV axis
  (**flash-decoding / split-K**) increases CTA count and improves latency hiding and load balance.
- Sequence-length imbalance across batch elements (variable `num_tokens`) argues for a token-axis
  split so work per CTA is bounded regardless of the longest sequence.

---

## 3. Correctness contract & edge cases

1. **LSE is base-2 of the scaled logits.** `lse = logsumexp(logits·sm_scale)/ln2 = log2(Σ exp(s_i))`
   where `s_i = (q·k_i)·sm_scale`. Working entirely in base-2 (see §4) yields `lse` directly.
2. **Empty sequence** (`start >= end`): must emit `output[b]=0` and `lse[b]=-inf` for all 32 heads.
   Flash math must not produce `NaN` here (guard `l==0 / count==0`). Must verify whether any feedback
   batch has zero tokens (all five have `num_kv_indices` slightly below `num_pages`, so at least the
   global set is nonempty, but per-batch emptiness must still be handled defensively).
3. **Tail masking**: when `num_tokens` is not a multiple of the KV tile `BLOCK_N`, masked lanes must
   contribute `-inf` to logits (→ `exp2 → 0`) and `0` to the V accumulation, and must not corrupt the
   running max `m`.
4. **Dtype discipline**: accumulate QK, softmax, and PV in **fp32**; cast to **bf16 only** when storing
   `output`; store `lse` as **fp32**. `kv_indptr`/`kv_indices` are int32 — use int64 offset arithmetic
   when forming global addresses to avoid overflow (`num_pages` up to 60071, page stride 1024 →
   max page-byte offset ~1.2e8, safe in int32 but int64 is safer for the pointer math).
5. **Layout / strides** (row-major, page_size=1):
   - `q[b,h,:]`  offset `b·(32·128) + h·128`.
   - `k_cache[page,0,kv,:]` offset `page·1024 + kv·128`; same for `v_cache`.
   - `output[b,h,:]` mirrors `q`; `lse[b,h]` offset `b·32 + h`.
6. **GQA mapping**: head `h` ↔ `kv_head = h // 4`; the 4 query heads `{4·kv, …, 4·kv+3}` share KV.

---

## 4. Numerical-risk analysis

- **Base-2 flash formulation (recommended).** Precompute `q_scaled = q · (sm_scale · log2(e))`.
  Then `p_i = q_scaled · k_i = s_i·log2(e)`. Running max `m = max_i p_i`, `l = Σ exp2(p_i − m)`,
  `acc = Σ exp2(p_i − m)·v_i`. Final `output = acc / l`, and **`lse = m + log2(l)`** — exactly the
  required base-2 LSE, no extra `ln2` conversion. `exp2`/`log2` map to fast hardware ops.
- **bf16→fp32 matching.** The reference multiplies fp32-upcast bf16 operands. A bf16×bf16 product is
  representable exactly in fp32, so either (a) fp32 FMA reduction after upcast, or (b) tensor-core
  `tl.dot` with bf16 inputs and **fp32 accumulation**, reproduces the reference dot to within fp32
  rounding of the summation order. Summation-order differences (tree vs sequential) are within bf16
  output tolerance; this is the main residual numeric risk and must be confirmed by the evaluator.
- **Max/`-inf` handling.** Initialize `m = -inf`, `l = 0`, `acc = 0`. Masked/out-of-range logits set to
  `-inf`. `exp2(-inf) = 0` keeps `l`, `acc` clean. Guard the final divide: if `l == 0` (no valid token),
  write `output = 0`, `lse = -inf`.
- **Split-K combine numerics.** Each split `j` yields `(m_j, l_j, acc_j)`. Combine with the standard
  log-sum-exp merge: `M = max_j m_j`; `L = Σ_j l_j·exp2(m_j − M)`; `A = Σ_j acc_j·exp2(m_j − M)`;
  `output = A / L`; `lse = M + log2(L)`. Skip splits with `l_j == 0` (empty). This is associative and
  numerically stable; the only risk is forgetting the rescale of `acc_j` by `exp2(m_j − M)`.
- **Small-M matmul precision.** With `M = G = 4` query rows, if tensor-core `tl.dot` is used, the `M` axis
  is padded (waste, but no correctness issue). fp32 FMA avoids padding and matches the reference domain
  most directly; given the memory-bound roofline, the FMA path costs no measurable time.
- **`log2`/`exp2` of subnormals / large negatives.** Scaled logits are bounded (`|q·k|` for random
  bf16 unit-ish inputs × `sm_scale` ~O(1)); overflow is not a concern once the running-max subtraction
  is applied. Empty-guard prevents `log2(0)`.

---

## 5. Triton design space

### 5.1 Parallelization scheme
- **Chosen base unit: one program per `(batch b, kv_head kv)`**, computing all `G=4` query heads that
  share `kv` together. This gives K/V single-load reuse (§2) — the decisive optimization.
  - Q block `[4, 128]`, accumulator `[4, 128]` fp32, running `m[4]`, `l[4]`.
  - Grid `(B, 8)` → base CTA counts in the table (§2).
- **Rejected: one program per `(batch, qo_head)`** — 4× KV reload, halves effective bandwidth. Only
  worth considering if register pressure from the `[4,128]` accumulator forces a spill (unlikely at 128 dim).

### 5.2 Flash-decoding / split-K along the KV axis
- Partition each sequence's token range into `S` splits; a first-pass kernel writes partial
  `(m, l, acc)` per `(b, kv, split)`; a lightweight second-pass **combine** kernel reduces over splits
  (§4). Needed to (a) fill SMs for **W3** (8 base CTAs) and (b) balance long sequences (W1/W5) and the
  intra-batch length imbalance.
- **`num_splits` heuristic** (to decide in plan.md): choose `S` so `B·8·S` is a small multiple of 108
  SMs while keeping each split's token count ≥ `BLOCK_N`. Rough targets: W3 `S~4–8`, W1 `S~4–8`,
  W4 `S~1–2`, W2 `S~1`, W5 `S~1–2`. A single autotuned/heuristic formula from
  `(B, avg_tokens, num_sms)` is preferred over per-workload constants.
- **Alternatives to two-pass**: atomic accumulation into a global `(m,l,acc)` with `atomicMax` on `m`
  is racy for the rescale and is rejected; a fixed-`S` padded partial buffer + combine pass is clean
  and deterministic. For `S==1` the combine pass is skipped (kernel writes final output directly).

### 5.3 Tiling
- `BLOCK_D = 128` (full head_dim, fits in registers/one MMA K-tile).
- `BLOCK_N` ∈ {32, 64, 128} tokens per KV tile — autotune candidate. Larger `BLOCK_N` amortizes index
  loads and loop overhead; smaller reduces wasted tail work for short sequences (W2/W3).
- `BLOCK_M = 4` (the GQA group) — small; drives the FMA-vs-dot decision (§4).

### 5.4 Paged gather
- Load `page_id = kv_indices[start + n]` for the tile, then gather
  `k_cache_ptr + page_id·1024 + kv·128 + arange(128)`. Because `page_size=1`, the 128-dim inner slice is
  contiguous (good coalescing per token); across tokens the access is a gather (page ids arbitrary).
- Load the `page_id` vector once per tile and reuse for both K and V.
- Mask tail lanes where `start + n >= end`.

### 5.5 QK / PV computation
- **Option A (fp32 FMA):** `p = tl.sum(q_scaled[:,None,:] * k[None,:,:], axis=2)` → `[4, BLOCK_N]`;
  `acc += (probs[:,:,None] * v[None,:,:])` reduced over `N`. Matches reference domain exactly, no MMA
  padding. Preferred given memory-bound roofline.
- **Option B (tensor-core `tl.dot`):** bf16 inputs, fp32 accumulate; `M=4` padded to 16. Only revisit if
  Option A shows unexpected compute overhead in evaluator timings.

### 5.6 Autotune / heuristic surface
- `BLOCK_N`, `num_splits` (or the heuristic that derives it), `num_warps` (1–4; small tiles favor 1–2),
  `num_stages` (software pipelining of the gather; 2–3). Keep the tuned config small and stable so a
  single immutable candidate covers all five workloads.

### 5.7 Launch plumbing (PyTorch, allowed)
- Allocate `output` (bf16) and `lse` (f32); compute strides; build the grid; if split-K, allocate the
  partial buffers `[B, 8, 4, S, 128]` (acc), `[B, 8, 4, S]` (m, l). Everything else runs in Triton.

---

## 6. Candidate ladder (planned progression, for plan.md)
1. **c001 — correctness-first baseline:** single-pass `(B, 8)` grid, GQA-grouped, fp32 FMA, `BLOCK_N`
   fixed, no split-K, full masking + empty guard. Establishes correctness and a timing reference.
2. **c002 — split-K / flash-decoding** with combine pass + `num_splits` heuristic (fixes W3 occupancy
   and long-sequence balance).
3. **c003+ — tuning:** `BLOCK_N`, `num_warps`/`num_stages`, gather pipelining, optional `tl.dot` path,
   refined `num_splits` formula. Each meaningful change is a new immutable candidate ID.

Progress from a provably-correct baseline before chasing bandwidth, since the roofline is generous and
correctness (LSE base-2, empty guard, tail mask) is the main risk.

---

## 7. Validation strategy
- **No local CUDA/torch harness** is permitted; correctness and timing come **only** from
  `./scripts/evaluate_candidate.sh feedback cNNN` over the five fixed workloads (one evaluation = all five).
- **Static invariants to self-audit before each evaluation:**
  - LSE computed as `m + log2(l)` in the base-2 domain (not divided by `ln2` twice).
  - Empty/`l==0` path returns `output=0`, `lse=-inf`, no `NaN`.
  - Tail mask sets logits to `-inf` and V-contribution to `0`; running max unaffected by masked lanes.
  - fp32 accumulation; bf16 store only for `output`; f32 for `lse`.
  - GQA mapping `kv = h//4`; K/V loaded once per `(b, kv)`.
  - Split-K combine rescales `acc_j` and `l_j` by `exp2(m_j − M)`.
  - int64 pointer arithmetic for gather offsets.
- **Coverage across workloads**: W3 exercises tiny/low-occupancy + potential empty edge; W1/W5 exercise
  long sequences and split-K; W2/W4 exercise short/medium sequences and tail masking. Passing all five
  gives confidence the single immutable kernel generalizes.
- **Decision recording**: after each evaluation, append one JSON object to `candidates.jsonl` with parent,
  source hash, hypothesis, validation outcome, per-workload result, geomean, decision, cumulative eval
  count, and skill usage. Never rewrite earlier records.
- **Budget discipline**: 100-evaluation cap; token soft/hard limits 1.0M/1.2M. Stop and write
  `SEARCH_COMPLETE` when geomean improvement converges. `final` only on explicit operator approval.

---

## 8. Open questions / risks
- **Baseline definition**: geomean speedup is measured against the reference (a Python double loop over
  `B×32` heads); a fused Triton kernel should yield a large speedup, so the real competition is between
  candidate variants (split-K, tile sizes). Confirm relative timings via the evaluator.
- **Tolerance**: `definition.json` does not print explicit atol/rtol; assume standard bf16-output
  tolerance and rely on the evaluator's correctness gate. Summation-order and bf16 rounding are the
  residual numeric risk — verify c001 passes before optimizing.
- **W3 launch-bound floor**: with only 65 tokens and 8→(split) CTAs, kernel-launch/overhead dominates;
  split-K may help occupancy but cannot beat launch latency. Keep the kernel single-launch (plus one
  combine launch) and avoid per-workload host-side branching that inflates launch cost.
- **`num_splits` selection** must be data-driven from `kv_indptr` (host-side max/avg token count) without
  changing the immutable kernel source between workloads — encode as a launch-time heuristic.

---

## 9. Skill usage note
`KernelWiki` and `ncu-report-skill` are scoped to **Blackwell (SM100/B200)** and **Hopper (SM90/H100)**.
This task targets **A800 (sm_80, Ampere)**, so neither skill applies (no tcgen05/TMEM/CLC/WGMMA, no
sm_100 profiling). No skills were invoked for this draft; if an Ampere-relevant question arises later it
will be reasoned about directly rather than via those Blackwell/Hopper-specific skills.
