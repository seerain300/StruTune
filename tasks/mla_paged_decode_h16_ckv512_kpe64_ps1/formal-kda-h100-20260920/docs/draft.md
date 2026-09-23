# Draft — `mla_paged_decode_h16_ckv512_kpe64_ps1`

Target: NVIDIA H100 (`sm_90`). Benchmark family: FlashInfer MLA paged decode, captured from
DeepSeek-V3 with tensor-parallel size 8 (`128/8 = 16` query heads). Deliverable: a Triton
kernel exposed through `solution/solution.py::run(...)` that matches the reference numerics
and beats it on geometric-mean speedup across the 47 feedback workloads.

> Scope note: this turn produces **only** this draft. No `plan.md`, no solution code.

---

## 1. What the operation computes

### 1.1 Signature (from `task/definition.json`)

Inputs:
- `q_nope`  : `[B, H=16, Dckv=512]` bf16 — query, "no positional encoding" part.
- `q_pe`    : `[B, H=16, Dkpe=64]`  bf16 — query RoPE part.
- `ckv_cache`: `[num_pages, page_size=1, 512]` bf16 — compressed latent KV cache.
- `kpe_cache`: `[num_pages, page_size=1, 64]`  bf16 — key RoPE cache.
- `kv_indptr`: `[B+1]` int32 — CSR row pointers per batch element into `kv_indices`.
- `kv_indices`: `[num_kv_indices]` int32 — page (== token, since `page_size=1`) indices.
- `sm_scale`: fp32 scalar — softmax scale (feedback value `0.1352337747812271`).

Outputs:
- `output`: `[B, H=16, 512]` bf16.
- `lse`   : `[B, H=16]` fp32 — **log2-base** log-sum-exp of the scaled logits.

### 1.2 Reference math (per batch element `b`, per head `h`)

Constants: `H=16`, `Dckv=512`, `Dkpe=64`, `page_size=1`. Constraints:
`len_indptr == B+1`, `num_kv_indices == kv_indptr[-1]`.

For each `b`:
```
beg  = kv_indptr[b]; end = kv_indptr[b+1]; L = end - beg   # tokens for this seq
if L <= 0:  output[b] = 0 ; lse[b] = -inf ; continue
tok = kv_indices[beg : end]                                # L token indices
Kc  = ckv_cache[tok].squeeze(1).float()                    # [L, 512]  (this is BOTH K and V)
Kp  = kpe_cache[tok].squeeze(1).float()                    # [L, 64]
logits        = q_nope[b].float() @ Kc.T + q_pe[b].float() @ Kp.T   # [16, L]
logits_scaled = logits * sm_scale
lse[b]        = logsumexp(logits_scaled, dim=-1) / ln(2)   # [16]  -> log2 base
attn          = softmax(logits_scaled, dim=-1)             # [16, L]
output[b]     = (attn @ Kc).to(bf16)                       # [16, 512]
```

Key structural facts that drive the whole design:

1. **The latent `ckv` tensor is used twice** — once transposed as the *key* in the score
   `q_nope @ Kc.T`, and once un-transposed as the *value* in `attn @ Kc`. There is no separate
   V tensor. This is the defining MLA-decode property (`kv_group_num = H = 16`, a single shared
   KV "head"). Loading `Kc` from HBM **once** and reusing it for both matmuls is the single most
   important traffic optimization (see §4.1).
2. **`page_size == 1`** ⇒ `kv_indices` are token indices directly; the gather is
   `row = kv_indices[beg + n]`, `addr = row * 512 (+ d)`. No intra-page offset arithmetic.
3. **Decode = single query token per sequence** (`qo_len = 1`), 16 heads. So the "M" dimension of
   the score matmul is only 16 — tiny. This is a GEMV-like / flat-MMA regime, memory-bound.
4. **LSE is base-2**: `lse = (e_max + ln(e_sum)) / ln2 = (e_max + ln(e_sum)) * log2(e)`.
   The reference divides `logsumexp` (natural) by `ln 2`. Must reproduce exactly, in fp32.
5. **Empty sequences** (`L == 0`) must yield `output = 0` and `lse = -inf`, not NaN.

### 1.3 `sm_scale` observation

The definition text says the "default" is `1/sqrt(192) ≈ 0.07217`, but every feedback workload
passes `sm_scale = 0.1352337747812271`. We treat `sm_scale` as an **opaque runtime fp32 scalar**
and apply it exactly where the reference does (multiply raw logits before max/exp/LSE). We do not
hard-code or re-derive it — that would be both wrong and fragile.

---

## 2. Workload characterization (the 47 feedback shapes)

`num_pages = 989669` (fixed; ~1.0 GB ckv cache + ~127 MB kpe cache — far larger than the working
set actually touched, so the gather is sparse/scattered across a huge buffer → the L2 hit rate on
`ckv`/`kpe` will be low and true HBM bandwidth dominates).

Three batch regimes (per `task/feedback_workloads.jsonl`):

| Regime | Lines | B  | `num_kv_indices` range | tokens / sequence (≈) |
|--------|-------|----|------------------------|-----------------------|
| single | 1–15  | 1  | 8 … 2708               | 8 … 2708 (one seq)    |
| medium | 16–31 | 16 | 457 … 17257            | ~28 … ~1079 avg       |
| large  | 32–47 | 64 | 9945 … 75145           | ~155 … ~1174 avg      |

Implications:
- **`B=1` low-token cases (8, 108, 208 …)** are pure *latency* bound: one sequence, a handful of
  tokens. Occupancy/tail dominate; there is almost no work. The win here is launch/parallelism
  overhead, not bandwidth. A naïve single-program kernel would leave 131 of 132 SMs idle ⇒
  split-K over the sequence is essential to spread even a small sequence across SMs (though for
  L=8 there is only ~1 token-tile, so splitting saturates at L/BLOCK_N).
- **`B=64` high-token cases** are *bandwidth* bound. Largest = 75145 tokens ×
  `(512+64)×2 = 1152 B/token ≈ 86.6 MB`. At H100 HBM3 ~3.35 TB/s the floor is ≈ **25.9 µs** if we
  hit the minimum-traffic ideal (each token's `ckv`+`kpe` read exactly once).
- **Parallelism budget**: H100 has 132 SMs. Stage-1 program count = `B × head_blocks ×
  num_kv_splits`. With all 16 heads in one block (`head_blocks = 1`, see §4.2), we need
  `num_kv_splits ≈ 132/B` to fill the machine: ~128 for B=1, ~8 for B=16, ~2 for B=64. So
  `num_kv_splits` must be chosen **adaptively per workload** (function of B and sequence length),
  not fixed. This is a primary tuning axis.

### 2.1 Roofline / arithmetic intensity

Per token, per sequence: score MACs `H·(Dckv+Dkpe) = 16·576 = 9216`; output MACs `H·Dckv =
16·512 = 8192`; total ≈ `17408` MACs → ≈ `34.8 kFLOP/token`. Bytes streamed = `1152 B/token`.
Arithmetic intensity ≈ **30 FLOP/B**. H100 bf16 tensor peak ≈ 990 TFLOP/s; HBM ≈ 3.35 TB/s ⇒
roofline knee ≈ 295 FLOP/B. Since `30 ≪ 295`, the kernel is **firmly memory-bandwidth bound** in
every non-trivial regime. Corollary (from KernelWiki `pattern-memory-bound`,
`technique-vectorized-loads`): optimize **bytes moved and effective bandwidth**, *not* compute;
compute-precision fidelity is cheap to keep high.

---

## 3. Reference algorithm mapping → flash-decoding (split-KV)

This operation is exactly the DeepSeek/MLA "grouped decode" that SGLang / lightllm / vLLM
implement as a two-stage split-KV flash-decoding kernel. KernelWiki evidence:

- `kernel-flashmla` (`wiki/kernels/flashmla.md`): MLA decode is memory-bound, paged KV,
  online-softmax accumulate; 656-B/token in the FP8 variant (here it is bf16, `ckv`+`kpe`
  separate). Confirms the "one warpgroup / head-block, stream KV pages, online softmax" shape.
- `lang-triton` (`wiki/languages/triton-blackwell.md`) ships the **verbatim upstream Triton MLA
  decode kernel** from vLLM PR-34597 (`triton_decode_attention.py`), adapted from SGLang /
  lightllm `gqa_flash_decoding_stage{1,2}`. It is the canonical reference for the exact structure
  I will adapt: `_fwd_grouped_kernel_stage1` (`Lk==576 ⇒ BLOCK_DMODEL=512, BLOCK_DPE=64`,
  `BLOCK_H=16`) + `_fwd_kernel_stage2` (cross-split LSE merge). It does `qk = tl.dot(q, k)`,
  `qk += tl.dot(qpe, kpe)`, online softmax, `acc += tl.dot(p, v)`, and stores `e_max + log(e_sum)`
  per split; stage-2 recombines. This matches our reference math one-to-one *except* for:
  (a) our indexing is **CSR ragged** (`kv_indptr` + `kv_indices`) not a rectangular block table;
  (b) our final `lse` must be **base-2** (upstream stores natural `e_max+log(e_sum)`);
  (c) here `v_buffer` and `k_buffer` are literally the *same* `ckv` data (reuse opportunity).

So the reference implementation is a well-trodden pattern; the optimization work is in the
H100-specific tuning (splits, block sizes, traffic reuse, warps/stages) and correctly porting the
CSR indexing + base-2 LSE + empty-sequence semantics.

---

## 4. Triton design space

### 4.1 Traffic minimization — reuse `ckv` for both K and V (highest-value lever)

Because it's memory bound, HBM bytes are the currency. The score needs `ckv` laid out as
`[Dckv, N]` (contract over `Dckv`); the output needs `ckv` as `[N, Dckv]` (contract over `N`). A
generic port (like upstream, which has distinct `K_Buffer`/`V_Buffer` pointers) issues **two**
loads of the same 512-wide `ckv` tile → ≈ `(512·2 + 64)·2 B = 2176 B/token`, ~1.9× the floor.

If instead I load each `ckv` token tile **once** into registers/SMEM and feed it to both
`tl.dot(q_nope, ckv.T)` (score) and `tl.dot(p, ckv)` (output), HBM traffic drops to the
`1152 B/token` floor. Options to realize the reuse:
- Load `ckv` tile as `[BLOCK_N, 512]`, use `tl.trans` for the score matmul, use it directly for
  the PV matmul. Trust the Triton scheduler to keep it resident (one global load).
- Or load once as `[512, BLOCK_N]` and `tl.trans` for PV.
This is candidate #1's core deviation from the stock kernel and should give a large bandwidth win
on the B=64 / long-sequence workloads. **Must verify the compiler doesn't re-load** (inspect via
`ncu` DRAM bytes if a candidate underperforms the byte floor).

`kpe` (64-wide) is only needed for the score, loaded once as `[Dkpe, N]`.

### 4.2 Head blocking

`H = 16`. All 16 heads share one KV, so put all 16 in one program (`BLOCK_H = 16`,
`head_blocks = 1`). Then the score matmul is `q[16, 512] · k[512, N] → [16, N]` and PV is
`p[16, N] · v[N, 512] → [16, 512]` — clean tensor-core tiles with M=16 (the minimum useful MMA M).
This maximizes KV reuse across the 16 heads (KV loaded once serves all heads) and matches
upstream's `BLOCK_H = 16`. Splitting heads across programs would re-read KV per head-block ⇒ more
traffic; avoid.

### 4.3 Split-KV (flash decoding) — two-stage

- **Stage 1** grid `(B, head_blocks=1, num_kv_splits)`. Each program handles one sequence's
  `[split_start, split_end)` token range, all 16 heads: online-softmax accumulate a partial
  `acc[16,512]` (fp32) + running `e_max[16]`, `e_sum[16]`; write partial `acc/e_sum` and the
  partial natural LSE `e_max + log(e_sum)` to a scratch `mid` buffer
  `[B, 16, num_kv_splits, 512+1]` fp32.
- **Stage 2** grid `(B, 16)`. Merge partials across splits with a second online softmax; write
  `output[B,16,512]` bf16 and `lse[B,16]` fp32 (**divide the merged natural LSE by `ln 2`**).

CSR adaptation: `seq_len[b] = kv_indptr[b+1] - kv_indptr[b]` (precompute in torch, plumbing-only);
`base[b] = kv_indptr[b]`; inner loop token index `= kv_indices[base + n]`, guarded by
`n < seq_len`.

### 4.4 `num_kv_splits` policy

Adaptive, driven by `B` and per-sequence length so that (a) `B·num_kv_splits` covers ~132 SMs and
(b) each split still has enough tokens to amortize its fixed cost. Sketch:
`num_kv_splits ≈ clamp(round(TARGET_PROGRAMS / B), 1, cdiv(max_seq_len, MIN_TOKENS_PER_SPLIT))`
with `TARGET_PROGRAMS ≈ 128–256`. Concretely ~64–128 for B=1, ~8–16 for B=16, ~2–4 for B=64. This
is a first-class tuning knob to sweep. (Upstream uses a fixed 4; we can do better per-regime.)
Note the scratch `mid` buffer scales with `num_kv_splits`; keep it bounded.

### 4.5 `BLOCK_N` (token tile)

Candidates: 16 / 32 / 64. Trade-offs: larger `BLOCK_N` ⇒ better load coalescing and fewer loop
iterations but more SMEM/registers and coarser split boundaries; the `acc[16,512]` fp32
accumulator already costs 32 KB of register state per program (8192 fp32), so register pressure is
high and occupancy is likely 1 block/SM regardless — meaning parallelism must come from split-K,
not from multiple resident blocks. Upstream uses `BLOCK_N=32`, `num_warps=4`, `num_stages=2`; a
sound starting point to sweep around.

### 4.6 Matmul vs elementwise for the score

With M=16 (a full MMA tile) `tl.dot` uses the tensor cores and is the right choice for both QK and
PV. (The non-grouped upstream path uses `tl.sum(q*k)` because there M=1; not our case.) Contract
dims: QK over 512 (+64), PV over `BLOCK_N`.

### 4.7 Single-stage fallback consideration

For the very small `B=1` low-token shapes, a two-stage kernel with a tiny stage-2 is fine, but the
stage-1/stage-2 launch pair has fixed overhead that can dominate when there are only 8 tokens. An
alternative is a fused single-pass kernel for tiny sequences. Keep two-stage as the baseline;
revisit a small-seq specialization only if profiling shows launch overhead dominating the
`num_kv_indices ∈ {8,108,208}` cases.

### 4.8 Layout / plumbing (PyTorch allowed, no compute)

- Pass `q_nope` and `q_pe` as **separate** pointers (avoid a `cat` copy). Two loads: `[16,512]` and
  `[16,64]`.
- `ckv_cache`/`kpe_cache` are `[num_pages, 1, D]`; stride to token `t` is `t*D` (page_size=1).
- Precompute `seq_lens = kv_indptr[1:] - kv_indptr[:-1]` and pass `kv_indptr` (for `base`) +
  `kv_indices`. These are trivial int32 ops (metadata plumbing, permitted).
- Allocate `output` (zeros, so empty seqs are 0), `lse`, and the fp32 `mid` scratch.
- Coalesce KV loads: 512-wide bf16 rows are contiguous ⇒ vectorized 128-bit loads naturally; keep
  `offs_d` contiguous in the fastest-varying dim.

---

## 5. Numerical risks & correctness requirements

Tolerances are not stated explicitly in `definition.json`; assume the standard KDA/FlashInfer
bf16-attention tolerance (output is bf16, so ~1e-2 relative is typical) but **do not gamble** —
keep fp32 accumulation everywhere the reference uses fp32.

Risks and mitigations:

1. **Accumulation precision.** Reference casts everything to fp32 and accumulates in fp32.
   → All online-softmax state (`e_max`, `e_sum`, `acc`) in fp32; `tl.dot` with fp32 accumulate.
2. **bf16 tensor-core inputs vs fp32 matmul.** `tl.dot(q_bf16, k_bf16)` accumulates in fp32 but
   rounds inputs to bf16 (reference multiplies fp32×fp32). Small error; output is bf16 so likely
   within tolerance. Risk area = the QK dot over 512 dims (accumulated error). Mitigation ladder if
   it fails: (a) keep as-is (fast, tensor-core); (b) upcast to fp32 elementwise multiply-reduce for
   QK only; (c) fp32 `tl.dot` (`input_precision="ieee"`/tf32-off) — slower. Start with (a), verify.
3. **PV probability precision.** `p` is fp32; casting `p→bf16` for `tl.dot(p, v_bf16)` loses
   mantissa. Reference does fp32 `attn @ Kc`. Given normalized probs and bf16 output, bf16 PV is
   usually fine; fallback = fp32 PV. Test both if needed.
4. **Base-2 LSE.** Final `lse = (e_max + log(e_sum)) * log2(e)`. A very common bug is emitting the
   natural-base LSE (upstream does). Explicitly multiply by `1/ln2 = 1.4426950408889634` in fp32 at
   stage-2 write. The intra-kernel exp can be natural (`tl.exp`) or base-2 (`tl.exp2` with logits
   pre-scaled by `log2(e)`) — either is fine for softmax as long as the final LSE conversion is
   correct; keep it simple with natural `exp` first.
5. **Empty sequence (`L==0`).** Reference ⇒ `output=0`, `lse=-inf`. In split-K, a program whose
   split is empty writes nothing; stage-2 that finds `e_sum==0` must emit `output=0` and
   `lse=-inf` (guard the `acc/e_sum` divide to avoid `0/0 = NaN`, and `-inf + log(0)`). Because
   `output` is zero-initialized, the main need is to *not* write NaN and to set `lse=-inf`. Must
   check whether any B=16/B=64 feedback shape actually contains a zero-length row (the ragged
   `kv_indptr` could); implement the guard unconditionally to be safe.
6. **Masking within a tile.** Tokens with `n >= split_end` (or `>= seq_len`) must contribute
   `-inf` to logits so `exp(-inf)=0`; guard the final tile of each split. Ensure an all-masked
   tile does not corrupt `e_max` (`max(qk, -inf)` stays finite once any real token seen; before
   any real token `e_max=-inf`, `re_scale=exp(-inf-(-inf))` → must be handled so `acc` stays 0).
   Upstream's ordering (`n_e_max = max(max(qk), e_max)`, `re_scale = exp(e_max - n_e_max)`) already
   handles the `e_max=-inf` start; reproduce it faithfully.
7. **Index dtype / range.** `kv_indices` up to `num_pages≈9.9e5`, fits int32; `base + n` also fits.
   Compute addresses in int32/int64 consistently to avoid overflow (`row*512` ≈ 5.1e8 < 2^31, safe
   in int32 but use int64 offsets in Triton to be safe).
8. **Determinism across splits.** Cross-split reduction order can change results slightly vs the
   single-pass reference; online softmax merge is mathematically invariant to split count, so
   within bf16 tolerance this is fine. (Upstream even exposes a batch-invariant `num_kv_splits=1`
   mode; not required here.)
9. **`page_size=1` assumption baked in.** The reference asserts `page_size==1`; hard-coding the
   token-index gather (`row = kv_indices[...]`) is valid for this task and simpler/faster than a
   general paged path.

---

## 6. Validation strategy

Constraints from `CLAUDE.md` / `TASK.md`: correctness is judged **only** by the trusted evaluator
(`./scripts/evaluate_candidate.sh feedback <cid>`), which runs the full 47-workload set (warmup 2 /
10 iters) and checks every shape including boundaries. I **cannot** run CUDA, torch, or a private
harness directly (Bash is locked down; only the two provided launchers are permitted). Therefore:

1. **First candidate must be conservative and mathematically faithful.** Port the known-good
   SGLang/vLLM two-stage grouped MLA-decode structure, adapted to (a) CSR ragged indexing,
   (b) base-2 LSE conversion, (c) empty-sequence guards, (d) `page_size=1` direct gather. Keep
   fp32 accumulation. Do *not* fold in the aggressive `ckv`-reuse traffic optimization on the very
   first candidate if it risks a double-load-vs-single-load correctness/scheduling surprise —
   establish a correct, passing baseline first, then optimize.
2. **One evaluation = one full feedback run = one immutable candidate.** Budget is 100 evals /
   ~9–11M tokens. Spend deliberately: change one dominant variable per candidate
   (e.g., `num_kv_splits` policy, then `BLOCK_N`, then `ckv` single-load reuse, then num_warps /
   num_stages) so each eval yields a clean signal. Record parent, source hash, hypothesis,
   per-workload result, geomean, decision, cumulative eval count, skill usage in `candidates.jsonl`
   (append-only, never rewrite).
3. **Profile only between evaluations, never concurrently.** Use `./scripts/ncu_profile.sh` with
   the `ncu-report-skill` workflow to confirm the hypothesis that a candidate is HBM-bound and to
   check achieved DRAM bytes vs the `1152 B/token` floor (validates the `ckv`-reuse optimization)
   and achieved bandwidth vs ~3.35 TB/s. Profiling and evaluation must be strictly serialized
   (concurrent GPU use ⇒ return code 3, wasted eval).
4. **Correctness gate before performance.** A candidate that fails any workload's correctness is
   invalid regardless of speed; the geomean only counts if all selected workloads pass. So the
   iteration loop is: (i) confirm pass on all 47, (ii) then push geomean.
5. **Convergence / stop.** Stop at budget or when geomean improvement genuinely plateaus across
   consecutive candidates; then write `SEARCH_COMPLETE` with the reason. `final` only on explicit
   operator approval.

### 6.1 Candidate roadmap sketch (to be formalized in `plan.md`, not now)

- **c001** — Correct two-stage baseline: grouped stage-1 (`BLOCK_H=16`, `BLOCK_DMODEL=512`,
  `BLOCK_DPE=64`, `BLOCK_N=32`, `num_warps=4`, `num_stages=2`), CSR indexing, base-2 LSE,
  empty guards, fixed modest `num_kv_splits` (e.g. 4 or an initial adaptive rule). Goal: pass 47/47
  and get a first geomean.
- **c002+** — Adaptive `num_kv_splits` policy tuned to the B∈{1,16,64} regimes (fill 132 SMs).
- **later** — single-load `ckv` reuse (K==V) to hit the byte floor; `BLOCK_N` sweep;
  num_warps/num_stages; possible tiny-sequence specialization for B=1 low-token shapes.

---

## 7. Key references used (KernelWiki skill)

- `wiki/kernels/flashmla.md` (`kernel-flashmla`): MLA decode memory-bound paged-KV online-softmax
  structure; H800 dense decode ~3000 GB/s / 660 TFLOPS anchor (bandwidth-bound confirmation).
- `wiki/languages/triton-blackwell.md` (`lang-triton`) + verbatim
  `artifacts/prs/vllm/PR-34597/.../triton_decode_attention.py` and `.../mla/triton_mla.py`:
  canonical two-stage grouped MLA-decode Triton kernel (`Lk==576 → 512+64` split, `BLOCK_H=16`,
  stage-2 LSE merge) — the structural template I adapt.
- `wiki/patterns/memory-bound.md`, `wiki/techniques/vectorized-loads.md`,
  `wiki/techniques/register-budgeting.md`: memory-bound optimization priorities (bytes, coalesced
  wide loads, occupancy) and the "profile first" discipline that governs this kernel.

All reflect KernelWiki knowledge cutoff 2026-04-27; the vLLM PR-34597 kernel is `source-reported`
upstream code, adapted (not copied) to this task's CSR indexing and base-2 LSE.
