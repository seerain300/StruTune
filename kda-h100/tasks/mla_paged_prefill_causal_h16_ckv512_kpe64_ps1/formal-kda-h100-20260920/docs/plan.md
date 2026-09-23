# Plan — `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`. Target: H100
(`sm_90`), Triton-primary. Ranking metric: geomean speedup over the reference across the
full 38-workload feedback set; **every** workload must pass correctness or the candidate is
invalid. Budget: 100 evaluations; token soft/normal/hard limits 9M / 10M / 11M.

## 0. Operating rules (do every candidate)

- One immutable `solution/solution.py` = one candidate id = one full 38-workload eval via
  `./scripts/evaluate_candidate.sh feedback cNNN`. Any meaningful source/config/launch
  change ⇒ **new** id. Never reuse an id for changed source; never rewrite earlier
  `candidates.jsonl` records.
- Triton does all attention math; torch only for metadata/launch/output-alloc. **No** Torch/
  CPU/NumPy/CUDA-ext fallback. A failing Triton kernel is invalid — fix or abandon, never
  substitute a fallback.
- Profiling only via `./scripts/ncu_profile.sh …`, **never** overlapping an evaluation
  (concurrent GPU use → controller return code 3 = wasted budget slot). Finish one before
  starting the other.
- `final` only after explicit operator approval.
- After each eval, append exactly one JSON record to `candidates.jsonl` (schema in §7).

## 1. Solution contract (fixed for all candidates)

`solution/solution.py` exposes:
```python
def run(q_nope, q_pe, ckv_cache, kpe_cache,
        qo_indptr, kv_indptr, kv_indices, sm_scale):
    # returns (output, lse)
    #   output: [total_q, 16, 512] bf16
    #   lse:    [total_q, 16]      fp32, base-2 logsumexp of scaled logits
```
Host side (allowed torch): read shapes; allocate `output` (zeros, bf16) and `lse`
(fill `-inf`, fp32); build any launch metadata; launch Triton grid; return. No python loop
over sequences/tokens; no `.item()` syncs on data-dependent values in the hot path. `batch =
len_indptr-1` and `total_q` come from tensor shapes (`.item()` on the small
scalars/indptr lengths is fine at launch time but prefer passing indptr tensors into the
kernel and looping inside).

### Canonical math the kernel must implement (re-derived from reference)
Per sequence `b`: `q_len = qo_indptr[b+1]-qo_indptr[b]`,
`kv_len = kv_indptr[b+1]-kv_indptr[b]`, `prefix_len = kv_len - q_len`. For query row `i`
(0-based within seq), head `h`, KV token `j` (0-based within seq, cache index
`t = kv_indices[kv_indptr[b] + j]`):
- key = `[ckv_cache[t,0,:512] | kpe_cache[t,0,:64]]`, value = `ckv_cache[t,0,:512]`
- `logit = (q_nope[row,h] · Kc_t) + (q_pe[row,h] · Kp_t)` where `row = qo_indptr[b]+i`
- **mask**: keep iff `j <= prefix_len + i`
- base-2 softmax: `s2 = logit * (sm_scale * LOG2E)`; `m2 = max_j s2`;
  `p = exp2(s2 - m2)`; `denom = Σ p`; `lse[row,h] = m2 + log2(denom)`;
  `output[row,h] = (Σ_j p_j Kc_t) / denom`
- empty seq (`q_len==0` or `kv_len==0`): leave `output=0`, `lse=-inf`.

`LOG2E = 1.4426950408889634`, applied exactly once.

## 2. Correctness checks (apply before every eval; the evaluator is the only gate)

We cannot run the evaluator's dataset ourselves, so gate each candidate on a **static
review checklist** before spending an evaluation:

- **C1 causal boundary**: mask is `j <= prefix_len + i` (equivalently `j > prefix_len+i →
  -inf`), matching reference `arange(kv_len) > (prefix_len+i)`. Off-by-one is the highest-
  risk bug; verify with `prefix_len = kv_len - q_len` and the extreme rows (`i=0`,
  `i=q_len-1`).
- **C2 base-2 LSE**: `lse = m2 + log2(denom)` with `s2 = logit*sm_scale*LOG2E`. Confirm it
  equals `logsumexp(logit*sm_scale)/ln2` algebraically. LOG2E applied once.
- **C3 V≡Kc**: PV uses `ckv_cache` (the 512 latent), not `kpe`. `kpe` enters logits only.
- **C4 dtypes/shapes**: output bf16 `[total_q,16,512]`, lse fp32 `[total_q,16]`. Accumulators
  fp32; only `tl.dot` operands bf16.
- **C5 empty/padding**: `output` pre-zeroed, `lse` pre-filled `-inf`; store masks prevent
  writing padding rows (`i>=q_len`) and padding KV cols; use `row_max_fixed =
  where(row_max==-inf, -1e20, row_max)` so an all-masked tile can't poison the running max
  / produce NaN; guard `denom==0`.
- **C6 index range**: `t = kv_indices[...]` indexes `ckv_cache`/`kpe_cache` rows (page_size=1
  ⇒ page index == token index); masked-out KV lanes load with `other=0.0` and a safe index.
- **C7 scale placement**: `sm_scale` is a runtime kernel arg (never baked as a literal).

If C1–C7 all pass on review → spend the evaluation. If the evaluator fails any workload,
diagnose against C1–C7 (empty-seq, tiny, and the 16384 single-seq case are the likely edge
failures) and open a **new** candidate id with the fix.

## 3. Candidate ladder (sequential, adaptive)

Each rung is a distinct immutable candidate. Rungs after `c001` are conditional on the prior
result and on profiling; the exact next rung is chosen from the decision rules in §4. Only
change **one** major axis per rung so the geomean delta is attributable.

> **Decision log — c001 (evaluated, REJECT, 1 eval spent).** The kernel failed to compile
> on all 38 workloads: `NameError` — the module-level `LOG2E` global was referenced inside
> the `@triton.jit` kernel, which Triton forbids (globals must be `tl.constexpr`). This is a
> pure compile bug, not a math/algorithm error; nothing reached correctness or timing. The
> baseline design (per-(seq,head,q-block), unified causal loop, exp2 base-2 softmax, causal
> early-exit, fp32 accumulators, BLOCK_M=32/BLOCK_N=64) is unchanged and still the intended
> first valid parent. **Next rung `c002`**: identical kernel with the base-2 constant inlined
> as a trace-time Python literal (fold `sm_scale * LOG2E` on the host, or use a `tl.constexpr`
> for the constant). No other axis changes until a fully-correct baseline exists.
>
> **Decision log — c002 (evaluated, ADOPT, first VALID parent, 2 evals spent).** The
> compile fix worked: **38/38 pass, valid_run=True, geomean = 107.67x** (arith mean 201.85x,
> min 9.88x, max 915.87x). All edge/prefix cases correct → causal boundary and NaN guards
> confirmed. c002 is now the parent. Timing bottlenecks (from the eval, to be confirmed by
> profiling before committing a rung): the big single-seq compute case #26 (16384/16387) is
> the outlier at **43.27x / sol=56.7ms** (>4x slower than any other large case); short-Q
> long-KV #17 (22/17759) is the min at **9.88x** (low occupancy). Both point at plan
> hypotheses 2 (16x latent-KV reload across heads) and 6 (occupancy). **Next rung `c003`:**
> head folding — fold the 16 heads (all sharing one KV) into the matmul M dimension so the
> latent KV is read once per query-tile instead of 16x. Profile #26 and a medium case first
> to confirm memory-bound, per §4.
>
> **Decision log — c003 (evaluated, ADOPT, 3 evals spent).** Head folding for the
> short-query regime (dispatch when max per-seq q_len <= 16 to a single-token, all-16-heads
> kernel; large-q keeps the byte-identical c002 kernel). **38/38 pass, geomean = 112.57x >
> c002 107.67x (+4.55%)** → new parent. Tiny cases improved (10/12 29->38, 13/14 35->41,
> 1/34 18.5->19.0); large regime within noise (guaranteed — same kernel). Two soft spots
> remain: (a) **#17 22/17759** is still the MIN (9.39x): 22 seqs x 1 query x ~800 KV/seq =
> grid of only 22 programs = under-occupied; single-token head folding does NOT parallelize
> the long KV loop → needs a **KV-split / flash-decoding** (plan Axis B / c005). (b) **#26
> 16384/16387** stays the large-regime outlier (~43x, 55.8ms): causal-triangle compute-bound,
> latent KV still read 16x across heads for large-q → needs **combined head x q-tile M**
> folding in the prefill kernel (plan Axis A / c003'). **Next rung `c004`:** decide via
> profiling (ncu_profile.sh on #26 and #17, serialized vs eval) between the combined-tile
> prefill kernel (#26; largest absolute time, but one workload) and KV-splits (#17; the min).
> Lean toward the combined head x q-tile prefill kernel since it also removes the residual
> 16x KV reload for ALL medium/large-q workloads, not just #26.

### c001 — correct fused baseline (must pass all 38 before any optimization)
- Single `@triton.jit` kernel, grid `(batch, num_heads=16, cdiv(max_q_len, BLOCK_M))`.
  Program handles `BLOCK_M` query rows of one (seq, head). Loop KV in `BLOCK_N` chunks.
- Splits: `BLOCK_DMODEL=512` (ckv), `BLOCK_DPE=64` (kpe), `BLOCK_DV=512` (value).
- Load Q `[BLOCK_M,512]` and Qpe `[BLOCK_M,64]` once. KV loop: gather `t` from
  `kv_indices`; load `Kc` transposed `[512,BLOCK_N]` and `Kp` transposed `[64,BLOCK_N]` for
  QK; load `V=Kc` normal `[BLOCK_N,512]` for PV (SGLang extend pattern — reads latent KV
  twice, accept for baseline).
- **Causal early-exit**: iterate `start_n` only up to `prefix_len + (cur_block_m*BLOCK_M +
  BLOCK_M)` (unified single loop with `j <= prefix_len+i` mask), not the whole `kv_len`.
- Online softmax in base-2 (`exp2`), fp32 accumulators, `p→bf16` for `tl.dot(p, V)`.
- Empty-seq / padding guards per C5. Store output + base-2 lse with masks.
- Launch heuristic (Hopper, `Lq=576`): start from SGLang values `BLOCK_M=32, BLOCK_N=64,
  num_warps=8, num_stages=1` (SMEM ~ fits with single buffer). If it fails to compile / SMEM
  OOM, reduce `BLOCK_M=16` — but that is a config change ⇒ new id (`c001b`/`c002`).
- **Hypothesis**: a single fused Triton kernel replacing the python double-loop reference is
  a large speedup on every workload; geomean ≫ 1. Establishes the valid parent.
- **Decision**: if all 38 pass → adopt as parent, record geomean, proceed. If any fail →
  new id fixing the specific check; do not optimize until a fully-correct parent exists.

### c002 — de-duplicate the latent-KV load (Axis C)
- Load `Kc` **once** per KV chunk as `[BLOCK_N,512]` and use `tl.trans` for the QK matmul,
  instead of loading Kc twice (transposed for QK + normal for PV). Halves latent-KV global
  traffic and SMEM footprint.
- **Hypothesis**: memory-bound workloads (tiny + short-Q + the many-seq prefills) speed up;
  frees SMEM to later grow blocks / add a pipeline stage. Compute-bound 16384 case roughly
  neutral.
- **Decision**: keep if geomean ↑ and no regression on the big compute cases; else revert to
  c001 lineage. (Only pursue if profiling of c001 shows memory-bound behavior on small/medium
  workloads — see §4.)

### c003 — head folding / KV reuse across the 16 heads (Axis A)
- All 16 heads share the same KV ⇒ read latent KV once for all heads. Fold heads into the
  matmul M dimension. Two sub-forms; pick per profiling:
  - **c003 (decode-style, short-Q)**: grid `(batch, cdiv(q_len,·))`; put the 16 heads of a
    single query token into M (rows = 16), KV read once. Best for #2/4/6/15/17/28/32.
  - **c003' (combined head×q-tile)**: M rows = `(q_tile × 16)` for a shared KV block
    (e.g. q_tile=4 → M=64, full wgmma tile). Output row `(t,h) → output[q_base+t,h,:]`;
    causal mask depends on `t` only (identical across the 16 heads). Most general.
- Start with whichever the c001/c002 profile says matters most; the other becomes a later
  rung if warranted.
- **Hypothesis**: eliminates the 16× latent-KV reload → large win on all memory-bound
  regimes, especially #17 (22 seqs × ~807 kv) and the tiny cases.
- **Decision**: keep if geomean ↑ with no big-case regression. If it helps small but hurts
  large (or vice versa), that motivates c00x regime dispatch (§ c004).

### c004 — regime dispatch / autotune block sizes (Axis D)
- Host-side heuristic selects kernel variant + `(BLOCK_M,BLOCK_N,num_warps,num_stages)` by
  regime keyed on `total_q`, per-seq `q_len` bucket, and `batch` (e.g. tiny/decode →
  head-folded small blocks; large prefill → bigger `BLOCK_M/BLOCK_N`, `num_stages=2` if SMEM
  allows after c002). Optionally `@triton.autotune` over a small config set keyed on a
  `q_len` bucket constexpr.
- **Hypothesis**: no single config is optimal across the 3 regimes (draft §2); per-regime
  dispatch lifts geomean beyond any single-config candidate.
- **Decision**: keep the dispatch table that maximizes geomean; freeze configs (any change =
  new id). Guard against autotune cache nondeterminism affecting the timed eval.

### c005 — flash-decoding KV split for short-Q-long-KV (Axis B, conditional)
- For low-occupancy short-Q/long-KV workloads (esp. #17), split KV across
  `NUM_KV_SPLITS` programs + a small stage-2 combine (SGLang `_fwd_grouped_kernel_stage1`
  pattern), to raise SM occupancy.
- **Hypothesis**: raises utilization on the few workloads where a single program per (seq)
  underfills the GPU; neutral elsewhere.
- **Decision**: only build if profiling shows those specific workloads occupancy-bound; keep
  only if geomean ↑ without regressions and correctness (multi-split LSE combine) holds.

### Reserve rungs (only if budget + evidence justify)
- Kc-dedup + num_stages=2 pipeline (needs SMEM headroom from c002).
- Separate prefix/triangle two-loop vs unified single loop A/B.
- `tl.trans` vs separate-load micro-tuning; `num_warps` sweep on the dominant big cases.

## 4. Profiling & decision workflow (between candidates)

Use `./scripts/ncu_profile.sh --set basic -o profile/<tag> python <harness>` **only when no
evaluation is running**, to answer the branch questions cheaply (profiling does not consume
eval budget, but eval budget is the scarce resource — profile to avoid blind eval spend):
1. After c001 passes: profile a **small/medium memory-ish** workload and a **large
   compute** workload (e.g. the 96/98 or 199/203 case and the 16384/16387 case). Determine
   memory-bound vs compute-bound and whether the 16× KV reload dominates the small cases.
   → chooses c002 (Kc dedup) vs c003 (head folding) as the next rung.
2. After each perf rung: confirm the intended bottleneck moved (bandwidth ↓, or tensor-core
   utilization ↑) and that no regime regressed, before committing the next rung.
3. Never launch profiling while `evaluate_candidate.sh` is running, and never launch an eval
   while profiling; serialize strictly.

Interpretation cues (via `ncu-report-skill`): low DRAM throughput + low SM busy on tiny
cases → launch/occupancy bound (favor head folding, smaller grid, fewer heads×reload). High
DRAM throughput on medium → memory bound (favor Kc dedup + head folding). High tensor-core
pipe utilization on 16384 → compute bound (favor block-size/pipeline tuning, not memory
changes).

## 5. Candidate lineage strategy

- **Parent = current best valid candidate** (highest geomean with all-38 passing). Each new
  rung's `parent` in `candidates.jsonl` is that best-so-far id, unless deliberately branching
  from an older parent (record which and why).
- Advance the parent only when a rung is **valid (all 38 pass) and geomean ≥ parent** (treat
  <~1% as noise → keep simpler parent). A rung that regresses or fails is recorded as
  `decision:"reject"`, and the next rung branches from the unchanged parent.
- Change **one** major axis per rung; keep the diff minimal and attributable. Config-only
  tweaks (block sizes, warps, stages) still get new ids since they change launch behavior.
- Keep a short running "lineage log" at the top of each `candidates.jsonl` reasoning field
  (parent → child, what changed, result) so the search tree is reconstructable.

## 6. Performance hypotheses (ranked, to confirm/refute via evals + profiling)

1. **Fusion** (c001): the python double-loop reference is the baseline; one fused kernel is a
   very large speedup on all 38 → geomean ≫ 1. (Highest confidence.)
2. **16× KV reload is the dominant avoidable cost on memory-bound regimes** (tiny + short-Q +
   many-seq prefill). Head folding (c003) and Kc dedup (c002) attack it; expect the largest
   geomean gains here because most workloads are small.
3. **Causal early-exit** materially cuts work on the large single-seq cases (16384: only the
   lower triangle) — already in c001; verify via the 16384 timing.
4. **No single block config is geomean-optimal** across regimes → dispatch/autotune (c004)
   beats any fixed config.
5. **exp2 base-2 softmax** is both correctness-exact for the LSE and slightly faster than
   natural `exp` (secondary effect).
6. Flash-decoding KV split (c005) helps only the handful of occupancy-starved short-Q/long-KV
   workloads; low aggregate geomean weight.

## 7. Evidence format (one JSON object appended to `candidates.jsonl` per eval)

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<sha256 of solution/solution.py at eval time>",
  "hypothesis": "fused Triton kernel replacing python double-loop; establish valid baseline",
  "change_from_parent": "initial implementation",
  "validation": {
    "all_pass": true,
    "num_workloads": 38,
    "num_pass": 38,
    "failed_workloads": []
  },
  "per_workload": [
    {"uuid": "...", "total_q": 33, "kv": 34, "speedup": 0.0, "correct": true}
    // one entry per workload: speedup vs reference + correctness
  ],
  "geomean_speedup": 0.0,
  "decision": "adopt|reject|parent-unchanged",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "notes": "profiling observations, next-rung choice, edge-case findings"
}
```
Rules: append-only; never edit prior records; `source_sha256` computed from the exact file
that was evaluated; `cumulative_evaluations` is a monotonic counter across all evals
(feedback set = 1 each). Record the concrete geomean and the per-workload table from the
evaluator output verbatim; if an eval is discarded (return code 3 from interference) note it
and that it still consumed a budget slot.

## 8. Stopping criteria & completion

- **Stop** when any of: (a) evaluation budget approached (leave margin, e.g. stop new rungs
  by ~90/100 unless a rung is mid-flight); (b) token soft limit (9M) reached — begin winding
  down, hard stop before 11M; (c) geomean improvement has genuinely converged (≥2
  consecutive rungs yield <~1–2% geomean gain and profiling shows no clear remaining
  bottleneck).
- On genuine convergence, write `SEARCH_COMPLETE` stating the best candidate id, its geomean,
  the reason (which limit / plateau), and the remaining unexplored axes (for the record).
- Do **not** run `final` unless the operator explicitly approves; `final` targets the single
  best valid candidate.

## 9. Skill usage log (ongoing)
- **KernelWiki** (used in draft): `kernel-flashmla` (MLA KV layout, V≡latent-KV, SM90 perf
  context); `lang-triton` → verbatim `pr-vllm-34597` grouped MLA decode (head folding +
  KV splits) and `pr-sglang-22079` extend/prefill (the `Lq==576` 512+64 split, Hopper block
  heuristics, transposed-K load, causal early-exit, unified mask). These templates directly
  seed c001–c005.
- **ncu-report-skill**: to be invoked per §4 between candidates to choose/confirm rungs;
  always serialized against evaluations.
