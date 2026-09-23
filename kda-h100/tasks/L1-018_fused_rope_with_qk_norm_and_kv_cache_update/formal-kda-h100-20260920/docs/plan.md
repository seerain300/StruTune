# Plan — L1/018 Fused RoPE + QK-Norm + KV-Cache Update (H100 / sm_90)

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`. No candidate is
implemented or evaluated in this turn — this document defines *how* candidates will be built,
checked, evaluated, recorded, and stopped.

---

## 1. Objective & success metric
- **Primary metric**: geometric-mean speedup over the reference across all 13 feedback workloads,
  subject to **every** workload passing correctness. A single correctness failure invalidates the
  candidate regardless of speed.
- **Goal**: maximize geomean while staying inside the eval budget (100) and token budget
  (soft 9M / normal 10M / hard 11M). Because the op is memory-bandwidth-bound (draft §5), the
  practical ceiling is "eliminate redundant HBM traffic + launch overhead and saturate HBM";
  we target the achieved-BW SOL, not compute.

## 2. Guardrails (must hold for every candidate)
- Triton is the compute path. PyTorch only for metadata/allocation/launch. **No** Torch/CPU/NumPy/
  CUDA-extension computational fallback; a failed Triton kernel is an invalid candidate, not a
  reason to fall back.
- `solution/solution.py` exposes `run(...)` with the exact reference signature; returns
  `(query_rotated, key_rotated, key_cache, value_cache)` where the two caches are the *same*
  tensors updated in place.
- Candidates are immutable and sequential (`c001`, `c002`, …). Any meaningful source / config /
  launch change ⇒ a new ID. Never reuse an ID for changed source. Never rewrite earlier records.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback <cid>` (full 13-workload set = 1 eval).
- Profiling **only** via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow). **Never** profile
  while an eval is running, and never launch an eval while profiling — a foreign process on the
  locked GPU ⇒ rc=3, discarded measurement, wasted budget slot.
- No direct CUDA / `nvidia-smi` / evaluator / alternate harness. No web/subagents/MCP. External
  knowledge only from `KernelWiki` and `ncu-report-skill`.
- `final` only after explicit operator approval.

## 3. Correctness contract (re-derived from draft §1.2, checked before each eval)
Per length-128 row `x` of `(b, head, s)`:
1. fp32 RMS: `xf=x.float(); var=mean(xf^2) over 128; xn = xf*rsqrt(var+eps); xn *= weight.float()`.
2. RoPE with `freqs = position * inv_freq[0:64]`, `c=cos(freqs)`, `s=sin(freqs)` (length 64):
   - `o[0:64]   = xn[0:64]*c   - xn[64:128]*s`
   - `o[64:128] = xn[64:128]*c + xn[0:64]*s`
   - cast to bf16 at store.
3. Cache: `key_cache[b,kv,cache_position[s],:] = key_rotated`; `value_cache[b,kv,cache_position[s],:] = value` (**raw** value).
Non-negotiable invariants (draft §7): V written unmodified; caches updated in place & returned;
S-dimension masking for non-multiples (293, 541); read `cache_position` for the destination row
(no `cache_len` alignment assumption); **int64** cache offset arithmetic (2^31 margin, draft §2);
fp32 internal math (draft §4.2); general (non-unit) norm weights.

### 3.1 Pre-eval checklist (manual, no local CUDA — forbidden)
Before spending an eval on any candidate, verify by inspection:
- [ ] Output dtypes bf16; shapes match definition; caches returned as the passed-in tensors.
- [ ] `value` path has no norm/rope.
- [ ] cos/sin recomputed in fp32 from `inv_freq`+`position`; both halves use the same `c,s`.
- [ ] Load/store masks cover S tail and (if flattened grid) row tail.
- [ ] Cache offset uses tensor strides and int64; index = `cache_position[s]`.
- [ ] Norm reduction over full D=128 in one tile; `rsqrt(var+eps)`.
- [ ] Grid covers all rows for every shape incl. `S=1`.

## 4. Candidate lineage strategy
Sequential, one meaningful change per ID. Each candidate has a parent (the current best valid
candidate) unless it is a fresh baseline. Lineage is greedy-with-record: keep the best valid
geomean as "champion"; branch new candidates from the champion; abandon a direction after ≤2
non-improving candidates.

### Phase 0 — Correct baseline
- **c001**: Simplest fully-correct fused implementation.
  - Two kernels: (a) **Q kernel** RMS+RoPE→`query_rotated`; (b) **KV kernel** K RMS+RoPE→
    `key_rotated`+`key_cache`, V copy→`value_cache`.
  - Grid strategy **A** (structured `(B,H,ceil(S/BLOCK_S))`), fixed `BLOCK_S=32`, `num_warps=4`,
    `num_stages=2`, `BLOCK_D=128`. In-kernel cos/sin recompute. No autotune yet.
  - Purpose: prove correctness on all 13 and establish baseline geomean. **Do not tune before c001 passes.**

### Phase 1 — Structural / traffic
- **c00x**: fuse vs split V-copy (test whether K+V in one kernel beats K kernel + dedicated
  vectorized bf16 copy kernel for `value_cache`).
- **c00x**: vectorized bf16 loads/stores (e.g. treat 128-dim row for coalesced 16-byte access;
  ensure contiguous store to `query_rotated`/cache slice).
- **c00x**: minimize redundant work — precompute per-`(b,s)` cos/sin once per S-block and reuse
  across the D halves; avoid recomputing sin/cos per head (Q kernel handles 96 heads sharing the
  same `(b,s)` cos/sin — consider grid ordering so a block reuses cos/sin across heads).

### Phase 2 — Occupancy / block tuning (guided by ncu)
- **c00x**: autotune `BLOCK_S∈{16,32,64,128}`, `num_warps∈{2,4,8}`, `num_stages∈{1,2,3}`,
  keyed on shape buckets (decode / small-prefill / large-prefill). Keep config list small to
  bound compile time.
- **c00x**: decode-shape grid (strategy **B**, flattened/head-packed rows) for `S=1` (#1,#3) if
  ncu shows GPU under-filled or launch-bound. Applied only where it helps; keep structured grid
  elsewhere.

### Phase 3 — Micro / polish
- **c00x**: tighten register/precision (e.g. keep weights in registers, avoid redundant casts),
  `tl.multiple_of`/`max_contiguous` hints, eviction policy on cache stores (streaming, no reuse).
- Consolidate: pick champion; if a further change gives <~2% geomean, stop the branch.

*(Exact IDs assigned at implementation time in strict `c001,c002,…` order; the phases above are the
priority queue, not fixed numbering. Only the next single change is committed to an ID.)*

## 5. Performance hypotheses (each tested by exactly one candidate + evidence)
- **H1 (baseline win)**: Fusing 5+ launches + removing intermediate materialization yields a large
  geomean speedup, biggest on decode shapes (launch-overhead-bound) and large prefill
  (redundant-traffic-bound). *Evidence*: c001 per-workload speedups; expect decode ≫ 1, prefill > 1.
- **H2 (BW saturation)**: On large prefill (#8,#9,#12) the Q kernel dominates (~83% traffic, draft §5)
  and should approach achieved-HBM SOL. *Evidence*: ncu memory throughput % on the Q kernel; tune
  until achieved BW plateaus.
- **H3 (V-copy fusion)**: Fusing V-copy into K kernel vs separate copy — one has better BW/occupancy.
  *Evidence*: A/B geomean + ncu on the KV path.
- **H4 (decode grid)**: Flattened/head-packed grid raises SM occupancy for `S=1`. *Evidence*: ncu
  achieved occupancy + wall-time on #1,#3.
- **H5 (block/warps)**: A BW-bound kernel needs enough resident warps + `num_stages` to hide latency;
  best `BLOCK_S`/`num_warps` differs by shape bucket. *Evidence*: autotune sweep geomean.

Each hypothesis: change → predict direction → evaluate/profile → confirm or reject in the record.

## 6. Profiling protocol (ncu-report-skill)
- Invoke the **ncu-report-skill** and drive it through `./scripts/ncu_profile.sh` only.
- Profile the **current champion**, never concurrently with any eval; ensure no eval is in flight.
- Focus metrics: DRAM/HBM throughput %, memory vs compute bound, achieved occupancy, launch
  overhead on decode. Profile a representative large shape (#8) and a decode shape (#1/#3).
- Use findings to choose the *next single* candidate change; record which metric motivated it.

## 7. Consult KernelWiki
- Before Phase 1/2 performance work, consult **KernelWiki** for H100 (sm_90) memory-bound
  elementwise-fusion patterns: vectorized bf16 access, warp/occupancy targets, `num_stages` for BW
  saturation, streaming store/eviction hints. Record any concrete guidance used per candidate.

## 8. Evaluation & evidence format
After each `./scripts/evaluate_candidate.sh feedback <cid>`, append **one** JSON object to
`candidates.jsonl` (never rewrite prior lines) with fields:
```json
{
  "candidate": "cNNN",
  "parent": "cM or null",
  "source_sha256": "<hash of solution/solution.py>",
  "hypothesis": "what this change tests (ref H1..H5)",
  "change_summary": "one-line diff from parent",
  "validation": {"all_pass": true, "failed_workloads": []},
  "per_workload": [
    {"uuid": "...", "axes": {"batch_size":_,"seq_len":_,"cache_len":_},
     "pass": true, "speedup": 0.0, "max_abs_err": null}
  ],
  "geomean_speedup": 0.0,
  "decision": "keep-as-champion | reject | superseded",
  "cumulative_evals": 0,
  "skills_used": ["ncu-report-skill", "KernelWiki"],
  "notes": "profiling takeaways / next step"
}
```
- `source_sha256`: compute over the exact evaluated `solution/solution.py`.
- `speedup` = reference_time / candidate_time per workload (as reported by the evaluator output).
- `decision` records lineage; champion is the highest valid `geomean_speedup` so far.
- Also note cumulative eval count so budget usage is auditable.

## 9. Stopping criteria
Stop and write `SEARCH_COMPLETE` (with the reason) when any holds:
- Geomean improvement over the last 2 evaluated candidates is < ~2% **and** ncu shows the champion
  is HBM-BW-bound near achieved SOL (no headroom left).
- Eval budget nearly exhausted (reserve ≥1 slot; keep margin for the operator-approved `final`).
- Token budget approaching soft limit (9M): consolidate to champion and stop exploring.
- Two consecutive candidates fail to beat the champion across all active directions.
`SEARCH_COMPLETE` records: champion ID, its geomean, why converged, and remaining budget.

## 10. Risk register (carried from draft, tracked across candidates)
- Tolerance rule (combined atol+rtol assumed, draft §4.1) — if c001 fails tight-atol shapes
  (#1,#3,#4), re-examine this **before** touching precision.
- 2^31 cache-offset margin — int64 indexing mandatory.
- Decode under-fill — measure before switching grid (H4).
- Autotune compile time vs coverage — keep config lists small.
- V-copy must never be normed/roped (recurring-bug watch).

## 11. Immediate next action (after this plan is approved to proceed)
Implement **c001** exactly as specified in §4 Phase 0, run the §3.1 pre-eval checklist, then a
single `feedback c001` evaluation, and append its record per §8. No performance tuning until c001
is confirmed correct.

## 12. Progress log / decision notes
- **c001 (eval 1/100) — REJECT (correctness).** 12/13 pass, geomean 10.46x over passers.
  Fused 2-kernel structure works and is fast (~9.4–12.9x). Only failure: workload 8f5402ae
  (B4 S1 cache_len2048), `INCORRECT_NUMERICAL` at atol=1e-5.
  - Diagnosis (updates risk-register §10, tolerance assumption draft §4.1): the tight atol=1e-5
    shapes are NOT under a pure combined rule — the evaluator is strict enough that near-zero
    entries need bf16-exact matching. c001 computes RoPE in fp32 (more accurate than the
    reference), but the reference computes RoPE in **bf16** (norm→bf16, cos/sin→bf16, each
    product/add rounded to bf16). Under the cancellation `x1*c − x2*s` the two differ by ~1
    bf16-ulp, which busts 1e-5 where |expected|≈0. Workload #1 (also atol=1e-5) passed only
    because cache_len=0 ⇒ position=0 ⇒ RoPE is identity.
  - **Next candidate c002**: match reference bf16 op semantics in the RoPE stage precisely —
    cast the normed vector to bf16, cast cos/sin to bf16, perform `x*cos`, `rotate_half(x)*sin`,
    and their sum in bf16 following the reference operation order. Keep RMSNorm in fp32 (reference
    does the norm in fp32 then casts to bf16). This is a single, meaningful numerical-fidelity
    change; expect all 13 to pass with performance essentially unchanged. Do not tune perf until
    correctness is green.
- **c002 (eval 2/100) — KEEP-AS-CHAMPION (first valid).** 13/13 pass, geomean **10.53x**
  (avg 10.59x). The single change (bf16-exact RoPE via `_rope_bf16`: round normed→bf16, cos/sin→bf16,
  each product+sum rounded to bf16, matching the reference op order) fixed 8f5402ae; all other
  shapes held (±~1x run-to-run noise). Confirms the c001 tolerance diagnosis.
  - **Next candidate c003 (performance, Phase 1/2)**: op is memory-BW-bound; first profile the
    champion with ncu-report-skill via `./scripts/ncu_profile.sh` (never concurrent with an eval) —
    achieved HBM BW % on the Q kernel for the largest shape (8603b54e) and achieved occupancy on a
    decode shape (2ff5eaf3). Then commit ONE change: likely `@triton.autotune` over
    BLOCK_S∈{16,32,64,128} × num_warps∈{2,4,8} × num_stages∈{1,2,3} keyed on shape bucket, or a
    single better fixed config if profiling points clearly. Keep the numerics identical so
    correctness stays 13/13.
- **c003 (eval 3/100) — KEEP-AS-CHAMPION.** 13/13 pass, geomean **14.64x** (avg 14.81x), up
  from c002's 10.53x. Single change: `@triton.autotune(key=[S])` over BLOCK_S∈{16,32,64} ×
  num_warps∈{2,4,8} × num_stages∈{2,3} on both kernels; numerics byte-identical to c002. The
  win is largest on decode/small shapes (2ff5eaf3 12.9→18.3x, 8f5402ae 11.6→17.6x), confirming
  the fixed BLOCK_S=32/warps=4 under-served them. Weakest remaining: d1dbaf22 (B2 S128) 9.70x.
  - **Next candidate c004 (profile-guided)**: profile the champion with ncu-report-skill via
    `./scripts/ncu_profile.sh` (never concurrent with an eval) — HBM BW% on the Q kernel for the
    largest shape (8603b54e) and achieved occupancy on decode (2ff5eaf3). Then ONE change chosen
    from: (a) fuse-vs-split the V-copy (H3), (b) add BLOCK_S=128 / more warps for large prefill,
    or (c) trim the config set to cut autotune cost. Keep numerics identical (13/13 must hold).
