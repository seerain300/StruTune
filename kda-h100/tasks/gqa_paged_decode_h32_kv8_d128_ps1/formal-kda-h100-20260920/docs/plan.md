# Plan — `gqa_paged_decode_h32_kv8_d128_ps1` (FlashInfer, H100 / sm_90)

Executable, sequential KDA plan derived from `docs/draft.md`. This turn creates the plan
only; no candidate is implemented or evaluated here.

## 0. Ground rules (from CLAUDE.md / TASK.md)

- Primary compute in **Triton**; PyTorch only for allocation/launch/metadata. No Torch,
  CPU/NumPy, or CUDA-extension fallback — a failed Triton kernel is invalid, never swapped
  for a fallback.
- Only correctness+timing oracle: `./scripts/evaluate_candidate.sh feedback cNNN`. One call
  runs the **full 48-workload set** and counts as **one** of the **100** evaluations.
- Profiling only via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow). **Never**
  overlap profiling with an evaluation (foreign process on the locked GPU ⇒ return code 3,
  measurement discarded, one eval slot burned).
- Candidates are **immutable**: any meaningful source/config/launch change ⇒ new id. Never
  reuse an id for changed source. Never rewrite earlier `candidates.jsonl` records.
- Token budget: soft 9.0M / normal 10.0M / absolute 11.0M. Candidate budget: 100 evals.
- `final` only after explicit operator approval.

## 1. Objective and gate

- **Hard gate:** every one of the 48 feedback workloads passes correctness (`output` bf16 +
  `lse` fp32 within tolerance). A candidate that fails any workload cannot be selected.
- **Ranking metric:** geometric mean speedup vs the reference across all passing workloads.
- **Where the wins are** (from draft §3–4): (a) GQA packing so K/V is read once (~4× traffic
  cut), (b) saturate HBM on batch=64 / large batch=16, (c) low launch overhead on tiny
  batch=1.

## 2. Candidate ladder (sequential, one source version at a time)

Each rung is implemented and evaluated **only after** the previous rung's evidence is
recorded. Later rungs are conditional on measured evidence, not executed blindly. Config
values are starting points; exact numbers may shift based on the prior rung's result, but any
change still spawns a new id.

### Phase A — correct baseline

- **c001 — GQA-packed baseline (Approach B).**
  Grid `(batch, num_kv_heads=8)`; each program owns one `(b, kv_head)` and its 4 query heads.
  - Read `kv_indptr[b]`, `kv_indptr[b+1]` inside the kernel (no host sync); `seq_len =
    end-start`.
  - Loop tiles of `BLOCK_N=64` over the sequence: gather page ids `kv_indices[start+offs_n]`,
    load `K_tile[BLOCK_N,128]`, `V_tile[BLOCK_N,128]` (contiguous 256 B/row, §draft-2).
  - `logits = q_group @ K_tileᵀ` (M padded to `BLOCK_H=16`, rows 4..15 masked), fp32 accum,
    **no tf32**.
  - Online softmax base-2 via folded `qk_scale = sm_scale*log2e`, `tl.exp2`; keep `m[16]`,
    `l[16]`, `acc[16,128]` in fp32.
  - Second dot `p@V`: start with **fp32** `p`×fp32 `V` (closest to reference; correctness
    first).
  - Tail mask `offs_n < seq_len` ⇒ logits `-inf`; empty-seq guard on store: `out =
    where(l>0, acc/l, 0)`, `lse = where(l>0, m+log2(l), -inf)`.
  - Config: `BLOCK_N=64`, `BLOCK_H=16`, `BLOCK_D=128`, `num_warps=4`, `num_stages=2`.
  - **Purpose:** establish correctness on all 48 and a baseline geomean. Not tuned for speed.

### Phase B — precision/throughput of the inner math (only after c001 passes)

- **c002 — bf16 second dot.** Cast `p` to bf16, `tl.dot` with bf16 `V` (tensor-core path).
  Hypothesis: faster and still within attention tolerance. Keep c001's fp32 second dot as the
  accuracy fallback if c002 fails the gate.
- **c003 — pipeline depth.** From the better of c001/c002, raise `num_stages` (2→3, maybe 4)
  to hide gather latency. Hypothesis: helps the bandwidth-bound batch=64/large-16.

### Phase C — tile/occupancy tuning (only after Phase B; guided by one ncu pass)

Run **one** `ncu_profile.sh` pass (separate from any eval) on a representative batch=64
workload to confirm HBM-bound behavior, achieved BW%, and that K/V is read once; and on a
batch=1 workload to check occupancy/launch overhead. Then:

- **c004 — `BLOCK_N` sweep.** Try `BLOCK_N ∈ {32,128}` vs 64. Larger ⇒ fewer iterations, more
  smem (`BLOCK_N*128*2B` each for K/V ×num_stages). Pick the geomean winner.
- **c005 — `num_warps` sweep.** {2,4,8} on the winning tile. Hypothesis: with M effectively 4
  and D=128, fewer warps may reduce overhead on small batches; more warps help big ones.
- **c006 — optional small autotune.** A *small* config set (keep warmup compile cost low,
  since it's inside the coarse feedback run) if manual sweeps disagree across regimes.

### Phase D — occupancy fix for low-parallelism regime (conditional)

Only if Phase C ncu evidence shows batch=1 (8 CTAs) and/or small batch=16 are
occupancy/latency-limited rather than launch-bound:

- **c007 — split-KV / flash-decoding.** Split each sequence into `S` chunks over extra
  programs (grid `(batch, 8, S)`) writing partial `(acc, m, l)`, then a small combine kernel
  merges partials via LSE. Hypothesis: raises occupancy for batch=1/16 without adding traffic;
  batch=64 already fills the machine so guard with a `seq_len`/batch heuristic so it does not
  regress the large cases. This is a genuinely different launch ⇒ new id (and any tuning of
  `S` ⇒ further ids).

## 3. Candidate lineage strategy

- Linear spine `c001 → c002 → c003 → …`; each candidate names its **parent** (the source it
  was derived from) in `candidates.jsonl`.
- When a change **regresses** or **fails the gate**, the next candidate's parent is the last
  *accepted* candidate, not the failed one. The failed record is still appended (never
  deleted) with `decision:"reject"` and the reason.
- Only ever change **one conceptual thing per candidate** (dtype OR pipeline OR tile OR grid),
  so each eval attributes a single cause. Multi-knob changes only via an explicit autotune
  candidate (c006).
- `source_sha256` recorded per candidate; identical source must never be re-evaluated under a
  new id, and changed source must never reuse an id.

## 4. Correctness checks (reason-first, before each eval)

Pre-implementation paper checks (evaluation is the only oracle — get it right before
spending a slot):
1. **Base-2 LSE fold:** `qk_scale = sm_scale*log2e`, `t=q·k*qk_scale`, `lse = m+log2(l)`
   where `m=max t`, `l=Σ exp2(t-m)`. Confirm `m+log2(l) == log2(Σ e^{scaled})`. A missing
   fold ⇒ LSE off by `ln2`.
2. **Empty-seq / 0/0:** `where(l>0, …, 0)` for output, `-inf` for lse. Matches reference.
3. **Layout/stride:** `k_cache`/`v_cache` `[num_pages,1,8,128]`; offset `p*1024 + kh*128 +
   d`; gather row is contiguous 256 B. Cast page ids to **int64** before pointer math.
4. **Head mapping:** `kv_head = h//4`; program's 4 query heads are `kv_head*4 + {0,1,2,3}`;
   store to exactly those output/lse rows; masked rows 4..15 never stored.
5. **Tail mask:** `offs_n < seq_len` ⇒ `-inf` logits; K/V `other=0` with clamped pointers.
6. **No tf32** in `q·kᵀ`; fp32 accumulators for `m,l,acc`; grid from static shapes only.

Post-eval checks (from the harness report):
- All 48 `passed=true`. If any fail, classify by regime (empty/tiny seq ⇒ guard; wrong LSE ⇒
  base fold; wrong values on large seq ⇒ masking/accumulation) and fix in the next id. **Never
  loosen the math to force a pass.**

## 5. Performance hypotheses (each tied to a candidate)

| # | Hypothesis | Rung | Expected signal |
|---|---|---|---|
| H1 | GQA packing (K/V read once) beats reference sharply on batch=64/large-16 | c001 | geomean ≫ 1 on those workloads |
| H2 | bf16 second dot speeds inner loop, stays within tol | c002 | faster, still 48/48 pass |
| H3 | More pipeline stages hide gather latency (bandwidth-bound) | c003 | batch=64 improves |
| H4 | Optimal `BLOCK_N` balances iteration count vs smem | c004 | one tile wins geomean |
| H5 | `num_warps` trades small-batch overhead vs large-batch throughput | c005 | per-regime shift |
| H6 | Split-KV raises occupancy for batch=1/16 without hurting batch=64 | c007 | small-batch up, large-batch flat |

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- Geomean improvement over the last **2–3** accepted candidates is < ~1–2% (converged), or
- Phase A–C exhausted and Phase D is either not indicated by ncu evidence or evaluated and not
  beneficial, or
- Approaching the token soft limit (9.0M) or the 100-eval budget — leave margin so the final
  best candidate and records are safely written.
Then select the best **valid** (48/48 passing) candidate as the final pick. Do **not** run
`final` without operator approval.

## 7. Evidence format — one JSON object appended per evaluated candidate to `candidates.jsonl`

Never rewrite earlier records. Schema:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_sha256": "<sha256 of solution/solution.py at eval time>",
  "hypothesis": "GQA-packed baseline reads K/V once; establish correctness + baseline geomean",
  "change_summary": "grid (batch,8); 4 q-heads/program; BLOCK_N=64; fp32 second dot",
  "config": {"BLOCK_N":64,"BLOCK_H":16,"num_warps":4,"num_stages":2,"second_dot":"fp32"},
  "validation": {
    "all_passed": true,
    "num_passed": 48,
    "num_total": 48,
    "failures": []
  },
  "per_workload": [
    {"uuid":"e2142798-...","batch":1,"num_kv_indices":73,"passed":true,"speedup":0.0}
  ],
  "geomean_speedup": 0.0,
  "decision": "accept | reject | baseline",
  "reason": "why accepted/rejected and what the next candidate should try",
  "cumulative_evals": 1,
  "skill_usage": ["ncu-report-skill: <when/what>", "KernelWiki: <if consulted>"],
  "notes": "profiling/observations, if any (never from an eval that overlapped profiling)"
}
```

Conventions:
- `speedup` = reference_time / candidate_time per workload; `geomean_speedup` over all
  passing workloads. Record raw enough per-workload data to compare regimes (batch 1/16/64).
- `cumulative_evals` increments by exactly 1 per `feedback` call (the 48-set = one eval).
- If a run returns code 3 (profiler/eval overlap or foreign process), record it as a wasted
  slot with `decision:"invalid-run"` and do **not** treat its numbers as real.
- `skill_usage` cites any `ncu-report-skill` / `KernelWiki` use for that candidate.

## 8. Progress log / next step

- **c001 — ACCEPTED (eval 1/100).** GQA-packed baseline. 48/48 pass; geomean **378.12x**
  (arith 596.45x, min 43.18x, max 1122.41x). Confirms H1 and all §4 correctness checks
  (base-2 LSE fold, empty-seq guard, layout, head mapping). Spine.
  - Regime read-out: batch1 latency-bound (~0.046–0.14ms, 8 CTAs); the HBM-bound cases with
    the largest solve times are batch16 kv>12k (0.169–0.196ms) and all batch64 (~0.30–0.33ms).
    Tuning should target these.

- **Next: c002** — Phase B. Cast `p`→bf16 and run the second dot `p@V` on tensor cores (bf16×bf16,
  fp32 accum) instead of fp32-ieee. Hypothesis H2: faster inner loop on the large-seq workloads
  while staying 48/48 within tol. Parent = c001. If it fails the gate, revert to c001's fp32
  second dot and move to Phase B/C pipeline/tile tuning instead.

- **c002 — ACCEPTED (eval 2/100).** bf16 tensor-core second dot. 48/48 pass; geomean
  **466.88x** (from 378.12x, **+23%**). Confirms H2. Win concentrated on the HBM-bound
  targets: batch64 ~700–800x → ~1170–1310x (0.30–0.33ms → 0.19–0.20ms); large batch16
  1e15ed03 0.169→0.141ms, ccdc67b6 0.196→0.154ms. batch1 unchanged (latency-bound). abs error
  ≤1.56e-2, matched_ratio gate passes everywhere. New spine.

- **Next: c003** — Phase B. Raise `num_stages` 2→3 to pipeline the paged-gather latency on the
  bandwidth-bound batch64 / large-batch16 cases. Parent = c002. Keep if geomean improves and
  48/48 hold; else revert and proceed to Phase C tile/warp tuning.

- **c003 — ACCEPTED (eval 3/100).** `num_stages` 2→3. 48/48 pass; geomean **489.78x** (from
  466.88x, **+4.9%**). Confirms H3. Large batch16 improved (1e15ed03 0.141→0.108ms, ccdc67b6
  0.154→0.128ms) and batch64 0.19–0.20ms → 0.17–0.19ms. batch1 slightly noisier/lower (deeper
  pipeline gives no benefit on tiny latency-bound cases) but net geomean clearly up. New spine.

- **Next: c004** — Phase C tile tuning. Try `BLOCK_N=128` (fewer loop iterations, larger gather
  bursts per tile) on the large-seq cases. Parent = c003. Keep if geomean improves and 48/48
  hold; else revert and try `BLOCK_N=32` / `num_warps` sweep. Remaining headroom is
  concentrated in batch1 (8 CTAs, latency-bound) and the 2 biggest batch16 single-long-seq
  cases — both split-KV (Phase D) territory if tile/warp tuning stalls.

- **c004 — ACCEPTED (eval 4/100).** `BLOCK_N` 64→128. 48/48 pass; geomean **521.27x** (from
  489.78x, **+6.4%**). Confirms H4. batch64 ~1400–1730x (0.156–0.174ms); big batch16
  1e15ed03 0.108→0.094ms, ccdc67b6 0.128→0.100ms. batch1 flat (latency-bound). New spine.
  Note: smem/stage now ~192KB (K+V×3 stages) — near H100 cap; a num_warps bump may not fit.

- **Next: c005** — Phase C. Sweep `num_warps` (try 8) on the BLOCK_N=128 config for long-seq
  MMA/gather throughput. Parent = c004. If it offers nothing (or hits smem/compile limits),
  move to Phase D split-KV for batch1 + the 2 big batch16 cases.

- **c005 — REJECTED (eval 5/100).** `num_warps` 4→8. 48/48 pass but geomean **487.61x**
  (−6.5% vs c004's 521.27x). 8 warps over-subdivide the small BLOCK_H=16/D=128 MMA tiles and
  hurt batch64 (max ~1440x vs ~1730x); no batch16 benefit. Reverted source to c004
  (sha256 88aaa1e3… confirmed). **c004 remains the spine.** num_warps knob closed.

- **Next: c006** — Phase D split-KV / flash-decoding. Split each sequence into chunks across
  extra programs (grid `(batch, 8, S)`) writing partial `(acc,m,l)`, then a combine kernel
  merges via LSE. Target the only remaining headroom: batch1 (8 CTAs, latency-bound) and the
  2 big batch16 single-long-seq cases (1e15ed03 kv=12942, ccdc67b6 kv=20911). Guard with a
  seq-len/batch heuristic so batch64 (already saturating) does not regress. Parent = c004.

- **c006 — REJECTED (eval 6/100).** Split-KV (S=16) for the `batch*8<64` regime + LSE combine
  kernel. 48/48 pass (split+combine math fully correct) but geomean **417.16x** (−20% vs c004),
  entirely from batch1: split path made batch1 **worse** (~45x / 0.090–0.099ms vs c004 ~64–90x /
  0.051–0.066ms). Root cause: batch1 sequences are short (kv 2..547) → launch/latency-bound, not
  occupancy-bound; a second kernel launch + fp32 partial buffers doubles the fixed cost on tiny
  data. The long-seq batch16 cases that could benefit kept the fast path (batch*8=128≥64), so the
  heuristic never reached them. Reverted source to c004 (sha256 88aaa1e3… confirmed). **c004
  stays the spine. Phase D closed** — split-KV is the wrong tool for this workload mix.

- **Convergence status.** Geomean progression: 378 (c001) → 467 (c002) → 490 (c003) → 521 (c004);
  c005 (num_warps) and c006 (split-KV) both rejected. The kernel is bandwidth-optimal on the
  large cases (K/V read once, bf16 TC dots, BLOCK_N=128, 3 stages) and batch1 is at its launch
  floor. **Next: at most one more low-risk tile/stage probe** (e.g. `num_stages=4`, or
  `BLOCK_N=96`) as c007; if it does not beat c004, declare convergence and write
  `SEARCH_COMPLETE` selecting c004.
