# Plan — L1/058 MoE Expert Token Radix Sort with Prefix Sum

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`. This turn writes the plan
only; no candidate is implemented or evaluated here.

Governing facts (from draft):
- Op = **stable counting/radix sort on bounded keys 0..255** + prefix sum of the histogram. Integer,
  no floating point. Exact-match required (integer tolerance).
- Workloads tiny (`16K ≤ N ≤ 64K`, ≤256KB, L2-resident); every N is a multiple of 256.
- Bottleneck is **launch/allocation overhead**, not bandwidth/compute. Win = collapse the reference's
  ~5–6 kernels + several allocations into 1–2 Triton launches with lean allocation.
- Budget: 100 evaluations; token soft 9M / normal 10M / hard 11M. Feedback set (16 shapes) == final set.

---

## 1. Ground rules for every candidate

1. Each candidate is an **immutable** `solution/solution.py` version with id `cNNN`. Any meaningful
   source/config/launch change → new id. Never reuse an id for changed source.
2. `solution.py` exposes `run(topk_idx)` returning `(sorted_token_indices int32 (N,), expert_offsets
   int32 (257,))`. Torch used only for reshape/view, output `torch.empty`, grid math, dtype/shape.
   **No** `torch.sort/argsort/bincount/cumsum/histc` or any Torch/CPU/NumPy/CUDA-ext compute fallback.
   A failing Triton kernel is invalid — never paper over it with Torch.
3. **Correctness is argued by construction BEFORE evaluating** (offline numeric checks are sandbox-
   denied). Only evaluate once the invariant checklist (§3) passes on paper.
4. Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN`. One run = full 16-workload set =
   1 evaluation. Append exactly one JSON record to `candidates.jsonl` per evaluated candidate
   (append-only; never rewrite earlier records).
5. Profiling only via `./scripts/ncu_profile.sh ...`, and **never concurrently with an evaluation**
   (foreign process on the locked GPU → return code 3, measurement discarded, one evaluation wasted).
6. Keep the working `solution/solution.py` equal to the best-known-good candidate at all times so an
   operator `final` (operator-approved only) would run the intended source.

---

## 2. Candidate lineage strategy

Strategy: establish a trivially-correct, allocation-lean baseline first, then reduce launch/alloc
overhead and improve parallelism, guided by evaluator geomean and (if ambiguous) ncu. Each step is one
immutable candidate. Parent is the current best-valid candidate unless noted.

### Phase 0 — probe primitives + baseline (must not skip)
- **c001 — Option C: expert-parallel ordered gather (2 launches).**
  - K1 (1 program, or small grid): histogram of `flat` over 256 bins + exclusive prefix →
    `expert_offsets` (257, int32), with `expert_offsets[0]=0`, `[256]=N`.
  - K2 (grid = 256, one program per expert `e`): stream all N in tiles, `mask=(flat==e)`, carried
    `tl.cumsum(mask)` gives compacted rank, write source indices consecutively from
    `expert_offsets[e]`. Ascending stream ⇒ stable by construction; scratch-free.
  - Purpose: known-good, known-fast reference geomean AND probe of `tl.histogram`, `tl.cumsum`,
    carried-scan behavior in the installed Triton. If `tl.histogram` is absent/unsafe, use one-hot-sum
    histogram (record which was used).
  - Hypothesis: 2 launches + 2 allocations beats reference (~5–6 kernels + several allocs) → geomean
    > 1 on most shapes; 256× L2-resident reads are cheap at this scale.

### Phase 1 — reduce launches / allocations
- **c002 — Option A: single-block fully fused (1 launch).** One program: tile-loop histogram →
  in-register exclusive scan of 256 counts → write `expert_offsets`; second tile-loop with carried
  `pos[256]` (init = exclusive offsets) does ordered placement via within-tile keyed rank (one-hot
  cumsum, S1) then `pos += tile_hist`. Fewest launches; tests whether launch savings beat single-SM
  serialization. Parent = best of c001.
- **c003 — Option D/B: 2-launch multi-block radix.** K1 per-block 256-bin histogram → `(num_blocks,256)`
  scratch (+ atomics for global counts if cheaper); K2 exclusive prefix over experts and blocks →
  `expert_offsets` + per-block bases, then ordered block-scatter (S1 one-hot cumsum, BLOCK≤128 for
  SRAM). Multi-SM parallel; expected to help most at N=32K–64K shapes. Parent = best so far.

### Phase 2 — tuning of the winning design
- **c004+ — autotune / micro-opts** on the Phase-0/1 winner: `BLOCK` ∈ {256,512,1024} (divides 256-aligned
  N), `num_warps` ∈ {2,4,8}, tile-loop unrolling, `other=256` masking sentinel vs bin-0 correction,
  `empty` vs `zeros` for offsets (ensure `[0]` explicitly zeroed and every entry written), avoiding
  int64 anywhere, minimizing Python-side host overhead (single grid computation, no redundant `.item()`).
  Each distinct config = a new id.

### Branch/backtrack rule
- If a candidate fails any workload's correctness → **reject**, do not build on it; return parent to
  `solution.py`; fix the invariant violation in a new id.
- If a candidate is correct but slower than parent → keep parent as best; may still branch from it to
  test an orthogonal idea. Best-valid candidate = highest geomean among all-pass candidates.

---

## 3. Correctness checks (per-candidate, pre-evaluation checklist)

Prove these by inspecting the code before spending an evaluation:

1. **Offsets identity.** `expert_offsets[0]==0`; `expert_offsets[e]==#{i: flat[i]<e}` (exclusive
   prefix of histogram); `expert_offsets[256]==N`. Confirm inclusive-vs-exclusive is not swapped and
   all 257 entries are written (nothing left uninitialized if using `empty`).
2. **Placement identity.** Token `i` written at position `expert_offsets[flat[i]] + #{j<i: flat[j]==flat[i]}`.
   Confirm the rank scan runs in strictly ascending `i` (stability), ties broken by original order.
3. **Value stored = source index `i`** (0..N-1), NOT the expert id.
4. **Empty experts.** Zero-length ranges write nothing, no OOB. (Random data at N/256≈64 may empty some experts.)
5. **Tail masking.** Masked/OOB lanes excluded from histogram (sentinel `other=256` or bin-0 correction)
   and from compaction; never written. (N%256==0 here, but keep general.)
6. **Dtype.** All accumulators int32; both outputs cast to int32; no int64 detour.
7. **Triton-only compute.** No forbidden Torch ops in the compute path; Torch limited to reshape/empty/grid.

A candidate failing any single workload's correctness in evaluation is rejected regardless of speed.

---

## 4. Performance hypotheses (to confirm/refute via evaluator geomean, then ncu if ambiguous)

- **H1 (launch-bound).** Speedup tracks launch+alloc count, not FLOPs/bytes. → fewer launches/allocs win.
  Test: c001 (2 launches) vs reference; c002 (1 launch) vs c001.
- **H2 (single-SM ceiling).** Option A's 1-launch saving may be offset by single-SM serialization of
  many tiles at the larger N (32K–64K). → A wins at small N, B/D win at large N. Test via per-workload
  speedups of c002 vs c003.
- **H3 (256× reads are free).** Option C's re-reading input 256× is L2-resident and negligible at
  ≤256KB. → C not memory-bound. Confirm via ncu (l2 hit rate, dram bytes) only if C underperforms.
- **H4 (grid-size overhead).** A 256-program grid (C) vs 1-program grid (A) launch cost is small
  relative to reference savings. → C still net-positive at N=16K.
- **H5 (autotune warmup excluded).** Autotune recompiles across 16 shapes add warmup only (2 warmup
  iters excluded from timing) → safe to autotune a *lean* config set; avoid config explosion.

Record which hypotheses each candidate supports/refutes in its evidence entry.

---

## 5. Profiling protocol (only between evaluations)

- Trigger: a correct candidate underperforms expectation or two designs are within noise.
- Command form: `./scripts/ncu_profile.sh --set basic -o profile/rN python harness.py` (via workspace
  launcher only; never invoke `ncu` directly; never while an evaluation runs).
- Read: launch count, per-kernel duration, achieved occupancy, L2 hit rate, DRAM bytes, whether time
  is dominated by fixed overhead vs a specific kernel. Use `ncu-report-skill` to interpret.
- Feed conclusions back into design choice (A vs B/C) and BLOCK/num_warps.

---

## 6. Stopping / convergence criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
1. **Converged:** best geomean improves < ~2% across 2–3 successive candidates spanning distinct designs.
2. **Design space exhausted:** Options A, B/D, C and their leading tuned configs evaluated; no untested
   idea with plausible upside remains.
3. **Budget:** approaching evaluation budget (100) or token soft limit (9M) — leave margin for the
   record-keeping; never exceed hard limit (11M).
Never run `final` without explicit operator approval, even at convergence.

---

## 7. Evidence format (`candidates.jsonl`, one JSON object per evaluated candidate, append-only)

```json
{
  "id": "c001",
  "parent": null,
  "design": "Option C: expert-parallel ordered gather, 2 launches",
  "source_sha256": "<hash of solution/solution.py at evaluation time>",
  "hypothesis": "2 launches + 2 allocs beat reference's ~5-6 kernels; stable by ascending stream scan",
  "correctness_checklist": {"offsets_identity": true, "placement_stability": true,
    "value_is_source_index": true, "empty_experts_ok": true, "tail_masking_ok": true,
    "int32_only": true, "triton_only": true},
  "primitives_probed": {"tl_histogram": "available|fallback-onehot", "carried_cumsum": "ok|spills"},
  "per_workload": [
    {"uuid": "e259b448-...", "bs": 8, "seq": 256, "N": 16384, "pass": true, "speedup": 0.00}
  ],
  "geomean_speedup": 0.00,
  "all_pass": true,
  "decision": "keep-as-best|reject|superseded",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki"],
  "notes": "hypotheses supported/refuted; next step"
}
```

Rules:
- Fill every field. `per_workload` has all 16 shapes with pass + speedup (speedup vs reference as
  reported by evaluator). `all_pass=false` ⇒ `decision="reject"`.
- `geomean_speedup` = geometric mean over the 16 per-workload speedups (only meaningful if all pass).
- `cumulative_evaluations` = running count of feedback evaluations spent (budget 100).
- Never edit a prior record; corrections go in a new record's `notes`.

---

## 8. Progress log

- **c001 (Option C, evaluated, REJECT).** 0/16 pass, all `RUNTIME_ERROR` at every N; `max_abs/max_rel`
  all 0.0 ⇒ the kernel never produced output (compile/trace error, not a numerical mismatch). Genuine
  failure, NOT a code-3 timing invalidation (`monitor.foreign_process_detected=false`). The controller
  log does not capture the Python traceback and offline `python` is sandbox-denied, so the exact line
  is inferred. Most likely cause: the JIT kernels take `NUM_TILES`/`PAD` as **runtime int args used as
  a Python `range()` bound and in tensor arithmetic** — this Triton build likely requires a
  `tl.constexpr` loop bound. Also suspect: `tl.histogram` availability/signature, and
  `tl.store(offsets_ptr, 0)` of a bare Python int.

- **c002 (Option C hardened, evaluated, REJECT).** 0/16, same identical `RUNTIME_ERROR`. Root cause
  now **confirmed** by running `harness.py` through the `ncu_profile.sh` launcher (offline debug, no
  evaluation running): Triton 3.5.0 (torch 2.9.0+cu130, cuda 13.0) raises
  `CompilationError: Cannot access global variable NUM_EXPERTS from within @jit'ed function ... unless
  instantiated as triton.language.constexpr`. Both c001 and c002 read the module-level python int
  `NUM_EXPERTS` **inside** their `@triton.jit` kernels → compile fails on every shape. The Option C
  *algorithm* is sound; this is purely an API/env issue.

- **c003 (Option C, global-fixed) — EVALUATED, VALID, NEW BEST.** 16/16 pass; geomean **3.47x**,
  arithmetic mean 3.51x. Offline correctness pre-checked on 9 shapes (incl. boundary seq lens and
  max N) via `harness.py`. Confirmed exact-match vs the torch reference.
  - **Key perf signal:** speedup is *lowest at large N* — N=65536 → 2.27x; the four N=32768 shapes →
    ~3.03–3.13x; every N≤18432 shape → 3.5–4.0x. The 257-way K2 grid re-reads all of `flat` once per
    program (≈256× total reads), so at larger N the scatter kernel dominates and erodes the win. This
    supports H1 (launch-bound at small N) and shows H3 (256× reads free) breaking down for N≥32768.
  - **Next (Phase 1):** c004 = Option A single-block fused (1 launch) to shave launch overhead further
    at small N; c005 = Option B/D multi-block radix to remove the 256× re-read that hurts large N.
    Then Phase-2 tuning of BLOCK/num_warps.
  - `solution/solution.py` now holds the best-valid source (c003).

- **Env pin (confirmed):** torch 2.9.0+cu130, cuda 13.0, triton 3.5.0. Lesson for all future
  candidates: never reference a module-level python variable inside a `@triton.jit` body — pass it as
  a `tl.constexpr` arg or define it with `triton.language.constexpr`.

## 9. Original next actions (superseded by §8 once c001 evaluated)

1. Implement candidates per §2, evaluate one at a time, append records, keep best-valid as source.
