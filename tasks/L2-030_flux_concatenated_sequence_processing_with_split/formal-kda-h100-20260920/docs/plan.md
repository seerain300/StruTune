# Plan — L2/030 Flux Concatenated Sequence Processing With Split

Executable, sequential KDA optimization plan. Builds on `docs/draft.md`. This turn produces
the plan only — **no candidate is implemented or evaluated here.**

---

## 0. Goal & north star

Maximize **geometric-mean speedup** over the fp32 reference across the 16 feedback workloads,
subject to **every selected workload passing correctness** (`atol ∈ [0.0018, 0.0021]`,
`rtol = 1e-5`, `match_ratio ≥ 0.98`). Implementation is **Triton-primary**; Torch only for
metadata/launch plumbing. Budget: 100 candidate evaluations; token soft/normal/hard = 9M/10M/11M.

The mathematical target (from the draft): eliminate the concat, compute two weight-shared
row-major GEMMs `A @ W.T` with fp32 accumulate/store, using `tl.dot(input_precision="tf32x3")`
as the compute path (accuracy-safe ~70–90× margin, ~2× faster on compute-bound shapes).

---

## 1. Operating rules (enforced every candidate)

1. One immutable source version per candidate ID (`c001`, `c002`, …), created sequentially,
   one at a time. Any meaningful source / config / launch change ⇒ **new ID**. Never reuse an
   ID for changed source; never rewrite earlier `candidates.jsonl` records.
2. Evaluate **only** with `./scripts/evaluate_candidate.sh feedback <id>`. The full 16-workload
   run = one evaluation.
3. After each evaluation, **append exactly one** JSON object to `candidates.jsonl` (schema §6).
4. Profiling only via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow), **never**
   concurrently with an evaluation. Serialize strictly. Never call `ncu`/`nvidia-smi`/CUDA/the
   external evaluator directly.
5. No Torch/CPU/NumPy/CUDA-extension computational fallback. A failing Triton kernel is invalid;
   fix or replace with a different **Triton** approach, never with `torch.matmul` as the compute.
6. `final` is operator-only — never run without explicit approval.
7. Work only inside this workspace; external knowledge only from KernelWiki + ncu-report-skill.

---

## 2. Solution module contract (`solution/solution.py`)

Every candidate exposes the same entry point; only the internals change.

```python
def run(hidden_states, encoder_hidden_states, process_weight):
    # returns (processed_encoder, processed_hidden)
```

Invariants held across all candidates:
- Output dtypes = fp32; shapes `[B,T,H]` and `[B,I,H]`; contiguous.
- `H = 3072` read from tensors (not hard-coded), but blocks may assume divisibility of N=K.
- Inputs reshaped `[B,S,H] → [M,H]` as a **view** (zero-copy for contiguous input); pass strides
  to the kernel rather than forcing `.contiguous()` unless a candidate deliberately tests that.
- Weight accessed transposed (`Bmat = W.T`) via stride swap — no materialized transpose copy.
- Compute path: `tl.dot(a, b, input_precision="tf32x3")`, fp32 accumulator. (`"ieee"` reserved
  as a correctness-safe Triton fallback candidate if tf32x3 ever misses.)
- No `torch.cat`; split realized by writing into two separate output tensors / launching over
  two regions.

---

## 3. Candidate lineage strategy (sequential search tree)

Each rung is a hypothesis-driven, immutable step. Later rungs are **conditional** on measured
evidence from earlier ones (branch decisions noted). Estimated evaluations in brackets; the
100-eval budget is ample, so the constraint is really the token budget and convergence, not evals.

### Phase A — Correctness-first baseline
- **c001 — Two independent tiled GEMMs, tf32x3, no concat.**
  Simplest correct Triton implementation: one `matmul` kernel launched twice (encoder rows, then
  image rows), sharing `W` (hot in L2). Fixed, conservative tile (e.g. `BM=64, BN=128, BK=32,
  warps=4, stages=3`), mask on M only.
  - Purpose: prove correctness of the tf32x3 + no-concat reduction on **all 16** workloads and
    get the first speedup datum + per-workload timing profile.
  - Gate: if any workload **fails correctness**, do not proceed to perf tuning — first create
    **c002 = c001 with `input_precision="ieee"`** to isolate whether the failure is precision
    (expected pass) or a logic/masking bug (would also fail under ieee ⇒ fix logic).
  - [1–2 evals]

### Phase B — Launch fusion (reduce overhead, help small-M)
- **c00x — Fused single-launch, two-region grid.**
  One kernel, grid over `ceil(M_text/BM) + ceil(M_img/BM)` M-tiles across both regions; a per-tile
  branch selects (encoder vs image) base pointers/M. Removes second launch overhead; keeps `W`
  hot across both regions.
  - Hypothesis: helps the tiny-M / launch-bound workloads (#4, #1, #2, #5-text, #8-text) and is
    neutral-to-positive on large ones.
  - Branch decision: compare geomean **and** per-workload timings vs the Phase-A winner. Keep the
    fused variant only if it does not regress the large compute-bound workloads. If fusion adds
    branch overhead that hurts big-M, keep two-launch for big-M and fuse only small-M (a later
    candidate), or keep whichever wins overall.
  - [1–2 evals]

### Phase C — Tiling / config tuning across the wide M range
- **c00x — M-bucketed heuristic or `triton.autotune` keyed on M.**
  Because M spans 77 → 16384, pick tile configs per M-bucket:
  - tiny M (≤256): small `BM` (32/64), enough `BN` to keep N-side efficient, minimize wasted
    rows; this tier is memory/launch-bound on the 37.7 MB weight read.
  - medium M (256–4096): balanced `BM=128, BN=128/256, BK=32/64`.
  - large M (≥4096, esp. #11): larger `BM/BN`, more `num_stages` (tf32x3 = 3 MMAs/K-step),
    `GROUP_M` swizzle for L2 reuse.
  - Explore, as **separate immutable candidates**, a small set of promising configs (each config
    change = new ID). Prefer a hand-tuned bucketed heuristic over broad autotune to keep the
    source deterministic and the config space auditable; if using `triton.autotune`, fix the
    config list in source (immutable) and key on `M`.
  - Evidence source: ncu profiling (serialized) on the dominant workloads (#11, #5, #8, #9, #10,
    #13) to classify compute- vs memory-bound and read achieved occupancy / tensor-core util /
    L2 hit-rate before choosing configs — avoids blind autotune churn.
  - [several evals, one per distinct config candidate]

### Phase D — Scheduling for the dominant large workloads (conditional)
- **c00x — Persistent / swizzled-raster scheduler**, only if Phase-C ncu evidence shows tail
  effect or L2 misses on #11 (M_img=16384) / #9 / #13.
  Persistent grid (~#SMs programs) iterating a group-swizzled tile list for L2 reuse of `W`.
  - Branch decision: adopt only if it improves the large workloads without regressing others;
    otherwise discard and keep the Phase-C winner.
  - [1–2 evals]

### Phase E — Micro-tuning of the current best (conditional)
- Small deltas on the best candidate: `num_warps 4↔8`, `num_stages 3↔5`, `BK 32↔64`, `GROUP_M`,
  vectorization of the fp32 store, `evict_last` hint on `W` loads (L2-resident reuse). Each is a
  new immutable ID. Stop when deltas fall inside measurement noise.
  - [a handful of evals]

Lineage recorded via `parent` in each `candidates.jsonl` record so the tree is reconstructable.

---

## 4. Correctness checks (per candidate)

Correctness is owned exclusively by the evaluator (no local CUDA/alternate harness allowed).
For each candidate:
1. Run `./scripts/evaluate_candidate.sh feedback <id>` once (full 16-workload set).
2. Read the per-workload pass/fail and record **all 16** results.
3. A candidate is **valid** only if all 16 workloads pass correctness. A candidate that fails any
   workload is recorded as invalid with the failing UUID(s) and the failure reason; it is **not**
   used as a parent for perf tuning until fixed under a new ID.
4. Pre-eval self-review checklist before spending an evaluation (cheap insurance):
   - `input_precision="tf32x3"` explicitly set (not defaulted to `"tf32"`).
   - fp32 accumulator; output tensors fp32, correct shapes, contiguous.
   - M-masking correct at region boundaries (esp. non-multiple-of-BM M: 77, 262, 308, 844, 1087,
     2846); N/K assumed exact only because 3072 is divisible — assert divisibility in host code.
   - Encoder rows read from `encoder_hidden_states` and write `processed_encoder`; image rows read
     `hidden_states` and write `processed_hidden` (no cross-wiring in the fused grid).
   - `W.T` stride swap correct (`stride_bk`, `stride_bn`).
   - Source hash differs from every prior candidate (guarantees a genuinely new version).

---

## 5. Performance hypotheses (falsifiable, tied to evidence)

| # | Hypothesis | Evidence to confirm/refute |
|---|-----------|----------------------------|
| H1 | tf32x3 passes correctness on all 16 (margin shape-independent, K fixed) | c001 evaluator correctness results |
| H2 | Removing concat + tf32x3 gives geomean > 1.0 already at c001 | c001 geomean |
| H3 | Compute-bound large-M (#11,#9,#8,#5,#13) reach ~1.8–2.5× via tensor cores | per-workload timing vs baseline; ncu tensor-core util ≥ ~60% |
| H4 | Small-M (#4,#1,#2) are weight-read/launch-bound (~1.0–1.4×), improved by fusion | per-workload timing before/after fusion; ncu DRAM throughput near roofline |
| H5 | Single fused launch beats two launches on small-M without hurting large-M | Phase-B per-workload comparison |
| H6 | M-bucketed tiling beats one-size-fits-all across the 2-orders-of-magnitude M range | Phase-C per-workload comparison across config candidates |
| H7 | Persistent/swizzle helps #11 only if tail/L2 is the bottleneck | Phase-D conditional on Phase-C ncu (tail-effect / L2 hit-rate) |

Rule: never adopt a candidate as the new best on geomean alone — also inspect per-workload
timings so a big-workload win doesn't mask small-workload regressions (all count in geomean).

---

## 6. Evidence format (`candidates.jsonl`, one line per evaluated candidate)

Append-only; never edit prior lines. One JSON object per evaluated candidate:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "tf32x3 + no-concat two-GEMM reduction is correct and >1x on all 16",
  "change_from_parent": "initial implementation",
  "config": {"BM":64,"BN":128,"BK":32,"num_warps":4,"num_stages":3,
             "input_precision":"tf32x3","launch":"two-kernel"},
  "validation": {"all_pass": true, "failed_uuids": []},
  "per_workload": [
    {"uuid":"9de8abcb-...","B":2,"T":128,"I":256,"pass":true,
     "ref_ms":0.0,"cand_ms":0.0,"speedup":0.0}
    /* ... one entry per the 16 workloads, in file order ... */
  ],
  "geomean_speedup": 0.0,
  "decision": "keep|reject|new-best",
  "reason": "why kept/rejected; which hypothesis confirmed/refuted",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "notes": "profiling findings, next branch chosen"
}
```

Field rules:
- `source_sha256`: hash of the exact `solution/solution.py` evaluated. Must be unique per ID.
- `per_workload`: all 16 entries, in `feedback_workloads.jsonl` order, with pass flag and timing;
  fill `ref_ms`/`cand_ms`/`speedup` from evaluator output (leave/mark unknown if the evaluator
  does not surface a given field, but always record the pass flag and geomean).
- `geomean_speedup`: geometric mean of per-workload speedups over passing workloads (as the
  evaluator reports it); if any workload fails, still record it and set `decision` accordingly.
- `cumulative_evals`: running count of evaluations spent (monotone).
- `skills_used`: list the skills actually consulted for that candidate.
- `decision`: `new-best` only when valid AND geomean improves over the prior best without a
  per-workload regression judged unacceptable.

Track the running best candidate ID and its geomean in `notes`/decisions so lineage is auditable.

---

## 7. Profiling protocol (ncu-report-skill)

- Invoke the ncu-report-skill and use `./scripts/ncu_profile.sh --set <set> -o profile/<name>
  python <harness>` as documented; pick a small representative subset (one large workload like
  #11, one small like #4, one mixed like #8) rather than profiling all 16.
- **Never** while an evaluation is running (foreign process on the locked GPU ⇒ discarded eval,
  rc 3, wasted budget). Finish one before starting the other.
- Use profiling to decide Phase-C configs and the Phase-D branch: read tensor-core (pipe)
  utilization, DRAM throughput vs roofline, achieved occupancy, L2 hit-rate, and tail/wave
  quantization. Record findings in the relevant candidate's `notes`.

---

## 8. Stopping criteria / convergence

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
1. Successive candidates improve geomean by < ~1–2% (within measurement noise) for ~2–3
   consecutive attempts across different levers → converged.
2. Profiling shows the dominant workloads are at their roofline (compute-bound near tensor-core
   peak for tf32x3, or memory-bound near HBM roofline) with no remaining structural lever.
3. Approaching the token soft limit (9M) — wind down, finalize the best candidate, stop.
4. Evaluation budget nearly exhausted (well before 100; unlikely to bind first).

On completion: identify the single best **valid** candidate (all 16 pass, highest geomean),
state it explicitly, and note that `final` requires operator approval (do not run it).

---

## 9. Immediate next actions (subsequent turns, not this one)

1. Implement **c001** (`solution/solution.py`): two-launch tiled GEMM, tf32x3, no concat,
   M-masking; run the pre-eval checklist (§4.4).
2. Evaluate c001 once; append its `candidates.jsonl` record (§6); confirm H1/H2.
3. Branch per §3 (fix precision→ieee if any fail; else proceed to Phase B fusion).
4. Profile (serialized) to guide Phase C/D; iterate immutable candidates; record evidence.
5. Converge, write `SEARCH_COMPLETE`, report best candidate; await operator approval for `final`.
