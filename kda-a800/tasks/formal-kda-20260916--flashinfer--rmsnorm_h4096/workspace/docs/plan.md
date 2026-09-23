# Executable Optimization Plan — `rmsnorm_h4096` (FlashInfer, A800 / sm_80)

Derived from `docs/draft.md`. This is the ordered, executable KDA plan. It defines the
candidate lineage, the exact source/eval mechanics, per-candidate correctness checks,
falsifiable performance hypotheses, the stopping criteria, and the evidence format for
`candidates.jsonl`. **No candidate is implemented or evaluated in this turn.**

---

## 0. Ground rules (binding, from CLAUDE.md / TASK.md)

- Primary implementation is **Triton**. PyTorch only for metadata/launch plumbing
  (shape, dtype, output alloc, strides, grid). **No** Torch/CPU/NumPy/CUDA-extension/
  alternate-impl compute fallback. A failing Triton kernel is invalid → fix as Triton,
  never fall back.
- Candidates are **immutable and sequential** (`c001`, `c002`, …). Any meaningful change
  to source, config, or launch ⇒ **new candidate ID**. Never reuse an ID; never rewrite a
  past `candidates.jsonl` record.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback <id>`. The five fixed
  feedback workloads together = **one** candidate evaluation.
- Budgets: **100** evaluations; token soft **1.0M** / normal **1.5M** / absolute **1.65M**.
- I **cannot** run CUDA, a profiler, `nvidia-smi`, Python, the evaluator directly, or any
  alternate correctness harness. The **only** empirical signal is the feedback eval.
  ⇒ Front-load correctness by construction; spend evals deliberately.
- `final` (14-workload) runs **only** with explicit operator approval. Never auto-run it.
- `KernelWiki` skill is Blackwell/Hopper-specific ⇒ **not applicable** to sm_80; not used.
  (Recorded as `skill_usage: none` per candidate.)

---

## 1. Operation contract (what every candidate must compute)

Per row `i` of `x[B, H]`, `H = 4096` (compile-time constant), `EPS = 1e-5`:

```
ms_i      = (1/H) * Σ_j (fp32(x_ij))^2      # upcast bf16→fp32 BEFORE squaring; fp32 reduce
inv_rms_i = rsqrt(ms_i + 1e-5)              # fp32
y_ij      = (fp32(x_ij) * inv_rms_i) * fp32(w_j)   # fp32 throughout; order (x*inv)*w
out_ij    = bf16(y_ij)                      # single final downcast; RNE
```

Non-negotiable numerics (mirror the reference to de-risk the unseen tolerance):
fp32 accumulation, denominator `H=4096` (true mean), `+1e-5` added to the **mean of
squares**, **no** mean-subtraction (RMSNorm, not LayerNorm), weight upcast to fp32,
multiply order `(x*inv)*w`, single bf16 store.

Entry point: `solution/solution.py` exposing `run(hidden_states, weight) -> out` (bf16,
same shape/device as input).

---

## 2. Source & evaluation mechanics (per candidate)

For each candidate `cNNN` the implement/eval turn will:

1. Write the candidate's Triton kernel + `run(...)` plumbing to `solution/solution.py`.
2. Snapshot it immutably to `runs/candidates/cNNN/solution.py` (copy, never edit in place
   afterward) and record its source hash — this backs the "immutable ID + hash" contract
   and the `source_hash` evidence field. If the launcher expects the source only at
   `solution/solution.py`, keep both in sync at creation time and never mutate the
   snapshot afterward.
3. Run `./scripts/evaluate_candidate.sh feedback cNNN` **once**.
4. Parse per-workload PASS/FAIL + speedup, compute geomean, append **one** JSON record to
   `candidates.jsonl` (Section 7). Never modify prior records.

Rule: if a change is only in `num_warps`/`num_stages`/tile/launch, it is still a **new
candidate ID** because launch/config changed. One source version per ID, always.

---

## 3. Candidate lineage strategy

Linear backbone with single-variable steps so each eval yields an unambiguous signal.
Each node names its **parent**, the **one** variable changed, and its **hypothesis**.
Later branches are conditional on observed evidence; exact IDs assigned as we go.

```
c001 (anchor: design A, num_warps=8)
  └─ c002  num_warps sweep  (try the more promising of {4,16} first)
       └─ c003  num_warps sweep  (the other of {4,16}) — fix best warps W*
            └─ c004  num_stages tuning at W*  (e.g. 2→3/4 if it helps large regime)
                 └─ c005  row-blocking (design C) for the large regime  [conditional]
                      └─ c006  autotune (design B) keyed on batch bucket [conditional]
                           └─ c007  split-reduction (design D) for tiny [contingency only]
```

- **c001 — correctness anchor & baseline (design A).** Fused, one-pass,
  one-program-per-row; `grid=(B,)`; `BLOCK_SIZE = H = 4096` (power of two ⇒ **no
  reduction mask** on the hidden axis); fp32 accumulate; exact reference math; explicit
  row strides; `num_warps=8`, `num_stages=default`. Purpose: prove all five workloads
  PASS and establish per-regime baseline speedups. **Gate: do not tune until c001 passes.**
- **c002 / c003 — `num_warps` sweep.** Change only `num_warps` (128/256/512 threads ⇒
  32/16/8 fp32 elems/thread). This is the single biggest bandwidth knob for the large
  regime. Pick the winner `W*`.
- **c004 — `num_stages`.** At `W*`, try increasing pipeline stages to overlap global
  loads with compute on the large cases. Keep whichever wins.
- **c005 — row-blocking (design C).** Each program handles `R` rows (inner loop / 2D
  tile), `grid=ceil(B/R)`. Only pursued if the large regime remains below roofline and
  evidence suggests grid/setup overhead or MLP underutilization. Watch register/SRAM
  pressure (`R·4096` fp32).
- **c006 — autotune (design B).** `triton.autotune` over `num_warps`/`num_stages`/`R`
  keyed on a `batch_size` bucket so tiny and large regimes each get their best config.
  Only if a single static config cannot serve both regimes well **and** we confirm the
  evaluator's timing excludes autotune warmup (else it adds noise; if unclear, prefer the
  best static config and skip autotune).
- **c007 — split-reduction (design D).** Contingency **only** if the tiny regime clearly
  lags and evidence indicates the reduction (not launch latency) is the bottleneck. Adds a
  second launch + global round-trip, which likely hurts the launch-bound tiny regime, so
  this is low priority and may never be built.

Deprioritized (from draft, not scheduled unless evidence forces it): grid-stride/
persistent kernel (design F), manual vectorization beyond Triton's automatic bf16
coalescing (design E) — the contiguous unit-stride rows already vectorize.

---

## 4. Per-candidate correctness checklist (static, before every eval)

Because the only empirical channel costs tokens, run this static review on the source
before each evaluation:

1. **Cast order:** bf16 load → `.to(tl.float32)` **before** squaring; reduce with fp32
   `tl.sum`. No bf16 accumulation anywhere.
2. **Epsilon:** `inv = tl.rsqrt(sum_sq / 4096.0 + 1e-5)` — eps on the mean-of-squares,
   not on variance and not on RMS. `1e-5` hardcoded as fp32.
3. **Denominator:** divide by `H = 4096` (constexpr), a true mean.
4. **No centering:** confirm no row-mean subtraction slipped in from a LayerNorm template.
5. **Multiply order & dtypes:** `y = (xf * inv) * w_f`, `w` upcast to fp32; all fp32 until
   the final cast.
6. **Final store:** cast to bf16 (RNE) into a bf16 output tensor; no fp32 output.
7. **Pointer/stride math:** use `x.stride(0)` for the row stride, unit inner stride for
   both input and output; `BLOCK_SIZE == H == 4096` so no hidden-axis mask.
8. **Grid & bounds:** `grid=(B,)` correct for `B ∈ {7, 14, 15, 64, 14418, 14509}`; program
   id maps 1:1 to a valid row (no OOB); handles the smallest `B` and 1D-grid-legal for the
   largest.
9. **Constants:** `EPS`, `H`, `BLOCK_SIZE` are constants/`constexpr`.
10. **No fallback path:** `run(...)` has exactly one compute path (the Triton launch); no
    Torch/NumPy/CUDA-ext branch, no `try/except` that computes an alternate result.
11. **Metadata-only Torch:** only shape/dtype/alloc/stride/grid use Torch; optionally a
    `.contiguous()` **safeguard** only if a strided input is possible (prefer explicit
    strides to avoid a copy). No compute in Torch.
12. **Determinism:** no data-dependent control flow that changes numerics across rows.

Correctness is defined by the **official evaluator's** per-workload PASS/FAIL (its
tolerance is unseen; mirroring reference math in fp32 is the mitigation). A candidate with
any FAIL is **not** eligible as the best/selected candidate regardless of speedup.

---

## 5. Performance hypotheses (falsifiable, tied to the roofline)

Roofline (draft §3): large cases move ~226 MB HBM (≈113 MB read + ≈113 MB write; weight
8 KB negligible) ⇒ ideal ≈ **100–145 µs** (≈125 µs @1.8 TB/s). Tiny cases are
**launch/latency-bound** (tens–hundreds of KB total).

- **H1 (fusion win, both regimes).** A single fused one-pass bf16-in/bf16-out kernel beats
  the un-fused torch reference (which materializes fp32 intermediates and launches several
  kernels) in **all five** workloads. *Falsified if* any workload shows speedup ≤ 1.0×.
- **H2 (large regime near roofline).** c001's large-case time is within a modest factor of
  the 100–145 µs floor; achievable speedup vs reference is large because the reference
  pays extra fp32-intermediate traffic. *Measure* achieved-vs-roofline to size headroom.
- **H3 (`num_warps` matters for large, not tiny).** Sweeping `num_warps` moves the large
  regime (bandwidth/occupancy) but barely moves tiny (launch-bound). *Falsified if* tiny
  swings more than large under warp changes.
- **H4 (`num_stages` overlap).** More pipeline stages help the large regime by overlapping
  loads with compute, up to SRAM/occupancy limits. *Falsified if* extra stages don't
  improve or regress large-case time.
- **H5 (row-blocking, large only).** Row-blocking helps the large regime (fewer/larger
  blocks, better MLP) but **hurts** tiny (fewer blocks ⇒ less parallelism). *Falsified if*
  it fails to improve large or it improves tiny.
- **H6 (split-K hurts tiny).** A two-pass split-reduction adds a launch + round-trip and
  therefore does **not** help the launch-bound tiny regime. *Falsified if* c007 improves
  tiny geomean over the best single-pass kernel. (Also gates whether c007 is ever built.)

Interpretation policy: attribute each eval's deltas to the tiny vs large regime; compare
large-case times to the roofline to decide remaining headroom; don't overspend evals on
the tiny regime once it's launch-floor-bound.

---

## 6. Stopping criteria (convergence)

Stop and stop-tuning when **any** of:

1. **Roofline + floor reached:** large-case time is close to the 100–145 µs HBM floor
   *and* tiny-case time is at the launch-latency floor (single fused launch), i.e. no
   plausible mechanism remains.
2. **Geomean convergence:** two or three successive candidates fail to improve the feedback
   geomean by a meaningful margin (≲ ~2%), i.e. diminishing returns.
3. **Budget guard:** approaching the evaluation budget (≤ a few of the 100 evals left) or
   the token soft limit (1.0M) — then lock in the best valid candidate and stop.
4. **All hypotheses resolved:** the scheduled/conditional candidates are exhausted and none
   beats the current best valid candidate.

On stop: identify the **best valid candidate** (highest feedback geomean with **all five**
workloads PASS), write `SEARCH_COMPLETE` with the rationale (which criterion, best ID, its
geomean, per-regime standing vs roofline/launch floor). Do **not** run `final` — it is
operator-approved only; note that the best candidate is ready for operator-approved final.

---

## 7. Evidence format (one JSON object per candidate, appended to `candidates.jsonl`)

Append exactly one line per evaluated candidate; never rewrite earlier lines. Schema:

```json
{
  "candidate_id": "c001",
  "parent_id": null,
  "source_path": "runs/candidates/c001/solution.py",
  "source_hash": "<sha256 of the immutable candidate source>",
  "design": "A: fused one-pass, one-program-per-row, BLOCK_SIZE=4096",
  "config": {"num_warps": 8, "num_stages": null, "rows_per_program": 1, "BLOCK_SIZE": 4096},
  "changed_vs_parent": "initial anchor",
  "hypothesis": "H1/H2: fused one-pass kernel passes all 5 and beats reference in both regimes",
  "static_checks": {"casts_fp32": true, "eps_on_meansq": true, "denom_H": true,
                    "no_centering": true, "mul_order_x_inv_w": true, "bf16_store": true,
                    "strides_ok": true, "grid_ok": true, "no_fallback": true},
  "per_workload": [
    {"uuid": "9d403d2b-7859-4dab-aaf4-12e53c555001", "batch_size": 15,    "pass": true, "speedup": null, "time_us": null},
    {"uuid": "14531e96-f6e6-4515-abb4-10855f72c80e", "batch_size": 64,    "pass": true, "speedup": null, "time_us": null},
    {"uuid": "841b0afa-80fa-449a-9e1d-f294da92d02f", "batch_size": 14509, "pass": true, "speedup": null, "time_us": null},
    {"uuid": "33bf737d-3b37-4e38-be80-ea39b4b46ae6", "batch_size": 7,     "pass": true, "speedup": null, "time_us": null},
    {"uuid": "f0f508c3-e880-4ec8-b8be-1062db313d36", "batch_size": 14418, "pass": true, "speedup": null, "time_us": null}
  ],
  "all_pass": true,
  "geomean_speedup": null,
  "regime_notes": {"tiny": "…", "large": "… vs ~100-145us roofline"},
  "decision": "keep|reject|new-best|baseline",
  "next_action": "c002: num_warps sweep",
  "cumulative_evals": 1,
  "skill_usage": "none",
  "tokens_note": "optional running token observation"
}
```

Field rules:
- `speedup`/`time_us` filled from the evaluator's actual output (units as reported); leave
  `null` only if the evaluator does not report that field.
- `geomean_speedup` = geometric mean of the five per-workload speedups (only meaningful if
  `all_pass` is true; if any FAIL, mark `decision: "reject"` and geomean as reported/`null`).
- `parent_id` names the lineage parent (Section 3); `changed_vs_parent` states the single
  variable changed.
- `cumulative_evals` is the running count of feedback evaluations used (max 100).
- `decision`: `new-best` iff `all_pass` and geomean strictly beats the prior best valid
  candidate; else `keep`/`reject`/`baseline` as appropriate.

---

## 8. Turn-by-turn execution loop (what each future turn does)

1. Pick the next candidate ID per Section 3 (respecting conditionals/contingencies).
2. Implement its single-variable change in `solution/solution.py`; snapshot to
   `runs/candidates/cNNN/solution.py`; record source hash.
3. Run the Section 4 static checklist; fix statically before spending an eval.
4. `./scripts/evaluate_candidate.sh feedback cNNN` (once).
5. Parse results; append one `candidates.jsonl` record (Section 7); update the best-valid
   tracker; note tokens/evals used.
6. Decide next action via hypotheses (Section 5) and stopping criteria (Section 6).
7. On convergence/budget: write `SEARCH_COMPLETE`; do **not** run `final`.

---

## 9. First concrete step (next turn)

Implement **c001** (design A) exactly per Sections 1–4: fused one-pass, one-program-per-row,
`BLOCK_SIZE=4096`, fp32 accumulate, exact reference numerics, explicit strides, no mask on
the hidden axis, `num_warps=8`, single bf16 store, no fallback path. Run the static
checklist, evaluate once on feedback, and record the anchor/baseline. Tune only after c001
passes all five workloads.

---

## 10. Decision log

### c001 — DONE (eval #1). Baseline. All 5 PASS, geomean **4.55x**.
Evaluator ground truth (previously unseen): device **A800-SXM4-80GB**, tolerance
`atol=0.01 rtol=0.01 matched_ratio=0.99`, `warmup=3 iters=100` (so autotune warmup should
be excluded → design B remains viable). Per-workload:
- tiny b7/15/64: sol **~39–40 µs**, **2.72–2.80x** vs ref. Bit-exact (abs=rel=0).
- large b14418/14509: sol **~163–164 µs**, **9.57–9.62x** vs ref, abs=3.12e-2 rel~7.8e-3
  (well inside tol). Roofline: 226 MB / 2.0 TB/s ≈ **113 µs** ⇒ c001 large ≈ **69%** of
  peak BW ⇒ real headroom for the large regime (H2 confirmed with room).

Hypothesis status: **H1 confirmed** (fused kernel beats reference in all 5). **H2 confirmed
with headroom** (large ≈69% of roofline).

Read on regimes:
- Tiny cases are floored at ~39–40 µs regardless of `batch_size` (7→64 nearly identical),
  i.e. **launch/fixed-overhead bound**, essentially independent of the kernel body. Little
  to gain here; avoid overspending evals (do NOT build split-K / design D unless something
  changes — H6 stands).
- Large cases carry the optimization weight and dominate geomean upside. Priority = push
  the large regime toward the 113 µs floor.

### Next: c002 — `num_warps` sweep, try **16** first (parent c001).
Rationale: at `num_warps=8`, each program (BLOCK_SIZE=4096 fp32) uses 8 elems/thread;
more warps (16 → 4 elems/thread) can raise per-SM memory-level parallelism and better
saturate HBM on the large regime (H3). Change **only** `num_warps=8→16`; new ID c002.
If 16 wins, c003 confirms trend / tries 32 or backs off; if 16 loses, c003 tries 4. Then
c004 tunes `num_stages` at the best warps. Row-blocking (c005) is the main lever if warp/
stage tuning stalls below roofline. Stop per Section 6 when large nears ~113 µs or geomean
converges.

### c002 — DONE (eval #2). num_warps 8→16. All 5 PASS, geomean **4.57x** (≈flat).
Sol times essentially unchanged vs c001: tiny ~40.6–41 µs (identical, launch-bound),
large **164.0/165.1 µs** vs c001's 163.4/164.5 µs — i.e. **no real large-regime win** from
doubling warps. The geomean nudge (4.55→4.57) is ref-timing variance (ref b14509 measured
1.64 ms here vs 1.58 ms in c001), not a solution improvement. **H3 not supported**: at
BLOCK_SIZE=4096 a single program already issues enough in-flight loads that warp count
isn't the bandwidth lever. Large still stuck at ~69% of the ~113 µs roofline.

Read: with one program per row, each SM runs few rows concurrently and the bottleneck is
**per-block memory-level parallelism / occupancy**, not warp count. Doubling warps within a
block doesn't add independent rows in flight. The lever is likely **more concurrent rows
per SM** (occupancy) — which one-program-per-row already maximizes at 14.5k blocks — OR
better load scheduling. Since warps are flat, next probe the other end (num_warps=4) to
confirm the plateau, then pivot to the structural lever: **row-blocking (design C)** and/or
vectorized wider loads, which change the load/issue pattern rather than just warp count.

### Next: c003 — `num_warps` sweep low end: **num_warps=4** (parent c001, the best flat baseline).
Rationale: complete the warp sweep to confirm the plateau. 4 warps → 32 fp32 elems/thread,
fewer threads/block → potentially more blocks resident per SM (higher occupancy) which is
the plausible large-regime lever. Change **only** `num_warps=8→4`; new ID c003. If also
flat, warp count is confirmed a non-lever and c004 pivots to row-blocking (design C).
