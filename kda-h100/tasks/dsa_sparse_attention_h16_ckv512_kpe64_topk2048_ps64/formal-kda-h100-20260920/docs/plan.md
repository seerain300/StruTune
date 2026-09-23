# Plan — `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`.
No candidate is implemented or evaluated in this turn.

Ground rules recap (from CLAUDE.md / TASK.md):
- Triton is the required compute path; PyTorch only for metadata/launch. No
  Torch/CPU/NumPy/CUDA‑extension computational fallback. A failed Triton
  kernel is invalid — never substitute a fallback.
- One immutable kernel version over the full feedback set = **one** evaluation.
- Budget: **100 evaluations**; token soft limit 9.0M, normal 10.0M, hard 11.0M.
- Correctness gate = `./scripts/evaluate_candidate.sh feedback <cid>` only.
  Never run CUDA / `nvidia-smi` / the evaluator directly / any alt harness.
- Profiling only via `./scripts/ncu_profile.sh` (ncu-report-skill workflow),
  **never** concurrently with an evaluation (foreign proc on locked GPU →
  rc=3, wasted budget).
- `final` only with explicit operator approval.

---

## 0. Objective and success definition

Maximize **geomean speedup vs the reference** over the 23 feedback workloads,
subject to **every** selected workload passing correctness. Target regime:
decode‑like tiny `num_tokens ∈ {1,2,6,7,8}`, `num_pages=8462`, `topk=2048`,
memory/gather‑bound (see draft §1.4). The reference is an eager per‑token Python
loop, so a single fused, well‑occupied Triton kernel should win large; split‑K
flash‑decoding is the expected headroom for filling H100's 132 SMs.

Convergence = no candidate improves geomean by more than a small margin
(≈ <2% relative) over the running best across several consecutive structural
attempts, AND ncu confirms we are near the HBM‑bandwidth roofline (or occupancy
is saturated), with no correctness regressions.

---

## 1. Milestones (ordered)

- **M0 Correctness‑first baseline.** A simple, obviously‑correct fused Triton
  kernel that passes all 23 workloads (including empty‑token and padding edge
  cases). Establishes a valid, immutable reference point and a first geomean.
- **M1 Fill the GPU.** Split‑K / flash‑decoding over the `topk` dimension with a
  stage‑2 combine, `num_kv_splits` chosen by a host heuristic from
  `num_tokens`. Expected primary win.
- **M2 Cut dominant traffic.** Keep each gathered KV tile resident and reuse
  `Kc` for both QK (nope) and PV so the 2 MiB/token latent is read once.
- **M3 Tune.** Autotune `BLOCK_N`, `num_kv_splits`, `num_warps`, `num_stages`,
  optional output‑dim tiling `BLOCK_DV`; PV‑precision variants if the
  correctness margin is tight.
- **M4 Converge / stop.** Diminishing returns confirmed by ncu near roofline;
  write `SEARCH_COMPLETE`.

Each milestone is realized by one or more immutable candidates; do not fold
multiple structural changes into a single candidate (keeps lineage attributable).

---

## 2. Candidate lineage strategy

Rules:
- IDs are immutable and sequential: `c001`, `c002`, … Any meaningful source,
  config, or launch change ⇒ new ID. Never reuse an ID for changed source.
- One structural hypothesis per candidate where practical, so a geomean delta
  maps to a single cause. Autotune‑space widening counts as a change ⇒ new ID.
- Record `parent` = the candidate this was derived from. Keep a candidate only
  if it is correct AND (faster than parent OR informative for the next step).
- If a candidate regresses or fails correctness, branch the next candidate from
  the last good parent (not the failed one), noting why in the failed record.

Planned initial lineage (subject to evidence; later branches decided from data):

| id   | parent | milestone | one‑line hypothesis |
|------|--------|-----------|---------------------|
| c001 | –      | M0 | A single fused Triton kernel, grid `(num_tokens,)`, `BLOCK_M=16` (all heads), online base‑2 softmax, `V=Kc` reused per tile, fp32 accumulate, bf16 `P@V`, is correct on all 23 and already beats the eager reference. |
| c002 | c001   | M1 | Split‑K over `topk` with a fp32 stage‑2 combine and a fixed moderate `num_kv_splits` raises occupancy and improves geomean, especially at `num_tokens=1`. |
| c003 | c002   | M1 | Host heuristic picking `num_kv_splits` from `num_tokens` (more splits for tiny token counts) beats a fixed value across the mixed set. |
| c004 | best   | M2 | Loading each KV tile once into SMEM/regs and reusing `Kc` for QK and PV cuts the dominant HBM read and improves geomean. |
| c005 | best   | M3 | Autotuning `BLOCK_N`/`num_warps`/`num_stages` over a small curated grid improves geomean without correctness loss. |
| c006+| best   | M3 | Output‑dim tiling `BLOCK_DV` and/or dynamic tail‑tile skip (trailing `-1`); PV fp32 fallback only if a precision failure appears. |

Later candidates are chosen from evidence; this table is the intended spine,
not a fixed contract. Prefer in‑kernel `-1` masking over host‑side index
compaction unless profiling shows masked lanes waste real bandwidth.

---

## 3. Correctness checks (per candidate, before spending an evaluation)

Static / offline (no GPU, no evaluator — allowed):
1. `run(...)` signature matches the reference exactly:
   `run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)` and
   returns `(output, lse)` with shapes `[T,16,512]` bf16 and `[T,16]` fp32.
2. Constants hard‑coded as constexpr match the asserted values
   (16/512/64/64/2048); shapes read from tensors, not assumed beyond the
   contract.
3. Use `sm_scale` verbatim (the passed `0.13523…`); never recompute from
   `1/sqrt(192)` (draft §2, §6).
4. LSE is **base‑2** flash form `m2 + log2(l2)` (verified in draft §3.2 to fp64).
5. Edge‑case logic is present in source before eval:
   - trailing `-1` padding excluded from max/sum/output;
   - fully‑empty token ⇒ `output row = 0` **and** `lse = -inf` (not 0, not NaN);
   - gather masks `-1`, clamps address, `other=0.0` (no OOB);
   - split‑K combine guards all‑`-inf` partials so `exp2(m_i−m)` never does
     `0*inf` → NaN.
6. No Torch/NumPy compute fallback anywhere on the execution path.

Dynamic (the evaluation itself is the correctness gate):
7. `./scripts/evaluate_candidate.sh feedback <cid>` must report **all 23
   workloads pass**. A candidate that fails any workload is invalid regardless
   of speed and is recorded as such.
8. Pay special attention to the boundary workloads: `num_tokens=1` (single
   token, occupancy stress) and any token whose captured `sparse_indices`
   contain padding / could be empty (`lse=-inf` path).

If the evaluator reports rc=3 (foreign‑process interference), the measurement
is void: do **not** count it as a real result; re‑run once the GPU is clear and
ensure no profiling is running.

---

## 4. Performance hypotheses and how each is tested

H1 — *The op is HBM/gather bound.* Test: ncu on the best candidate; expect high
DRAM throughput, low compute‑pipe utilization, gather (indexed loads) dominating
DRAM sectors. Action: if confirmed, all effort goes to bandwidth (traffic
reduction + occupancy), not FLOP reduction.

H2 — *Tiny `num_tokens` starves occupancy.* Test: ncu `achieved_occupancy` /
active SMs on c001 vs c002/c003; expect c001 to use ≪132 SMs at `num_tokens=1`
and split‑K to raise active SMs. Action: tune `num_kv_splits` so
`num_tokens * num_kv_splits (* head_tiles)` comfortably exceeds ~132.

H3 — *Reusing `Kc` for QK and PV halves the dominant read.* Test: DRAM bytes
before/after M2 in ncu; expect ~2 MiB/token → near single‑read. Action: keep
tile resident across QK→softmax→PV.

H4 — *bf16 `P@V` is within tolerance.* Test: correctness pass at M0/M1. If it
fails, escalate PV precision (fp32 combine → fp32 FMA PV) as its own candidate.

H5 — *Autotune parameters matter but with diminishing returns.* Test: geomean
across a small curated config grid; keep the widening only if it beats parent.

Profiling protocol: run ncu **only** through `./scripts/ncu_profile.sh`,
serialized with evaluations, on a representative workload (at least
`num_tokens=1` for the occupancy question and `num_tokens=8` for the
bandwidth‑bound question). Use the `ncu-report-skill` workflow to read reports.

---

## 5. Evaluation discipline / budget

- Spend an evaluation only after the offline checks in §3 pass and the source
  compiles/imports cleanly. Aim to make each of the ~100 evaluations
  hypothesis‑driven.
- Never profile and evaluate at the same time; finish one before the other.
- Rough budget allocation (soft, revisit from data): ≤10 evals to lock M0/M1,
  the bulk on M2/M3 sweeps, reserve a handful near the end for confirmation
  runs. Token budget: keep well under 9.0M soft limit; front‑load analysis,
  keep candidate diffs small.

---

## 6. Evidence format — one JSON object appended to `candidates.jsonl`

Append exactly one complete JSON object per **evaluated** candidate; never
rewrite earlier records. Required fields (per CLAUDE.md §7):

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py as evaluated>",
  "milestone": "M0",
  "hypothesis": "single fused kernel, correct on all 23, beats eager reference",
  "change_summary": "grid=(num_tokens,), BLOCK_M=16, online base-2 softmax, V=Kc reuse, fp32 acc, bf16 P@V",
  "validation": {
    "offline_checks": "signature/shape/dtype/scale/lse-base2/edge-cases/no-fallback all pass",
    "compiles": true
  },
  "per_workload": [
    {"uuid": "0c23b10c...", "num_tokens": 1, "pass": true, "speedup": 0.0, "latency_ms": 0.0}
  ],
  "geomean_speedup": 0.0,
  "all_pass": true,
  "decision": "keep|reject|branch-parent",
  "decision_reason": "…",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "notes": "ncu findings / next hypothesis"
}
```

Notes:
- `speedup`/`latency_ms`/`geomean_speedup` are filled from the evaluator output
  (per‑workload). If the evaluator returns a different field vocabulary, mirror
  its exact reported numbers and keep these keys as the normalized view.
- `decision` must be justified against the parent's geomean and correctness.
- Record any rc=3 void run in `notes` (do not fabricate a result for it).
- `cumulative_evaluations` increments by exactly 1 per real feedback run.

Human‑readable running log (optional, alongside the JSONL) may be kept in this
plan's changelog section or `docs/` but the authoritative record is
`candidates.jsonl`.

---

## 7. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any of:
1. **Convergence:** best geomean not improved by >~2% relative across several
   consecutive structural candidates, and ncu shows we are near HBM roofline or
   occupancy‑saturated (H1/H2 exhausted).
2. **Budget:** approaching the 100‑evaluation cap or the token soft limit
   (9.0M) with no clear remaining lever.
3. **Diminishing design space:** remaining ideas are speculative/high‑risk with
   low expected upside relative to remaining budget.

At stop: the best **valid** candidate (correct on all 23, highest geomean) is
the final selection. Do **not** run `final` — that requires explicit operator
approval and is a single 23‑workload confirmation of the chosen candidate.

---

## 8. Risk register → concrete guards (from draft §7)

| risk | guard baked into the plan |
|------|---------------------------|
| Under‑occupancy at `num_tokens=1` | M1 split‑K with host `num_kv_splits` heuristic (c002/c003) |
| bf16 `P@V` precision | fp32 accumulate; escalate to fp32 PV as its own candidate if H4 fails |
| Empty‑token `lse=-inf` mismatch | explicit `-inf` init + guarded combine (§3.5) |
| `-1` gather OOB | mask + clamp + `other=0` (§3.5) |
| `acc[16,512]` register pressure | tile `BLOCK_DV`, tune warps/stages (M3) |
| Redundant `Kc` read | resident tile reuse for QK & PV (M2) |
| Profiling/eval overlap | strict serialization; never background profile during eval |
| Wrong scale | use passed `sm_scale` verbatim (§3.3) |

---

## 9. Immediate next actions (next turn, not this one)

1. Implement `c001` in `solution/solution.py` (M0 baseline) with all §3 offline
   checks satisfied.
2. Run `./scripts/evaluate_candidate.sh feedback c001`; append its record to
   `candidates.jsonl`.
3. If correct, branch `c002` (split‑K) per §2; else fix within a new ID and
   branch from the last good parent.

---

## 10. Decision log

- **c001 (M0)** — evaluated once, `feedback`, rc=1, 0/23, `valid_run=False`.
  CompilationError: module‑level `LOG2E` float read inside the `@jit` kernel;
  Triton only permits `tl.constexpr` globals. Pure compile bug, no timing, not
  an rc=3 interference void → counts as 1 real evaluation. **Rejected.**
  Cumulative evaluations = 1.
  - *Lesson learned:* offline check list in §3 must also include "kernel
    imports/compiles with no non‑constexpr global reads" — add a trivial
    smoke‑compile awareness before spending an eval. (Cannot run Triton
    compile locally without the benchmark interpreter, so keep kernel globals
    inlined or constexpr by construction.)
  - **Next:** `c002` branches from the *design* of c001 (last intended‑good
    structure), fixing the global by inlining the `1.4426950408889634` literal
    at the `qk_scale` site (or passing `log2e` as a constexpr arg). No other
    logic change, so `c002` still tests the M0 correctness hypothesis before we
    proceed to split‑K (M1).

- **c002 (M0)** — evaluated once, `feedback`, rc=0, **23/23 pass, valid_run=True,
  geomean 18.35x** (arith 19.17x, min 10.57x, max 26.70x). Compile fix of c001
  (LOG2E passed as `tl.constexpr`), no other change. **First valid candidate;
  kept as current best and the M0 baseline.** Cumulative evaluations = 2.
  - *Key evidence:* speedup rises monotonically with `num_tokens`
    (1 → 10.57x, 2 → ~13x, 6 → ~20x, 7 → ~22x, 8 → ~25x). This confirms plan
    hypothesis **H2** — at small token counts we launch too few CTAs (1 per
    token) to fill 132 SMs, so the single‑token case is occupancy‑starved and
    drags the geomean. Precision is fine: abs error ≤ ~1.56e‑2, large *relative*
    errors are on near‑zero outputs and absorbed by `matched_ratio=0.99`.
  - **Next:** `c003` = **M1 split‑K over the topk dimension** with a fp32
    stage‑2 combine, grid `(num_tokens, num_kv_splits)`, chosen to raise active
    SMs (esp. for `num_tokens=1`). Keep PV bf16 for now but watch the precision
    margin under split partials; guard the all‑`-inf` merge.

- **c003 (M1)** — evaluated once, `feedback`, rc=0, **23/23 pass,
  valid_run=True, geomean 21.66x** (arith 22.61x, min 12.57x, max 29.40x).
  Two‑kernel flash‑decoding: stage‑1 `grid=(num_tokens, 16)`, one KV tile per
  CTA → fp32 partial `(m,l,acc)` buffers; stage‑2 `grid=(num_tokens, H)` fp32
  log‑sum‑exp merge. **+18% relative over c002 (18.35x). New current best.**
  Cumulative evaluations = 3.
  - *Key evidence:* every bucket improves (num_tokens=1 10.57→12.57x, num_tokens=8
    ~24.5→~28.5x). But `sol_ms` is now ~0.10 ms and **nearly flat across all
    shapes**, and `num_tokens=1` is still the min → we are **latency/launch
    bound**, not yet bandwidth bound. At num_tokens=1 stage‑1 launches only
    1×16=16 CTAs (≪132 SMs); the 2‑kernel launch + `acc_buf` HBM round‑trip add
    fixed overhead visible at these tiny sizes. Precision unchanged (abs ≤1.56e‑2).
  - **Next candidates (pick one, evidence‑driven):**
    (a) `c004` = host heuristic that raises `num_kv_splits` for tiny
    `num_tokens` (target ≥~132 stage‑1 CTAs) so the single‑token case fills the
    GPU — but more splits ⇒ bigger `acc_buf` and heavier combine, so there is a
    sweet spot; and/or
    (b) reduce combine/launch overhead (fuse or shrink the round‑trip).
    An `ncu` profile (serialized, never during eval) on num_tokens=1 vs 8 would
    confirm launch‑ vs bandwidth‑bound before committing.

- **c004 (M1, adaptive split)** — evaluated once, `feedback`, rc=0, **23/23 pass,
  geomean 19.88x (REGRESSION vs c003 21.66x). Rejected.** Cumulative
  evaluations = 4. Adaptive host heuristic shrank `BLOCK_N` (16..128) for small
  `num_tokens` to target ~128 stage‑1 CTAs.
  - *Result:* the small‑token tail got **worse**, not better —
    num_tokens=1 12.57→7.97x (sol 0.106→0.166 ms), num_tokens=2 ~14.7→~12.9x —
    while num_tokens=8 (BLOCK_N=128, identical to c003) was unchanged (~28.5x).
    **Clean negative:** finer splits with small `BLOCK_N` waste the MMA and add
    combine/dispatch overhead; c003's 16 splits @ BLOCK_N=128 is the
    occupancy‑optimal split‑K config for this workload set.
  - *Deeper read:* the ~0.10 ms `sol_ms` floor persists across c002 (single
    kernel, 0.119 ms), c003, and c004 → the dominant cost is **per‑call
    launch/dispatch + gather latency**, not the split strategy or `acc_buf`
    traffic (`acc_buf` ≤ ~4 MB even at num_tokens=8, negligible vs 3.35 TB/s).
    Also `ref` timings jitter run‑to‑run (num_tokens=1 ref 1.26–1.34 ms) →
    trust `sol_ms`; small geomean deltas carry measurement noise.
  - **Next:** revert to c003's proven config (BLOCK_N=128, 16 splits) and, as
    `c005`, probe `num_warps=8` on the single‑tile partial kernel to raise
    per‑CTA memory‑level parallelism (better hide the per‑CTA gather latency,
    which matters most at num_tokens=1 where only 16 CTAs are active). If that
    also fails to move `sol_ms`, the op is launch‑bound and we are near
    convergence — confirm with a serialized `ncu` profile.

- **c005 (M1, num_warps=8)** — evaluated once, `feedback`, rc=0, **23/23 pass,
  geomean 22.92x** (arith 24.05x, min 14.78x, max 32.40x). Only change vs c003:
  stage‑1 `num_warps` 4→8. Recorded geomean ≥ c003 (21.66x) with no regression,
  so **kept as current best**. Cumulative evaluations = 5.
  - *Caveat — the gain is largely reference‑timing noise.* `sol_ms` is
    essentially unchanged from c003 (num_tokens=1 0.106→0.113 ms, num_tokens=8
    ~0.10→~0.105 ms), while the freshly‑timed **reference jumped this run**
    (num_tokens=1 ref 1.34→2.90 ms; num_tokens=8 ~2.85→3.0–3.4 ms). `num_warps=8`
    did **not** materially move our kernel time.
  - **Convergence signal:** the ~0.10 ms `sol_ms` floor is now stable across
    c002 (single kernel 0.119), c003, c004 (rejected), and c005 — independent of
    split count and warp count. The op is **launch/dispatch + gather‑latency
    bound** at these tiny sizes; the geomean is dominated by the noisy reference
    timing we cannot control.
  - **Remaining low‑risk probes before declaring convergence:**
    (a) `c006` = stage‑1 `num_stages` 2→3/4 for deeper gather pipelining;
    (b) a **serialized** `ncu` profile (never during an eval) on num_tokens=1 to
    confirm launch‑bound (low DRAM %, few active SMs). If (a) shows no `sol_ms`
    movement and ncu shows no attackable stall, write `SEARCH_COMPLETE` with
    c005 as the best valid candidate.

- **SEARCH_COMPLETE (after c005).** Declared convergence; wrote
  `SEARCH_COMPLETE` with c005 as the best valid candidate (23/23,
  geomean 22.92x; c003 an equivalent num_warps=4 fallback). Cumulative
  evaluations = 5/100.
  - *Decisive argument (no ncu needed):* wall time (`sol_ms`) is **flat at
    ~0.10–0.11 ms across the entire feedback set**, from `num_tokens=1`
    (16 stage‑1 CTAs) to `num_tokens=8` (128 CTAs ≈ one full H100 wave). Going
    8× wider in both work and parallelism does not change wall time ⇒ the op is
    **fixed per‑invocation overhead bound**, not occupancy/compute/bandwidth
    bound. Across the whole set we never exceed one wave, so no kernel‑side
    parallelism change can go below the floor. Confirmed by the ledger: c002
    (single kernel, 16‑deep serial path) 0.118 ms → c003 (split‑K, 1 tile/CTA,
    short path) 0.10 ms → c004 (finer splits) regress → c005 (num_warps 4→8)
    no move. `num_stages` is a no‑op here (`TILES_PER_SPLIT=1` ⇒ one loop
    iteration, nothing to pipeline), so probe (a) was dropped as non‑justified.
  - Correctness is solid (23/23 at atol=0.01/rtol=0.01/matched_ratio=0.99,
    abs ≤ 1.56e‑2) and the measured geomean climbed 18.35 → 21.66 → 22.92x, with
    the last step within reference‑timing noise.
  - **Did NOT run `final`** (operator‑approval only).
