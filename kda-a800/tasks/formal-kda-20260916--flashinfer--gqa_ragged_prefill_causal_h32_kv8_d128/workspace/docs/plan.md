# Executable Optimization Plan — `gqa_ragged_prefill_causal_h32_kv8_d128`

Target: **NVIDIA A800 (`sm_80`, Ampere)**. Benchmark family: FlashInfer.
Entry point: `solution/solution.py` exposing `run(q, k, v, qo_indptr, kv_indptr, sm_scale) -> (output, lse)`.
Primary compute must be **Triton**; PyTorch only for metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-ext
computational fallback. This plan operationalizes `docs/draft.md`; do not restate all semantics here.

Ground rules carried from `CLAUDE.md` / `TASK.md`:
- Only signal is `./scripts/evaluate_candidate.sh feedback <cid>` (5 fixed workloads = 1 evaluation).
- Never run CUDA/profiler/nvidia-smi/evaluator directly; never `final` without operator approval.
- Candidates are immutable; any meaningful source/config/launch change ⇒ new candidate ID.
- Budget: 100 evals; token soft 1.0M / normal 1.5M / absolute 1.65M.

---

## 0. Workflow discipline (per candidate)

For every candidate `cNNN` the loop is strictly:

1. **Edit source** `solution/solution.py` to the new immutable version (one change-set at a time).
2. **Static self-check** (offline, no GPU): run the pre-eval checklist in §4. If any check fails, fix
   before evaluating — do not spend an evaluation on a known-broken version.
3. **Freeze** the source: record `sha256(solution/solution.py)` (see §7 evidence). This hash *is* the
   candidate identity; if the file changes afterward, it is a new candidate ID.
4. **Evaluate exactly once**: `./scripts/evaluate_candidate.sh feedback cNNN`.
5. **Append** one JSON object to `candidates.jsonl` (schema in §7). Never rewrite prior records.
6. **Decide** keep/revert per §6, pick the next lineage step, repeat.

Source-version management (immutability without a VCS):
- The live file under evaluation is always `solution/solution.py`.
- Before overwriting it for the next candidate, snapshot the exact evaluated bytes to
  `docs/candidates/cNNN_solution.py` (archival copy for lineage/diff/rollback). This snapshot is
  documentation only; the evaluator reads `solution/solution.py`. Snapshotting *after* recording the
  hash guarantees the archived copy matches the evaluated hash.
- If the evaluator instead expects the source under `runs/candidates/cNNN/` (its harness decides), that
  is handled entirely by `evaluate_candidate.sh`; we only ever author `solution/solution.py` and never
  touch evaluator/controller internals.

One knob per candidate where practical, so each speedup/regression is attributable. Bundling is allowed
only for tightly coupled changes that cannot be evaluated independently (noted explicitly in the record).

---

## 1. Target design (what `c001` builds)

A single **fused, ragged (varlen), causal GQA flash-attention prefill** Triton kernel, launched **once**,
with a lean host wrapper and **zero host↔device syncs** (grid path A from the draft). Rationale: 4/5
feedback workloads are minuscule (≤35 tokens, some 1×1) and thus **overhead-bound**; the win comes from
collapsing the reference's per-batch Python loop + ~dozen small torch kernels + repeated `.item()` syncs
into one launch. The 982-token/15-seq case additionally rewards a real fused varlen kernel over the
Python batch loop.

Kernel contract (numerics from draft §3; base-2 flash):
- Grid: `(pid_m, pid_h, pid_b)` = `(cdiv(max_grid_M, BLOCK_M), num_qo_heads=32, batch=len_indptr-1)`.
  - Path A `max_grid_M`: use `cdiv(total_q, BLOCK_M)` as a safe upper bound on any single seqlen (a
    sequence can be at most `total_q` long). Programs beyond their sequence's own `Nq` early-exit after
    reading two int32 offsets from `qo_indptr`. Zero syncs.
- Each program: load one `[BLOCK_M,128]` q tile for head `pid_h` inside sequence `pid_b`; kv head =
  `pid_h // 4`; online-softmax loop over `[BLOCK_N,128]` k/v tiles within `[kv_start,kv_end)`.
- Causal boundary: query local `i` attends kv local `j` iff `j <= i + delta`, `delta = Nk - Nq`. Skip kv
  tiles fully above the block's max `q_eff`.
- Accumulate `acc` fp32 `[BLOCK_M,128]`, track `m`,`l` in the `s2 = q·k*sm_scale*log2(e)` domain.
- Write `output = (acc/l)` → bf16; `lse = m + log2(l)` fp32. Guard `Nk==0`/`l==0` rows → `output=0`,
  `lse=-inf` (robustness for the 21-wl final; not exercised by feedback).
- `tl.dot(bf16,bf16)->fp32` accumulate (Ampere HMMA). Scale applied to logits after the dot (avoid extra
  bf16 rounding of q).

Host wrapper (`run`):
- Read shapes only; no `.item()` on data-dependent values (path A).
- `output = torch.empty([total_q,32,128], bf16)`, `lse = torch.empty([total_q,32], fp32)`. If any q row
  could be uncovered by the grid, switch to `zeros`/`full(-inf)` prefill (cheap at these sizes). With the
  full-coverage grid, `empty` is fine.
- Pass indptr pointers; strides explicit; `sm_scale`, `BLOCK_*`, head counts as (mostly `constexpr`) args.
- Fold `sm_scale*log2(e)` into a single scalar passed to the kernel.
- Return `(output, lse)` in exact declared order/dtype.

Fixed starting config for `c001`: `BLOCK_M=32`, `BLOCK_N=64`, `BLOCK_D=128`, `num_warps=4`,
`num_stages=2`, no autotune. Correctness first; tuning later.

---

## 2. Candidate lineage strategy

Linear-then-branch. Each node = one immutable candidate ID. Keep a node only if it passes all 5 workloads
**and** does not regress geomean vs the current best (§6). Lineage is recorded via `parent` in each JSON
record.

Phase 0 — Correctness baseline
- **c001**: minimal correct fused kernel per §1 (per-head program, grid path A, fixed small config,
  bf16-in/fp32-acc). Goal: all 5 pass; establish baseline geomean. This is the make-or-break correctness
  gate; do not tune until it passes.
- If c001 fails correctness: diagnose from the failing workload signature (§4/§5), spin **c002** as a
  corrected version (not a tuning change), repeat until a correct baseline exists. Numerics/masking bugs
  take priority over everything.

Phase 1 — Overhead / launch levers (biggest expected geomean movers, per H1/H2)
- **Grid-sync A vs B**: candidate for path B (one `.item()` `max_seqlen`, tighter grid for wl5) to test
  H2 (path A should win on tiny wls). Keep whichever geomean is higher.
- **Host-wrapper leanness**: remove any residual syncs/contiguity copies/allocations; ensure single
  launch. Candidate only if a concrete change is identified (e.g., `empty` vs prefill, dropping a
  `.contiguous()`).
- **`constexpr`-ing** everything static (head counts, gqa_ratio, head_dim, strides where constant) to cut
  per-launch arg/compile overhead — one candidate if it changes generated code meaningfully.

Phase 2 — Kernel-shape levers (second-order per H3)
- **Tile sizes**: sweep `BLOCK_M ∈ {16,32,64}`, `BLOCK_N ∈ {32,64,128}` as separate candidates, guided by
  the size profile (tiny cases favor small `BLOCK_M` + single kv tile; wl5 favors `BLOCK_M=32/64`,
  `BLOCK_N=64`). Pick the best single fixed config.
- **`num_warps ∈ {2,4,8}`, `num_stages ∈ {1,2,3}`**: sweep as candidates; low kv depth ⇒ expect
  `warps=4`, `stages=1–2` best. Deep pipelining unlikely to help.
- **GQA head-grouping**: one program per kv-head group processing all 4 sibling q-heads (load K/V once per
  kv tile, pack a `[4·BLOCK_M,·]` or `[4,BLOCK_M,·]` M dimension). Reduces program count on 1-token cases;
  a single candidate to test whether it beats per-head. Keep only if it wins.

Phase 3 — Optional / conditional
- **Autotune** keyed on a coarse `max_seqlen` bucket, only if a single fixed config cannot serve both the
  tiny cases and wl5 well. Weigh first-call benchmarking overhead (counts against timing) vs gain; prefer
  fixed config. One candidate to A/B.
- **fp32-dot / `input_precision`**: only if bf16-in/fp32-acc is marginal on correctness (H4 says
  unlikely). Correctness rescue, not a perf lever.

Branching rule: always branch tuning candidates off the current best-correct node, never off a
known-worse or failing node. At most one active hypothesis per candidate.

---

## 3. Performance hypotheses (falsifiable, tied to candidates)

- **H1 (dominant):** one fused launch replacing the reference's per-batch loop + small torch kernels/syncs
  gives large speedup on all 5, maximal on the 1×1 cases (wl2/wl3, near-pure overhead). *Tested by c001
  geomean; expect >1 on every workload, largest on wl2/wl3.*
- **H2:** zero-sync grid (path A) beats one-sync tight grid (path B) on the tiny cases; extra early-exit
  grid rows for wl5 are cheaper than a `.max().item()` sync. *Tested by the A-vs-B candidate pair.*
- **H3:** tile/warp/stage tuning is second-order; a small fixed config (`BLOCK_M=32`, `BLOCK_N=64`,
  `warps=4`, `stages=2`) is near-optimal. *Tested by the Phase-2 sweep giving small deltas.*
- **H4:** bf16-in/fp32-acc passes tolerance; fp32-dot unnecessary. *Tested by c001 correctness.*
- **H5:** GQA head-grouping gives at most a minor gain (kv tiny ⇒ little reload to save), mainly on
  program-count for 1-token cases. *Tested by the grouping candidate.*

Explicit non-hypotheses (not pursued on `sm_80` at this scale): tensor-core occupancy tuning, deep
cp.async pipelines, warp specialization, persistent/split-KV kernels. The `KernelWiki` skill targets
Blackwell/Hopper and does **not** apply to A800; it will not be used. Profilers cannot be run here.

---

## 4. Pre-evaluation correctness checklist (offline, every candidate)

Run this reasoning gate before spending an evaluation. All must hold:

1. **Return contract:** returns `(output, lse)`; `output` bf16 `[total_q,32,128]`; `lse` fp32
   `[total_q,32]`; order exact.
2. **Base-2 LSE:** `lse = m + log2(l)` with `m,l` in the `s2 = (q·k)*sm_scale*log2(e)` domain. Closed-form
   1×1 check: single logit `s=(q·k)*sm_scale` ⇒ `lse = s*log2(e)`, `output = v` (softmax of one element).
   This directly validates wl2/wl3.
3. **Causal-with-delta:** allowed iff `kv_local <= q_local + delta`, `delta=Nk-Nq`. For feedback
   `delta=0` ⇒ standard `kv_local <= q_local`. Hand-trace seqlen 7 (wl1) row 0 sees only kv 0; row 6 sees
   kv 0..6. Verify `<` vs `<=` and the `+1` are consistent with the reference.
4. **GQA mapping:** kv head = `q_head // 4`; verify k/v pointer offset uses `pid_h//4` and correct
   head-stride (inner stride 128, head stride 128, token stride `heads*128`).
5. **Ragged confinement:** every program stays within `[q_start,q_end) × [kv_start,kv_end)`; no straddle
   across sequences; early-exit when `pid_m*BLOCK_M >= Nq`.
6. **Tail masks:** explicit M-mask (query tail) and N-mask (kv tail) with `other=0` on loads and masked
   logits `-inf` (so `p=0`). Trace non-pow2 seqlens 7 and 35.
7. **Degenerate guard:** `Nk==0` or `l==0` rows ⇒ `output=0`, `lse=-inf` (no div-by-zero NaN).
8. **No fallback:** compute path is Triton only; no torch attention math, no `repeat_interleave`
   materialization, no full-logits materialization.
9. **No stray sync (path A):** no `.item()`/`.cpu()`/`.tolist()` on data-dependent values in `run`.

---

## 5. Failure-diagnosis map (feedback signatures)

- **wl2 or wl3 fail (1×1), others pass** ⇒ LSE base-2 formula or single-element softmax/scale bug (check
  #2). These are the cleanest analytic oracles.
- **wl1 (7) or wl4 (35) fail, 1×1 pass** ⇒ causal off-by-one/tail-mask bug (checks #3/#6).
- **wl5 (982/15-seq) fails, single-seq pass** ⇒ ragged offset/straddle or per-batch grid mapping bug
  (checks #5/#1), or a per-sequence causal `delta` handling issue.
- **All fail** ⇒ contract/shape/dtype/return-order or a global scale/`log2(e)` error (checks #1/#2).
- **NaN/inf** ⇒ missing degenerate guard or masked-row handling (checks #6/#7).
- **Correct but slow** ⇒ overhead in `run` (syncs, copies, extra launches) or bad tile/grid — move to
  Phase 1/2 tuning, not a correctness fix.

Any correctness fix = new candidate ID; never mutate an evaluated version.

---

## 6. Keep / revert decision rule

For each evaluated candidate compare against the current **best-correct** node:
- **Reject (do not branch from it)** if any of the 5 workloads fails correctness — invalid regardless of
  speed.
- **Adopt as new best** if all 5 pass **and** geomean speedup improves beyond noise
  (treat < ~1–2% geomean change as noise; require a clear, repeatable improvement to adopt; if in doubt,
  keep the simpler/lower-risk version).
- **Reject** if all pass but geomean does not improve (or regresses); keep the parent as best, record the
  negative result, move to the next hypothesis.
- Prefer the **simplest** version among statistically-tied candidates (fewer knobs, no autotune, no
  sync).

Best-so-far is tracked explicitly in each record (`is_new_best`) so the final submission choice is
unambiguous.

---

## 7. Evidence format (one JSON object per evaluation, appended to `candidates.jsonl`)

Append-only; never edit prior lines. Schema:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_sha256": "<sha256 of the exact evaluated solution/solution.py>",
  "timestamp": "<ISO-8601>",
  "hypothesis": "Minimal correct fused varlen causal GQA flash kernel; single launch, zero sync.",
  "change_from_parent": "initial implementation",
  "config": {
    "grid_path": "A",
    "BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_D": 128,
    "num_warps": 4, "num_stages": 2,
    "gqa_grouping": "per_head",
    "matmul_precision": "bf16_in_fp32_acc",
    "autotune": false
  },
  "static_checks": {
    "return_contract": true, "base2_lse": true, "causal_delta": true,
    "gqa_mapping": true, "ragged_confinement": true, "tail_masks": true,
    "degenerate_guard": true, "triton_only": true, "no_host_sync": true
  },
  "validation": "offline checklist §4 all pass; 1x1 closed-form lse=s*log2(e) confirmed",
  "per_workload": [
    {"uuid": "b8f94163", "shape": "1seq/7", "correct": true, "speedup": 0.0},
    {"uuid": "bdc49f9e", "shape": "1seq/1", "correct": true, "speedup": 0.0},
    {"uuid": "f4c23a33", "shape": "1seq/1", "correct": true, "speedup": 0.0},
    {"uuid": "32f5e961", "shape": "1seq/35", "correct": true, "speedup": 0.0},
    {"uuid": "2328b031", "shape": "15seq/982", "correct": true, "speedup": 0.0}
  ],
  "all_correct": true,
  "geomean_speedup": 0.0,
  "decision": "adopt|reject|corrected-retry",
  "is_new_best": true,
  "cumulative_evaluations": 1,
  "skills_used": ["none: A800/sm_80 out of KernelWiki scope; profilers unavailable"],
  "notes": "raw evaluator output summary; anomalies; next step"
}
```

Rules:
- `source_sha256` recorded **before** evaluation and matches the archived `docs/candidates/cNNN_solution.py`.
- `per_workload.speedup`/`geomean_speedup` copied verbatim from the evaluator output; do not fabricate.
- `cumulative_evaluations` is monotonic across all candidates (the running eval count vs the 100 budget).
- `skills_used` documents skill consideration each candidate (expected: none, with the A800-scope reason).
- If evaluation errors (compile/launch failure), record `all_correct=false`, capture the error class in
  `notes`, and treat as a correctness failure (new ID for the fix).

---

## 8. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with the reason) when **any** holds:
1. **Convergence:** the best-correct geomean has not improved beyond noise (~1–2%) across the last ~3–4
   tuning candidates spanning the remaining meaningful levers (grid-sync, tiles, warps/stages, grouping),
   and no untested hypothesis with plausible upside remains.
2. **Design space exhausted:** all Phase 1–2 levers evaluated; Phase 3 either tested or judged
   not worthwhile (documented).
3. **Budget:** approaching the token soft limit (1.0M) with only marginal expected gains, or the 100-eval
   cap. Given the small overhead-bound design space, expect convergence in **far fewer than 100 evals**.

`SEARCH_COMPLETE` must state: chosen best candidate ID, its geomean, why further search is not
worthwhile, and remaining budget. Do **not** run `final` — that is operator-approval-only and out of
scope for this search loop.

---

## 9. Immediate next actions (next turn, not this one)

## 10. Progress log

### c001 — adopted (new best), 1 eval used
- Source sha256 `a0d4f6aa76f54922f896c4a74cd25ff7136ecb7ecc9f5111bc20a2e7bab4cca9`; archived
  `docs/candidates/c001_solution.py`.
- Result: **VALID 5/5**, geomean **13.89x** (arith 21.83x, min 8.76x, max 72.38x).
  - wl1 7-tok 9.58x, wl2 1-tok 8.88x, wl3 1-tok 8.76x, wl4 35-tok 9.58x, wl5 982/15-seq 72.38x.
- **H1 confirmed** (single fused launch beats the reference loop+kernels on every workload; largest
  multiplicative win on the big varlen case). **H4 confirmed** (bf16-in/fp32-acc passes; wl2/wl3
  abs<8e-7 match the 1×1 closed form exactly).
- Key signal: `sol_ms` is a near-constant floor (~0.065ms) for all four small workloads and only rises
  to 0.104ms at 982 tokens. The small cases are therefore **fixed launch/dispatch-overhead-bound**, not
  compute-bound. Tile-size/warp tuning (Phase 2) will move these almost nothing; the only lever that can
  raise the tiny-case speedups is shrinking that fixed launch overhead.

### Next candidate (planned): c002 — Phase-1 grid overhead
- Test **H2** and probe the ~0.065ms floor. Options ranked by expected impact:
  1. Reduce the launched grid's degenerate program count. Path A launches
     `cdiv(total_q,BLOCK_M) * 32 * batch` programs; for wl5 that is `31*32*15 = 14880` programs, most of
     which early-exit — this is the likely cause of wl5 being the only compute-visible case and may also
     add dispatch overhead on the small cases (still `1*32*1=32`). A tighter grid over `max_seqlen`
     (path B, one `.item()` sync) or a flattened/persistent program-id scheme reduces launched programs.
  2. Weigh path B's single host↔device sync against path A's zero-sync: on the tiny cases the sync may
     cost more than it saves (H2 predicts A wins there); on wl5 the tighter grid may help. One immutable
     candidate to A/B, keep the higher geomean.
- Because c001 already achieves double-digit geomean and the small-case floor is fixed overhead,
  remaining upside is bounded; if c002 (and one tile/warp probe) do not beat 13.89x beyond noise, the
  search is near convergence (plan §8) and should stop with `SEARCH_COMPLETE`.

---

### (original next-action list, now completed for c001)

1. Implement `c001` in `solution/solution.py` per §1 (fused varlen causal GQA flash, path A, fixed small
   config, bf16-in/fp32-acc). — done
2. Run the §4 offline checklist; fix statically until it passes. — done
3. Record `source_sha256`, archive to `docs/candidates/c001_solution.py`. — done
4. `./scripts/evaluate_candidate.sh feedback c001`. — done (geomean 13.89x, 5/5)
5. Append the §7 record; decide per §6; proceed to Phase 1. — done (adopted; c002 = Phase-1 grid)
