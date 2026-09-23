# Executable Plan — L2/043 Mamba-2 Chunk Scan with Segment Sum

Companion to `docs/draft.md`. This file is the *executable* optimization plan:
concrete kernel decomposition, sequential candidate lineage, correctness checks,
performance hypotheses, stopping criteria, and evidence format. No candidate is
implemented or evaluated in this step.

Target: NVIDIA **A800 / sm_80 (Ampere)**, 108 SMs. Compute must be **Triton**;
PyTorch only for shapes/allocation/launch. No Torch/CPU/NumPy/CUDA-extension
computational fallback. Ranking metric: **geometric-mean speedup** vs the fp32
reference, gated on every selected workload passing correctness.

---

## 0. Fixed facts to build against

Constants (from `task/definition.json`): `H=16`, `P=head_dim=64`, `N=state_size=256`,
`n_groups=1`, `Q=chunk_size=256`. Variable: `B=batch_size`, `L=seq_len`.
Because `n_groups=1`, `B`/`C` params are **shared across all 16 heads**.

Feedback workloads (fixed five; one immutable kernel over all five = one evaluation):

| # | B | L | pad | L_pad | NC | last-chunk valid rows | (b,nc,h) tiles |
|---|---|-----|-----|-------|----|-----------------------|----------------|
| 1 | 2 | 293 | 219 | 512   | 2  | 37                    | 64  |
| 2 | 4 | 1024| 0   | 1024  | 4  | 256                   | 256 |
| 3 | 4 | 256 | 0   | 256   | 1  | 256                   | 64  |
| 4 | 1 | 1024| 0   | 1024  | 4  | 256                   | 64  |
| 5 | 4 | 541 | 227 | 768   | 3  | 29                    | 192 |

Per-workload tolerance (all `max_rtol=0.05`, `required_match_ratio=0.98`):
`max_atol` = 0.020 / 0.029 / 0.024 / 0.022 / 0.029 for workloads 1..5.
`NC ≤ 4` in every case — inter-chunk recurrence is tiny. Parallel tiles 64–256, so
**occupancy is a first-order concern** for the 64-tile workloads (1, 3, 4).

Entry point contract: `solution/solution.py` must expose
`run(hidden_states, A, B, C, D, initial_states) -> (output, final_state)` with
`output: [B, L, H*P] bf16` and `final_state: [B, H, P, N] bf16`, matching the
reference signature exactly. Do not import or replicate the evaluator.

---

## 1. Reference algebra we implement (condensed from draft §2)

Per `(b, chunk c, head h)`, with `a[t] = cumsum_t A` (fp32) over the chunk, `a_last=a[Q-1]`:

- **Diagonal / intra-chunk:** `L[i,j] = exp(a[i]-a[j])` for `i≥j` else 0 (lower-tri).
  `G[i,j] = Σ_s C[i,s]·B[j,s]` (**head-independent**). `M = G∘L`.
  `Y_diag[i,:] = Σ_{j≤i} M[i,j]·X[j,:]` → `[Q,P]`.
- **Per-chunk state (right factor):** `decay_state[t]=exp(a_last-a[t])`;
  `states[p,n] = Σ_t decay_state[t]·X[t,p]·B[t,n]` → `[P,N]`.
- **Inter-chunk (middle):** `A_end_pad = [0, a_last(0..NC-1)]`,
  `decay_chunk = exp(segsum(A_end_pad))` `[NC+1,NC+1]`; recurrence over
  `states_with_init = concat(initial_states, states)`:
  `new_states[i] = Σ_{j≤i} decay_chunk[i,j]·states_with_init[j]`.
  `states_out = new_states[:-1]`, `final_state = new_states[-1]`.
- **Off-diagonal / state→out (left):** `Y_off[t,:] = exp(a[t])·Σ_s C[t,s]·states_out[s,:]` → `[Q,P]`.
- **Combine:** `y = Y_diag + Y_off`; reshape `[B,L_pad,H,P]`; add `D_residual=D[h]·X`;
  crop to `L`; reshape `[B,L,H*P]`; cast bf16. `final_state` cast bf16.

Key exploitable levers: never materialize `[Q,Q]`/`[Q,Q,H]` to DRAM (the reference's
multi-GB intermediates are the headroom); compute head-independent `G` once per group;
`NC≤4` ⇒ trivial recurrence; `N=Q=256`, `P=64` are matmul-friendly tiles.

**Numerical fidelity invariants (must hold in every candidate):**
1. Always form the difference *then* exponentiate (`exp(a[i]-a[j])`, `exp(a_last-a[t])`,
   `exp(a[t])`) — never `exp(a[i])/exp(a[j])`.
2. Keep `A`, cumsum, all exp arguments, and all reductions in **fp32**; cast to bf16
   only when writing `output` and `final_state`.
3. Load padded/out-of-range seq rows with `other=0.0` (A-pad=0 keeps cumsum flat;
   X/B/C-pad=0 contribute nothing). Mask output stores to `< L`.
4. Add `D_residual` in fp32 before the final bf16 cast.
5. Respect the transposed layout of `A: [B,H,L]` vs seq-major `X/B/C: [B,L,·]`.

---

## 2. Kernel decomposition (Option A — faithful 4-stage SSD)

`c001` implements Option A; later candidates fuse/tune. Working intermediates in fp32.

- **K1 `chunk_cumsum`** — grid `(B, H, NC)` (or fold into K2). Loads `A[b,h,chunk]`
  with row mask, `a = tl.cumsum(A_row)` along Q. Emits `a: [B,H,NC,Q]` and
  `a_last: [B,H,NC]`. (Alternative: recompute inside K2/K4 to save a DRAM round-trip —
  a tuning axis, not for c001.)
- **K2 `chunk_state`** — grid over `(b, nc, h)`. Accumulate
  `states[p,n] = Σ_t exp(a_last-a[t])·X[t,p]·B[t,n]` with an fp32 `[P=64,N=256]`
  accumulator via `tl.dot` (reduce over t in blocks of Q). Emit `states: [B,NC,H,P,N]` fp32.
- **K3 `state_passing`** — grid `(B, H)`. Sequential recurrence over `c=0..NC-1`
  carrying an fp32 `[P,N]` state seeded by `initial_states`, applying per-chunk decay
  `g_c=exp(a_last(c))` exactly as `decay_chunk`. Emit `states_out: [B,NC,H,P,N]` fp32 and
  `final_state: [B,H,P,N]`. (NC≤4 ⇒ trivial; can also build the `[NC+1,NC+1]` matrix and
  do a small matmul — equivalent, pick the clearer one at implement time.)
- **K4 `chunk_scan`** — grid over `(b, nc, h[, i-block])`. Recompute/reload `a`. Compute
  `G=C·Bᵀ` (`tl.dot [Q,N]·[N,Q] → [Q,Q]`), apply lower-tri mask + `exp(a_i-a_j)` in
  registers → `M`; `Y_diag = M·X` (`[Q,Q]·[Q,P]`). Compute
  `Y_off = exp(a[t]) ⊙ (C·states_out)` (`[Q,N]·[N,P]`). `y=Y_diag+Y_off + D[h]·X`.
  Store to `output[b, c*Q + i, h*P : h*P+P]` masked to `< L`.

**Layout/stride checklist for K1–K4** (verify statically before every eval, draft §4.8):
`A` is head-major `[B,H,L]`; `X` is `[B,L,H,P]`; `B/C` are `[B,L,1,N]` (group broadcast to
all H). Chunk c, row i ⇒ seq index `c*Q+i`; guard `< L` (and `< L_pad` implicitly by grid).

**Alternative decompositions (later candidates, see §4):**
- Option B: two fused kernels (state stages 1–2; scan stages 1,3,4) — fewer launches,
  less intermediate DRAM.
- Option C: single mega-kernel per `(b,h)` looping chunks internally, carrying state in
  registers/SRAM (exploits NC≤4) — removes K3 and the `states` DRAM round-trip; higher
  register pressure, serializes chunks within a block.

---

## 3. Candidate lineage strategy

Rules: candidates are **immutable and sequential** (`c001`, `c002`, …); any meaningful
source/config/launch change ⇒ new ID; never reuse an ID for changed source; append one
JSON record per evaluated candidate to `candidates.jsonl` (never rewrite). Each candidate
changes **one primary variable** so the evaluation attributes cause→effect. Lineage is a
tree: every candidate names its `parent`; a regressed/failed candidate is abandoned and the
next candidate branches from the last *known-good* parent (kept in `solution/solution.py`
only when it is the current best).

Branching policy:
- **Correctness first:** `c001` prioritizes passing all five workloads over peak speed
  (conservative precision). Do not start speed tuning until a passing baseline exists.
- **One knob per step:** precision mode, then head-shared `G` reuse, then fusion, then
  tiling/occupancy, then launch/dtype micro-tuning.
- **Keep-if-better:** accept a candidate as the new parent only if it (a) passes all five
  workloads and (b) improves geomean (or ties geomean with lower risk/complexity). Otherwise
  revert to the parent source for the next candidate.
- **Budget-aware:** 100-eval hard cap; token soft 1.0M / normal 1.5M / absolute 1.65M.
  Prefer high-information candidates; avoid speculative micro-sweeps once converged.

### 3.1 Planned sequence (each = one evaluation; later steps are hypotheses, revised by evidence)

- **c001 — Correct Option-A baseline.** 4 kernels (K1–K4), fp32 accumulation everywhere,
  `input_precision="ieee"` on the exp-feeding matmuls (`C·Bᵀ`, `C·states`) and on
  `X·B`; `M·X` value matmul may use tf32 but start with ieee to lock correctness. Simple
  grid = one program per `(b,nc,h)` (plus per-`(b,h)` K3, per-`(b,h,nc)` K1). Goal: all
  five pass; establish baseline geomean.
- **c002 — Precision relaxation (tf32).** Parent c001. Switch the large matmuls to
  `input_precision="tf32"` (rtol=0.05 is loose). Hypothesis: sizeable speedup, still
  passing. If any workload drops <0.98 match ratio, keep ieee selectively on the offending
  matmul only (that selective variant becomes its own candidate).
- **c003 — bf16 matmul inputs.** Parent = best of c001/c002. Feed `tl.dot` bf16-loaded
  X/B/C with fp32 accumulate for the value/score matmuls. Hypothesis: faster loads + tensor
  cores; watch exp-amplified paths (`C·Bᵀ`, `C·states`) — keep those higher-precision if
  they fail.
- **c004 — Head-shared G / group reuse.** Parent = best so far. Compute head-independent
  `G=C·Bᵀ` (and the `X·B`, `C·states` group contractions where legal) once per `(b,chunk)`
  and reuse across 16 heads (loop heads inside the block or restructure grid group-major).
  Hypothesis: up to ~16× less CB matmul work in K4 → large speedup on the compute-heavy
  workloads (2, 5).
- **c005 — Fuse cumsum (drop K1).** Recompute `a` inside K2/K4 instead of a separate
  kernel + DRAM round-trip. Hypothesis: fewer launches / less DRAM; neutral-to-positive.
- **c006 — Occupancy for 64-tile workloads.** Parent = best. Add an `i`-block (query) split
  and/or head/`P`/`N` splitting in K2/K4 to expose ≥108 concurrent blocks for workloads
  1/3/4. Tune `num_warps`∈{4,8}, `num_stages`∈{2,3,4}, `BLOCK_i`∈{64,128,256}. Hypothesis:
  fills SMs on the low-parallelism cases without hurting the 256-tile case.
- **c007 — Fusion (Option B/C).** Parent = best. Either merge K2/K4 into two kernels
  (Option B) or a per-`(b,h)` chunk-looping mega-kernel carrying state in registers
  (Option C, removing K3 + `states` round-trip). Hypothesis: removes DRAM traffic and
  launch overhead; risk = register pressure/occupancy. Adopt only if it beats c006.
- **c008+ — Micro-tuning.** Remaining knobs: tile shapes, `num_warps`/`num_stages` per
  kernel, `tl.dot` accumulator layout, vectorized/coalesced load ordering for the
  transposed `A`, optional `tl.exp2`-based decay (multiply cumsum by log2e) if faster and
  still in tolerance. One knob per candidate; stop when converged (§6).

This is a *plan*, not a script: after each evaluation, the observed per-workload
pass/fail + speedups redirect the next candidate (e.g. if tf32 already passes cleanly,
skip straight to head-shared reuse; if occupancy is the bottleneck, prioritize c006).

---

## 4. Correctness checks

### 4.1 Static self-checks (no GPU) — run before *every* evaluation
Re-derive against draft §2/§4 and §1 above:
1. Shapes/strides for all loads/stores, including transposed `A: [B,H,L]` vs seq-major
   `X/B/C`, and `B/C` group broadcast (n_groups=1 → all 16 heads).
2. Seq index math `c*Q+i`; every seq-indexed load masked with `other=0.0`; every output
   store masked to `< L` (and final chunk’s valid-row count for workloads 1/5).
3. cumsum axis is the in-chunk Q axis and matches `torch.cumsum`; padded A rows load 0.
4. Exp-difference ordering (invariant §1.1) everywhere; all exp args fp32 (invariant §1.2).
5. `states_with_init` ordering (index 0 = initial_states), `states_out=new_states[:-1]`,
   `final_state=new_states[-1]`.
6. `D_residual` added in fp32 pre-cast; final bf16 casts on `output` and `final_state`.
7. Kernel launch grid covers all `(b,nc,h[,i])`; no out-of-bounds program IDs.
8. Output dtype/shape exactly `[B,L,H*P] bf16` and `[B,H,P,N] bf16`.

### 4.2 Harness correctness (the only correctness signal)
`./scripts/evaluate_candidate.sh feedback cNNN` over the five fixed workloads. Per
workload: `atol ≤ max_atol`, `rtol ≤ 0.05`, fraction-within-tolerance `≥ 0.98`, on **both**
`output` and `final_state`. A candidate is **valid** only if all five pass; an invalid
candidate cannot be a parent for speed claims. Never run CUDA/profilers/nvidia-smi/the
external evaluator or any alternate correctness harness directly.

### 4.3 Coverage argument
The five workloads already span: heavy-padding partial last chunk (1: 37 valid rows; 5: 29),
exact multiples (3: single chunk; 2/4: four chunks), and batch/occupancy variety
(B=1,2,4 → 64–256 tiles). Passing all five is a strong joint signal for masking/cumsum/decay
correctness and speed generality.

### 4.4 Failure handling
A failed/erroring Triton kernel is **invalid** — fix the kernel; never substitute a
Torch/CPU/NumPy path. If a precision-relaxation candidate fails match ratio on a subset of
workloads, the next candidate re-tightens precision only on the implicated matmul(s), as a
new ID. Record the failure and its diagnosis in `candidates.jsonl`; do not silently retry
the same source under a new ID.

---

## 5. Performance hypotheses

Ranked by expected impact (validated/refuted by evaluations):

1. **Eliminating DRAM intermediates dominates.** The reference writes multi-GB
   `L:[B,H,NC,Q,Q]` and `G,M:[B,NC,Q,Q,H]` to DRAM; the op is memory-bound there. A fused
   Triton kernel that never materializes `[Q,Q]` should yield the largest speedup even
   before precision tuning. → biggest gain expected at c001→(fused) already.
2. **Head-shared `G` (n_groups=1)** removes ~16× redundant `C·Bᵀ` work in the diagonal
   stage → strong gains on compute-heavy workloads 2 and 5. (c004)
3. **tf32/bf16 matmul inputs** cut matmul latency with rtol=0.05 headroom; expected
   moderate uniform speedup if tolerance holds. (c002/c003)
4. **Occupancy** governs workloads 1/3/4 (64 tiles < 108 SMs): exposing extra grid
   parallelism (i-block / P / N splits) should lift those specifically without hurting the
   256-tile workload 2. (c006)
5. **Launch/round-trip reduction** (fuse cumsum, Option B/C) trims fixed overhead —
   relatively larger for the small workloads (1/3). (c005/c007)
6. **Micro-tuning** (`num_warps`/`num_stages`/tile shapes, `exp2` decay) — small,
   workload-dependent; only after structure is settled. (c008+)

Metric definition: per-workload speedup = reference_time / candidate_time (as reported by
the harness); primary score = **geometric mean of the five speedups**, reported only when
all five pass. Track both geomean and the per-workload vector to catch regressions masked by
the mean (esp. the low-occupancy trio 1/3/4 vs compute-heavy 2/5).

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any of:
1. **Convergence:** two or three consecutive accepted candidates improve geomean by
   < ~2% each, and the remaining ideas in §3 are exhausted or judged sub-2% — i.e.
   marginal returns below noise.
2. **Budget:** approaching the 100-evaluation cap, or the token **soft limit (1.0M)** with
   a valid best candidate in hand (hard-stop before 1.5M normal / 1.65M absolute).
3. **Structural ceiling:** the kernel is compute/occupancy-bound with no untried lever
   expected to matter on these five shapes (confirmed by the per-workload speedup pattern,
   reasoned — not by running a profiler here).
4. **No valid candidate risk:** if correctness cannot be achieved within tolerance after
   focused effort, keep the best *valid* candidate as the answer and stop tuning speed.

At stop: ensure `solution/solution.py` holds the best **valid** candidate’s source, the
`candidates.jsonl` lineage is complete, and `SEARCH_COMPLETE` states the reason and the
selected candidate ID. Never run `final` (16-workload) without explicit operator approval.

---

## 7. Evidence format

One JSON object appended to `candidates.jsonl` per evaluated candidate (never rewrite prior
records). Schema:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "2026-09-17THH:MM:SSZ",
  "hypothesis": "Faithful Option-A 4-kernel SSD, fp32/ieee — establish passing baseline.",
  "change_from_parent": "initial implementation",
  "validation": {
    "static_checks": "pass  (strides/masks/cumsum/exp-order/dtypes verified vs plan §4.1)",
    "harness_stage": "feedback"
  },
  "per_workload": [
    {"uuid": "4bf2c843-...", "B": 2, "L": 293,  "passed": true, "max_atol_used": 0.020,
     "match_ratio_output": null, "match_ratio_final_state": null, "speedup": null},
    {"uuid": "2907639d-...", "B": 4, "L": 1024, "passed": true, "speedup": null},
    {"uuid": "a85c5733-...", "B": 4, "L": 256,  "passed": true, "speedup": null},
    {"uuid": "6dc47920-...", "B": 1, "L": 1024, "passed": true, "speedup": null},
    {"uuid": "5fd24475-...", "B": 4, "L": 541,  "passed": true, "speedup": null}
  ],
  "all_passed": true,
  "geomean_speedup": null,
  "decision": "accept|reject|revert",
  "decision_reason": "baseline established / regressed on WL1 / correctness fail on WL5 ...",
  "cumulative_evaluations": 1,
  "skill_usage": {"KernelWiki": "not_used (A800=sm_80 Ampere, out of skill scope)"},
  "notes": "observations, next-candidate direction"
}
```

Fill `speedup`/`match_ratio`/`geomean` from the harness output for each real evaluation
(placeholders `null` here only because no candidate is evaluated in this step). Keep a
running `cumulative_evaluations` counter. Record `decision` (accept and promote to parent,
reject-and-branch-from-parent, or revert) with a one-line reason, and the per-candidate
skill usage honestly.

---

## 8. Risk register (carry from draft §8; resolve during search)

1. Distribution/scale of `random` bf16 inputs → how often `exp` blows up; whether tf32/bf16
   matmul inputs stay within tolerance. Resolve empirically at c001/c002.
2. Best precision per matmul (ieee vs tf32 vs bf16-input) balancing speed vs 0.98 match.
3. Whether head-shared `G` reuse pays off vs. simpler per-head recompute.
4. Option A vs C for inter-chunk state given NC≤4 (register-carry mega-kernel removing a
   DRAM round-trip).
5. Occupancy tuning for the 64-tile workloads (1/3/4) on 108 SMs.
6. Exact stride handling for `A:[B,H,L]` vs seq-major X/B/C during chunking (static-check
   every candidate).

---

## 8b. Decision log (updated per evaluation)

- **c001 (eval #1) — REJECT.** Faithful Option-A 4-kernel SSD, fp32 accumulation,
  `tl.dot(..., input_precision="ieee")` on every matmul. Result: **all 5 workloads
  `RUNTIME_ERROR`** (uniform, including the clean single-chunk L=256 case) ⇒ a
  compile/launch failure, not an algebra or tolerance problem. Harness surfaces no
  traceback; bash `python` is blocked in this sandbox so the Triton version could not be
  probed. Leading suspects: (a) the `input_precision=` kwarg is not accepted by the
  installed Triton build (older API uses `allow_tf32=`), which would fail identically on
  every shape; (b) fp32 `tl.dot` on the large `[64,256]`/`[256,64]` operand tiles with
  `num_warps=8` + default pipelining exceeding A800 shared memory.
  **Next (c002):** drop the `input_precision` kwarg entirely and feed **bf16 operands**
  into all `tl.dot` calls (default fp32 accumulation retained), which both removes the
  kwarg-compat risk and halves dot-operand SRAM; keep cumsum / decays / `exp` /
  accumulators / D-residual in fp32; pin `num_warps=4`, `num_stages=2` to bound SRAM.
  K1–K4 algebra is unchanged (re-verified against reference §1).

- **c002 (eval #2) — ACCEPT (new best/parent).** Same 4-kernel algebra, but every
  `tl.dot` now consumes **bf16 operands** with default fp32 accumulation and the
  `input_precision` kwarg is removed; `num_warps=4`, `num_stages=2` on K2/K4. Result:
  **5/5 PASS, geomean 10.64×** (WL1 10.49, WL2 12.25, WL3 8.97, WL4 8.94, WL5 13.23).
  Confirms c001's failure was a compile/launch issue (kwarg or SRAM), not algebra, and
  that bf16 operands + fp32 accumulation are within tolerance. The occupancy hypothesis
  is visible in the data: the 64-tile workloads (WL3 L=256, WL4 B=1) are the slowest,
  while the high-parallelism ones (WL2 256-tile, WL5 192-tile) lead.
  **Next (c003):** head-shared `G = C·Bᵀ` reuse in K4 — with n_groups=1, `G` is identical
  across all 16 heads, so restructure K4 to compute `C·Bᵀ` once per `(b,chunk)` and reuse
  it while applying the head-dependent `L`/decay per head. Cuts diagonal matmul work ~16×
  and raises arithmetic intensity, especially on the compute-heavy WL2/WL5. Alternatively
  expose more grid parallelism for the 64-tile cases if occupancy dominates.

- **c003 (eval #3) — REJECT (revert to c002).** Head-shared `G` via a *separate* kernel
  `Kg` that writes `G:[B,NC,Q,Q]` fp32 to DRAM, with K4 loading the G tile instead of
  recomputing `C·Bᵀ`. Result: **5/5 PASS but geomean 10.08× vs c002's 10.64× (−5.3%)**.
  The added full `[B,NC,Q,Q]` fp32 DRAM round-trip (Kg write + K4 read) plus an extra
  launch outweighs the saved matmul FLOPs. Per-workload: compute-heavy WL2 improved
  (12.25→12.77) but WL5 dropped (13.23→12.96), and the 64-tile trio all regressed
  (WL1 10.49→9.57, WL3 8.97→8.09, WL4 8.94→8.10). **Key lesson:** the redundant `C·Bᵀ`
  recompute in c002 is essentially *free* relative to memory/launch overhead — c002 is not
  bound by that matmul. Head-shared reuse only helps if `G` is kept **in-SRAM**, never
  round-tripped to DRAM. Reverted `solution/solution.py` to the c002 source.
  **Next (c004):** in-SRAM head fusion — one K4 program per `(b,chunk,i-block)` loops all
  16 heads internally, computing each `G=C·Bᵀ` block once and reusing it across heads while
  `X`/`states`/decay vary per head (G stays resident, no DRAM). This should realize the
  ~16× G-matmul saving without c003's traffic penalty. Risk: register/SRAM pressure and
  reduced grid parallelism (grid shrinks by 16× → may hurt the already low-occupancy
  64-tile WL3/WL4); if so, keep an i-block split to preserve enough blocks.

- **c004 (eval #4) — REJECT (revert to c002).** Removed kernel K1 and its
  `acum`/`alast` DRAM tensors, recomputing `a=cumsum(A)` inline in K2/K3/K4 (in K4, extract
  `a_i`/`a_j` sub-blocks from the full-Q cumsum via one-hot mask-sum gathers). Result:
  **5/5 PASS but geomean 7.66× vs c002's 10.64× (−28%)**. The inline recompute — a full
  Q=256 cumsum *plus* a `[64,256]` one-hot gather per i-block in K4, repeated across every
  `B·NC·H·(Q/BLOCK_M)` program — costs far more than the single cheap K1 launch + tiny
  `acum`/`alast` round-trip it replaced. Compute-heavy WL2/WL5 collapsed (12.25→8.00,
  13.23→6.77) and even the 64-tile cases regressed, so **K1 is not the fixed-overhead
  culprit**. Reverted `solution/solution.py` to c002.
  **Combined lessons (c003+c004):** c002 is not bound by redundant `C·Bᵀ` matmul nor by K1;
  it is already memory/launch-lean on these tiny (NC≤4) shapes. The remaining large traffic
  is the `states`/`states_out`:`[B,NC,H,P,N]` **fp32** intermediates (~0.27 GB each for WL2,
  written by K2, round-tripped through K3, read by K4).
  **Next (c005):** keep the exact c002 structure but store `states` and `states_out` in
  **bf16** to halve that traffic. Numerically low-risk: K4 already casts `states_out`→bf16
  before the `C·states` dot (off-diagonal term unchanged), and K3's recurrence stays fp32
  in-register with only the per-chunk increment rounded to bf16 (≤4 adds, well within
  rtol=0.05). If c005 helps, follow with the in-SRAM head-fusion idea (former c004 plan).

- **c005 (eval #5) — ACCEPT (new best/parent).** Kept the exact c002 structure but
  allocated `states` and `states_out`:`[B,NC,H,P,N]` in **bf16** (K2 stores `states.to(bf16)`;
  K3 carries the recurrence in fp32 registers, stores `states_out` bf16; K4 loads
  `states_out` bf16 directly). All arithmetic stays fp32. Result: **5/5 PASS, geomean
  11.49× vs c002's 10.64× (+8%)**, every workload improved (WL1 10.49→11.03, WL2
  12.25→14.22, WL3 8.97→9.36, WL4 8.94→9.38, WL5 13.23→14.55). `max_abs`/`max_rel` are
  identical to c002, confirming the bf16 rounding moved no element across the tolerance
  boundary. This validates that the `states`/`states_out` fp32 round-trip was a genuine
  bottleneck; the compute-heavy WL2/WL5 (largest state tensors) gained most.
  **Next (c006):** the 64-tile WL3/WL4 still floor at ~9.4× (launch/occupancy-limited).
  Attack that with more grid parallelism, or try storing `acum` in bf16 too (smaller
  effect). Prefer the occupancy lever for the low-parallelism workloads.

- **c006 (eval #6) — ACCEPT (new best/parent).** Folded the tiny K1 cumsum kernel into
  K2: K2 now computes `a=cumsum(A)`/`a_last` inline and **stores** them (from its `nblk==0`
  program) to the same `acum`/`alast` tensors, so K3/K4 keep reading them from DRAM exactly
  as in c005 — no expensive K4 recompute (the mistake that sank c004). 4 launches → 3.
  Result: **5/5 PASS, geomean 14.45× vs c005's 11.49× (+26%)**, every workload up
  substantially (WL1 11.03→14.38, WL2 14.22→16.80, WL3 9.36→12.08, WL4 9.38→12.15, WL5
  14.55→17.77). The small workloads gained most, confirming their floor was launch/overhead
  bound. `max_abs`/`max_rel` identical to c005 → numerics unchanged. Key contrast with c004:
  *computing the cumsum once in K2 and storing* is a big win, whereas *recomputing it per
  i-block in K4* was fatal.
  **Next (c007):** only 3 launches remain (K2/K3/K4). K3 is tiny but the inter-chunk
  recurrence is sequential over chunks so it cannot merge with the per-`(b,chunk)` K2/K4
  trivially. Option C — a per-`(b,h)` mega-kernel looping chunks in registers — could drop
  K3 + the `states`/`states_out` round-trip, at the risk of serializing chunks and lowering
  parallelism on WL3/WL4. Otherwise micro-tune `num_warps`/`BLOCK_N`. Expect smaller gains;
  c006 is already a strong best.

- **c007 (eval #7) — ACCEPT (new best/parent).** Fused K2 (chunk_state) + K3
  (state_passing) into one kernel `KS` (grid `(B,H,N/BLOCK_N)`) that carries the
  `[P,BLOCK_N]` SSM state in fp32 registers and loops chunks: store `states_out[c]` =
  carried-in state, compute `contrib` via bf16 dot, `state = exp(a_last)·state + contrib`.
  This removed the entire `states:[B,NC,H,P,N]` intermediate tensor *and* a launch (3→2).
  Result: **5/5 PASS, geomean 19.23× vs c006's 14.45× (+33%)**, every workload up (WL1
  14.38→20.25, WL2 16.80→19.79, WL3 12.08→17.03, WL4 12.15→17.07, WL5 17.77→22.60). The
  feared chunk serialization on WL4 (B=1, NC=4) did **not** bite — WL4 still gained +40%,
  so the removed round-trip/launch dominates the lost NC-parallelism on A800.
  `max_abs`/`max_rel` identical to c006 (the register-fp32 `contrib` is at least as
  accurate). Cumulative arc: c002 10.64 → c005 11.49 → c006 14.45 → **c007 19.23**.
  **Next (c008):** only 2 launches remain (KS, K4); `acum` + `states_out` are the only
  cross-kernel DRAM. Diminishing returns expected. Try single-knob micro-tuning first
  (`BLOCK_M`∈{128,256} or `num_warps`∈{2,8} on K4); larger structural ideas (fuse `acum`
  away — needs single-i-block K4 to avoid c004's fatal per-i-block recompute; or fold the
  off-diagonal into KS) are higher-risk.

- **c008 (eval #8) — REJECT (revert to c007).** Single-knob micro-tune: K4 query-block
  `BLOCK_M` 64→128. Result: **5/5 PASS but geomean 14.72× vs c007's 19.23× (−23%)**, hurting
  every workload (WL1 20.25→16.51, WL2 19.79→13.91, WL3 17.03→13.46, WL4 17.07→13.37, WL5
  22.60→16.74). The larger `[128,·]` tiles raise register/SRAM pressure and cut
  occupancy/pipelining more than the shorter lower-tri j-loop saves; with tiny K4 tiles
  (P=64) the extra grid parallelism from `BLOCK_M=64` (Q/BLOCK_M=4) matters more on A800.
  **`BLOCK_M=64` is confirmed optimal for K4.** Reverted `solution/solution.py` to c007.
  **Next (c009):** remaining single knobs — `num_warps`∈{2,8} on K4 or KS, `num_stages`∈{3,4},
  `BLOCK_N`∈{128,256} on KS. Given the tile-size knob regressed and structural fusion is
  exhausted (2 launches, minimal DRAM), expect <2% swings; one or two more targeted knob
  candidates, then converge.

- **c009 (eval #9) — REJECT (revert to c007).** Single-knob micro-tune: K4 `num_warps`
  4→8. Result: **5/5 PASS but geomean 14.81× vs c007's 19.23× (−23%)**, hurting every
  workload (WL1 20.25→17.21, WL2 19.79→13.16, WL3 17.03→14.22, WL4 17.07→14.26, WL5
  22.60→15.51). K4's tiles are tiny (`BLOCK_M=64`, P=64 output), so 8 warps oversubscribe
  the small `[64,64]`/`[64,256]` dots and cut per-SM occupancy. `num_warps=4` is optimal.
  Reverted to c007.
  **Convergence assessment:** two independent K4 knob directions — tile size (c008
  `BLOCK_M=128`) and warp count (c009 `num_warps=8`) — both regressed ~−23%, so c007's K4
  config (`BLOCK_M=64`, `num_warps=4`, `num_stages=2`) is a local optimum. All structural
  levers are exhausted: `[Q,Q]` never materialized; states materialization removed; launch
  count down to 2 (KS, K4); only `acum`+`states_out` cross DRAM; bf16 tensor-core operands
  with fp32 accumulation. The rejected branches (c003 DRAM-G, c004 inline-cumsum) confirm
  no further fusion/traffic lever helps on these NC≤4 shapes. **The search has converged;
  c007 (geomean 19.23×, 5/5 pass) is the best valid candidate.** Next turn: create
  `SEARCH_COMPLETE` unless a genuinely new lever appears.

- **c010 (eval #10) — REJECT (revert to c007). SEARCH CONVERGED.** Single-knob micro-tune:
  KS `num_warps` 4→2 (applying c008/c009's "fewer resources" lesson to the other kernel).
  Result: **5/5 PASS but geomean 12.73× vs c007's 19.23×**, driven by a catastrophic WL3
  outlier (B4,L256, single-chunk NC=1): 17.03× → **3.00×** (0.083 ms → 0.472 ms, ~5.7×
  slower); the others were flat/slightly down (WL1 20.16, WL2 16.57, WL4 17.04, WL5 19.53).
  With only 2 warps the KS `[64,64]`-output dot over K=256 and the Q=256 cumsum are badly
  under-parallelized within each of the 256 KS programs, and NC=1 offers no recurrence
  amortization. `num_warps=4` is optimal for KS. Reverted to c007.
  **Convergence:** three consecutive micro-tunes rejected (c008 K4 BLOCK_M, c009 K4
  num_warps, c010 KS num_warps). Both `num_warps` directions (2 and 8) and the tile-size
  direction (128) regress, so c007's config (KS num_warps=4/num_stages=2; K4 BLOCK_M=64/
  num_warps=4/num_stages=2) is a confirmed local optimum. All structural levers are
  exhausted or shown to hurt: `[Q,Q]` never materialized; states tensor removed (c007);
  launch count minimal at 2; only `acum`+`states_out` cross DRAM; bf16 tensor-core operands
  with fp32 accumulation; DRAM-G (c003) and inline-cumsum (c004) both regressed.
  **Best valid candidate: c007, geomean 19.23×, 5/5 pass.** `SEARCH_COMPLETE` created.

## 9. Immediate next action (next turn, not now)

Implement `c001` = the correct Option-A 4-kernel Triton solution (fp32/ieee, simple grid),
run static self-checks (§4.1), then evaluate once via
`./scripts/evaluate_candidate.sh feedback c001` and append its record to
`candidates.jsonl`. No code is written or evaluated in this planning step.
