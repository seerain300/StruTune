# Plan — `gdn_decode_qk4_v8_d128_k_last`

Executable, sequential optimization plan. Derived from `docs/draft.md`. Target: **A800, sm_80**.
No code is written or evaluated in this turn — this is the roadmap the candidate loop follows.

---

## 0. Ground rules (recap, binding)

- Triton owns all computation (gates + delta update + read-out). PyTorch only for allocation,
  shapes/strides, grid math, dtype/scale plumbing. No Torch/CPU/NumPy/CUDA-ext fallback.
- One immutable kernel version over all 5 feedback workloads = **one evaluation**.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback <cNNN>`.
- Any source/config/launch change ⇒ new candidate ID. Never mutate an evaluated candidate or a
  prior `candidates.jsonl` record.
- Budgets: ≤100 evaluations; token soft 1.0M / hard 1.2M.
- `final` (54-workload) is operator-only — never run without explicit approval.

---

## 1. Target contract (what `solution/solution.py` must expose)

- Entry point: `run(q, k, v, state, A_log, a, dt_bias, b, scale)`.
- Returns `(output, new_state)`:
  - `output`: `[B,1,8,128]` **bf16** (V-dim last), `= scale * new_state @ q_h`.
  - `new_state`: `[B,8,128,128]` **f32**, k-last `[B,H,V,K]`.
- Must handle `state is None` → treat as zeros (allocate-zero in wrapper; math reduces to
  `new_state = outer(beta*v_h, k_h)`, `output = scale*new_state@q_h`).
- `scale is None or == 0.0` → `1/sqrt(K)`. Feedback always passes `0.08838834764831843`; still
  handle the default in the wrapper.
- Head mapping: v-head `h` uses q/k-head `h // 2` (`G = num_v_heads/num_q_heads = 2`); but
  compute `G` from shapes, don't hardcode, so the kernel stays valid for the 54-workload final.
- Keep dims general where cheap (read `B, num_q/k/v_heads, K, V` from tensors); the const axes
  (`4/4/8/128`) are known but avoiding hardcodes reduces final-eval risk.

### Reference algebra to implement (from draft §2, per `(b,h)`, all f32)
```
g    = exp(-exp(A_log[h]) * softplus(a[b,0,h] + dt_bias[h]))
beta = sigmoid(b[b,0,h])
old_v[v]        = g * dot_K(S[v,:], k_h)          # S = state[b,h] in [V,K]
delta_v[v]      = beta * (v_h[v] - old_v[v])
new_state[v,k]  = g*S[v,k] + delta_v[v]*k_h[k]
output[v]       = scale * dot_K(new_state[v,:], q_h)
```
Reductions over `K` via `tl.sum` in f32. **No `tl.dot`, no TF32, no tensor cores.**

---

## 2. Candidate lineage strategy

Sequential, single active source version at a time. Each candidate is immutable once evaluated.
Branch from the current best-valid parent. Fallback rule: if a candidate regresses or fails,
its children revert to the last known-good parent and change exactly one axis.

```
c001  correctness baseline (fixed BLOCK_V, gates in-kernel, no autotune)
  └─ c002  add @triton.autotune(BLOCK_V, num_warps) keyed on batch_size
       ├─ c003  tune num_stages / grid shape (1D vs 2D) / vector-width & .to placement
       ├─ c004  state=None fast path + minor launch-overhead trims (only if evidence)
       └─ c005+ micro-tuning of the winning config; optional gate-prologue experiment
                 (only if a hypothesis predicts a win)
```

One-axis-per-candidate discipline: never change tiling AND gating AND grid in the same ID —
otherwise attribution of the geomean delta is impossible.

### Decision log (updated as candidates land)
- **c001 (parent, best-valid so far):** geomean **182.09x**, all 5 pass. sol_ms sits at a
  ~0.064ms floor for B≤16 and 0.090ms for B=64 → the op is *launch-overhead-bound*, not
  tile-bound. GPU compute is trivial (memory-bound, tiny state).
- **c002 (rejected):** added `@triton.autotune` (BLOCK_V×warps, key=[B]) + `grid=lambda`.
  geomean **134.26x** — a uniform regression: every workload ~0.025ms slower, sol_ms floor
  rose 0.064→0.091ms. The autotune runtime dispatch (per-launch key hash + config lookup) and
  the callable grid add fixed per-launch overhead that dominates a launch-bound op. **Confirms
  H2/H3 are unreachable here** — differences between BLOCK_V configs live below the launch
  floor, so tuning tiles cannot help; the real lever is *reducing per-launch overhead*.
- **Revised strategy:** revert to the c001 kernel body (fixed BLOCK_V, static grid, no
  autotune) and attack the launch floor: minimize kernel argument count / Python-side plumbing
  (enforce contiguous inputs, pass constexpr shapes, drop redundant stride scalars), which is
  the only axis that can move 4 of 5 (launch-bound) workloads.

---

## 3. Sequential execution steps

### c001 — correctness baseline (highest priority: must pass all 5)
- Single fused kernel; parallel unit `(bh, v_block)`, `bh = b*H + h`, fixed `BLOCK_V = 16`.
- Steps 1–10 from draft §6.1: decode `b,h`; `qk_head = h // G`; recompute `g,beta` in-kernel
  (guarded softplus, threshold 20); load `q_h,k_h` (bf16→f32), `v_tile`, `S_tile=[BLOCK_V,K]`;
  compute `old_v, delta_v, new_state_tile, out`; store f32 `new_state`, bf16 `output`.
- No autotune, no masking (all candidate `BLOCK_V | 128`), `num_warps=4`, `num_stages=1`.
- **Gate: correctness on all 5 workloads.** If any fail → do NOT tune; debug algebra/indexing/
  dtype in a new candidate (c001b lineage) until correctness holds.

### c002 — occupancy via autotune
- Add `@triton.autotune` over `BLOCK_V ∈ {8,16,32,64,128}` × `num_warps ∈ {1,2,4,8}`, `key=[B]`.
- Hypothesis (draft §3): small `B` (1,4) needs small `BLOCK_V` to fill 108 SMs; `B=64` prefers
  fat tiles (`BLOCK_V=128`) for max reuse / fewest gate recomputes.
- Re-verify correctness, then compare geomean vs c001.

### c003 — pipelining / grid / cast placement
- Sweep `num_stages ∈ {1,2}`, grid 1D vs 2D `(bh, v_tiles)`, and where the bf16→f32 casts and
  the f32→bf16 output cast happen (minimize register pressure / redundant conversions).
- One knob at a time if any shows noise; otherwise fold compatible knobs.

### c004 — launch overhead & optional paths
- Only if small-`B` workloads dominate downside: trim Python-side plumbing (avoid extra
  `.contiguous()`, precompute strides once), add `state is None` fast path.

### c005+ — converge
- Micro-tune winning config. Optional gate-prologue experiment **only** if a concrete hypothesis
  (e.g. gate recompute shows up as a bottleneck) predicts a win; otherwise skip.
- Stop per §6.

---

## 4. Correctness checks (per candidate, before trusting speedup)

Because the evaluator is the sole oracle and budget is finite, apply a static pre-eval checklist
first, then rely on the evaluator's built-in correctness gate:

1. **Algebra**: matches draft §2 (mat-vec `old_v`, rank-1 `new_state`, `scale` on read-out only).
2. **Indexing**: `qk_head = h // G`; k-last strides for `state`/`new_state` (`S[b,h,v,:]`
   contiguous over K); `output` layout `[B,1,H,V]` bf16; `a,b` indexed `[b,0,h]`.
3. **Dtype discipline**: bf16→f32 upcast on `q,k,v,a,b`; f32 accumulators; f32 `new_state`
   store (never bf16); RNE f32→bf16 for `output`.
4. **Gate numerics**: guarded softplus `where(x>20, x, log(1+exp(x)))`; `exp(A_log)` then `g`;
   `sigmoid(b)`; all f32.
5. **Masking**: none required while `BLOCK_V | 128`; if a non-dividing `BLOCK_V` is ever added,
   add V-tail masks on load AND store.
6. **Evaluator gate**: every selected workload must pass correctness or the candidate is invalid
   regardless of speed. Record pass/fail per workload.
7. **General-shape safety** (for future `final`): `G`, `B`, `H`, `K`, `V` read from tensors, not
   hardcoded, so the kernel remains correct on the 54-workload set.

A candidate that fails Triton compilation or correctness is invalid — no fallback substitution.

---

## 5. Performance hypotheses (falsifiable, tied to candidates)

| # | Hypothesis | Test | Expected signal |
|---|---|---|---|
| H1 | Op is memory-bound; f32 streaming pass beats reference's per-head loop | c001 geomean | speedup > 1 on all B, largest at B=64 |
| H2 | Small B is occupancy/launch-bound; fine `BLOCK_V` helps B∈{1,4} | c002 vs c001 | B=1,4 speedup improves with small BLOCK_V |
| H3 | Large B prefers fat tiles (BLOCK_V=128) for reuse | c002 autotune pick | B=64 selects BLOCK_V=128, warps≈4 |
| H4 | Tensor cores / TF32 give nothing (trivial FLOPs) and hurt accuracy | not tried; asserted | (rejected by design) |
| H5 | num_stages>1 gives little (single S load, no K-loop pipeline) | c003 | ≤ noise improvement |
| H6 | In-kernel gate recompute cheaper than extra prologue launch+round-trip | c005 (optional) | prologue ≤ in-kernel |

Interpretation: geomean is the primary metric, but inspect per-workload — B=1,4 (latency-bound)
set the downside; B=16,64 (bandwidth-bound) set the upside.

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- Geomean improvement over the best-valid candidate is < ~2% across two consecutive new
  candidates (convergence).
- Best candidate is within a small factor of the roofline for the bandwidth-bound workloads
  (B=16,64 near HBM limit) AND small-B workloads are launch-bound with no further split to give.
- Evaluation budget (100) or token budget (soft 1.0M) is approaching.
- All plausible one-axis changes in §2 lineage are exhausted without net gain.

Never run `final` on convergence; it requires explicit operator approval.

---

## 7. Evidence format (one JSON object appended per evaluated candidate to `candidates.jsonl`)

Append-only; never rewrite earlier records. Fields:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "memory-bound f32 streaming kernel beats reference per-head loop",
  "change_summary": "one-axis description vs parent",
  "validation": {"compiled": true, "all_correct": true},
  "per_workload": [
    {"uuid": "3daa0974...", "batch_size": 1,  "correct": true, "speedup": 0.0},
    {"uuid": "9a92acbc...", "batch_size": 4,  "correct": true, "speedup": 0.0},
    {"uuid": "abec5d32...", "batch_size": 8,  "correct": true, "speedup": 0.0},
    {"uuid": "f0507025...", "batch_size": 16, "correct": true, "speedup": 0.0},
    {"uuid": "b1931bc6...", "batch_size": 64, "correct": true, "speedup": 0.0}
  ],
  "geomean_speedup": 0.0,
  "decision": "keep|reject|parent-for-next",
  "cumulative_evaluations": 1,
  "skill_usage": "none (Ampere; KernelWiki=Blackwell/Hopper, ncu=B200 — N/A)",
  "notes": "observations, autotune picks, next-step rationale"
}
```

Rules: fill `per_workload` from evaluator output verbatim; `speedup`/`geomean` are recorded, not
invented; `decision` drives the next parent; `cumulative_evaluations` is monotonic.

---

## 8. Risk register & mitigations

- **c001 correctness miss** → most likely k-last indexing or `h//G` mapping; mitigate by keeping
  c001 minimal and re-checking §4 statically before eval.
- **Autotune cache staleness** across candidates → each candidate is a fresh immutable file with
  its own configs; don't share tuning state between IDs.
- **Small-B noise** in speedup (µs-scale, launch-dominated) → judge by geomean trend across
  candidates, not a single small-B number.
- **Over-fitting to feedback B set** → keep kernel shape-general (§1, §4.7) so `final` holds.
- **Budget creep** → prefer folding compatible knobs into one candidate when attribution is
  clear; reserve separate IDs for genuinely independent axes.

---

## 9. Skill usage (confirmed N/A)

- KernelWiki: Blackwell/Hopper-only → not applicable to A800/sm_80.
- ncu-report-skill: B200/sm_100 profiling and profilers are disallowed here → not applicable.
- Plan rests on reference algebra (draft §2) and the A800 roofline (draft §3).
