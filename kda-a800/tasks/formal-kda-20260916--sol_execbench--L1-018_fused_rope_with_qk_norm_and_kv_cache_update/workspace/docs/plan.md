# Plan — L1/018 Fused RoPE + QK-Norm + KV-Cache Update

Target: **A800 (`sm_80`, Ampere)**. Primary compute in **Triton**; PyTorch only for allocation,
strides, grid math, and scalar plumbing. **No** Torch/CPU/NumPy/CUDA-extension computational fallback —
a broken Triton kernel is an invalid candidate, not something to paper over with `torch`.

This plan operationalizes `docs/draft.md` into concrete kernel signatures, grid math, `constexpr`
parameters, an autotune space, an ordered candidate ladder, correctness checks, performance
hypotheses, stopping criteria, and the evidence format. **No code is written or evaluated this turn.**

> Skill note: `KernelWiki` is scoped to Blackwell (SM100) / Hopper (SM90). This task is Ampere `sm_80`,
> so `KernelWiki` is out of scope and will not be consulted. Any candidate record will report
> `skill_usage: "none (KernelWiki is SM90/SM100-only; task is sm_80 Ampere)"`.

---

## 0. Ground rules recap (binding)

- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback cNNN`. Never call `kda-eval`,
  a profiler, `nvidia-smi`, CUDA directly, or any alternate correctness harness.
- Five fixed workloads = one candidate evaluation. Any meaningful source/config/launch change ⇒ **new**
  candidate ID. Never mutate an already-evaluated candidate's source or reuse its ID.
- Append exactly one JSON object per evaluated candidate to `candidates.jsonl`; never rewrite prior rows.
- Budget: 100 evaluations; token soft 1.0M / normal 1.5M / absolute 1.65M.
- `final` (13-workload) only after explicit operator approval. Do not run it proactively.
- Submission entry point: `solution/solution.py` exposing `run(...)` with the reference signature.

### 0.1 Submission mechanics (how a candidate is materialized)

1. Author/overwrite `solution/solution.py` (the single source of truth for the current candidate).
2. Run `./scripts/evaluate_candidate.sh feedback cNNN`. The trusted controller snapshots the source,
   locks a GPU, hashes the source (immutable-ID check), and runs the official evaluator over the 5
   feedback workloads.
3. The controller archives the snapshot under `runs/candidates/cNNN/` (do not hand-edit that tree).
4. Read the evaluator's per-workload pass/fail + timing, then append the candidate record to
   `candidates.jsonl`.

**Discipline:** one `solution/solution.py` edit → one candidate ID → one evaluation → one record.
Never evaluate `cNNN`, then edit source, then evaluate as `cNNN` again.

---

## 1. Operation spec (frozen from `task/definition.json`)

Constants: `NH_Q=96`, `NH_KV=8` (GQA group = 12), `HEAD_DIM=128`, `HALF=64`,
`MAX_POS=262144`, `rope_theta=1e7`, `eps=1e-6`. Inputs bf16 except `position_ids`/`cache_position` int64
and `inv_freq`/`eps` fp32.

Per element the fused math (for each `(b, head, s)` row of length 128):

```
# RMS norm (fp32 accumulate, bf16 round on store)
x32   = x.float()                      # [128]
var   = mean(x32^2) = sum(x32^2)/128   # fp32
inv   = rsqrt(var + eps)               # fp32
xn    = x32 * inv * weight.float()     # fp32, weight is q_norm_weight (Q) / k_norm_weight (K)

# RoPE (halves-paired: dim j pairs with j+64, shared angle theta_j)
theta_j = position_ids[b,s].float() * inv_freq[j]     # j in [0,64)
c_j, s_j = cos(theta_j), sin(theta_j)                 # fp32 (ref rounds to bf16 first — see §5.1)
out[j]    = xn[j]    * c_j - xn[j+64] * s_j
out[j+64] = xn[j+64] * c_j + xn[j]    * s_j           # store bf16

# outputs
query_rotated[b, :96, s, :] = out(Q)
key_rotated[b, :8,  s, :]   = out(K)
key_cache[b, :8, cache_position[s], :]   = out(K)     # rotated K written to cache
value_cache[b, :8, cache_position[s], :] = value[b, :8, s, :]   # V copied verbatim (NOT normed/roped)
```

Cache tensors are mutated **in place** and returned as the same objects; only the `S` selected rows per
`(b, kv_head)` change.

### 1.1 Correctness-critical invariants (must hold in every candidate)

1. Variance accumulated in **fp32**, divided by 128, `+eps` inside `rsqrt`.
2. Weight applied as `weight.float() * xn`, rounded to bf16 on store.
3. RoPE is **halves-paired** `(j, j+64)`, not adjacent-pairs.
4. **V is copied unchanged** into `value_cache`. Never norm/rope V.
5. `position_ids[b,s]` drives the angle; `cache_position[s]` drives the cache row. Read both
   independently (they coincide in these workloads but must not be assumed equal).
6. Cache pointer offsets computed in **int64** (`[B,8,262144,128]` reaches ~5e8 for B=2).
7. In-place scatter touches only `S` rows; all other cache rows stay bit-identical to input.
8. Fresh output tensors via `torch.empty_like`; cache tensors are the passed-in objects.
9. Masked loads/stores for non-power-of-2 `S` (541) and `S=1` (decode).

---

## 2. Feedback workloads (fixed)

| # | uuid-prefix | B | S | cache_len | atol | rtol | character |
|---|---|---|---|---|---|---|---|
| 1 | 6a214883 | 1 | 1024 | 512 | 4.5e-3 | 0.05 | prefill, pos 512..1535 |
| 2 | 13f5b933 | 1 | 541  | 0   | 4.6e-3 | 0.05 | non-pow2 S, pos 0..540 |
| 3 | 2ff5eaf3 | 1 | 1    | 0   | 1e-5 | 0.05 | decode, single token, pos 0 |
| 4 | 052e17c9 | 2 | 512  | 256 | 4.4e-3 | 0.05 | batch 2, pos 256..767 |
| 5 | 39eb0040 | 1 | 2048 | 0   | 3.4e-3 | 0.05 | largest prefill, pos 0..2047 |

Effective per-element tolerance ≈ `max(atol, rtol·|ref|)`. Outputs are O(1) (RMS-normed), so `rtol=0.05`
dominates everywhere except the exactly-zero-angle decode case (workload 3), where the op reduces to pure
RMS norm (`cos=1, sin=0` exactly) and fidelity of the norm path is what matters.

---

## 3. Kernel architecture (baseline = draft Option B)

Two Triton kernels launched from `run(...)`:

- **`qknorm_rope_q_kernel`** — Q only: RMS-norm + RoPE → `query_rotated`. Grid over the 96 Q heads.
- **`qknorm_rope_kv_kernel`** — K+V: RMS-norm + RoPE on K → `key_rotated` **and** `key_cache` scatter;
  copy V → `value_cache` scatter. Grid over the 8 KV heads.

Separating them is natural because Q has no cache write and 12× the head count of K/V; a combined kernel
would need per-program role branching (deferred to Option C, candidate ladder step 5).

### 3.1 Program → work mapping

Program handles one **block of `BLOCK_S` consecutive seq tokens** of one `(b, head)`:

```
grid_q  = (B * NH_Q,  ceil(S / BLOCK_S))
grid_kv = (B * NH_KV, ceil(S / BLOCK_S))
pid_bh  = program_id(0)           # decode to (b, head)
pid_s   = program_id(1)           # seq block
b       = pid_bh // NH            # NH = 96 (Q) or 8 (KV)
h       = pid_bh %  NH
s_off   = pid_s * BLOCK_S + arange(0, BLOCK_S)      # [BLOCK_S]
s_mask  = s_off < S
```

Load the row as two halves `a = x[:, 0:64]`, `b_ = x[:, 64:128]` (each `[BLOCK_S, 64]`, bf16→fp32). This
makes `rotate_half` free (reuse `a`,`b_`) and gives a clean `HALF=64` inner tile.

```
# RMS norm over full 128 = sum over both halves
ssq  = sum(a*a, axis=1) + sum(b_*b_, axis=1)        # [BLOCK_S]
inv  = rsqrt(ssq / 128 + eps)                       # [BLOCK_S]
an   = a  * inv[:,None] * w_lo[None,:]              # w_* = weight halves, fp32
bn   = b_ * inv[:,None] * w_hi[None,:]
# angles
pos    = load(position_ids + b*S + s_off, mask=s_mask).to(fp32)     # [BLOCK_S]
theta  = pos[:,None] * inv_freq[None,:]                             # [BLOCK_S, 64]
c, sn  = cos(theta), sin(theta)
out_lo = an*c - bn*sn                                                # [BLOCK_S,64]
out_hi = bn*c + an*sn
store(out_ptr lo cols, out_lo, mask); store(out_ptr hi cols, out_hi, mask)   # bf16
# KV kernel only:
crow   = load(cache_position + s_off, mask=s_mask).to(int64)        # [BLOCK_S]
cache_off = ((b*NH_KV + h).to(int64)*MAX_POS + crow)*HEAD_DIM
store(key_cache   + cache_off, out(K), mask)
store(value_cache + cache_off, v_row, mask)   # v_row loaded bf16, stored verbatim
```

`inv_freq` (`[64]` fp32), `w_lo`/`w_hi` (`[64]` each) are loaded once per program (tiny, cached in L2).

### 3.2 `constexpr` parameters

- `BLOCK_S` — seq tokens per program (autotune: 1, 2, 4, 8, 16, 32).
- `HALF=64`, `HEAD_DIM=128` — compile-time constants.
- `NH` — passed per launch (96 or 8) as a regular arg (grid decode), not constexpr, to share kernel code;
  or specialize with two constexpr instantiations if the divide/mod cost matters (measure in ladder).
- `num_warps` (1,2,4,8), `num_stages` (1,2,3) — autotune.
- `EMULATE_BF16_XN` (constexpr bool, default False) — optional fidelity lever (§5.1), only if needed.

### 3.3 Launch plumbing (`run`)

- `query_rotated = torch.empty_like(query)`, `key_rotated = torch.empty_like(key)`.
- Pass raw strides for all tensors; do **not** assume contiguity beyond what the evaluator provides
  (derive strides from `.stride()`; the input layout is standard contiguous `[B,H,S,D]`).
- Pass `S`, `NH`, `eps` as scalars; `MAX_POS`, `HEAD_DIM`, `HALF` as constexpr.
- Grid computed on host. `S=1` and `S=541` handled purely by `s_mask` (no special-casing needed).
- Return `(query_rotated, key_rotated, key_cache, value_cache)` — cache objects unchanged identities.
- All data math stays in Triton (no torch cos/sin/rms/scatter on the payload).

---

## 4. Candidate ladder (sequential; each rung is one evaluation)

Each rung changes exactly one lever vs its parent, so the evaluation attributes the effect cleanly.
Rungs after c001 are conditional — a rung is only run if its hypothesis is still live given prior
evidence. Parentage is recorded per record.

| ID | parent | change / hypothesis | keep-if |
|---|---|---|---|
| **c001** | — | Correct baseline: Option B (2 kernels), `BLOCK_S=8`, `num_warps=4`, fp32 internal, no autotune. Establish green + baseline geomean. | all 5 pass |
| **c002** | best | Autotune `BLOCK_S ∈ {1,2,4,8,16,32}` × `num_warps ∈ {1,2,4,8}` (`@triton.autotune` keyed on `S`). | geomean ↑, all pass |
| **c003** | best | Single 128-wide load per row (`[BLOCK_S,128]`) then slice halves in-register vs two 64-loads — test coalescing. | geomean ↑ |
| **c004** | best | Fold B into the seq/grid axis: fully-flattened 1-D grid to cut launch overhead on small-S (esp. workload 3 decode & 541). | geomean ↑ (watch W3) |
| **c005** | best | Option C: single fused kernel over Q and K+V with per-program role branch, to cut one launch. | geomean ↑ and all pass |
| **c006** | best | Micro: hoist `inv_freq`/weights, precompute reciprocals, tune `num_stages`; drop redundant fp32↔bf16. | geomean ↑ |
| **c007+** | best | Only if a specific workload lags: targeted `BLOCK_S`/grid tuning for that regime, or the `EMULATE_BF16_XN` fidelity lever if a borderline correctness miss appears. | resolves the lag |

**Branch policy:** "best" = current best *valid* candidate (all 5 pass, highest geomean). A rung that
regresses geomean or breaks a workload is abandoned; the next rung branches from the prior best, not the
regressor. Record every evaluated rung regardless of outcome.

**Fidelity fallback (correctness, not speed):** if any workload fails on numerics, the first remedy is
`EMULATE_BF16_XN=True` (round `xn` to bf16 and back before RoPE) and/or rounding `cos/sin` to bf16 to
mirror the reference's rounding order — each as a **new** candidate ID. This is a lever of last resort;
`rtol=0.05` on O(1) outputs is expected to make it unnecessary.

---

## 5. Numerical fidelity & risks

### 5.1 Rounding-order difference vs reference
Reference rounds `xn`, `cos`, `sin` to **bf16 before** the RoPE multiply; a fused kernel keeps them fp32
and rounds only the final products. The fused result is *more* precise; worst-case deviation from the
(rounded) reference is ≈ `2^-8·|xn| ≈ 0.004·|xn|`, far inside `rtol·|ref| ≈ 0.05·O(1)`. Being more
precise is safe because tolerance is measured against the rounded reference. If a workload is borderline,
`EMULATE_BF16_XN` reproduces the reference's rounding order to close the gap.

### 5.2 Other risks
| risk | mitigation |
|---|---|
| bf16 variance loss | fp32 accumulate, always |
| wrong RoPE pairing | halves-paired `(j,j+64)`; unit-check formula vs reference |
| int32 offset overflow on cache | int64 cache offsets |
| clobbering untouched cache rows | scatter only `S` rows; return same tensors |
| V accidentally transformed | copy V verbatim; V never enters norm/rope path |
| tiny/non-pow2 S | masked loads/stores; grid via `ceil(S/BLOCK_S)` |
| launch overhead on S=1/541 | fuse to ≤2 kernels; c004/c005 test 1-D grid / single kernel |
| decode tight atol (W3) | angle exactly 0 ⇒ identity RoPE; faithful fp32 RMS suffices |

---

## 6. Correctness checks (per candidate, before appending record)

**A. Static self-review of `solution/solution.py`** (before evaluating):
1. RMS: fp32 accumulate, `/128`, `+eps` inside `rsqrt`, weight `.float()`.
2. RoPE halves-paired `(j,j+64)`; both use `theta_j = pos·inv_freq[j]`.
3. Q path writes only `query_rotated`; K path writes `key_rotated` **and** `key_cache`.
4. V copied verbatim to `value_cache`; V not normed/roped.
5. `position_ids` → angle; `cache_position` → cache row; both loaded independently.
6. Cache offset arithmetic in int64; only `S` rows written; masks correct for `S=1` and `S=541`.
7. Outputs `empty_like`; cache tensors returned as passed-in objects.
8. No torch compute on payload; no non-Triton fallback anywhere.

**B. Official evaluation** — `./scripts/evaluate_candidate.sh feedback cNNN`. This is the *only* source of
correctness truth. Each of the 5 workloads must PASS its `atol/rtol`. Any FAIL ⇒ candidate invalid for
selection (still recorded).

**C. Regime coverage** is already provided by the 5 workloads (decode / non-pow2 / batch>1 / large
prefill / nonzero cache_len). No extra harness is permitted or built.

---

## 7. Performance hypotheses

- **H1 (fusion win):** collapsing 5+ reference launches and eliminating materialization of
  `cos/sin/emb` (`[B,S,128]` bf16 ×3) into 1–2 fused kernels yields a large geomean speedup; the op is
  memory-bound and the reference wastes bandwidth on intermediate tensors. *Test:* c001 geomean > 1.
- **H2 (block/warp tuning):** `BLOCK_S`/`num_warps` materially affect occupancy on this tiny 128-wide
  inner dim; the best config differs between decode (S=1) and prefill (S=2048). *Test:* c002 autotune
  beats fixed c001; per-workload timings shift with `S`.
- **H3 (coalescing):** a single 128-wide load may coalesce better than two 64-wide loads (or vice
  versa). *Test:* c003 vs parent.
- **H4 (launch overhead):** small-S workloads (3: S=1, 2: S=541) are launch/overhead bound; a flattened
  1-D grid (c004) and/or single fused kernel (c005) help those specifically. *Test:* W3/W2 timing ↓.
- **H5 (traffic floor):** Q dominates traffic (96 vs 8 heads); once bandwidth-bound, further kernel
  restructuring yields diminishing returns → signals convergence.

Primary metric: **geometric mean speedup** over the 5 passing workloads vs the reference (as reported by
the evaluator). Every selected workload must pass correctness. Secondary: per-workload speedups to see
which regime binds.

---

## 8. Stopping criteria (convergence)

Stop and write `SEARCH_COMPLETE` when any of:
1. **Converged:** two consecutive ladder rungs improve geomean by **< 3%** each and no live hypothesis
   (§7) remains untested, i.e. the kernel is bandwidth-bound (H5).
2. **Budget:** approaching the token soft limit (~1.0M) with no in-flight improving candidate, or nearing
   the 100-evaluation cap.
3. **Diminishing space:** all ladder rungs c002–c006 explored and best is stable.

`SEARCH_COMPLETE` states the reason, the best candidate ID, its geomean, and the remaining budget.
`final` (13-workload) is **operator-only** and never run without explicit approval.

---

## 9. Evidence format (one JSON object appended to `candidates.jsonl` per evaluation)

Append-only; never rewrite a prior row. Schema:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "timestamp": "2026-09-17T00:00:00Z",
  "source_file": "solution/solution.py",
  "source_sha256": "<hash of the evaluated solution.py>",
  "hypothesis": "Correct Option-B baseline (2 kernels), BLOCK_S=8, num_warps=4, fp32 internal.",
  "change_vs_parent": "initial baseline",
  "config": {"kernels": 2, "block_s": 8, "num_warps": 4, "num_stages": 2, "autotune": false},
  "static_checks": {"rms_fp32": true, "rope_halves_paired": true, "v_verbatim": true,
                    "int64_cache_offset": true, "masks_ok": true, "no_fallback": true},
  "validation": {
    "stage": "feedback",
    "all_pass": true,
    "per_workload": [
      {"uuid": "6a214883", "B": 1, "S": 1024, "cache_len": 512, "pass": true,
       "speedup": 0.0, "ref_ms": 0.0, "cand_ms": 0.0, "max_atol": null, "max_rtol": null},
      {"uuid": "13f5b933", "B": 1, "S": 541,  "cache_len": 0,   "pass": true, "speedup": 0.0},
      {"uuid": "2ff5eaf3", "B": 1, "S": 1,    "cache_len": 0,   "pass": true, "speedup": 0.0},
      {"uuid": "052e17c9", "B": 2, "S": 512,  "cache_len": 256, "pass": true, "speedup": 0.0},
      {"uuid": "39eb0040", "B": 1, "S": 2048, "cache_len": 0,   "pass": true, "speedup": 0.0}
    ]
  },
  "geomean_speedup": 0.0,
  "decision": "keep|reject|new-best",
  "decision_reason": "all pass, baseline established",
  "cumulative_evaluations": 1,
  "skill_usage": "none (KernelWiki is SM90/SM100-only; task is sm_80 Ampere)",
  "notes": "..."
}
```

Rules for filling it:
- `speedup`/`ref_ms`/`cand_ms`/`max_atol`/`max_rtol` copied from the evaluator output (0.0 placeholders
  above are illustrative only). If the evaluator does not emit a field, record `null` and note it.
- `geomean_speedup` = geometric mean of the 5 per-workload `speedup` values (only meaningful if
  `all_pass`; if a workload fails, still record but mark `decision: reject`).
- `decision`: `new-best` if valid and beats prior best geomean; `keep` if valid but not best; `reject` if
  any workload fails or geomean regresses.
- `cumulative_evaluations` increments by 1 per feedback evaluation (5 workloads = 1).
- `source_sha256` is the hash of the exact `solution/solution.py` evaluated (immutability audit).

---

## 10. Execution order (next turns, one action per step)

1. **c001** — write `solution/solution.py` (Option B baseline, §3), static-check (§6A), evaluate,
   append record. Goal: green + baseline geomean.
2. If green, **c002** autotune (§4). Else apply the smallest correctness fix as a new ID.
3. Proceed down the ladder (§4), each rung one lever, branching from current best, recording every
   evaluation.
4. Watch stopping criteria (§8). On convergence, write `SEARCH_COMPLETE`.
5. Do **not** run `final` without operator approval.

No solution code is created in this planning turn; implementation begins at step 1 of §10 in a later turn.

---

## 11. Decision log (live)

- **c001** (baseline, Option B, BLOCK_S=8, num_warps=4, num_stages=2): **all 5 pass**, geomean
  **15.38x**. Per-workload speedups 14.46–17.27x; W3 (decode, angle=0) exact. **new-best.**
  H1 (fusion win) confirmed strongly. Next: **c002** — `@triton.autotune` over
  `BLOCK_S ∈ {1,2,4,8,16,32}` × `num_warps ∈ {1,2,4,8}` keyed on `S` (H2), branching from c001.
- **c002** (autotune BLOCK_S×num_warps, num_stages=1): **all 5 pass**, geomean **15.31x** —
  **reject** (-0.5% vs c001). Autotune only helped W3 decode (17.27→17.73x); prefill W1/W4/W5 each
  regressed slightly. **H2 largely dead**: the op is bandwidth-bound on the 128-wide inner dim, so
  block/warp choice is second-order. best stays **c001**. Next live levers: **c003** single 128-wide
  load per row (H3, coalescing) and **c005** single fused Q+K+V kernel to cut a launch on small-S
  (H4). Branch from c001. If c003 also fails to beat c001, the H5 bandwidth-bound convergence signal
  strengthens.
- **operator final(c001)** (13-workload, out-of-band): c001 scored 12/13 with **one
  INCORRECT_NUMERICAL** at `8f5402ae` (B=4, S=1, cache_len=2048 → decode token at position 2048).
  Root cause: c001 keeps `xn`, `cos`, `sin`, and the RoPE products in **fp32**, while the reference
  rounds each to **bf16** *before* multiplying/summing. At large RoPE angles (`cos`/`sin` span [-1,1])
  this rounding-order mismatch makes a large fraction of a single-position row's elements disagree
  with the (rounded) reference, dropping the matched-ratio below 0.99. The feedback five never
  exposed it because their positions are small (≤2047 but W3's is exactly 0, and the prefill cases
  average over many rows so the matched-ratio stays ≥0.99). This redirected the ladder: fidelity, not
  throughput, is the binding lever. → **c003**.
- **c003** (fidelity: emulate reference bf16 rounding order — round `xn`, `cos`, `sin`, and each RoPE
  product to bf16; config = c001's BLOCK_S=8/num_warps=4/num_stages=2): **all 5 pass**, geomean
  **15.62x** — **new-best** (+1.5% vs c001). Decisive signal: feedback `max_rel` collapsed from
  3.77/5.35 (c001) to 0.87/1.15 (c003), i.e. c003 now tracks the reference's rounding, which is
  exactly what fixes the large-angle `8f5402ae` failure mode. Also marginally faster (bandwidth-bound;
  the extra casts are register-only). c003 is now both the geomean leader and the robust final-run
  choice. Remaining live levers are second-order throughput ideas (H3 single-128 load, H4 single fused
  kernel) whose parent should now be c003 (carry the bf16 emulation forward), but H2 already showed
  block/warp tuning is dead and H5 (bandwidth-bound) is strongly indicated — expected gains are small.
