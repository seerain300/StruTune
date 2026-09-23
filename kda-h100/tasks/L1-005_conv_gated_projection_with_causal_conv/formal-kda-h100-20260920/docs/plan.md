# Plan — L1/005 `conv_gated_projection_with_causal_conv`

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`, `TASK.md`,
`task/definition.json`, `task/feedback_workloads.jsonl`, and CLAUDE.md rules.

Target: H100 `sm_90`. Compute must be **Triton**; Torch only for metadata/launch. No fallback.
Budget: 100 evaluations, 9M-token soft limit. `final` is operator-only.

---

## 0. Ground rules (operational, re-stated for execution)

- One immutable candidate `cNNN` per meaningful source/config/launch change. Never reuse an id
  for changed source; never rewrite earlier `candidates.jsonl` records (append-only).
- Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN`. One full 16-workload run =
  one evaluation. Correctness must pass on **all 16** for a candidate to be valid.
- Profiling only via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow), **never**
  concurrent with an evaluation (co-resident process ⇒ return code 3, discarded, wasted eval).
- No local CUDA / `nvidia-smi` / direct `ncu` / alternate harness. Local Python exec is denied
  in this sandbox, so correctness is argued on paper first; the evaluator is the ground truth.
- A failing Triton kernel is invalid — fix it, never route around with Torch/CPU/NumPy.
- Write `SEARCH_COMPLETE` only when geomean genuinely converges; never run `final` without
  explicit operator approval.

---

## 1. Reference contract (frozen — the target to match)

Per token, in bf16 at every stage (each stage rounds to bf16). `H=2048`, `K=4`, `3H=6144`,
`N=B·S`.

Chunk map (verified in draft §1.2):
- `Bg[b,s,h] = BCx[b,s, h]`      (rows `[0:H]`, input gate)
- `Cg[b,s,h] = BCx[b,s, H+h]`    (rows `[H:2H]`, output gate)
- `V [b,s,h] = BCx[b,s, 2H+h]`   (rows `[2H:3H]`, value)

Compute:
```
BCx    = x @ in_proj_weightᵀ + in_proj_bias          # F.linear: out[n,o]=Σ_k x[n,k]·W[o,k]
Bx     = Bg * V                                       # bf16
conv[b,s,h] = conv_bias[h] + Σ_{k=0..3} conv_weight[h,0,k] · Bx[b, s-3+k, h]   # Bx[<0]=0, per-batch
y      = Cg * conv                                    # bf16
output = y @ out_proj_weightᵀ + out_proj_bias         # F.linear
```
Causal taps: `w0·Bx[s-3] + w1·Bx[s-2] + w2·Bx[s-1] + w3·Bx[s] + bias`. Left padding is
per-sequence; halo must not cross batch boundaries.

Accumulation policy to match cuBLAS/cuDNN: bf16 loads, **fp32 accumulate**, bias added in fp32,
round to bf16 at the same points the reference does (`Bx`, `conv`, `y`, both GEMM outputs).

---

## 2. Architecture of the solution

`solution/solution.py` exposes `run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias,
out_proj_weight, out_proj_bias)` returning `(B,S,H)` bf16.

Kernel decomposition (draft §5.1). Flatten `(B,S)→N` for GEMMs (per-token independent); keep
explicit `(b, local-s)` indexing in the conv kernel so causal masking is per batch.

- **Option A (3 kernels, correctness-first baseline):**
  - **K1 `triple_gemm_gate`** — tile `BLOCK_M` tokens × `BLOCK_H` channels, contract `K=H`,
    accumulate three products (`Bg`,`Cg`,`V`) from weight row-blocks `h`, `H+h`, `2H+h` sharing
    one `x` tile. Epilogue: `Bx=Bg*V` (store bf16), store `Cg` (bf16). Grouping by hidden channel
    is mandatory so `Bx` pairs columns `h`/`2H+h`.
  - **K2 `causal_conv_gate`** — memory-bound; load `Bx` tile + 3-row halo (masked at seq/batch
    boundaries), 4-tap FMA with per-channel `conv_weight`/`conv_bias`, multiply by `Cg`, store `y`.
  - **K3 `out_gemm`** — `(N×H)·(H×H)ᵀ + out_proj_bias`, bf16, fp32 accumulate.
- **Option B (2 kernels):** fuse K2 into the K3 prologue — materialize the `y` tile on the fly
  (conv+gate) then feed directly as the `A` operand of `out_proj`. Removes a full `y`
  round-trip. Adopt only after Option A numerics are trusted.
- **Option C (mega-kernel):** deferred / likely rejected (cross-tile conv coupling + double
  contraction). Only if evidence shows a large residual gain.

`tl.dot` with bf16 operands + fp32 accumulate → wgmma on sm90 (KernelWiki `lang-triton`). Weight
`B`-operand addressed transposed via strides to satisfy `out[n,o]=Σ_k in[n,k]·W[o,k]`.

---

## 3. Candidate lineage strategy

Front-load correctness (cheap to reason, expensive to get wrong with no local run); back-load
throughput tuning (guided by ncu). Prefer many autotune configs **inside** one candidate over
many single-config candidates, to conserve the eval budget.

| id | parent | change | hypothesis (what & why) | primary goal |
|---|---|---|---|---|
| **c001** | — | Option A, fixed conservative tiles, full masking, no autotune/persistence | Removing transposes/pad/contiguous + fusing gating already beats the reference on small/medium `N`; proves correctness on all 16 | correctness + baseline geomean |
| **c002** | c001 | add `triton.autotune` to both GEMMs keyed on shape/`N` regime | tiny-`N` wants small tiles (low overhead), large-`N` wants throughput tiles; single fixed tile is suboptimal across 32× `N` range | geomean, esp. tails |
| **c003** | best(c001,c002) | tune conv kernel (`BLOCK_M`/`BLOCK_H`, vectorized bf16 loads, tap-as-register-shift) | conv is pure memory traffic; coalesced/vectorized loads + halo-in-registers reduce its cost | small/medium `N` |
| **c004** | best | Option B: fuse conv+gate into out_proj prologue | removes full `y` (2·N·H·2 bytes) write+read; biggest at medium `N` | traffic-bound geomean |
| **c005** | best | persistent / tile-scheduling for large-`N` GEMMs (swizzle, `num_stages` sweep) | reduce tail effects & improve L2 locality at `N=8192` (KernelWiki `pattern-tail-effect`, `technique-tile-scheduling`) | large-`N` geomean |
| **c006+** | best | targeted, ncu-guided (dtype policy, K1 three-accumulator layout, split-K if `K=2048` underfills) | address the specific bottleneck ncu attributes | residual |

Rules for lineage:
- Each row is a distinct `cNNN` with its own source hash. Only advance the parent to a candidate
  that (a) passed all 16 and (b) improved geomean (or is neutral but enables a later step).
- If a candidate regresses or fails, keep the parent as the working best and branch a new id with
  a corrected hypothesis; do not mutate the failed id.
- Abandoned branches are recorded (decision=`reject`) but never deleted.

---

## 4. Correctness checks (per candidate, before spending an eval)

Paper checklist (must all pass by inspection before evaluation):

1. **Chunk map**: `Bg=rows[0:H]`, `Cg=rows[H:2H]`, `V=rows[2H:3H]`; `Bx=Bg*V`, `y=Cg*conv`.
2. **Weight transpose in `tl.dot`**: both GEMMs compute `Σ_k in[n,k]·W[o,k]` (weight is `[out,in]`).
3. **Causal taps**: `conv[s]=bias+Σ_{k=0..3} w[k]·Bx[s-3+k]`, `Bx[s'<0]=0`.
4. **Per-batch boundary**: halo reads masked when `local-s < 0`; a token tile never pulls halo
   from a different batch. Verify with the flatten scheme (`row=b·S+s` ⇒ guard on `s`, not `row`).
5. **Ragged-`S` / remainder masking**: `N%BLOCK_M`, `S%BLOCK_S`, `H%BLOCK_H` all masked.
6. **Bias**: add in fp32 accumulator, then round to bf16; biases are bf16 inputs.
7. **Rounding placement**: `Bx`,`conv`,`y`, GEMM outputs rounded to bf16 to mirror the reference.
8. **dtype/shape of `run` output**: `(B,S,H)` bf16, contiguous.

Evaluator-driven checks (the ground truth):
- **Correctness canaries** (must pass or the candidate is invalid): ragged `S` → `b662596d`
  (4×541), `36e77f89` (2×1879), `d9432839` (8×997); large-`B`/short-`S` → `121c8e28` (16×256),
  `58e3ac47` (32×256), `dcc98535` (2×128). These exercise per-batch causal masking and remainders.
- Read per-workload pass/fail + timing from evaluator output; a single failure ⇒ fix as a new id.
- Tolerances are bf16-level (`atol` 0.008–0.028, `rtol` 0.05); fp32 accumulation keeps us well
  inside them. If a candidate fails *only* on a cancellation-sensitive workload, consider keeping
  `conv` in fp32 (more accurate than reference) as a new candidate — but bf16 rounding is default.

---

## 5. Performance hypotheses (with evidence to gather)

- **H1 — Fusion beats reference at small/medium `N`.** Reference does ~8 launches + transposes +
  pad + contiguous. Collapsing to 2–3 kernels removes launch overhead and redundant `N·H`/`N·3H`
  passes. *Evidence:* c001 per-workload speedup on `N≤2048` workloads; expect the biggest relative
  wins there.
- **H2 — `in_proj` (75% of FLOPs) dominates large-`N`.** At `N=8192` the two GEMMs set the floor;
  Triton `tl.dot` must approach cuBLAS bf16 or the fusion win erodes. *Evidence:* ncu on c001/c002
  attributing time across K1/K2/K3; SM/tensor-core utilization of K1 vs K3.
- **H3 — Autotune closes the `N`-range gap.** One tile can't serve 256 and 8192. *Evidence:*
  c002 geomean vs c001, decomposed into tiny/medium/large buckets.
- **H4 — conv is memory-bound and cheap once vectorized.** `O(N·H·K)` FLOPs but several loads.
  *Evidence:* ncu DRAM-throughput / achieved-bandwidth on K2; c003 delta on small/medium `N`.
- **H5 — Option B removes a full `y` round-trip.** Saves `2·N·H·2` bytes W+R. *Evidence:* c004 vs
  best on medium-`N` (traffic-bound) workloads; ncu DRAM bytes before/after.
- **H6 — Large-`N` GEMM has tail/scheduling slack.** *Evidence:* ncu wave/occupancy + tail on
  `N=8192`; c005 delta on the three `N=8192` workloads.

Profiling protocol: after a *correct* baseline exists, run ncu **serially** (never during an
eval) with the ncu-report-skill workflow via `./scripts/ncu_profile.sh --set … -o profile/<tag>
python <harness>`; attribute time per kernel, then form the next candidate's hypothesis. Choose
one or two representative shapes (e.g. a large `N=8192` and a small `N=256`) to keep profiling
cheap; do not profile all 16.

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- **Convergence:** best geomean improves < ~1–2% across two consecutive accepted candidates, and
  ncu shows the dominant kernel near a hardware roofline (tensor-core-bound GEMM or
  bandwidth-bound conv) with no clear remaining lever.
- **Budget:** approaching the 100-evaluation cap or the 9M-token soft limit — leave margin to
  record evidence and (on operator approval) run `final`.
- **Diminishing branches:** the last 2–3 hypotheses each failed to beat the working best.

At stop: identify the best valid candidate (passes all 16, highest geomean), record it, and wait
for **explicit operator approval** before any `final` run.

---

## 7. Evidence format (append one JSON object per evaluated candidate to `candidates.jsonl`)

Append-only; never rewrite prior records. Each object contains:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py as evaluated>",
  "hypothesis": "Option A fused 3-kernel pipeline removes transposes/pad/contiguous and fuses gating; establishes correctness on all 16 and a baseline geomean.",
  "change_from_parent": "initial implementation",
  "kernels": ["triple_gemm_gate", "causal_conv_gate", "out_gemm"],
  "config": {"tiling": "fixed", "autotune": false, "fusion": "optionA"},
  "validation": {
    "paper_checklist_passed": true,
    "all_workloads_pass": true,
    "correctness_canaries": {"b662596d": "pass", "36e77f89": "pass", "121c8e28": "pass", "58e3ac47": "pass", "d9432839": "pass", "dcc98535": "pass"}
  },
  "per_workload": [
    {"uuid": "dcc98535-...", "B": 2, "S": 128, "N": 256, "pass": true, "speedup": 0.0},
    {"uuid": "...", "...": "..."}
  ],
  "geomean_speedup": 0.0,
  "decision": "accept | reject | keep-as-baseline",
  "decision_reason": "…",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki:lang-triton", "KernelWiki:technique-kernel-fusion"],
  "notes": "ncu findings, next hypothesis, anomalies"
}
```

Field rules:
- `per_workload` lists all 16 with `pass` and measured `speedup` (baseline/candidate as reported
  by the evaluator); include `B`,`S`,`N` for readability.
- `geomean_speedup` = geometric mean of per-workload speedups over the 16 (primary ranking metric).
- `cumulative_evaluations` = running count of feedback evaluations spent (monotonic).
- `decision` records whether this candidate becomes the new working best.
- `skills_used` records each KernelWiki page or ncu-report-skill invocation that informed the
  candidate.
- A candidate that fails correctness records `all_workloads_pass=false`, the failing uuid(s),
  `decision="reject"`, and the corrective hypothesis for the next id.

Profiling artifacts go under `profile/<tag>/` and are referenced by path in `notes`; they are not
evaluations and do not increment `cumulative_evaluations`.

---

## 8. Execution order (next turns)

1. Implement `solution/solution.py` as **c001** (Option A), completing the §4 paper checklist.
2. Evaluate c001 via `./scripts/evaluate_candidate.sh feedback c001`; record per §7.
3. If any workload fails → fix as `c002`(+) with corrected hypothesis; else profile (serially)
   and proceed down the §3 lineage (autotune → conv tuning → Option B → persistence).
4. Advance the working best only on validated improvements; log every candidate.
5. Stop per §6; write `SEARCH_COMPLETE`; await operator approval for `final`.

No candidate is implemented or evaluated in this turn — this document is the plan only.

---

## 9. Decision log

- **c001 (accept, working best).** Option A fused 3-kernel pipeline. **16/16 pass**, geomean
  **1.2918x** (avg 1.2973x). Speedup is largest at small `N` (1.42–1.49x: `66fd2ad8`, `dcc98535`,
  `b0a9e3f0`, `8b678b13`) and smallest at large `N` (1.10–1.13x: `58e3ac47` 32×256, `6033fd0f`
  2×4096, `d9432839` 8×997, `01719394` 1×8192). Confirms H1 (fusion win at small `N`) and H2
  (untuned GEMMs set the floor at large `N`). Correctness canaries (ragged-`S` + large-`B`) all
  pass, so per-batch causal masking and remainder masking are correct. cumulative_evaluations=1.
  Next: **c002** — add `triton.autotune` to K1/K3 GEMMs keyed on the `N` regime to lift the
  large-`N` tail; optionally ncu-profile a large-`N` and a small-`N` shape first (serially, never
  during an eval) to confirm K1 `in_proj` dominates at large `N`.
- **c002 (accept, new working best).** Added `triton.autotune` (keyed on `N,H`) to the two
  GEMMs K1/K3; K2 conv unchanged. **16/16 pass**, geomean **1.4833x** (avg 1.4901x), up from
  c001's 1.2918x (**+14.8%**). The large-`N` tail was lifted exactly as H2/H3 predicted:
  `6033fd0f` 1.12→1.48, `58e3ac47` 1.11→1.47, `d9432839` 1.13→1.53, `01719394` 1.13→1.50, and
  medium-`N` climbed to 1.5–1.70x. **Regression at the two tiniest `N=256` workloads**
  (`66fd2ad8` 1.49→1.24, `dcc98535` 1.45→1.27) and ragged `b662596d` (1.27→1.22): at very small
  `N` the GEMMs are trivial and the fixed conv K2 (BLOCK_S/H=64) + per-launch overhead now
  dominate. cumulative_evaluations=2. Next: **c003** tune K2 conv (vectorized bf16 loads, larger
  BLOCK_H, tap-as-register-shift) to recover tiny-`N`, or **c004** Option B (fuse conv+gate into
  out_proj prologue) to drop the `y` round-trip. Tiny-`N` is now K2/overhead-bound, not
  GEMM-bound.
- **c003 (accept, new working best).** Added canonical `GROUP_M=8` super-grouping to the
  program-id→(pid_m,pid_n) mapping of both GEMMs (K1/K3) for L2 weight-tile reuse (pure bijective
  pid remap; results unchanged). K2 unchanged. **16/16 pass**, geomean **1.5012x** (avg 1.5066x),
  up from c002's 1.4833x (**+1.2%**). Helped the largest-`N` GEMMs (`6033fd0f` 1.48→1.52,
  `d9432839` 1.53→1.55, `01719394` 1.50→1.54, `121c8e28` 1.50→1.55) and recovered part of the
  tiny-`N` regression (`66fd2ad8` 1.24→1.26, `dcc98535` 1.27→1.30, `b662596d` 1.22→1.27). Modest
  gain ⇒ GEMM scheduling is near its ceiling. cumulative_evaluations=3. **Remaining lever:** the
  tiny/medium-`N` regime where the separate conv K2 + full `y` DRAM round-trip dominate. Next:
  **c004** Option B — fuse conv+gate into the out_proj (K3) prologue so `y` never touches DRAM
  (saves 2·N·H·2 bytes), most helpful at `N=256..2048`.
- **c004 (reject).** Option B: fused conv+gate into the out_proj prologue. **16/16 pass**
  (numerically correct, all canaries pass) but **geomean 0.8696x** — a hard regression, slower
  than the reference on 14/16 workloads. Root cause: the fused kernel recomputes the conv+gating
  once per output-channel tile (`num_pid_n = H/BLOCK_N = 8..16×`), and those redundant
  memory-bound `bx` tap loads + `cg` loads far outweigh the single `y` round-trip saved. Worst on
  ragged `b662596d` (0.67x). This **falsifies H5** — the `y` round-trip was not the bottleneck;
  Option B is the wrong fusion for this shape. **c003 remains the working best (1.5012x);
  the Option B branch is abandoned.** cumulative_evaluations=4. Next: **c005** branch from c003 —
  keep the standalone 3-kernel structure but tune the conv K2 kernel (autotune larger BLOCK_H,
  wider vectorized bf16 loads) so `y` is computed exactly once but that single pass is faster;
  targets tiny-`N` still capped ~1.26–1.30x. Orthogonal to the failed fusion.
- **c005 (reject).** Branched from c003; autotuned the standalone conv K2 kernel (8 configs over
  BLOCK_S/BLOCK_H/num_warps, key=[S,H]). **16/16 pass**, geomean **1.4588x** — below c003's
  1.5012x (**−2.8%**). Conv autotune did not help and slightly hurt: tiny-`N` regressed
  (`66fd2ad8` 1.26→1.09, `dcc98535` 1.30→1.15). Falsifies H4 for this workload set — the conv K2
  was not the bottleneck and its fixed 64×64 tile (c003) was already adequate; the `key=[S,H]`
  conv autotune likely mis-shares configs across same-`S` different-`B` shapes and adds do_bench
  overhead. **c003 remains the working best (1.5012x); conv-tuning direction abandoned.**
  cumulative_evaluations=5. Two consecutive non-improvements (c004 0.87x, c005 1.46x); c002→c003
  was already only +1.2%. Next (final lever): **c006** = c003 with the fixed conv restored + an
  expanded K3 out_proj GEMM config space (add BLOCK_K=128 to halve the K=2048 loop). If c006 also
  fails to beat c003, declare convergence (3 consecutive non-improvements + main axes exhausted)
  and write SEARCH_COMPLETE with c003 as best.
