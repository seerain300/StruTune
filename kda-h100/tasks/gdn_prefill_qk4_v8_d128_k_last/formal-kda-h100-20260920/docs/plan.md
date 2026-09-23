# Plan — `gdn_prefill_qk4_v8_d128_k_last` (GDN prefill, k-last, H100/sm_90)

Status: executable optimization plan. No candidate is implemented or evaluated in this turn.
Builds directly on `docs/draft.md` (operation, constraints, numerical risks, design space).

---

## 0. Guardrails (repeated so they bind every step)

- Work only inside this workspace. External knowledge only via `KernelWiki` / `ncu-report-skill`.
- Primary implementation is **Triton**; PyTorch only for tensor metadata / launch plumbing.
  **No** Torch/CPU/NumPy/CUDA-extension computational fallback. A failing Triton kernel is
  invalid — it is never "rescued" with a fallback; it becomes a recorded failed candidate.
- Immutable candidates: `c001`, `c002`, … one source version at a time, sequential.
  Any meaningful source/config/launch change ⇒ new ID. Never reuse an ID for changed source.
- Evaluate only with `./scripts/evaluate_candidate.sh feedback cNNN` (full 100-workload set =
  **one** evaluation). Budget: **100** evaluations. Tokens: soft 9M / normal 10M / absolute 11M.
- Profiling only via `./scripts/ncu_profile.sh` (ncu-report-skill workflow). **Never** run
  profiling and evaluation at the same time (foreign process on the locked GPU → rc 3, one
  wasted eval). Finish one before starting the other.
- `final` runs only on explicit operator approval. Create `SEARCH_COMPLETE` on genuine convergence.
- Do not modify evaluator, dataset, controller, launcher, shared config, or `feedback_workloads.jsonl`.

---

## 1. Objective and metric model

- **Metric:** geometric mean speedup vs the reference `run(...)`, over all 100 feedback
  workloads; **every** selected workload must pass correctness or the candidate is invalid.
- **Consequence for prioritization:** geomean weights all 100 shapes equally. The census
  (draft §2.4) is: `T≤64`: 28, `65–512`: 32, `513–4096`: 22, `>4096`: 18 (16 exactly 8192).
  So **60 of 100 workloads are ≤512 tokens** and are launch-overhead / fixed-cost bound, while
  the 18 large ones are throughput bound. A candidate that only wins the 8192 cases but regresses
  the small tail can *lose* geomean. Both regimes must be addressed; small-shape overhead is
  first-class, not an afterthought.
- **Baseline is very slow** (pure sequential fp32 double-loop in Python), so even a naive but
  correct fused Triton kernel should show large speedups everywhere. The real contest is between
  our own candidates, not against the reference.

---

## 2. Local offline scaffolding (host-only, no GPU kernel, no eval budget)

Before any candidate, build a **non-evaluated** local harness under `docs/` or a scratch file
that does NOT touch the evaluator. Purpose: de-risk math and transpose bugs cheaply.

- **T1 — algebra oracle (float64, CPU):** reimplement the reference recurrence and the compact
  form (draft §1.4); confirm equality on random asymmetric toy tensors. Already done once for the
  compact reduction; extend to the chunked/WY math *before* writing chunked Triton.
- **T2 — transpose oracle:** with an **asymmetric** toy (fake K=3, V=5) confirm the k-last
  `[V,K]`↔`[K,V]` mapping for both `state` input and `new_state` output. K=V=128 hides this bug.
- **T3 — gate oracle:** confirm the stable `softplus` identity
  `softplus(x)=max(x,0)+log1p(exp(-|x|))` matches `F.softplus` to fp32 over a wide `x` range, and
  that `g=exp(-exp(A_log)*softplus(x))` stays in `(0,1]`.

These use only Python/torch on CPU for *design validation of our own math*; they are not a
solution fallback and never appear in `solution/solution.py`.

---

## 3. Candidate lineage strategy

Linear-with-branches lineage. Each candidate changes **one** conceptual thing so its eval
attributes cause→effect cleanly. Parent is the best *valid* ancestor unless noted.

```
c001  Correctness anchor + tolerance probe        (algorithm A: token-recurrent, fp32)
  │   → learn: does it pass everywhere? what is the effective tolerance headroom?
  ├─ c002  Fuse gate computation + remove per-seq launch overhead (single/persistent launch)
  │        parent = c001                            (algorithm A, faster plumbing)
  ├─ c003  Chunked matmul delta rule (algorithm B/WY), chunk C=64, fp32 accumulate
  │        parent = best valid of {c001,c002}       ← throughput path for large shapes
  │     ├─ c004  Tune chunk size C ∈ {32,64,128}
  │     ├─ c005  Head batching / GVA q,k reuse (2 v-heads per q/k tile)
  │     ├─ c006  num_warps / num_stages / pipelining sweep on the chunk loop
  │     └─ c007  Small-shape specialization: route T≤threshold to the c002 path,
  │              large T to the chunked path (single dispatcher, two code paths)
  └─ (contingency) c00x  hybrid C (chunk-across-seq + sequential inner) if B fails tolerance
```

Branch rule: if a chunked candidate **fails correctness** on the fp32 `new_state` check, do not
keep pushing chunk size; fall back to the hybrid (C) or reduce reordering (smaller C, sequential
inner) as an explicit new candidate, and record why.

Fork rule: only fork from a **valid** parent (passes all 100). If the current line is invalid,
the parent for the next attempt is the last valid candidate, and the change is the *fix*.

---

## 4. Candidate specifications (executable detail)

### c001 — token-recurrent correctness anchor + tolerance probe
- **Parent:** none.
- **Hypothesis:** a straightforward fused Triton kernel that mirrors the reference recurrence
  order will (a) pass every workload and (b) reveal how much numerical headroom the evaluator
  allows, which dictates whether chunked reordering is safe later.
- **Design:** grid over `(num_seqs × num_v_heads)` = N×8 programs. Each program:
  loads its sequence slice via `cu_seqlens`; loads/zeros `S[128,128]` in **fp32** (k-last
  transpose handled by strided load); computes gates in fp32 (stable softplus); loops tokens with
  the compact rank-1 update `S = g·S + kᵀ·(β·(v − k·(g·S)))`; writes `output=scale·(q·S)` in bf16
  and `new_state` back in k-last fp32. q/k indexed by `hv//2` (no repeat_interleave materialize).
- **Config:** `num_warps=4` start; no autotune yet (keep it simple/deterministic).
- **Correctness focus:** exact-order match → tightest possible agreement with reference; this is
  the anchor other candidates are diffed against.
- **Success:** all 100 pass. **Primary deliverable beyond speed:** per-workload pass margins to
  infer tolerance. If any fail, first suspect the k-last transpose (T2 oracle) or fp32/bf16 cast.

### c002 — launch-overhead reduction (small-shape win)
- **Parent:** c001.
- **Hypothesis:** the 60 workloads ≤512 tokens and the many `N=1` cases are dominated by
  kernel-launch / fixed overhead; collapsing to a **single kernel launch** (persistent grid that
  internally iterates sequences, or one grid sized N×8 with in-kernel seqlen lookup and no host
  Python loop) improves geomean via the small tail without touching the math.
- **Change vs c001:** only the launch/plumbing structure and gate fusion; recurrence identical.
- **Success:** geomean ↑, no correctness regression. If neutral, keep whichever is simpler as the
  base for c003.

### c003 — chunked matmul delta rule (throughput path)
- **Parent:** best valid of {c001, c002}.
- **Hypothesis:** replacing the O(T) serial loop with `T/C` chunk steps of `[C,128]×[128,128]`
  matmuls (tensor cores via `tl.dot`, fp32 accumulate) yields large speedups on the 18 big
  workloads while the tolerance headroom from c001 keeps `new_state` valid.
- **Design:** per (seq, head) chunk loop with WY/UT intra-chunk resolution (draft §4.2 B):
  intra = causal-masked `(Q̃K̃ᵀ)U`; inter = `Q̃·S`; carry `S ← γ_C·S + K̃ᵀU`. Multiply-by-decay
  formulation (avoid divide-by-γ). Ragged final chunk masked by seqlen. fp32 accumulate.
- **Chunk size:** start **C=64** (draft §3 decay-stability rationale).
- **Success:** geomean ↑ vs parent AND all 100 still pass. **Watch:** fp32 `new_state` drift on
  the longest sequences (~2854 tokens/seq). If it fails, branch to smaller C / hybrid.

### c004–c006 — tuning (only after c003 is valid)
- **c004 chunk size** C∈{32,64,128}: MFU vs tail-waste vs decay stability. One value per candidate.
- **c005 head batching / GVA reuse:** load q/k tile once, serve v-heads `2h,2h+1`; halve q/k traffic.
- **c006 warps/stages:** sweep `num_warps∈{2,4,8}`, `num_stages∈{2,3,4}` guided by ncu (see §6).
Each is an isolated single-knob change with its own ID.

### c007 — dual-path dispatcher
- **Parent:** best valid tuned chunked candidate.
- **Hypothesis:** a size threshold `T*` routing small sequences to the low-overhead c002-style path
  and large ones to the chunked path beats a single path on geomean (best of both regimes).
- **Change:** launcher-side branch on per-sequence (or per-batch) length; both kernels already
  validated individually. Choose `T*` from measured crossover, not guessed.

Contingency candidates get the next free ID and a recorded rationale; the numbering stays
sequential and immutable.

---

## 5. Correctness checks (per candidate, before spending an eval)

Cheap gate → expensive gate ordering, to avoid wasting eval-budget on broken kernels:

1. **Static:** source compiles/imports; `run(...)` signature matches; handles `state=None`;
   outputs have exact shapes/dtypes (`output` bf16 `[T,8,128]`, `new_state` fp32 `[N,8,128,128]`).
2. **Offline oracle (host, no eval budget):** run the candidate's `run` on a *few* small
   self-constructed tensors and compare to the in-repo reference `run` (copied read-only from
   `definition.json` into a scratch checker) at fp32 with an asymmetric-shape transpose probe.
   Must match to ~1e-4 rel before we spend an eval. This is validation of *our* output vs the
   *reference formula*, run locally — not a shipped fallback.
3. **Only then** spend one `./scripts/evaluate_candidate.sh feedback cNNN`.

If the evaluator returns rc 3 (interference), the measurement is void — do **not** count it as a
result; re-run when the GPU is clean, and never have profiling running concurrently.

Every evaluated candidate must have **all 100 workloads pass**; a single failure ⇒ candidate is
invalid regardless of speed, and the next candidate is a fix branched from the last valid one.

---

## 6. Performance hypotheses (falsifiable, mapped to candidates)

| # | Hypothesis | Tested by | Kill/keep signal |
|---|---|---|---|
| H1 | A fused fp32 Triton recurrence passes all 100 and is already ≫ reference. | c001 | all pass + large geomean ⇒ keep as anchor |
| H2 | Small shapes (≤512, N small) are launch/overhead bound; one-launch design lifts geomean. | c002 | geomean ↑ mainly from small-shape rows |
| H3 | Chunked matmul (C=64, fp32 accum) is algebraically-close enough to pass and much faster on 8192 cases. | c003 | big-shape speedup + still 100/100 |
| H4 | Chunk size has an interior optimum (too small = overhead, too big = decay drift + tail waste). | c004 | geomean vs C is unimodal |
| H5 | GVA q/k reuse cuts memory traffic with no accuracy cost. | c005 | geomean ↑, correctness unchanged |
| H6 | warps/stages tuning closes a measured pipeline-stall / occupancy gap. | c006 + ncu | ncu shows stall reason improved |
| H7 | A dual-path dispatcher beats any single path on geomean. | c007 | geomean ↑ over both single paths |

**Profiling policy (H6 and diagnosis):** use `./scripts/ncu_profile.sh --set basic -o profile/<id>
python <harness>` via the ncu-report-skill on **one large workload** (e.g. an 8192/32-seq case)
to read the dominant stall reason (mem vs SFU-from-gates vs tensor-core idle) and occupancy.
Never while an eval is running. Small shapes are addressed by design (launch count), not profiling.

---

## 7. Evidence format (one JSON object appended per evaluated candidate to `candidates.jsonl`)

Append-only; never rewrite earlier records. Each record:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<ISO8601>",
  "hypothesis": "<one-line falsifiable claim>",
  "change_vs_parent": "<the single conceptual change>",
  "algorithm": "token-recurrent | chunked-WY | hybrid | dispatcher",
  "config": {"chunk_size": null, "num_warps": 4, "num_stages": 1, "grid": "NxHV"},
  "offline_check": {"max_rel_err_vs_reference": <float>, "asym_transpose_probe": "pass"},
  "validation": {"all_workloads_pass": true, "num_pass": 100, "num_fail": 0},
  "per_workload": [
    {"uuid": "<uuid>", "total_seq_len": 8192, "num_seqs": 32,
     "passed": true, "speedup": <float>}
  ],
  "geomean_speedup": <float>,
  "decision": "keep-as-base | superseded-by-cXXX | rejected(reason) | invalid(reason)",
  "cumulative_evaluations": <int>,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "notes": "<tolerance-headroom findings, stall reason, next step>"
}
```

Rules: record parent, source hash, hypothesis, validation, **per-workload** result, geomean,
decision, cumulative eval count, and skill usage for every candidate (per CLAUDE.md §7). If a
candidate fails correctness, still append a full record with `validation.all_workloads_pass=false`
and the failing rows, then branch a fix. `decision` must name the successor when superseded.

---

## 8. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when **any** of:

1. **Budget:** 100 evaluations reached.
2. **Token limit:** approaching the 9M soft limit — wind down toward a clean stop before 10M;
   never exceed 11M.
3. **Convergence:** three consecutive candidates fail to improve geomean by a meaningful margin
   (e.g. < ~1–2% relative) and no untested hypothesis in §6 remains with plausible upside.
4. **Design exhaustion:** the throughput path is compute/roofline-bound per ncu and the small-shape
   path is launch-bound to the floor, i.e. no structural lever left.

At stop, the best **valid** candidate (all 100 pass, highest geomean) is the submission. `final`
is run **only** after explicit operator approval, as a single full evaluation of that candidate.

---

## 9. Immediate next actions (next turn)

1. Build the offline oracles T1–T3 (§2) — host-only, no eval budget.
2. Implement **c001** (§4) as `solution/solution.py`.
3. Run offline correctness gate (§5 steps 1–2).
4. Evaluate `c001` once; append its evidence record (§7); read tolerance headroom.
5. Decide c002 vs jump-to-c003 based on where geomean leaves the most on the table.

---

## 10. Decision log

- **c001 (evaluated, VALID).** Token-recurrent anchor. 100/100 pass; geomean **89.56x**
  (amean 105.47x, min 17.40x, max 402.51x). Evaluator tolerance revealed: **atol=rtol=0.01,
  matched_ratio=0.99** — loose, so summation-order changes (chunking/splitting) are safe
  numerically; the fp32-state-drift worry from draft §3 is not the binding constraint.
  Cost structure: lowest speedups are (a) launch-bound tiny single-seq shapes (12–40 tok,
  29–40x) and (b) **occupancy-starved long, low-`num_seqs` shapes** (5709/2-seq 71.8x,
  2107/1-seq 88.6x, 8192/20-seq 163x) where only `num_seqs*8` programs exist to cover 132 SMs.
  The many mid shapes already sit 90–160x.
- **c002 (next, planned).** Change vs c001: **V-column splitting** — partition the state's
  independent V=128 columns into `V/BV` blocks handled by separate programs, so the grid
  becomes `num_seqs * num_v_heads * (V/BV)`. V columns are mathematically independent given the
  per-token k-reduction over K, so this is exact (no tolerance risk). Expected to lift the
  occupancy-starved (b) cases most and thus the geomean, at the cost of recomputing gates/k per
  column block (cheap). This supersedes the earlier c002 "launch-overhead" idea, which the data
  shows is secondary to occupancy on the low-`num_seqs` long shapes. If V-splitting underdelivers
  on the throughput frontier, c003 (chunked matmul) remains the next lever.

- **c002 (evaluated, VALID).** V-column splitting with host-side adaptive `NSPLIT` (power-of-2,
  applied only when `num_seqs*8 < 128`). **Geomean 89.56x → 130.25x (+45.4%)**, 100/100 pass,
  numerics identical to c001 (confirms exactness). Big broad gains exactly where predicted — the
  occupancy-starved low-`num_seqs` regime: 134/1-seq 80→138x, 983/2-seq 91→181x, 574/2-seq
  107→202x, 1377/1-seq 88→177x, 2107/1-seq 88→179x, 5709/2-seq 72→124x, 4124/15-seq 299→482x.
  High-`num_seqs` 8192 cases were intentionally left unsplit and are unchanged (as designed).
  Only noise-level movement on the tiniest launch-bound shapes (6/1-seq 17.4→16.8x).
- **c003 (next, planned).** Change vs c002: **chunked / matmul (WY) delta rule**. The remaining
  headroom is the ~13 high-`num_seqs` 8192 workloads at 150–240x: they now have adequate
  occupancy but each program serially walks ~140–410 tokens doing rank-1 outer products (poor
  tensor-core use). Convert the per-token loop into chunked `[C,128]×[128,128]` `tl.dot` matmuls
  (UT/WY intra-chunk resolution, inter-chunk state carry with multiply-by-decay). Loose tolerance
  (atol/rtol 0.01) makes chunk-order reordering safe. Risks: ragged-tail masking at sequence
  ends; avoid divide-by-γ (multiply-by-decay form). Keep the c002 V-split/low-occupancy path for
  small shapes — likely a dispatcher (chunked for long sequences, c002 path for short) so the
  broad c002 wins on the 60 small/mid shapes are preserved.
