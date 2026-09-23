# Plan — L1/008 Expert-Output Weighted Index-Add Accumulation

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`,
`task/definition.json`, `task/feedback_workloads.jsonl`, `TASK.md`, and `CLAUDE.md`.
This turn produces the plan only — no candidate is implemented or evaluated.

---

## 0. Goal and ground rules

- **Objective:** maximize geometric-mean speedup over the reference across the full
  16-workload feedback set, with *every* workload passing correctness (atol per row of the
  table in the draft, rtol 0.05).
- **Compute in Triton only.** PyTorch allowed only for metadata, output allocation, and
  launch plumbing. The accumulation adds must live in the Triton kernel. No Torch / CPU /
  NumPy / CUDA-extension computational fallback. A failed Triton kernel is invalid — never
  swap in a torch fallback.
- **Sequential, immutable candidates.** Implement `c001`, `c002`, … one source version at a
  time. Any meaningful source/config/launch change → new candidate ID. Never reuse an ID or
  rewrite an earlier `candidates.jsonl` record.
- **Evaluate only** via `./scripts/evaluate_candidate.sh feedback cNNN`. One full 16-set run
  = one evaluation. Budget 100 evals; token soft/normal/absolute 9M/10M/11M.
- **Profiling only** via `./scripts/ncu_profile.sh …` (ncu-report-skill workflow), and
  **never** while an evaluation is running (foreign process on the locked GPU → rc 3,
  discarded measurement, wasted eval). Serialize: finish eval, then profile, then next eval.
- **`final` is operator-only.** Do not run it without explicit approval.
- Consult `KernelWiki` for H100/sm_90 specifics (atomics, vectorization, warp count) and
  `ncu-report-skill` for profiling; record skill usage per candidate.

---

## 1. Invariants every candidate must satisfy (pre-eval checklist)

Before spending an evaluation on any candidate, statically confirm:

1. **Fresh output, no input mutation.** `run()` returns a new buffer initialized from
   `final_hidden_states` and never writes into `final_hidden_states`, `expert_outputs`, or
   `token_indices`. (Evaluator reuses inputs across warmup+timed iters; in-place accumulation
   would corrupt iterations 2..N.) Verify by code inspection: the only tensor written is the
   freshly allocated output / scratch.
2. **Semantics:** for each `i`, `output[token_indices[i]] += expert_outputs[i]`, with
   `output` starting as a copy of `final_hidden_states`. Empty rows → identity; duplicate
   destination rows → all contributions summed.
3. **dtype/shape:** output is bf16, shape `[batch_seq_len, 3072]`. `token_indices` read as
   int64 then cast to int32 for offset math (max offset ≈25.2M < 2³¹, safe).
4. **Hidden tiling divides 3072** (block ∈ {128,256,384,512,768,1024,1536,3072}) → no hidden
   mask needed. Any masking is only on the row/selected-token axis.
5. **Signature:** `run(final_hidden_states, expert_outputs, token_indices)` exactly; entry
   point importable as `solution.solution.run`.
6. **Determinism of shape handling:** kernel must be correct for non-power-of-2 row counts
   (541, 1879, 131) and both tiny (131 rows) and large (8192 rows) shapes without special
   casing that breaks any of them.

If any check fails, fix in the *same* candidate ID (it hasn't been evaluated yet) rather
than burning an evaluation.

---

## 2. Candidate lineage strategy

Tree, not a chain: keep a known-good anchor and branch experiments off it. Each node lists
parent, the single change, and the hypothesis. Only *evaluated* nodes consume budget and get
a `candidates.jsonl` record; abandoned pre-eval edits keep the same unused ID.

```
c001 (anchor: Design A, fp32-scratch atomic scatter)
 ├─ c002  BLOCK_H / num_warps / grid-shape tuning of c001
 ├─ c003  Design A-bf16 (direct bf16 atomics on output)  ── correctness gate first
 ├─ c00x  rows-per-program / vectorization variants of the winning A form
 └─ c0xx  Design C (sort + fused no-atomic gather)  ── main speed experiment
      └─ c0xx+ C tuning (offset-build method, BLOCK_H, gather-in-kernel vs pre-gather)
```

### Phase 0 — `c001` correctness + speed anchor (Design A, fp32 internal)
- **Build:** `output`-side fp32 scratch initialized from `final.float()` (allocation +
  copy = plumbing), Triton scatter kernel does fp32 `tl.atomic_add` of `expert_outputs`
  rows into the scratch at `dst*3072 + col`, then a Triton cast kernel writes bf16 `output`.
  All adds/casts in Triton.
- **Hypothesis:** correct on all 16 shapes with wide margin (fp32 accumulation is more
  accurate than the reference bf16 atomic sum); modest speedup or near-parity vs baseline —
  establishes the correctness floor and a timing reference.
- **Decision rule:** must pass all 16. If it fails correctness, the design/logic is wrong —
  debug before any tuning. If it passes but is slower than expected, that's fine; it's the
  anchor.

### Phase 1 — tune the atomic design (`c002`, …)
- One knob-group per candidate: `BLOCK_H ∈ {256,512,768,1024}`; grid 1D
  (`num_selected`, internal hidden loop) vs 2D (`num_selected`, hidden_blocks);
  `num_warps ∈ {2,4,8}`; `num_stages`; rows-per-program; `sem="relaxed"` atomics; vectorized
  loads. Change **one** group at a time so each eval attributes cleanly.
- **Hypothesis:** memory-bound; best `BLOCK_H`/`num_warps` maximizes achieved HBM BW and
  atomic throughput. Use ncu (between evals) to pick candidates rather than brute-forcing all
  combos through the eval budget.

### Phase 2 — direct bf16 atomics (`c003`)
- **Build:** copy `final → output` (bf16), scatter with `tl.atomic_add` directly on bf16
  `output`; no fp32 scratch, no cast pass. Halves output-side bytes and removes a kernel.
- **Correctness gate:** bf16 atomic accumulation reproduces the reference's rounding class;
  must still pass all 16 tolerances. If Triton's bf16 atomic lowering on sm_90 is
  unsupported/slow/inaccurate, abandon and keep fp32-scratch as the atomic winner.
- **Hypothesis:** if correct, fastest of the atomic family (least traffic, fewest launches).

### Phase 3 — sort-based fused gather (Design C)
- **Build:** plumbing computes `perm = argsort(token_indices)` and per-row segment offsets
  (`bincount+cumsum` or sort-derived `searchsorted`); single fused Triton kernel loops each
  output row's segment adding `expert_outputs[perm[j]]` into an fp32 accumulator seeded with
  `final[r]`, stores bf16. No atomics, no separate copy (copy is fused into the accumulator
  init). Accumulation stays in Triton.
- **Compliance note:** sort/offset build is index *preprocessing* (plumbing), the
  accumulation itself is Triton; document this reasoning explicitly in the candidate record.
  If judged non-compliant, Design C is dropped and the atomic winner ships.
- **Hypothesis:** lowest HBM traffic (~10S, no atomic amplification) and fused copy → biggest
  wins on the four L2-spilling shapes (#4/#15/#16 = 50 MB, #7/#12 = 25/23 MB). Risk: sort +
  offset launch overhead hurts the tiny shapes (#3/#10/#14); non-contiguous permuted expert
  reads may cost L2 locality; data-dependent inner loop.
- **Tuning sub-branch:** offset-build method; `BLOCK_H`; gather-in-kernel via `perm` vs
  physically pre-gathering `expert_outputs[perm]` (extra 8S copy — likely not worth it);
  long-tail-segment handling / `num_stages`.

### Phase 4 — shape-adaptive dispatch (only if warranted)
- If ncu/evals show atomic wins on small shapes but Design C wins on large shapes, a single
  candidate may **dispatch on `batch_seq_len`** (and/or L2-fit) between the two paths. This is
  a launch/config change → its own candidate ID. Keep the threshold simple and justify it
  from measured crossover, not guesswork.

---

## 3. Correctness checks per candidate

Since local CUDA/torch execution and any alternate correctness harness are forbidden,
correctness is validated by (a) the static invariant checklist in §1 and (b) the evaluator's
per-workload pass/fail. Procedure:

1. Re-run the §1 checklist by code inspection before evaluating.
2. Evaluate: `./scripts/evaluate_candidate.sh feedback cNNN`.
3. A candidate is **valid** only if all 16 workloads pass correctness. Record per-workload
   atol/rtol status.
4. On correctness failure: diagnose (empty-row identity? duplicate-row summation? dtype/cast?
   atomic ordering? offset overflow? in-place mutation across iters?), fix under a **new**
   candidate ID (the failed one is immutable and already recorded), and re-evaluate.
5. Watch-list shapes for silent edge bugs: #10 (131 rows, tiny/launch-bound), #2/#12 (541/
   1879 non-pow2), #4/#15/#16 (50 MB, L2-spilling atomics), #14/#3 (small + tightest atol
   0.070/0.071).

---

## 4. Performance hypotheses (test, don't assume)

- **H1 — memory-bound.** All designs are dominated by the mandatory `8S` expert read;
  compute is trivial. Expect near-HBM-BW-bound kernels. Verify with ncu (DRAM throughput,
  memory vs compute).
- **H2 — atomic amplification scales with output size.** On L2-resident outputs (≤~13 MB,
  most shapes) atomic RMW stays on-chip; on the ≥25 MB shapes it leaks to HBM, inflating
  traffic ~1.6–1.8×. Predicts Design C's advantage concentrates on #4/#7/#12/#15/#16.
- **H3 — fp32 scratch costs ~1 extra S of output traffic + a cast pass** vs bf16-direct
  atomics (`c003`); worth it only if bf16 atomics are slow or inaccurate.
- **H4 — tiny shapes are launch-bound** (#3/#10/#14, <2 MB). Fewer kernel launches beats
  bandwidth tricks there → favors single-kernel designs; Design C's sort overhead may make it
  *lose* on these.
- **H5 — best `BLOCK_H`/`num_warps` differ by shape** given the 131→8192 row range; a single
  static config is a compromise, motivating Phase-4 dispatch only if the crossover is large.
- **Baseline is itself copy+atomic**, so realistic wins come from: removing atomic
  amplification (C), fusing the clone launch (C), and better-tuned vectorized atomics / fewer
  launches (A). Quantify each with ncu before committing eval budget.

Profiling loop (between evals only): `./scripts/ncu_profile.sh --set basic -o profile/<tag>
python <harness>` per ncu-report-skill; read DRAM throughput, L2 hit rate, atomic
throughput, achieved occupancy; use to select the next candidate's knobs. Never profile
while an eval runs.

---

## 5. Stopping / convergence criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- Geomean improvement over the current best has converged: <~1–2% gain across the last 2–3
  evaluated candidates spanning distinct design ideas (not just a knob wiggle).
- Both major design families (tuned atomic A/A-bf16 and, if compliant, sort-fused C) have
  been evaluated and the winner is stable under retuning.
- Evaluation budget (100) or token budget (soft 9M / normal 10M / absolute 11M) approached.
- ncu shows the winner is within a few percent of HBM roofline for the dominant large shapes
  (no meaningful headroom left).

Do **not** run `final` on convergence — record the best valid candidate and stop; `final`
requires operator approval.

## 6. Guardrails against wasting budget

- Never evaluate two logical changes in one candidate — attribution breaks.
- Never evaluate and profile concurrently.
- Fix pre-eval mistakes under the same unused ID; only bump IDs after an eval or after real
  source change.
- Prefer ncu-guided knob selection over brute-force sweeps through the eval budget.
- Keep the last known-good valid candidate identified at all times as the shippable fallback.

---

## 7. Evidence format (one JSON object appended to `candidates.jsonl` per evaluated candidate)

Append-only; never rewrite earlier records. Fields:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "design": "A-fp32-scratch-atomic | A-bf16 | C-sort-fused | dispatch",
  "hypothesis": "what change and why it should help",
  "config": {"BLOCK_H": 512, "num_warps": 4, "num_stages": 2, "grid": "2D", "rows_per_prog": 1},
  "invariants_checked": ["fresh-output/no-inplace", "empty-row", "dup-row", "dtype/overflow", "hidden-divides-3072"],
  "validation": "static checklist result + how correctness reasoned",
  "per_workload": [
    {"uuid": "2c1d8396-...", "axes": {"batch_size": 2, "seq_len": 1024}, "correct": true, "speedup": 1.00}
    // ... one entry per of the 16 workloads (correct flag + speedup vs reference)
  ],
  "geomean_speedup": 1.00,
  "all_correct": true,
  "decision": "keep-as-anchor | promote-best | reject(reason) | retune",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki:<topic>", "ncu-report-skill:<profile tag>"],
  "notes": "ncu findings, failure diagnosis, next step"
}
```

Rules: record parent lineage, exact source hash, hypothesis, validation, every workload's
correctness+speedup, geomean, decision, cumulative evaluation count, and skill usage. A
candidate with any `correct=false` is invalid regardless of speedup.

---

## 8. Immediate next action (next turn, not this one)

Implement `c001` (Design A, fp32-scratch atomic scatter) in `solution/solution.py`, run the
§1 checklist, then evaluate once with `./scripts/evaluate_candidate.sh feedback c001` and
append its record to `candidates.jsonl`. No implementation or evaluation happens in this
turn.
