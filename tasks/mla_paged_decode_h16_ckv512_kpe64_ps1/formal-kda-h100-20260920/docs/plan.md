# Plan — `mla_paged_decode_h16_ckv512_kpe64_ps1`

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`. Target: NVIDIA H100
(`sm_90`), Triton kernel behind `solution/solution.py::run(...)`. Primary metric: geomean speedup
over the 47 feedback workloads; **every workload must pass correctness or the candidate is invalid**.

This turn writes the plan only — no candidate is implemented or evaluated here.

---

## 0. Ground rules (from CLAUDE.md / TASK.md — binding)

- Implement immutable candidates `c001, c002, …` **sequentially, one source version at a time**.
- Evaluate **only** via `./scripts/evaluate_candidate.sh feedback <cid>`. One full 47-workload
  feedback run = **one** candidate evaluation. Budget: **100 evaluations**.
- Token budget: soft 9M / normal 10M / absolute 11M. Be economical — this plan front-loads the
  high-value levers so we converge in ≪ 100 evals.
- Any meaningful source/config/launch change ⇒ **new candidate ID**. Never reuse an ID for changed
  source; never rewrite earlier `candidates.jsonl` records (append-only).
- Primary implementation is Triton. PyTorch only for metadata/launch plumbing. **No Torch/CPU/NumPy/
  CUDA-extension/alternate fallback.** A failed Triton kernel is invalid — fix Triton, never fall back.
- Do **not** run CUDA / `nvidia-smi` / the evaluator internals / any private correctness harness.
- Profiling only through `./scripts/ncu_profile.sh` (ncu-report-skill workflow). **Never run
  profiling and evaluation at the same time** (concurrent GPU use ⇒ return code 3, wasted eval).
- `final` only on explicit operator approval. Create `SEARCH_COMPLETE` when converged.

---

## 1. Objective & success criteria

1. **Correctness (gate):** all 47 feedback workloads pass the evaluator's numeric check
   (output bf16, lse fp32 base-2), including the tiny `B=1` boundary shapes and any zero-length rows.
2. **Performance (rank):** maximize geometric-mean speedup vs the reference across all 47.
3. **Convergence:** stop when geomean improvement plateaus (< ~2% over 2 consecutive accepted
   candidates) or budget/token limits approach; then write `SEARCH_COMPLETE`.

Reference target model (from draft §2): memory-bandwidth bound; byte floor `1152 B/token`,
largest workload (75145 tokens) HBM floor ≈ 25.9 µs at ~3.35 TB/s. We use these to judge headroom,
not as pass criteria.

---

## 2. Fixed architecture (shared by all candidates unless a candidate explicitly changes it)

Two-stage flash-decoding (split-KV), adapted from the SGLang/vLLM grouped MLA-decode template
(`lang-triton`, PR-34597), with this task's CSR indexing + base-2 LSE + `page_size=1` gather:

- **Stage 1** grid `(B, 1, num_kv_splits)`; `BLOCK_H = 16` (all heads in one program → KV loaded
  once serves all 16 heads). Per program: online-softmax over its `[split_start, split_end)` token
  range; contract `q_nope[16,512]·ckv[512,N]` + `q_pe[16,64]·kpe[64,N]` for scores, `p[16,N]·ckv[N,512]`
  for output. Write partial `acc/e_sum` `[16,512]` fp32 and partial **natural** LSE `e_max+log(e_sum)`
  `[16]` to `mid[B,16,num_kv_splits,512+1]` fp32.
- **Stage 2** grid `(B,16)`: merge partials across splits with a second online softmax; write
  `output[B,16,512]` bf16 and `lse[B,16]` fp32 = merged-natural-LSE `× (1/ln2)`.
- **CSR indexing:** `seq_lens = kv_indptr[1:] - kv_indptr[:-1]` (torch, plumbing); `base = kv_indptr[b]`;
  token row `= kv_indices[base + n]`, masked by `n < seq_len`; `ckv`/`kpe` address `= row*D (+d)`.
- **fp32 everywhere the reference is fp32:** `e_max`, `e_sum`, `acc`, all `tl.dot` accumulate fp32.
- **Empty/masked guards:** all-masked tile keeps `e_max=-inf` start correct; stage-2 with `e_sum==0`
  emits `output=0`, `lse=-inf` (no `0/0`). `output` allocated with `torch.zeros`.

Tuning axes (the knobs candidates will move, one at a time): `num_kv_splits` policy; `BLOCK_N`
(token tile); `ckv` single-load reuse (K==V); `num_warps`; `num_stages`; QK/PV precision path;
possible tiny-sequence specialization.

---

## 3. Candidate lineage strategy

**Principle:** change exactly one dominant variable per candidate so each evaluation yields a clean
signal. Branch from the best correct-and-fast ancestor. Record the parent explicitly.

Planned line of descent (IDs are provisional; a candidate is only "spent" once evaluated):

| ID    | Parent | One change under test | Hypothesis (why it should help) |
|-------|--------|-----------------------|----------------------------------|
| c001  | —      | **Correct two-stage baseline.** `BLOCK_H=16`, `BLOCK_DMODEL=512`, `BLOCK_DPE=64`, `BLOCK_N=32`, `num_warps=4`, `num_stages=2`, fixed `num_kv_splits=4` (upstream default). Separate K/V loads (no reuse yet). | Establish a passing 47/47 baseline + first geomean; lowest-risk faithful port. |
| c002  | c001   | **Adaptive `num_kv_splits`** = `clamp(round(TARGET/B),1,cdiv(max_seq,MIN_TOK))`, `TARGET≈128`, `MIN_TOK≈256`. | Fill 132 SMs: B=1 grossly under-parallel at splits=4; expect large win on lines 1–15 and B=16. |
| c003  | best   | **Single-load `ckv` reuse** (load token tile once, feed both QK and PV via `tl.trans`). | Memory bound; drops traffic ~2176→1152 B/tok. Expect win on B=64 long-seq (lines 32–47). |
| c004  | best   | **`BLOCK_N` sweep** (try 64; also 16 if 64 regresses small shapes). | Coalescing vs register/SMEM pressure & split granularity trade-off. |
| c005  | best   | **`num_warps` / `num_stages` sweep** (warps∈{2,4,8}, stages∈{2,3,4}) — pick one change; may split into two candidates. | Pipeline depth vs occupancy for the fp32 `acc[16,512]` (32 KB regs) regime. |
| c006  | best   | **`num_kv_splits` policy refinement** using ncu evidence (retune TARGET/MIN_TOK per regime). | Second-order tuning once traffic + tiling are settled. |
| c007+ | best   | **Tiny-seq specialization** for B=1 low-token lines (8/108/208…) *iff* profiling shows launch/tail overhead dominates; e.g. fewer splits or fused single-pass. | Only if evidence justifies; avoids over-splitting overhead. |
| c00x  | best   | **Precision fallback** (fp32 QK and/or fp32 PV) — only if a candidate fails correctness. | Recover tolerance if bf16 tensor-core rounding fails the numeric gate. |

Rules:
- If a candidate **fails correctness**, its immediate successor must be a *correctness fix* (e.g.
  precision fallback, guard fix), not a new perf lever.
- If a candidate **regresses geomean**, abandon that branch; next candidate branches from the prior
  best. Do not stack an unproven change on another unproven change.
- Only promote a change into the "shared architecture" mentally once it is proven by an accepted
  candidate; keep the actual source immutable per ID.

---

## 4. Per-candidate execution loop (repeat for each `cNNN`)

1. **Write source** `solution/solution.py` implementing exactly one change vs the parent. Keep the
   `run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale) -> (output, lse)`
   signature. No fallback paths.
2. **Static self-check (no GPU):** re-read the diff against the correctness checklist (§5). Confirm
   base-2 LSE conversion present, empty/masked guards present, fp32 accumulation, CSR addressing,
   `page_size=1` gather, no `cat`/host compute.
3. **Evaluate:** `./scripts/evaluate_candidate.sh feedback cNNN` (this consumes 1 of 100). Ensure no
   profiling is running concurrently.
4. **Record** one JSON line appended to `candidates.jsonl` (§7). Never edit prior lines.
5. **Decide:** accept (new best), reject (regression/failure), or investigate.
6. **Profile only if needed** and only *between* evaluations: `./scripts/ncu_profile.sh --set …`
   via the ncu-report-skill to test a specific hypothesis (DRAM bytes vs `1152 B/tok` floor;
   achieved BW vs ~3.35 TB/s; SM occupancy vs split count; stall reasons). Never overlap with an eval.
7. **Plan next candidate** from the winner, one new variable.

---

## 5. Correctness checklist (verified statically before every eval; evaluator is the final judge)

The evaluator is the only correctness oracle available (Bash/CUDA locked). So each candidate must
pass this static review before spending an evaluation:

- [ ] **Signature/outputs:** returns `(output[B,16,512] bf16, lse[B,16] fp32)`.
- [ ] **Scale placement:** `sm_scale` (opaque fp32) multiplies raw logits *before* max/exp/LSE, as
      in the reference. Not hard-coded.
- [ ] **Base-2 LSE:** final `lse = (e_max + log(e_sum)) * 1.4426950408889634` (= `/ln2`) in fp32.
      Intra-kernel exp base is free (natural first); only the emitted LSE base matters.
- [ ] **fp32 accumulation:** `e_max`, `e_sum`, `acc`, and `tl.dot` accumulators all fp32.
- [ ] **CSR indexing:** `base=kv_indptr[b]`, `seq_len=kv_indptr[b+1]-kv_indptr[b]`, token row
      `=kv_indices[base+n]`, masked `n<seq_len`. Matches ragged layout, not a rectangular block table.
- [ ] **`page_size=1` gather:** address `= row*Dckv (+d)` for ckv, `row*Dkpe (+d)` for kpe. int64 offsets.
- [ ] **Empty sequence (`L==0`):** `output=0` (zeros alloc), `lse=-inf`; no `0/0`→NaN and no
      `-inf+log(0)` corruption. Guard the stage-2 `acc/e_sum` divide on `e_sum==0`.
- [ ] **Tile masking:** out-of-range tokens contribute `-inf` logits (→ `exp=0`); `e_max=-inf` start
      handled via `n_e_max=max(max(qk),e_max)`, `re_scale=exp(e_max-n_e_max)` ordering.
- [ ] **Split correctness:** cross-split online-softmax merge is math-invariant to `num_kv_splits`;
      `mid` buffer sized `[B,16,num_kv_splits,513]`; `assert num_kv_splits == mid.shape[2]`.
- [ ] **No fallback / no host compute** beyond metadata (seq_lens, allocations).

If the evaluator reports a numeric failure: the next candidate is a targeted fix (most likely
precision path in §3 or an LSE/guard bug), documented as such.

---

## 6. Performance hypotheses (each tested by exactly one candidate, confirmed by ncu when material)

- **H1 (parallelism, c002):** For B∈{1,16}, fixed `num_kv_splits=4` under-fills 132 SMs; adaptive
  splits to ~128/B raise occupancy and cut latency on lines 1–31. *Evidence:* geomean delta on the
  B=1/B=16 subset; ncu SM-active/achieved-occupancy vs c001.
- **H2 (traffic, c003):** Kernel is HBM bound; loading `ckv` once for both QK and PV cuts DRAM bytes
  ~1.9×→1.0× the floor, winning the B=64 long-seq lines. *Evidence:* ncu `dram__bytes.sum` vs
  `1152 B × total_tokens`; geomean delta on lines 32–47. **Risk:** compiler may re-load; if ncu shows
  no byte reduction, abandon and keep separate-load form.
- **H3 (tiling, c004):** `BLOCK_N=64` improves coalescing/loop overhead on long seqs but may hurt
  tiny seqs via coarser split boundaries / register pressure. *Evidence:* per-regime geomean; ncu
  memory-throughput and eligible-warps.
- **H4 (pipeline, c005):** more `num_stages` hides load latency where SMEM allows; `num_warps`
  balances the fp32 `acc[16,512]` register load. *Evidence:* ncu stall-long-scoreboard,
  issue-slot utilization; geomean.
- **H5 (tail, c007):** tiny B=1 shapes are launch/tail bound, not bandwidth; a lighter split policy
  or fused single-pass reduces two-kernel overhead. *Evidence:* ncu kernel duration vs launch gap;
  geomean on lines 1–9.

Discipline (from `pattern-memory-bound`): **profile before assuming.** Do not spend a candidate on a
compute optimization unless ncu shows compute is the bottleneck (it should not be, per roofline).

---

## 7. Evidence format (one appended JSON object per evaluated candidate in `candidates.jsonl`)

Append-only. One line per candidate, immediately after its evaluation. Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py as evaluated>",
  "timestamp": "<ISO8601>",
  "hypothesis": "Correct two-stage baseline; establish 47/47 pass and first geomean.",
  "change_vs_parent": "initial",
  "config": {
    "num_kv_splits": "fixed=4",
    "BLOCK_H": 16, "BLOCK_DMODEL": 512, "BLOCK_DPE": 64,
    "BLOCK_N": 32, "num_warps": 4, "num_stages": 2,
    "ckv_reuse": false, "qk_precision": "bf16", "pv_precision": "bf16"
  },
  "validation": {
    "all_pass": true,
    "num_workloads": 47,
    "num_pass": 47,
    "num_fail": 0,
    "failed_uuids": []
  },
  "per_workload": [
    {"uuid": "...", "batch_size": 1, "num_kv_indices": 8, "speedup": 0.00, "pass": true}
    /* ... one entry per workload ... */
  ],
  "geomean_speedup": 0.00,
  "geomean_by_regime": {"B1": 0.00, "B16": 0.00, "B64": 0.00},
  "decision": "accept|reject|fix-next",
  "decision_reason": "…",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "profiled": false,
  "profile_notes": null,
  "notes": "…"
}
```

Rules: `source_sha256` must match the exact evaluated file. If a candidate fails to build/run, still
record it with `all_pass:false` and the error class in `notes` (it still consumed an evaluation).
`cumulative_evaluations` increments by 1 each record. Keep `per_workload` complete so regime-level
regressions are visible.

---

## 8. Budget management

- Token budget favors few, high-signal evaluations. Target convergence in ~6–10 evaluations; hard
  reserve: stop initiating new candidates as the 9M soft token limit approaches, wrap up by 10M.
- Evaluation budget (100) is not the binding constraint; the one-variable-per-candidate discipline
  keeps each eval informative. Do not batch speculative changes to "save" evals — that destroys signal.
- Profile sparingly (each ncu run has setup cost and must be serialized against evals); profile only
  to resolve a specific hypothesis (H2 traffic and H1 occupancy are the highest-value profiles).

---

## 9. Stopping criteria & completion

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
1. **Converged:** best geomean improves < ~2% across 2 consecutive accepted candidates, and the
   remaining planned levers (§3) are exhausted or shown non-material by ncu.
2. **At floor:** ncu shows the dominant (B=64) workloads within a small margin of the HBM byte/BW
   floor and small shapes are launch-bound — no further structural headroom.
3. **Budget:** approaching token soft limit (9M) or evaluation budget.

`SEARCH_COMPLETE` records: best candidate ID, its geomean + per-regime, the reason, and the
evidence (ncu findings) supporting "converged/at-floor." **Do not run `final` without explicit
operator approval.**

---

## 10. Immediate next action (next turn, not now)

Implement **c001** exactly as specified in §2/§3 (faithful two-stage baseline, fixed
`num_kv_splits=4`, separate K/V loads), run the §5 static checklist, then a single
`./scripts/evaluate_candidate.sh feedback c001`, and append the §7 record.

---

## 11. Decision log

- **c001 (evaluated, cumulative=1): FAIL — compile error, `fix-next`.** Every one of the 47
  workloads failed identically with `NameError: Cannot access global variable LOG2E from within
  @jit'ed function` in the stage-2 kernel. Triton does not allow module-level globals inside
  `@triton.jit` unless declared `tl.constexpr`. This is a pure compile-time bug (evaluator exit
  code 1, **not** a GPU-interference rc-3 invalidation), so the evaluation is valid and counts
  against the budget. The algorithm/structure (stage-1 CSR gather, ckv reused as K and V, online
  softmax, fp32 accumulation, empty/masked guards, base-2 LSE placement) is otherwise as intended.
  **Fix (c002):** inline the `1/ln2 = 1.4426950408889634` literal directly in the stage-2 kernel
  (or declare it `tl.constexpr`), with **no other change**, so c002 realizes the intended baseline
  and produces the first real geomean. This supersedes the original §3 c002 (adaptive splits),
  which shifts to c003; the rest of the lineage shifts down by one. The static checklist (§5) will
  add an explicit item: "no bare module-level Python globals referenced inside `@triton.jit`."

- **c002 (evaluated, cumulative=2): PASS — 47/47, geomean 51.46x, `accept` (new best).**
  The compile fix (pass `1/ln2` as a `tl.constexpr` arg) produced the intended faithful baseline.
  arith-mean 61.83x, min 15.59x, max 145.15x. Per-regime (my approx geomeans): B1≈24x, B16≈56x,
  B64≈93x. Correctness note: several PASS rows report large `rel` (up to ~1e4) — expected under the
  `matched_ratio=0.99` tolerance, which permits a small fraction of near-zero reference elements to
  have large relative error; absolute error stays ~bf16 ULP (7.8e-3…1.6e-2). Baseline established.

  **Next lever (c003): adaptive `num_kv_splits`.** The fixed `num_kv_splits=4` is the dominant
  bottleneck at both ends:
  - **B=1 under-parallelized:** grid = `B·num_kv_splits = 4` programs on 132 SMs ⇒ ~3% occupancy.
    Longest B=1 seqs are worst (2708→15.59x, 2408→16.74x, 1908→19.41x) because each of the 4
    programs serially streams ~677 tokens.
  - **B=64 long seqs regress:** 75145→46.30x, 62345→43.07x, 68745→44.79x, vs short B=64 145x.
    With 4 splits, wall-clock ∝ max seq_len/4; long sequences dominate. More splits shorten each
    program's serial token stream and fill the machine (B=64·splits already ≥ 132 at splits≥3, but
    the long-seq programs still each stream seq/4 — need more splits to cut that serial length).
  Plan: `num_kv_splits` chosen adaptively from `B` and the max sequence length so that
  `B·num_kv_splits` comfortably exceeds 132 AND each split streams a bounded number of tokens
  (e.g. `MIN_TOKENS_PER_SPLIT ≈ 256`): `num_kv_splits = clamp(cdiv(TARGET_PROGRAMS, B) ,
  1, cdiv(max_seq_len, MIN_TOKENS_PER_SPLIT))` with `TARGET_PROGRAMS ≈ 256`. The `mid` scratch and
  stage-2 loop scale with `num_kv_splits`, so keep it bounded (cap, e.g., 64–128). This is a launch/
  host-side plumbing change plus a `NUM_KV_SPLITS` constexpr; the kernel bodies are unchanged, so
  correctness risk is low. Only after c003 do we pursue the `ckv` single-load traffic reuse (which
  chiefly helps the long-seq B=64 tail) and BLOCK_N / warps / stages sweeps.
