# Executable Plan: `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`

Target **A800 / sm_80**. Entry point `solution/solution.py::run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)`.
This plan operationalizes `docs/draft.md`. It defines the candidate lineage, the exact code skeleton each candidate builds on, the correctness checks done *before* every evaluation, the performance hypotheses, the evidence record format, and the stopping criteria. **No code or evaluation is produced in this turn.**

---

## 0. Ground rules (recap from CLAUDE.md / TASK.md)

- Triton is the primary compute; PyTorch only for metadata/launch plumbing. **No** Torch/CPU/NumPy/CUDA-ext computational fallback — a failed Triton kernel is invalid, not something to paper over.
- Only evaluation channel: `./scripts/evaluate_candidate.sh feedback <cNNN>` over the 5 fixed workloads = **one** candidate evaluation. No direct CUDA/profiler/`nvidia-smi`/evaluator runs.
- Candidates are **immutable** and **sequential**: `c001`, `c002`, … Any meaningful source/config/launch change ⇒ new ID; never reuse an ID for changed source.
- One JSON object appended per evaluated candidate to `candidates.jsonl`; earlier records never rewritten.
- Budget: **100 evaluations**; token soft 1.0M / hard 1.2M. Stop at budget or convergence; write `SEARCH_COMPLETE` on genuine convergence. `final` only with operator approval.

**Metric.** Primary = geometric-mean speedup over the reference across the 5 feedback workloads, **conditioned on every workload passing correctness**. A single failing workload ⇒ candidate invalid regardless of speed.

---

## 1. Solution contract & plumbing (fixed across all candidates)

`solution/solution.py` must expose `run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)` returning `(output, lse)` with:
- `output`: `[num_tokens, 16, 512]` **bf16**, on the same CUDA device as inputs.
- `lse`: `[num_tokens, 16]` **fp32**, base-2 log-sum-exp.

Plumbing (PyTorch, non-compute):
- Read shapes from tensors; assert the guaranteed constants (`H=16, ckv=512, kpe=64, ps=64, topk=2048`) but **do not** rely on asserts for control flow.
- Flatten cache views without copy: `ckv_cache.view(num_pages*64, 512)`, `kpe_cache.view(-1, 64)` (contiguous ⇒ free). Pass base pointers + strides to the kernel; index `flat_row = idx` directly (draft §1.2).
- Allocate `output = torch.zeros(...)` bf16, `lse = torch.full(..., -inf)` fp32 so untouched rows already satisfy the spec (empty-token case).
- Split-K scratch (design B): `acc_partial [num_tokens, splits, 16, 512] fp32`, `m_partial [num_tokens, splits, 16] fp32`, `l_partial [num_tokens, splits, 16] fp32`. Sizes for largest feedback (`T=8, splits≤64`): `8·64·16·512·4 ≈ 128 MiB` fp32 scratch worst case — acceptable, but prefer fewer splits (§4) to shrink it. Consider `splits=16` ⇒ 32 MiB.
- Grid launch computed on host from `num_tokens`; the kernel itself hard-codes the constexpr constants.
- `sm_scale` passed as fp32 scalar; fold `sm_scale * log2e` on host and pass the combined constant to use `exp2` in-kernel (draft §4.3).
- Guard `num_tokens == 0` (return the empty allocations without launching) — cheap defensive plumbing, not a compute fallback.

**One Triton file, one code path.** No alternate backend. If a Triton compile/launch fails, that candidate is recorded as invalid — we fix the kernel, not swap it out.

---

## 2. Reference-equivalence checklist (applied by static reasoning before every eval)

Because we cannot run a local harness, each candidate is gated by re-deriving these on paper against the reference in `definition.json`:

1. **Logit** `s = (q_nope·Kc + q_pe·Kp)`; scaled `s·sm_scale`. QK contraction over `ckv+kpe=576`.
2. **Softmax** over the valid-index axis only; `-1` (and any OOB) indices contribute probability 0 (`logit=-inf`).
3. **Output** `= softmax @ Kc` (the *ckv* cache is V), dim 512, cast to bf16 at store only.
4. **Base-2 LSE** `= m2 + log2(l2)` where `m2=max(s·sm_scale·log2e)`, `l2=Σ2^(s·sm_scale·log2e − m2)`. Cross-check: equals `logsumexp(s·sm_scale)/ln2` (draft §4.3).
5. **Empty token/head** (`l2==0` ⇒ no valid index): `output=0`, `lse=-inf`; no `0/0` NaN.
6. **Split-K combine**: global `M=max_k m2_k`; `L=Σ_k l2_k·2^(m2_k−M)`; `acc=Σ_k acc_k·2^(m2_k−M)`; `out=acc/L`; `lse=M+log2(L)`. Splits with `l2_k==0`/`m2_k=-inf` contribute nothing (guard the rescale so `2^(-inf−M)=0`, not NaN).
7. **Dtypes**: accumulators/stats fp32; final `output` bf16, `lse` fp32. QK uses bf16×bf16→fp32. PV dtype is the tunable knob (§4).
8. **Pointer arithmetic** in int64 for gather offsets (`idx.to(int64)`), draft §5.7.

Any candidate whose reasoning fails a checklist item is **not** evaluated until fixed (protects the eval budget).

---

## 3. Candidate lineage strategy

Depth-first, greedy: each candidate has exactly **one** parent and changes **one** dimension vs. its parent. Keep the best-valid candidate as the "current best"; branch new candidates from it. Abandon a branch when a change regresses geomean or breaks correctness.

**Phase 0 — Validity & baseline (get one green run).**
- **c001**: Simplest *correct* Triton design. Prefer design **B (split-K flash decode)** with conservative fixed params (`BLOCK_N=64`, `num_kv_splits=8`, `num_warps=4`, `num_stages=2`, PV in **fp32 with TF32 off** for safety), plus the combine kernel. Goal: pass correctness on all 5 workloads and record the first geomean. If a two-kernel design proves fiddly to get correct first-try, an even simpler **design A (one CTA/token, single pass, no combine)** is an acceptable c001 to establish correctness fast, then move to split-K in c002. Decision on which to write is made at implementation time; the plan permits either as the *baseline* since both are within the Triton-only rule.

**Phase 1 — Parallelization / occupancy (the dominant lever, draft §3–4.1).**
- Sweep `num_kv_splits` ∈ {8, 16, 32, 64} — one candidate each — to fill 108 SMs given only 2–8 tokens. Hypothesis: more splits → better latency hiding until combine/scratch overhead dominates. Expect an interior optimum.
- If c001 was design A, c002 = design B (split-K) is the first and largest expected jump.

**Phase 2 — Inner tiling.**
- Sweep `BLOCK_N` ∈ {64, 128, 256} at the best split count. Trade gather pipelining vs. register/smem pressure (draft §4.5).
- `num_stages` ∈ {2, 3, 4} and `num_warps` ∈ {4, 8} — sweep the promising 2–3 combos, not the full cross-product, to conserve evals.

**Phase 3 — Precision/throughput of PV (accuracy↔speed, draft §4.4).**
- Try PV matmul with `P` cast to **bf16** (faster) vs the fp32/TF32-off baseline. Only adopt if it still passes correctness on all 5 workloads. This is a pure speed play guarded hard by the correctness gate.
- Optionally test `tl.dot` `input_precision`/`allow_tf32` variants for QK if bf16×bf16 shows any tolerance pressure.

**Phase 4 — Structural refinements (only if Phase 1–3 haven't converged).**
- Tokens-per-CTA = 2 (M=32) to improve MMA utilization vs. occupancy (draft §4.2).
- Single-pass / fused reduction (design D) to remove the combine-kernel launch for these micro problems, if launch overhead is shown to matter.
- Split the 512 output dim across CTAs if register pressure caps occupancy.
- Autotune (`triton.autotune`) over the winning neighborhood — but note each config change is still *one immutable candidate* from the search's perspective; prefer a small fixed config chosen from Phase 1–3 evidence over shipping a broad autotune space (compile-time + nondeterminism risk).

Each phase advances only from the current best-valid candidate. If a phase yields no improvement within ~3–4 candidates, freeze that dimension and move on.

---

## 4. Performance hypotheses (falsifiable, ranked by expected impact)

| # | Hypothesis | Change | Expected effect | Falsified if |
|---|---|---|---|---|
| H1 | Reference is dominated by full-cache fp32 materialization (>1.25 GiB) + Python token loop; touching only selected bf16 rows wins big | any correct Triton kernel | Large geomean speedup even from a naive kernel | c001 geomean ≤ ~1× |
| H2 | With T∈{2..8} the kernel is occupancy/latency-bound; split-K fills SMs | ↑ `num_kv_splits` | Monotone speedup then plateau | no gain from 8→16→32 splits |
| H3 | There is an interior optimum in splits (combine + scratch overhead) | sweep splits | best at 16 or 32, worse at 64 | monotone to 64 |
| H4 | Gather latency hiding needs pipelining | ↑ `num_stages`, tune `BLOCK_N` | modest speedup | flat/negative |
| H5 | PV in bf16 is faster and still within tolerance | `P→bf16` | speedup, still correct | correctness fails on any workload |
| H6 | Removing the combine launch helps at micro sizes | single-pass (design D) | small speedup | flat/negative or breaks correctness |

Impact ranking: **H1 ≫ H2 ≈ H3 > H4 ≈ H5 > H6**. Budget is spent accordingly (most evals in Phases 0–2).

---

## 5. Evidence format (one JSON object appended per evaluated candidate to `candidates.jsonl`)

Append-only; never rewrite prior records. Each record:

```json
{
  "id": "c001",
  "parent": null,
  "timestamp": "2026-09-17T..Z",
  "source_sha256": "<sha256 of solution/solution.py at eval time>",
  "design": "splitK_flashdecode | single_cta | fused_singlepass",
  "params": {"BLOCK_N": 64, "num_kv_splits": 8, "num_warps": 4,
             "num_stages": 2, "pv_dtype": "fp32_tf32off", "tokens_per_cta": 1},
  "hypothesis": "H1/H2: establish correct baseline; expect large win vs fp32-materializing reference",
  "prechecks": {"lse_formula": true, "empty_row_guard": true,
                "splitk_combine": true, "tf32_disabled_where_needed": true,
                "int64_gather": true},
  "results": [
    {"uuid": "02d6ae9c...", "num_tokens": 8, "correct": true, "speedup": 0.0, "latency_ms": 0.0},
    {"uuid": "ddfa9e34...", "num_tokens": 6, "correct": true, "speedup": 0.0, "latency_ms": 0.0},
    {"uuid": "9d4a5f21...", "num_tokens": 2, "correct": true, "speedup": 0.0, "latency_ms": 0.0},
    {"uuid": "f77df5ce...", "num_tokens": 2, "correct": true, "speedup": 0.0, "latency_ms": 0.0},
    {"uuid": "564007ac...", "num_tokens": 8, "correct": true, "speedup": 0.0, "latency_ms": 0.0}
  ],
  "all_correct": true,
  "geomean_speedup": 0.0,
  "decision": "keep_as_best | reject_regression | reject_incorrect | branch_point",
  "cumulative_evaluations": 1,
  "skill_usage": ["none" | "KernelWiki:<topic>"],
  "notes": "what changed vs parent, and what the next candidate will test"
}
```

- Numeric fields (`speedup`, `latency_ms`, `geomean_speedup`) are populated from the evaluator's actual output; the zeros above are placeholders in the template only.
- `source_sha256` computed from the exact `solution/solution.py` submitted for that eval (immutability audit).
- `decision` drives lineage: `keep_as_best` updates the current best; `reject_*` prunes the branch; `branch_point` marks a candidate we will fork multiple children from.
- `cumulative_evaluations` increments by 1 per candidate (5 workloads = 1 eval) and is checked against the 100 budget.

A short running "search log" narrative may also be kept in `docs/` (optional), but `candidates.jsonl` is the source of truth.

---

## 6. Correctness-failure protocol

- If the evaluator reports an incorrectness on any workload: record `reject_incorrect`, diagnose against the §2 checklist (most likely LSE base/scale, split-K `-inf` combine, or PV precision), and open a **new** candidate ID with the fix. Never mutate the failed candidate's source under the same ID; never add a non-Triton fallback.
- If Triton fails to compile/launch: same — new ID with the fix; the failed one is recorded invalid.
- If bf16-PV (H5) fails tolerance: revert to fp32-PV best under a new ID; keep bf16 only where it passes.

---

## 7. Stopping criteria & convergence

Stop and (when genuinely converged) write `SEARCH_COMPLETE` with the reason when **any** holds:
1. **Converged**: no candidate improves geomean by > ~2% over the current best across ~3–5 consecutive candidates spanning the remaining live dimensions.
2. **Budget**: cumulative evaluations reach 100, or token usage nears the 1.0M soft limit (leave headroom before the 1.2M hard limit).
3. **Roofline ceiling**: measured latency approaches the memory-bound floor (draft §3, ~9 µs ideal for the largest workload) such that further tuning cannot plausibly move geomean.
4. **Diminishing structural options**: Phases 1–4 exhausted with the best config stable.

`SEARCH_COMPLETE` will name the best candidate ID, its geomean, the binding stop reason, and confirm all-correct. **Never run `final` without explicit operator approval.**

---

## 7b. Search log (running)

- **c001** (splitK_flashdecode, BLOCK_N=64, splits=8, PV fp32-ieee): **INVALID — OutOfResources**.
  Shared memory required 227584 B > A800 limit 166912 B. Root cause: the PV matmul used
  `kc.to(float32)`, forcing an fp32 `[BLOCK_N, 512]` K operand (128 KiB) into smem on top of
  the bf16 K tile and the fp32 accumulator. The fp32 K copy is the culprit, not BLOCK_N.
  **Fix → c002:** keep K in bf16 for the PV matmul (`tl.dot(P.to(bf16), kc)`, fp32 accumulate),
  eliminating the fp32 K materialization. Tolerance is loose (atol=rtol=0.01, matched_ratio=0.99),
  so bf16-PV should stay within bounds; this also moves us to the Phase-3 PV-dtype knob early
  because fp32-PV is simply infeasible in smem at this tile size on sm_80.

- **c002** (splitK_flashdecode, BLOCK_N=64, splits=8, PV bf16): **VALID — geomean 25.45x**, all 5 pass.
  Per-workload speedup 17.9x–34.3x; `max_abs=0.015625` on every workload (comfortably inside atol/rtol
  with matched_ratio ≥ 0.99). Confirms **H1** (touching only the selected bf16 rows crushes the
  fp32-materializing reference). **Key finding:** `sol_ms ≈ 0.128 ms` is essentially *flat* across
  num_tokens∈{2,6,8}; the reference alone scales with num_tokens, so the higher speedups just reflect
  a slower reference. Flat sol_ms ⇒ the kernel is **launch/latency/occupancy-bound** (only
  `num_tokens×8 = 16–64` CTAs on 108 SMs), **not** throughput-bound. This is exactly the regime
  Phase 1 (H2/H3) targets: raising `num_kv_splits` should add CTAs and shrink `sol_ms`, lifting the
  geomean on the small-token workloads (where speedup is currently lowest, 17.9x). This is now the
  **current best**.
  **Next → c003 (Phase 1, H2):** `num_kv_splits = 32` (BLOCK_N=64, PV bf16 unchanged) to fill SMs;
  scratch grows to `8·32·16·512·4 ≈ 64 MiB` fp32 — acceptable. If it helps, sweep 16 vs 32 vs 64 next.

- **c003** (splitK_flashdecode, BLOCK_N=64, splits=**32**, PV bf16): **VALID — geomean 25.20x**, all 5 pass.
  Essentially a tie with c002 (25.45x), slightly worse. `sol_ms ≈ 0.128–0.130 ms` — **flat vs c002's
  splits=8**. So **H2/H3 falsified**: adding CTAs (8→32 splits) did *not* shrink sol_ms. The split
  kernel's compute is already fully hidden; the ~0.128 ms floor is a *fixed per-call cost*, not an
  occupancy problem. c002 stays best. Hypothesis for the floor: two kernel launches + 3 fp32 scratch
  allocs (acc/m/l partials). → test by **removing** that overhead.

- **c004** (fused_singlepass, BLOCK_N=64, splits=1, PV bf16): **VALID — geomean 17.45x**, all 5 pass —
  **REGRESSION**. Fused one CTA/token, whole 2048-index loop, no scratch, no combine. `sol_ms` *rose*
  to 0.176–0.193 ms. **H6 falsified.** The ~0.128 ms floor was **not** launch/scratch overhead — the
  split-K version was genuinely faster because splitting the 2048-index loop across CTAs hides HBM
  gather latency; a single CTA serializing 32 BLOCK_N iterations with a large live fp32 `acc[16,512]`
  is slower. **Conclusion: split-K is the correct structure and c002 (splits=8) remains BEST.**
  The 8→32 flatness + the fused regression together suggest the optimum split count is *small* (≈4–8)
  and the remaining ~0.128 ms is a real latency floor. Micro-optimizing structure further looks
  low-yield; the memory-bound roofline floor for these micro problems is already close.
  **Next → c005:** stay in split-K, branch from c002. Two cheap probes remain before declaring
  convergence: (a) `num_kv_splits = 4` (fewer combine reads, larger per-CTA chunk — does the floor
  drop?), and (b) `num_warps`/`num_stages` tuning on the split kernel to probe the fixed floor. Try
  (a) first as c005.

- **c005** (splitK_flashdecode, BLOCK_N=64, splits=8, **num_warps=8**, PV bf16): **VALID — geomean
  24.94x**, all 5 pass — **marginal regression / tie** vs c002 (25.45x). `sol_ms 0.1276–0.1295 ms` ≈
  the same ~0.128 ms floor. **H4 falsified**: doubling warps/CTA (4→8, 256 threads) did *not* lower
  the floor, so the floor is not thread/MLP-limited. Chose the warp probe over splits=4 because it
  directly tests the memory-level-parallelism explanation for the floor; it came back flat.
  **Convergence read after 4 valid candidates:** the ~0.128 ms floor is **insensitive to split count
  (8 vs 32), warp count (4 vs 8), and fusion**, and `sol_ms` is flat across num_tokens∈{2,6,8}. Three
  consecutive candidates (c003, c004, c005) all failed to beat c002, spanning the occupancy, structural
  (fusion), and MLP/warp dimensions of the plan. This is the plan's §7 convergence signal (no >2%
  improvement over ~3 candidates across the remaining live dimensions).
  **One lever remains genuinely unprobed: inner tiling `BLOCK_N` (Phase 2).** Larger `BLOCK_N`
  (128/256) changes the loop-iteration count and per-tile issue width — a *different* mechanism than
  split/warp count — and could plausibly reduce the split-kernel execution portion of the floor.
  Because it is an untried, mechanistically-distinct lever with ample budget (5/100 evals used),
  the search is **not yet declared converged**.
  **Next → c006 (Phase 2, H4):** from c002 (splits=8, warps=4), set `BLOCK_N = 128`. smem check: bf16
  `kc` tile becomes `128×512×2 = 128 KiB` × num_stages(2) double-buffer ≈ 256 KiB > 166.9 KiB A800
  limit → would OOM at num_stages=2. So c006 must pair `BLOCK_N=128` with `num_stages=1` (single
  buffer, `kc` tile 128 KiB fits) OR keep num_stages=2 and drop to a smem-safe combination. Will verify
  the smem budget before eval. If BLOCK_N shows no gain either, declare SEARCH_COMPLETE with c002 best.

## 8. Immediate next actions (subsequent turns, not now)

1. Implement **c001** per §1 skeleton + §3 Phase 0 (simplest correct split-K, or design A if faster to correctness), verify against the §2 checklist, then `./scripts/evaluate_candidate.sh feedback c001`.
2. Append the c001 record (§5) to `candidates.jsonl`.
3. Proceed through Phases 1→4, one immutable candidate per change, updating the current best and pruning per §6.
4. Monitor budget (§7); on convergence write `SEARCH_COMPLETE`.
