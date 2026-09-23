# Plan — L1/007 `hyena_fft_size_padding_rfft`

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`. This turn writes the
plan only; no candidate is implemented or evaluated here.

Ground rules carried from `CLAUDE.md` / `TASK.md`:
- Triton-only compute; PyTorch for metadata/launch/dispatch only. No Torch/cuFFT/CPU/NumPy numeric
  fallback. A failed Triton path is invalid — never swap in a fallback.
- One immutable kernel version over the full 16-workload feedback set = one evaluation.
  Any meaningful source/config/launch change ⇒ new candidate ID; never reuse an ID or rewrite an
  earlier `candidates.jsonl` record.
- Budget: 100 evaluations; token soft 9M / normal 10M / hard 11M.
- Profiling only via `./scripts/ncu_profile.sh`, never concurrent with an evaluation.
- `final` only with explicit operator approval.

---

## 0. Objective and metric

Maximize the **geometric-mean speedup vs. the reference** across all 16 feedback workloads, subject to
**every** workload passing correctness at `atol = rtol = 1e-5`. Because geomean weights losses and
wins symmetrically in log-space, the strategy is: (a) never regress a shape below correctness, and
(b) attack whichever shape class currently drags the geomean most (early: the large power-of-two
shapes per draft §3.1).

---

## 1. Operation recap (implementation contract)

Per draft §1: with `M = batch·256`, `N = 2L`, `P = L+1`,
```
real_out[m,k] = (1/N) · Σ_{n=0}^{L-1} x[m,n]·cos(θ_{k,n})
imag_out[m,k] = (1/N) · Σ_{n=0}^{L-1} x[m,n]·(−sin(θ_{k,n})),   θ_{k,n} = π·k·n/L,  k=0..L
```
i.e. two real GEMMs `X·C` and `X·S` sharing left operand `X:(M×L)`; twiddles generated on the fly with
int64 index reduction `r=(k·n) mod N`, `θ=2π·r/N`. `run(x)` returns `(x_freq_real, x_freq_imag)`,
each `(batch,256,L+1)` contiguous float32.

Non-negotiable invariants (draft §1.2, §4.3):
- `imag_out[:,0] == 0` and `imag_out[:,L] == 0` (exactly, up to fp32 rounding).
- `real_out[:,0] = (Σ_n x[n])/N`; `real_out[:,L] = (Σ_n (−1)^n x[n])/N`.
- `k·n` formed in **int64** before `mod N`; only the small residue is cast to fp32 for `cos/sin`.
- `1/N` applied once in the epilogue on the fp32 accumulator.

---

## 2. Candidate lineage strategy

Sequential, immutable candidates. Each is a single source version of `solution/solution.py`. Lineage
is a chain unless a branch is explicitly justified in the record. Planned spine:

| ID   | Parent | Theme (Plan §draft 5) | Purpose |
|------|--------|-----------------------|---------|
| c001 | —      | Plan A, fp32/ieee dense-DFT fused GEMM | Correctness floor: all 16 shapes green; record geomean baseline. |
| c002 | c001   | Plan B: block/warp/stage tuning at fixed ieee precision | Speed without touching numerics. |
| c003 | c002   | Plan B: `tf32x3` for large-L shapes where margin allows | Cut MMA cost on compute-bound shapes safely. |
| c004 | c003   | Plan B: per-shape precision dispatch (ieee small-L, tf32x3/tf32 large-L) | Precision exactly where the error budget permits. |
| c005 | c004   | Plan B: autotune keyed on L (and M); small-shape FMA variant if `tl.dot` inefficient | Regime-specific configs. |
| c006+| best   | Plan C: FFT-structured Triton kernel for large pow2 shapes (#7,#9,#12,#15) | The main upside; only if §5 gate passes. |
| c0xx | best   | Plan D: shape-dispatched hybrid (GEMM for small/prime, FFT for large pow2) | Likely end state. |

Rules:
- Advance the spine only when a candidate **improves geomean and keeps all 16 correct**. Otherwise the
  parent for the next candidate stays the last accepted one (record the rejected branch, don't delete).
- One variable per candidate where practical (precision vs. tiling vs. dispatch), so evidence
  attributes the delta. Bundling is allowed only when changes are inseparable — state so in the record.
- Plan C candidates must first prove correctness on the **small** pow2 shapes (#6 L=256, #14 L=512,
  #1 L=1024) before enabling the large ones, to isolate FFT bugs cheaply.

---

## 3. Per-candidate execution loop (executable)

For each candidate `cNNN`:
1. **Design note** — one paragraph in `candidates.jsonl` hypothesis field: what changed vs. parent and
   the expected effect (which shapes, why).
2. **Implement** `solution/solution.py` (Triton). Keep the previous accepted version recoverable
   (git-independent: the accepted source is whatever produced the last accepted record; do not
   overwrite until the new candidate is about to be evaluated).
3. **Offline pre-eval audit (no GPU, no alternate numeric harness)** — see §4.1. Cheap; catches
   index/sign/shape/arg-reduction bugs before spending an evaluation.
4. **Evaluate exactly once:** `./scripts/evaluate_candidate.sh feedback cNNN`. Never run this while a
   profiler is active on the GPU.
5. **Record** one JSON line appended to `candidates.jsonl` (schema §6). Never edit prior lines.
6. **Decide** accept / reject / branch per §5 decision rules; update the spine pointer.
7. **(Optional) profile** the current accepted candidate via `./scripts/ncu_profile.sh` to inform the
   next hypothesis — strictly after the evaluation finishes, never concurrent.

Budget discipline: target ≤ ~15–25 evaluations total; each eval is precious. Do not evaluate a
candidate that fails the offline audit. Reserve ≥ 3 evaluations of headroom for the final polish.

---

## 4. Correctness checks

### 4.1 Offline (pre-evaluation, per candidate — no GPU / no alternate correctness harness)
Structural reasoning only, to avoid wasting evaluations:
- **Shape/contiguity:** outputs are two `(batch,256,L+1)` contiguous float32 tensors; `P=L+1` bins;
  flatten `(batch,256)→M` is a view, reshape back is free.
- **Sign convention:** forward transform `exp(−iθ)` ⇒ `real=+cos`, `imag=−sin`. Confirm both signs in
  source.
- **Argument reduction:** confirm `k*n` is int64 before `% N`, residue in `[0,N)`, `θ=2π·r/N`.
  Explicitly check the worst case (L=32768, k·n≈1.07e9) does not overflow int64 and gives an fp32
  angle in `[0,2π)`.
- **Endpoint invariants:** trace that k=0 yields `r=0` (imag term 0) and k=L yields `sin(π·n)=0`
  (imag 0) — draft §1.2.
- **Normalization:** single `*(1/N)` in epilogue on the fp32 accumulator.
- **Precision:** confirm accumulation dtype is fp32 regardless of MMA input precision.

### 4.2 Authoritative gate (the evaluation)
`./scripts/evaluate_candidate.sh feedback cNNN` over all 16 shapes at `atol=rtol=1e-5`. Pass = every
shape correct. This is the only sanctioned correctness measurement; a single failing shape invalidates
the candidate.

### 4.3 Precision-budget rule before relaxing accuracy
Before enabling TF32/tf32x3 on any shape, predict per-shape error from draft §4.2
(`≈ ε_mma/(2√L)`, `ε_tf32≈5e-4`, `ε_tf32x3≈2e-6`, `ε_ieee≈6e-8`) and only relax where the prediction
is safely `< ~1e-6` (order-of-magnitude margin under the 1e-5 gate). Small-L shapes (L≲512) stay on
ieee/tf32x3. Verify the prediction with the feedback eval; if any shape fails, revert that shape's
precision in the next candidate (new ID).

---

## 5. Performance hypotheses and decision rules

Hypotheses (to confirm/refute with evaluation + profiling), grounded in draft §3.1:
- **H1 (prime shapes win):** dense-DFT GEMM beats cuFFT on the 2·prime shapes (#2,3,4,5,8,10,11,13,16)
  because cuFFT Bluestein-pads. Expect the largest wins on small primes (#8 L=131, #16 L=211, #3 L=773).
- **H2 (small pow2 ~ neutral/win):** #6 (L=256), #14 (L=512), #1 (L=1024) are cheap enough that a fused
  GEMM ties or beats cuFFT + launch overhead.
- **H3 (large pow2 lose under dense DFT):** #7 (L=8192,b64), #15 (L=32768), #9 (L=4096,b32),
  #12 (L=2048,b16) are `Θ(M·L²)`-bound and lose to cuFFT's `N log N`; these cap the geomean and are
  the Plan C target.
- **H4 (precision step-down helps compute-bound shapes):** tf32x3 (and tf32 where §4.3 allows) cuts MMA
  time on the large-L shapes without breaking their (looser) error budget.
- **H5 (Plan C ROI):** an FFT-structured Triton kernel turns H3 losses into ~parity or wins, moving the
  geomean materially; feasibility limited by H100 SRAM for N≤16384 and a multi-block Stockham for
  N∈{32768,65536}.

Decision rules:
- **Accept** a candidate iff all 16 pass **and** geomean ≥ best-accepted geomean (ties broken toward
  simpler/more-robust source). Update spine.
- **Reject** if any shape fails or geomean regresses; keep the record, keep the parent as spine head.
- **Branch** only to test an isolated idea (e.g. an FFT prototype) without disturbing the spine; label
  it clearly and merge back only if it wins.
- **Plan C gate:** pursue c006+ FFT only if, after Plan B converges, profiling confirms H3 (large pow2
  compute-bound and dominating the geomean shortfall) AND ≥ ~10 evaluations + sufficient token budget
  remain. Otherwise ship the best Plan B.
- **Profiling triggers:** profile a shape when (i) it unexpectedly fails H1–H3, or (ii) to confirm
  compute- vs. memory-bound before committing to Plan C. Use `--set basic` first, deeper sets only if
  needed. Never concurrent with an evaluation (return-code-3 discard risk).

---

## 6. Evidence format (`candidates.jsonl`)

One JSON object per line, appended, never rewritten. Schema:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "plan_ref": "Plan A (draft §5.1)",
  "hypothesis": "Fused dense-DFT GEMM, int64 arg-reduction, fp32/ieee accumulate; expect all 16 correct, prime shapes win, large pow2 lose.",
  "changed_vs_parent": "initial",
  "precision": {"small_L": "ieee", "large_L": "ieee"},
  "offline_audit": {"shapes_ok": true, "signs_ok": true, "argred_int64": true, "endpoints_zero": true, "norm_once": true},
  "eval_cmd": "./scripts/evaluate_candidate.sh feedback c001",
  "eval_return_code": 0,
  "per_workload": [
    {"uuid": "beb07143-...", "batch": 8, "seqlen": 1024, "pass": true, "speedup": 0.0, "latency_ms": 0.0, "ref_ms": 0.0}
    /* ... one entry per 16 feedback workloads, in file order ... */
  ],
  "geomean_speedup": 0.0,
  "all_pass": true,
  "decision": "accept|reject|branch",
  "decision_reason": "…",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "notes": "profiling findings, next-step hypothesis"
}
```

Rules: fill `per_workload` with every shape's pass/latency/speedup from the evaluator output; compute
`geomean_speedup` over the 16; set `all_pass`; keep `cumulative_evals` monotonically increasing (count
only actual evaluations, including discarded return-code-3 ones as consumed budget with a note).
`source_sha256` binds the record to exact source (immutability audit).

---

## 7. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
1. **Convergence:** best geomean improves < ~2% over the last 3 accepted candidates and no untried
   high-ROI idea remains (Plan C either landed or gated out).
2. **Budget:** cumulative evaluations approach 100, or token usage approaches the 9M soft limit
   (leave margin for the final record and `SEARCH_COMPLETE`).
3. **Plan C infeasible/negative:** FFT prototype cannot be made correct within tolerance or does not
   beat the Plan B geomean after a bounded number of evaluations (≤ ~5 for the prototype) — revert to
   best Plan B and stop.
4. **Diminishing returns vs. risk:** remaining ideas risk regressing correctness for marginal geomean.

`SEARCH_COMPLETE` records: best candidate ID, its geomean, per-shape wins/losses, why further search
was abandoned, and the recommended candidate for operator-approved `final`.

Never run `final` without explicit operator approval.

---

## 8. Immediate next action (next turn)

Implement **c001** (Plan A): fused dense-DFT Triton GEMM producing both real and imag outputs, int64
argument reduction, fp32/ieee accumulation, `1/N` epilogue, `(batch,256)→M` flatten and free reshape
back. Run the §4.1 offline audit, then a single `feedback c001` evaluation, and record per §6.
Consult `KernelWiki` before writing the H100 tensor-core GEMM; use `ncu-report-skill` for post-eval
profiling only.
