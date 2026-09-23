# Plan — L1/007 `hyena_fft_size_padding_rfft`

Run: `formal-kda-20260916--sol_execbench--L1-007_hyena_fft_size_padding_rfft`
Target HW: **NVIDIA A800 / `sm_80` (Ampere)**. Compute must be **Triton**; PyTorch only for metadata/launch
plumbing; **no** Torch / CPU / NumPy / CUDA-extension computational fallback (a failing Triton kernel is
invalid — it is *not* replaced by a fallback).

This is the executable optimization plan. It operationalizes `docs/draft.md` into a sequential candidate
lineage with decision gates, correctness gates, performance hypotheses, stopping criteria, and the evidence
schema. **No candidate is implemented or evaluated in this turn.**

---

## 0. Operating rules (self-binding, from CLAUDE.md / TASK.md)

- Submission entry point: `solution/solution.py` exposing `run(x) -> (x_freq_real, x_freq_imag)`.
- One immutable source version = one candidate ID (`c001`, `c002`, …). Any meaningful source / config /
  launch change ⇒ **new** ID. Never reuse an ID for changed source; never rewrite earlier `candidates.jsonl`
  records.
- Evaluate **only** with `./scripts/evaluate_candidate.sh feedback <cNNN>` (five fixed feedback workloads =
  one candidate evaluation). No direct CUDA/profiler/`nvidia-smi`/foreign harness.
- Budgets: **100** candidate evaluations; tokens **soft 1,000,000 / normal 1,500,000 / absolute 1,650,000**
  (includes uncached input, cache-creation, cache-read, output).
- `final` (16-workload) is **operator-approval-only**; never run it autonomously.
- Skills: `KernelWiki` is Blackwell/Hopper-scoped; this task is Ampere. Consult only if an
  architecture-neutral Triton point transfers; record usage (or "none") per candidate either way.

**Eval-conservation principle.** An evaluation is spent only after all pre-eval reasoning gates (§3) pass
and the change is expected to move a tracked metric. Group *safe, non-behavioral* micro-tweaks into one
candidate where sensible — but every change that alters numerics, tiling, dispatch, or launch is a new ID.

---

## 1. Problem restatement (drives every candidate)

For input `x : (batch, d_model=256, seqlen)` float32 contiguous, with `M = batch·256` rows, `N = 2·seqlen`,
output bins `freq_len = seqlen + 1`:

```
X[k] = Σ_{n=0}^{seqlen-1} x[n]·W_N^{kn},   W_N = e^{-2πi/N},   k = 0 … seqlen
R[k] = (1/N)·Σ x[n]·cos(2π·kn/N)
I[k] = -(1/N)·Σ x[n]·sin(2π·kn/N)
```

Outputs: two float32 contiguous tensors `(batch, 256, seqlen+1)` = `(real, imag)`.

Feedback workloads (M = batch·256, N = 2·seqlen):

| WL | batch | seqlen | N | regime | M | out bins |
|----|------:|-------:|--:|--------|--:|---------:|
| 1 | 4  | 1801 | 3602  | 2·prime  | 1024  | 1802 |
| 2 | 2  | 211  | 422   | 2·prime  | 512   | 212  |
| 3 | 64 | 8192 | 16384 | power-of-2 (large) | 16384 | 8193 |
| 4 | 8  | 1024 | 2048  | power-of-2 | 2048 | 1025 |
| 5 | 8  | 1321 | 2642  | 2·prime  | 2048  | 1322 |

Tolerance (all): `max_atol = 1e-5`, `max_rtol = 1e-5`.

**Geomean gating.** Metric = geometric mean speedup over workloads that all pass correctness. A single
catastrophic workload dominates the geomean, so **WL3 (pow2, large) must never blow up** — a pure `O(seqlen²)`
dense DFT on WL3 is ~300× over the memory floor (draft §3.1) and would sink the geomean even with wins
elsewhere. cuFFT is strong on pow2 (WL3/WL4) and weak on `2·prime` (Bluestein: WL1/WL2/WL5 → our best
opportunities).

---

## 2. Candidate lineage strategy (phased, with decision gates)

Numbering is planned but **adaptive**: gates below decide whether to advance, branch, or discard. Each arrow
is `parent → child`. Only one source version exists in `solution/solution.py` at a time; it is snapshotted
by the controller under the candidate ID at eval time.

### Phase A — Correctness baseline + landscape map
- **c001 — dense DFT-as-GEMM, all sizes.** Two fp32 `tl.dot` GEMMs `R=(X·Cᵀ)/N`, `I=-(X·Sᵀ)/N`, fused
  `1/N` scale + real/imag split in the epilogue, twiddles generated **in Triton** with integer argument
  reduction (`kn mod N`), memoized at module scope keyed by `(seqlen, device)`. `input_precision="ieee"`
  on every dot (TF32 forbidden). Emit exact `0.0` for `I[:,:,0]` and `I[:,:,seqlen]`.
  - Purpose: (a) prove the numeric recipe passes 1e-5 on **both** prime and pow2 sizes; (b) measure the
    per-WL speedup map and the exact WL3 penalty; (c) confirm evaluator warmup/timing semantics (does a cold
    twiddle build get timed?).
  - Risk accepted: WL3 will be slow (compute-bound). If c001 OOMs or times out on WL3, immediately apply the
    c001-mitigation (below) rather than treating it as a dead end.
  - **c001-mitigation (only if c001 is intractable on WL3):** block the twiddle-`K` dimension so twiddles are
    streamed/regenerated on-the-fly per K-tile (no full `(seqlen+1)×seqlen` matrix materialized). New ID
    (`c00x`) since numerics/launch change.

### Phase B — Bluestein-size wins (WL1/WL2/WL5), keep dense pow2 for now
Attack the workloads where cuFFT is weakest first (cheapest wins, lowest risk).
- Tune GEMM tiling `BLOCK_M/N/K`, `num_warps`, `num_stages`; add a small `triton.autotune` set constrained to
  fp32-safe configs. (new ID)
- Twiddle representation trade study: live `tl.sin/tl.cos` (with int arg reduction) **vs** a precomputed
  `O(N)` root-of-unity table indexed by `kn mod N`. Pick per accuracy+speed. (new ID)
- **Optional structural win:** radix-2 DIT split of the length-`N` padded transform → two length-`seqlen`
  sub-DFTs done as shared small GEMMs; works on `seqlen` not `N`, reuses twiddles across even/odd halves.
  Only pursue if plain tuned dense-DFT on the prime sizes has not already comfortably beaten cuFFT. (new ID)

### Phase C — Power-of-two branch (WL3/WL4): stop WL3 being a geomean sink
Graduated by risk/reward; advance only through the gate in §2.1.
- **C-1 (preferred first): four-step / mixed-radix FFT via small GEMMs.** Pad real signal to length `N`
  (pow2), factor `N = N1·N2` (e.g. 16384 = 128·128, 2048 = 64·32), reshape `N1×N2`, DFT columns (small
  `N1×N1` twiddle GEMM), multiply the `W_N^{n1·k2}` twiddle grid, DFT rows (`N2×N2` GEMM), reindex; keep bins
  `0..seqlen`, fuse `1/N` + real/imag split. Cost ≈ `O(N·(N1+N2))` per row (~50× cheaper than dense on WL3),
  and each stage is still a numerically-clean fp32 `tl.dot`. Easier/lower-risk in Triton than hand-rolled
  butterflies and more general for `final`. (new ID)
- **C-2 (if C-1 insufficient near the memory floor): real-input packing.** Pack the length-`N` real signal
  into `N/2` complex, one length-`N/2` complex FFT, split/recombine to the one-sided spectrum — halves work.
  Validate cancellation at DC/Nyquist (§3). (new ID)
- **C-3 (only if C-1/C-2 stall and budget allows): iterative Stockham radix-2/4 butterfly FFT** along the
  transform axis, `M` rows in parallel, Hermitian symmetry (compute only `seqlen+1` bins). Highest
  risk/reward. (new ID)

### Phase D — Size-adaptive dispatch (unify best branches)
- Single `run()` dispatch: `seqlen` power-of-two → Phase-C FFT branch; else → best Phase-B DFT branch. All
  branches pure Triton. Confirm no regression on any WL vs the per-branch best. (new ID)
- Add a **correct-but-slower general fallback** (dense DFT-GEMM with streamed twiddles) for arbitrary
  `seqlen` so the hidden `final` workloads (which may be neither pow2 nor `2·prime`) still pass correctness
  rather than fail. (new ID)

### Phase E — Convergence tuning
- Final tiling/`num_warps`/table-vs-live sweeps on whichever branch dominates each WL's geomean contribution.
  Micro-tuning only; stop per §5.

### 2.1 Decision gate applied after every evaluation
For each candidate, per workload record `correct` and `speedup = ref_time / cand_time`; compute
`geomean over all-correct WLs`.
- **KEEP & advance** if `all_correct` and geomean improved beyond noise (~>2%) or a target regime improved
  without regressing another.
- **ITERATE** (same phase, new ID) if a specific WL underperforms its hypothesis but the approach is sound.
- **DISCARD / revert lineage** (branch from the prior best, not the failed node) if a candidate regresses
  geomean, fails correctness, or the approach hit a wall. Record why; never delete the record.
- **Correctness failure on any WL ⇒ candidate is non-viable** regardless of speed; diagnose (usually §3
  numerics), fix under a new ID.

---

## 3. Correctness checks (pre-eval gates — spend no eval until all pass)

Cheap, reasoning-only gates run before every `evaluate_candidate.sh` call:

1. **Shape/dtype/layout:** two outputs, each `(batch, 256, seqlen+1)`, float32, contiguous; order = (real,
   imag). `M = batch·256` rows handled.
2. **Precision:** every `tl.dot` uses `input_precision="ieee"` (no TF32 — TF32's ~1e-3 error fails 1e-5
   instantly). fp32 accumulators throughout.
3. **Phase argument reduction (mandatory):** compute `m = (k·n) mod N` in **int32/int64** first, then
   `angle = 2π·m/N` with `m ∈ [0,N)`. Never form `2π·k·n/N` directly in fp32 — `k·n` reaches ~6.7e7 on WL3,
   past fp32's exact-integer range (2²⁴≈1.6e7), which corrupts twiddles.
4. **Structural invariants (free):**
   - `I[:,:,0] == 0` exactly (DC), `I[:,:,seqlen] == 0` exactly (Nyquist, `N` even) — emit literal `0.0`.
   - `R[:,:,0] == (Σ_n x[n]) / N`.
   These are hard invariants; any nonzero there signals a bug (reduction/twiddle/indexing).
5. **FFT-branch-specific (Phase C):** validate DC/Nyquist bins and the `imag=0` invariants after
   packing/butterfly recombination — these steps subtract similar-magnitude terms (cancellation risk).
6. **Accumulation-order stability:** blocked `tl.dot` tree reduction (~`log(seqlen)·eps`) should pass 1e-5;
   if a WL fails by a hair, escalate accumulation (split/Kahan, or table twiddles) under a new ID before
   abandoning the approach.
7. **No fallback:** confirm the code path raises on Triton failure and does not route to Torch/NumPy/CPU.

Post-eval correctness is authoritative from the evaluator's per-WL pass/fail; the gates above only prevent
wasting evals on predictable failures.

---

## 4. Performance hypotheses (falsifiable, tested per candidate)

- **H1 (baseline map):** c001 dense DFT-GEMM passes correctness on all five WLs; wins WL2 (tiny, Bluestein),
  is competitive on WL1/WL5, loses badly on WL3 (compute-bound), and is near/loss on WL4. Confirms cuFFT's
  weak vs strong regimes and quantifies the WL3 penalty.
- **H2 (epilogue fusion):** fusing `1/N` + real/imag split + writing both outputs in one pass beats the
  reference's materialize-complex-then-3-extra-passes on bandwidth for *every* branch — a floor-level win
  contributing to geomean even where compute merely matches cuFFT.
- **H3 (Bluestein wins):** on WL1/WL2/WL5, cuFFT pays Bluestein (chirp-z: pad to ≥2N−1, three FFTs +
  pointwise) constant-factor + plan overhead; a tuned direct DFT/radix-2-split beats it (target >1.5×, WL2
  potentially larger).
- **H4 (pow2 four-step):** Phase-C C-1 cuts WL3 from ~300× over floor to within a small multiple of cuFFT
  (target: WL3/WL4 speedup ≥ ~0.7×), converting WL3 from geomean-sink to geomean-neutral.
- **H5 (geomean > 1):** even if pow2 only *matches* cuFFT (~0.7–1×), large Bluestein wins yield overall
  geomean > 1. Concretely, if WL1/2/5 ≈ {2×,3×,2×} and WL3/4 ≈ {0.8×,0.9×}, geomean ≈ `(2·3·2·0.8·0.9)^{1/5}`
  ≈ 1.6×.
- **H6 (twiddle caching):** module-scope memoization keyed by `seqlen` removes `O(seqlen²)` trig from the hot
  path across timed iterations — *conditional on* the evaluator not timing only a single cold call
  (resolved empirically at H1). If cold-timed, prefer `O(N)`-table or on-the-fly generation.

Each hypothesis is recorded with its candidate and marked confirmed/refuted from the per-WL evidence.

---

## 5. Stopping criteria (convergence)

Stop and write `SEARCH_COMPLETE` (with the reason) when any of:
1. **Converged:** geomean improvement < ~2% across 2–3 consecutive KEEP candidates, and each regime is at its
   realistic ceiling (pow2 branch within measurement noise of cuFFT; Bluestein branch plateaued).
2. **Budget:** approaching the 100-eval cap, or token usage nearing the **1,000,000 soft** limit (hard stop
   well before **1,500,000 / 1,650,000**). Reserve margin so `SEARCH_COMPLETE` + record-keeping fit.
3. **Diminishing returns:** remaining ideas are high-risk/high-cost (e.g. C-3 full butterfly) with low
   expected geomean delta given the current best.
`SEARCH_COMPLETE` names the best valid candidate ID, its geomean, per-WL speedups, and why further search is
not worthwhile. **`final` is not run without explicit operator approval.**

---

## 6. Evidence format (append-only `candidates.jsonl`, one object per evaluated candidate)

One complete JSON object appended per evaluated candidate; earlier records are never rewritten. Schema:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "timestamp": "<ISO-8601>",
  "phase": "A",
  "hypothesis": "dense DFT-as-GEMM baseline; passes all WLs, big WL3 loss (H1)",
  "approach": "two fp32 ieee tl.dot GEMMs; fused 1/N + real/imag split; int-reduced twiddles, memoized",
  "validation": {
    "pre_eval_gates": {"shape_dtype": true, "ieee_no_tf32": true, "int_arg_reduction": true,
                        "invariants_dc_nyquist": true, "no_fallback": true},
    "notes": "gates reasoned before eval"
  },
  "per_workload": [
    {"uuid": "59d23189-...", "wl": 1, "batch": 4,  "seqlen": 1801, "regime": "2*prime",
     "correct": true, "ref_ms": null, "cand_ms": null, "speedup": null},
    {"uuid": "da70c6cf-...", "wl": 2, "batch": 2,  "seqlen": 211,  "regime": "2*prime",  "correct": null},
    {"uuid": "2e37472c-...", "wl": 3, "batch": 64, "seqlen": 8192, "regime": "pow2",     "correct": null},
    {"uuid": "beb07143-...", "wl": 4, "batch": 8,  "seqlen": 1024, "regime": "pow2",     "correct": null},
    {"uuid": "4d7fea4e-...", "wl": 5, "batch": 8,  "seqlen": 1321, "regime": "2*prime",  "correct": null}
  ],
  "all_correct": null,
  "geomean_speedup": null,
  "decision": "keep|iterate|discard",
  "decision_rationale": "<why, and which node the next candidate branches from>",
  "cumulative_evals": 1,
  "skill_usage": "none (KernelWiki is Blackwell/Hopper-scoped; task is Ampere sm_80)"
}
```

- `ref_ms` / `cand_ms` / `speedup` filled from evaluator output; `correct` and `speedup` are per-WL.
- `geomean_speedup` computed over **all-correct** workloads (if any WL fails, note it and that geomean is not
  meaningful for ranking).
- Also maintain a short human-readable running summary at the top of subsequent draft/plan notes if useful,
  but the machine record is `candidates.jsonl`.

---

## 7. Immediate next action (next turn, not this one)

Implement **c001** (Phase A): `solution/solution.py` with the dense DFT-as-GEMM Triton kernel per §2/§3,
run pre-eval gates (§3), then `./scripts/evaluate_candidate.sh feedback c001`, and append the c001 record to
`candidates.jsonl` per §6. Use c001's per-WL map to choose between the Phase-B and Phase-C entry points and
to resolve the twiddle-caching / warmup question (H6).
