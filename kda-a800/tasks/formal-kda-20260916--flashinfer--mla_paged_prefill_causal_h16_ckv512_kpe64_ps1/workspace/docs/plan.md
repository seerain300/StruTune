# Executable Optimization Plan: `mla_paged_prefill_causal_h16_ckv512_kpe64_ps1`

Target: **A800 / `sm_80` (Ampere)**. Implementation: **Triton** (PyTorch only for metadata / launch
plumbing). Ranking: **geometric-mean speedup** over the fp32 reference across selected workloads,
**every** selected workload must pass correctness. This plan operationalizes `docs/draft.md` into a
sequential, one-lever-per-candidate search with explicit accept/reject gates.

Budget: **100 feedback evaluations**; token soft `1.0M` / normal `1.5M` / hard `1.65M`. One immutable
kernel version over all five feedback workloads = **one** evaluation. `final` is operator-only.

> Skill usage: `KernelWiki` is Blackwell/Hopper-scoped (tcgen05/TMEM/WGMMA/FP8/NVFP4) and does **not**
> apply to `sm_80`; it will be recorded as "not applicable — Ampere target" in each candidate record
> unless a genuinely portable idea surfaces. No profiler / `ncu` / `nvidia-smi` (disallowed).

---

## 0. Ground rules (procedural, non-negotiable)

- Work only inside this workspace. No parent dirs, no evaluator/controller internals, no other tasks.
- Write source to `solution/solution.py` exposing `run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr,
  kv_indptr, kv_indices, sm_scale)` returning `(output, lse)`.
- Candidates are **immutable**. Any meaningful source, config, or **launch** change ⇒ **new candidate
  ID** (`c001`, `c002`, …). Never reuse an ID for changed source; never rewrite a `candidates.jsonl`
  record.
- Evaluate **only** with `./scripts/evaluate_candidate.sh feedback <cNNN>`. Never run CUDA/Triton
  directly, a profiler, `nvidia-smi`, the evaluator, or an alternate correctness harness.
- A failed Triton implementation is **invalid**; never substitute a Torch/CPU/NumPy/CUDA-ext fallback.
- Append exactly one JSON object per evaluated candidate to `candidates.jsonl` (schema in §8).

---

## 1. Correctness contract (must match the reference exactly)

Per (sequence `b`, local query `i`, all 16 heads), with `page_beg=kv_indptr[b]`,
`page_end=kv_indptr[b+1]`, `kv_len=page_end-page_beg`, `q_len=qo_indptr[b+1]-qo_indptr[b]`,
`tok=kv_indices[page_beg:page_end]`, `Kc=ckv_cache[tok]` `[kv_len,512]`, `Kp=kpe_cache[tok]`
`[kv_len,64]`:

```
logits[h,j] = q_nope[i,h,:]·Kc[j,:] + q_pe[i,h,:]·Kp[j,:]        # contract dim 512+64=576
s[h,j]      = logits[h,j] * sm_scale                            # sm_scale = per-workload fp32 scalar
query_abs   = (kv_len - q_len) + i
s[h,j]      = -inf   where  j > query_abs                       # per-row causal cutoff w/ prefix
lse[q,h]    = logsumexp_j(s[h,:]) / ln(2)                       # BASE-2
p[h,j]      = softmax_j(s[h,:])
out[q,h,:]  = sum_j p[h,j] * Kc[j,:]                            # V == ckv latent (512 wide)
```

Non-negotiable correctness invariants (checked "by construction" then confirmed empirically):

1. **base-2 LSE.** Track running max `m`, sum `l` in the natural (`exp`) domain, emit
   `lse = (m + log(l)) / log(2)`. If `exp2` is used for speed, fold `sm_scale*log2(e)` into the QK
   scale and emit `lse = m2 + log2(l2)` consistently. This is the tightest (fp32-compared) check.
2. **sm_scale** applied before both softmax and LSE; take from the runtime scalar. Never hardcode
   `1/sqrt(192)`.
3. **Causal cutoff** is per-row: `valid(j) ⟺ j ≤ (kv_len - q_len) + i_local`. Masked lanes contribute
   nothing to `m`, `l`, or `acc` (`exp=0`).
4. **Empty / skipped sequence** (`kv_len==0` or skipped query range) ⇒ `output=0`, `lse=-inf`, no
   division by empty sum. Guard `l==0` to avoid `1/0` / NaN even though the five workloads are
   non-empty.
5. **fp32 accumulation** for logits, `m`, `l`, `acc[*,512]`; cast to **bf16 only** at final store.
   `output` bf16 `[total_q,16,512]`, `lse` fp32 `[total_q,16]`.
6. **QK stays in bf16 MMA** (bf16×bf16 product is exact in fp32; Ampere accumulates fp32). Any
   accidental fp32 `tl.dot` may lower to TF32 (10-bit) and diverge — set `allow_tf32=False` if ever
   used. **PV** casts `p→bf16` (standard FA); escalate precision only on empirical failure (§7).

---

## 2. Workload map (drives launch heuristics)

Constant across all five: `num_pages=989669`, `sm_scale=0.1352337747812271`, page_size=1.

| # | total_q | batch | num_kv_idx | regime | dominant cost | key lever |
|---|--:|--:|--:|---|---|---|
| 1 | 6053 | 11 | 6091 | large prefill | compute + KV BW | q-tiling, causal skip, KV reuse |
| 2 | 43 | 1 | 46 | tiny | launch / single-CTA latency | light kernel, 1 launch |
| 3 | 17 | 1 | 19 | tiny | launch / latency | light kernel, 1 launch |
| 4 | 22 | 22 | 17759 | decode-like (~1 q/seq, long ctx) | SM occupancy / BW | **split-KV** flash-decoding |
| 5 | 3842 | 20 | 3916 | medium prefill | compute + KV BW | q-tiling, causal skip |

Workload 4 is the parallelism trap: ~352 output rows but ~17.8k KV to stream ⇒ without KV splitting
only a few CTAs launch and ~108 SMs starve. Protecting #4 is the top perf risk.

---

## 3. Shared kernel architecture (MQA-shaped fused FlashAttention-MLA)

Put the **16 heads on the M (row) axis** so one paged gather feeds all heads:

- **QK:** `logits[M,BLOCK_N] = q_nope[M,512] @ Kcᵀ[512,BLOCK_N] + q_pe[M,64] @ Kpᵀ[64,BLOCK_N]`
  (two `tl.dot`s into one fp32 tile).
- **Online softmax** over KV tiles (running `m`,`l`, rescale `acc`).
- **PV:** `acc[M,512] += p[M,BLOCK_N] @ Kc[BLOCK_N,512]` (`p` cast bf16).
- **Paged gather:** `tok = kv_indices[base+n]`; `Kc = ckv_cache_ptr + tok[:,None]*512 + arange(512)`,
  `Kp` similarly (64). Mask tail lanes `n ≥ kv_len`.
- `M = BLOCK_Q_tok * 16` where `BLOCK_Q_tok ∈ {1,2,4}` query tokens share the sequence's KV, each row
  carrying its own causal cutoff. `acc[M,512]` fp32 is the dominant resource (16→32KB, 32→64KB,
  64→128KB per CTA); it caps `M`/occupancy on `sm_80`.
- **Tile→sequence mapping** computed host-side in PyTorch (allowed plumbing) from `qo_indptr`
  (prefix-sum / searchsorted), passed as an int32 table, to avoid in-kernel binary search. Reused by
  all candidates.

Both grid strategies live in the **same** source; a host heuristic selects which to launch per
workload. Because a launch change needs a new ID, heuristic thresholds are treated as part of the
candidate and only changed with a new ID.

---

## 4. Candidate lineage (sequential; one attributable lever each)

Each candidate: pick hypothesis → implement single change → hash source → evaluate feedback → record →
accept/reject by the gate in §7. Parent = last **accepted** candidate unless noted. IDs are planned;
actual branching depends on observed deltas (contingencies listed).

**Phase A — correct baseline**
- **c001 (parent: none).** Simplest fully-correct fused kernel: q-tiling grid over `total_q`,
  `BLOCK_Q_tok=1` (`M=16`), `BLOCK_N=64`, bf16 QK + bf16 PV, correct per-row causal mask (no skipping
  yet), base-2 LSE, fp32 accumulation, empty-seq guards, 4 warps, 2 stages. Goal: **establish a
  correct immutable baseline** and read per-workload speedups — especially whether #4 is starved.
  *Do not proceed to optimization until c001 is correct on all five.*

**Phase B — algorithmic wins (biggest expected geomean gain)**
- **c002.** Add **causal tile-skipping**: skip KV tiles fully above the tile's max valid index; skip
  mask compute on fully-unmasked tiles; only boundary tiles pay the compare. Expected ~1.5–2× on
  #1/#5; neutral on #4 (nearly full context) and tiny ones. Contingency: if LSE/causal boundary is
  fragile, keep the safe full-mask path and gate skipping behind exact index math.
- **c003.** Add **split-KV / flash-decoding path** + lightweight combine (either a second Triton
  reduce kernel or a Triton combine — no Torch compute) for low-query/long-KV. Host heuristic:
  choose split-KV when `total_q_rows` is small relative to SM count and `kv_len` large; else q-tiling.
  Expected: large win on **#4**; neutral elsewhere (path not taken). This is the highest-value perf
  item after c002.

**Phase C — reuse & tiling tuning (smaller, cumulative)**
- **c004.** Stack query tokens: `BLOCK_Q_tok=2` (`M=32`) in the prefill path to halve KV re-reads on
  #1/#5, honoring per-row causal cutoffs. Watch occupancy from the 64KB `acc`. Accept only if geomean
  improves without correctness loss.
- **c005.** Tune `BLOCK_N ∈ {32,64,128}` (gather granularity vs smem/regs). One value per candidate;
  branch to c005a/c005b as needed.
- **c006.** Tune `num_warps ∈ {4,8}` and `num_stages ∈ {2,3}` (pipeline gather+MMA).
- **c007.** Tune split count for the decode path (#4) against occupancy.
- **c008 (optional).** `exp`→`exp2` in online softmax for speed **iff** LSE base-2 bookkeeping stays
  exact (re-verify #1 invariant); perf-only, must not regress LSE.

**Phase D — contingency / precision**
- **cNNN (only if a correctness failure appears).** Escalate PV precision: (a) hi/lo bf16
  error-compensated PV dot (~2× MMAs, ~fp32), or (b) fp32 PV dot with `allow_tf32=False` (slow).
  Do not pay this cost pre-emptively.

Ordering rationale: correctness first (A), then the two changes with the largest expected geomean
impact and the biggest downside if missing (causal skip, split-KV for #4), then incremental tuning
whose deltas are small and workload-specific. Stop early if C stops moving the geomean (§6).

---

## 5. Correctness checks

**By construction (before every evaluation):**
- Re-derive base-2 LSE and `m+log(l)` combination on paper for the exact softmax base used.
- Verify causal predicate `j ≤ (kv_len-q_len)+i_local` at tile boundaries (off-by-one is the classic
  bug), and that split-KV combine reproduces the same global max / renormalization.
- Verify empty/skipped-seq path writes `0` / `-inf` and never divides by zero.
- Verify dtypes/layout of `output` (bf16) and `lse` (fp32), fp32 internal accumulation, final cast.
- Confirm `sm_scale` read from the argument; confirm bf16 QK MMA (no stray TF32 dot).
- Confirm no Torch/CPU compute path exists (Triton-only compute).

**Empirical (the only runtime signal):** `./scripts/evaluate_candidate.sh feedback <cNNN>` over all
five workloads. A candidate is **correct** only if the harness reports pass on **all five**; a fail on
any one makes the candidate invalid regardless of speed. Tolerance is unstated in `definition.json`,
so treat the first passing candidate's margins as the calibration for how aggressive PV precision can
be, and escalate precision (Phase D) if any fail.

---

## 6. Performance hypotheses (falsifiable, tied to expected per-workload deltas)

| Lever | Candidate | Hypothesis | Expected per-workload effect | Falsified if |
|---|---|---|---|---|
| Fuse Python double-loop into one Triton launch | c001 | Reference is loop-heavy fp32; fusion wins broadly | speedup > 1 on all five, esp. 2,3 (launch collapse) | any workload ≤ 1 |
| Causal tile-skipping | c002 | ~half of QK/PV masked in 1/5 | ~1.3–2× on #1,#5; ≈neutral #2,#3,#4 | no improvement on #1/#5 |
| Split-KV / flash-decoding | c003 | #4 is SM-starved w/o KV split | large gain on #4; neutral others | #4 unchanged/worse |
| Query-token stacking (M=32) | c004 | Halves KV re-reads in prefill | gain on #1,#5 | occupancy loss cancels reuse |
| BLOCK_N tuning | c005 | Gather granularity vs smem tradeoff | modest gain on BW-bound 1,4,5 | flat/negative |
| warps/stages | c006 | Better gather+MMA overlap | modest broad gain | flat/negative |
| split-count tuning | c007 | Match #4 splits to ~108 SMs | #4 gain | flat/negative |
| exp2 softmax | c008 | Cheaper exponentials | small broad gain, LSE unchanged | LSE regresses (reject) |

Bottleneck model (from draft §5.6): prefill (1,5) balanced compute+BW, causal halves it; decode (4)
pure gather BW (~20 MB) + occupancy; tiny (2,3) launch/latency bound. Geomean is protected primarily
by (a) never starving #4 and (b) keeping tiny-workload fixed overhead minimal.

---

## 7. Accept / reject gate and budget discipline

Per candidate, after feedback eval:
1. **Correctness gate:** all five pass → eligible; any fail → **reject** (invalid), do not carry as
   parent. If failure is precision-shaped, branch to Phase D; if it is a logic bug, fix as a **new**
   candidate ID.
2. **Performance gate:** accept as new parent iff **geomean speedup improves** over the current best
   accepted candidate **and no workload regresses below 1.0×** (never trade a passing workload into a
   loss). A neutral change (Δgeomean within noise) is recorded but not promoted; revert to prior best
   as parent.
3. **Attribution:** exactly one lever per candidate so each Δ is explainable. If a candidate bundles
   changes, it is still one ID but the record must state the confound.
4. **Budget:** track cumulative eval count (cap 100) and token usage. If token use approaches the
   **1.0M soft** limit, tighten to only high-expected-value candidates (Phase B); past **1.5M** move
   to close-out and write `SEARCH_COMPLETE`.

---

## 8. Evidence format (append-only `candidates.jsonl`, one object per evaluated candidate)

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_sha256": "<sha256 of solution/solution.py at eval time>",
  "hypothesis": "Simplest fully-correct fused MLA FlashAttention baseline (q-tiling, M=16, bf16 QK+PV, base-2 LSE).",
  "change_from_parent": "initial",
  "launch_config": {"grid": "q-tiling", "BLOCK_Q_tok": 1, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2, "splits": null},
  "correctness": {"passed_all": true, "per_workload": [
      {"uuid": "053a88a3-d024-4fbe-bc46-3f49d141de9f", "workload": 1, "pass": true},
      {"uuid": "a8cff3c6-0976-4dbb-833f-481091dc6a39", "workload": 2, "pass": true},
      {"uuid": "7572654f-7994-431a-bed1-65d7ec507b10", "workload": 3, "pass": true},
      {"uuid": "ed999c80-d4d1-4952-b063-cd39dfbf5612", "workload": 4, "pass": true},
      {"uuid": "805238ee-8470-4f4e-aa41-7c15f785173f", "workload": 5, "pass": true}
  ]},
  "speedup_per_workload": {"1": null, "2": null, "3": null, "4": null, "5": null},
  "geomean_speedup": null,
  "decision": "accept|reject|neutral",
  "decision_reason": "why promoted / rejected / not promoted",
  "cumulative_evals": 1,
  "tokens_used_estimate": null,
  "skill_usage": "KernelWiki: not applicable (Ampere sm_80; skill is Blackwell/Hopper-scoped)",
  "notes": "observations, correctness margins, next-lever pointer"
}
```

Rules: append only; never edit a prior record. Numbers (`speedup_per_workload`, `geomean_speedup`,
`per_workload.pass`) come **only** from the feedback evaluator output. `source_sha256` is computed on
the exact `solution/solution.py` that was evaluated. The five workload UUIDs above are fixed and must
not change.

---

## 9. Stopping criteria & completion

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
1. **Convergence:** the last 2–3 accepted candidates each improve geomean by less than a small
   threshold (e.g. < ~2%) and remaining levers have low expected value.
2. **Budget:** cumulative evals near 100, or token usage past the 1.5M normal-completion limit
   (never exceed 1.65M absolute).
3. **All high-value levers exhausted:** c002 (causal) and c003 (split-KV) landed, tuning (C) flattened.

The best **valid** (all-five-passing, highest geomean) accepted candidate is the submission. `final`
(full 38-workload eval) is run **only** after explicit operator approval — never proactively.

## 9a. Decision log

- **c001 (eval 1/100): REJECT — compile bug, no algorithmic signal.** Triton
  `CompilationError` on all 5 workloads: a module-level Python global `_INV_LN2`
  cannot be accessed inside a `@triton.jit` kernel. Fix in **c002** by inlining the
  `1/ln(2)` factor as a local literal (or a `tl.constexpr`); everything else in c001
  is unchanged. The algorithm itself was never exercised.
- **Tolerance discovered:** harness uses `atol=0.01, rtol=0.01, required_matched_ratio=0.99`
  (looser than feared). Implication: **bf16 PV is very likely sufficient** — do not pay
  Phase D precision cost preemptively; only escalate if a real correctness fail appears.
- **Selected trial indices** reported by harness: `[24, 26, 2, 16, 32]` (maps to the 5
  fixed feedback workloads in order 1..5). Warmup=3, iters=100, trials=1.
- **c002 (eval 2/100): ACCEPT — first valid baseline, geomean 68.46x.** c001 algorithm
  with `1/ln(2)` inlined (compile fix, no algorithmic change). All 5 workloads PASS.
  Per-workload speedup: #1=336.52x, #2=32.65x, #3=17.63x, #4=16.85x, #5=461.07x.
  Correctness margins fine (abs=1.56e-2 passes at atol/rtol=0.01, matched_ratio=0.99),
  so bf16 PV confirmed sufficient — no Phase D. New best/parent = c002.
  **Bottleneck:** geomean dragged by the low-CTA regimes #4 (decode-like: 22 q-rows
  × ~807 KV each ⇒ ~108 SMs starved) and #3 (tiny). The huge prefills (#1/#5) are
  already 300–460x. → **Next lever = split-KV / flash-decoding for #4** (Phase B c003),
  which is the plan's highest-value remaining item; causal tile-skip is deprioritized
  (only weakly moves geomean since #1/#5 are already fast).

## 10. Risk register (watch-list)

- **LSE base-2 mismatch** — most likely subtle fp32-compared bug; re-derive per softmax base.
- **Unstated tolerance** — governs whether bf16-`p` PV suffices; watch first passing margins.
- **#4 occupancy** — validate split-KV actually lands more CTAs; top perf risk.
- **512-wide fp32 `acc` register/smem pressure** — may cap `M` and split sizes on `sm_80`.
- **TF32 leakage** into any fp32 dot — keep dots bf16; `allow_tf32=False` if fp32 ever used.
- **tile→sequence mapping** correctness at sequence boundaries — host-precompute, unit-reason it.
