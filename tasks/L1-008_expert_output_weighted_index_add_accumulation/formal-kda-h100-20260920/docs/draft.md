# Draft — L1/008 Expert-Output Weighted Index-Add Accumulation

Task: optimize official SOL-ExecBench task
`L1/008_expert_output_weighted_index_add_accumulation` on NVIDIA **H100 (sm_90)**.
Submission is `solution/solution.py` exposing `run(...)`. Primary compute must be
Triton; PyTorch only for metadata / launch plumbing / buffer allocation. No Torch,
CPU, NumPy, or CUDA-extension computational fallback.

This document is analysis only. No `docs/plan.md` and no solution code are produced in
this turn.

---

## 1. Operation semantics

Reference (from `task/definition.json`):

```python
@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    output = final_hidden_states.clone()
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    return output
```

So, for every selected-token row `i`:

```
output[token_indices[i], :] += expert_outputs[i, :]
```

with `output` initialized to a *copy* of `final_hidden_states`. This is a general
**scatter-add / index_add** along dim 0 over a wide feature dimension. There is no
explicit routing-weight tensor in the signature: the "weighted" part is already baked
into `expert_outputs` (the description says expert outputs are pre-multiplied by routing
weights upstream). Therefore the kernel is a **pure accumulate** — no multiply — plus the
initial copy of the base buffer.

### Tensors and dtypes

| Tensor | Shape | Dtype | Role |
|---|---|---|---|
| `final_hidden_states` | `[batch_seq_len, 3072]` | bf16 | base accumulation buffer |
| `expert_outputs` | `[num_selected_tokens, 3072]` | bf16 | contributions to scatter |
| `token_indices` | `[num_selected_tokens]` | int64 | destination row per contribution |
| `output` (return) | `[batch_seq_len, 3072]` | bf16 | base + scattered contributions |

Constants: `hidden_size = 3072`, `num_experts_per_tok = 8`.
Derived: `batch_seq_len = batch_size * seq_len`,
`num_selected_tokens = batch_seq_len * 8`.

### Key structural facts

- `num_selected_tokens` is **exactly `8 * batch_seq_len`**. So on average each output row
  receives 8 contributions.
- `token_indices` are drawn **i.i.d. uniform** in `[0, batch_seq_len)`
  (`torch.randint`). They are *not* grouped by token or by expert — this is a fully
  general random scatter. The `num_experts_per_tok = 8` constant only fixes the 8:1 ratio;
  there is **no exploitable per-token block structure** to gather over.
- Per-row contribution counts follow ~Poisson(8): most rows get ~4–14, some rows get 0
  (identity: `output[r] = final[r]`), and a tail of rows get 20+. The kernel must handle
  0-count and high-count rows correctly.
- `hidden_size = 3072` factors cleanly: `3072 = 3·1024 = 6·512 = 12·256 = 24·128`. Any of
  `{128,256,384,512,768,1024,1536,3072}` divides it, so the feature axis can be tiled with
  **no masking** on the hidden dimension.

### Feedback workloads (full set = 16; one full set = one evaluation)

`README.md` says "five" but that is stale; `task/feedback_workloads.jsonl` contains 16
rows and `TASK.md` confirms feedback runs the FULL set and final is a 16-workload eval.

| # | batch×seq | `batch_seq_len` (rows) | `num_selected` | output bytes (bf16) | atol |
|---|---|---|---|---|---|
| 1 | 2×1024 | 2048 | 16384 | 12.6 MB | 0.093 |
| 2 | 4×541  | 2164 | 17312 | 13.3 MB | 0.094 |
| 3 | 2×128  | 256  | 2048  | 1.6 MB  | 0.071 |
| 4 | 1×8192 | 8192 | 65536 | 50.3 MB | 0.100 |
| 5 | 4×512  | 2048 | 16384 | 12.6 MB | 0.093 |
| 6 | 1×512  | 512  | 4096  | 3.1 MB  | 0.087 |
| 7 | 16×256 | 4096 | 32768 | 25.2 MB | 0.094 |
| 8 | 2×512  | 1024 | 8192  | 6.3 MB  | 0.078 |
| 9 | 4×256  | 1024 | 8192  | 6.3 MB  | 0.090 |
| 10| 1×131  | 131  | 1048  | 0.8 MB  | 0.085 |
| 11| 1×1024 | 1024 | 8192  | 6.3 MB  | 0.090 |
| 12| 2×1879 | 3758 | 30064 | 23.1 MB | 0.130 |
| 13| 2×256  | 512  | 4096  | 3.1 MB  | 0.087 |
| 14| 1×256  | 256  | 2048  | 1.6 MB  | 0.070 |
| 15| 64×128 | 8192 | 65536 | 50.3 MB | 0.100 |
| 16| 32×256 | 8192 | 65536 | 50.3 MB | 0.100 |

All tolerances are `rtol = 0.05` plus the `atol` shown. Row counts span 131 → 8192; three
`seq_len` values are non-power-of-2 (541, 1879, 131). Largest output (50.3 MB) is right
around H100's L2 (50 MB), which matters for the atomic-vs-sort tradeoff below.

---

## 2. Constraints (from CLAUDE.md / TASK.md)

- Work only in this workspace; do not inspect parents, baselines, evaluator, controller,
  datasets, or other tasks. Only `KernelWiki` and `ncu-report-skill` are permitted external
  knowledge.
- Primary implementation **must be Triton**. PyTorch allowed for metadata, launch
  plumbing, and buffer allocation only. The index-add arithmetic (the adds) must live in
  the Triton kernel. No computational fallback of any kind.
- Draft first (this file); then `docs/plan.md`; then immutable candidates `c001, c002, …`
  one source version at a time. Each source/config/launch change → new candidate ID; never
  reuse an ID.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback <id>`. One full 16-workload
  run = one evaluation. Budget: 100 evaluations; token soft/normal/absolute limits
  9M/10M/11M.
- Append one JSON record per evaluated candidate to `candidates.jsonl`; never rewrite.
- Profiling only through `./scripts/ncu_profile.sh …` (ncu-report-skill workflow), **never
  concurrently with an evaluation** (foreign process on the locked GPU → controller discards
  the measurement, return code 3, and burns one evaluation). Finish one before starting the
  other.
- `final` only with explicit operator approval.

### Critical correctness constraint: no in-place mutation of inputs

The reference `clone()`s every call, so each call is independent and idempotent. The
evaluator runs warmup (2) + timed (10) iterations reusing the **same input tensors**. If
`run()` accumulated into `final_hidden_states` in place, the buffer would grow across
iterations and every iteration after the first would be wrong. Therefore `run()` **must**
produce a fresh `output` buffer (a copy of `final_hidden_states`) each call and must never
write into any input tensor. The per-call copy of the base buffer is mandatory, not
optional.

---

## 3. Numerical analysis

### Reference is itself a bf16 atomic sum (non-deterministic)

`index_add_` on a bf16 CUDA tensor accumulates with bf16 atomic adds in
hardware-non-deterministic order, rounding to bf16 after each add. So the "expected"
output the evaluator compares against is one nondeterministic sample of an ~8-term bf16
accumulation. The generous tolerances (atol 0.07–0.13, rtol 0.05) almost certainly encode
that nondeterminism plus rounding slack.

### bf16 precision budget

bf16 has 8 mantissa bits → relative rounding ε ≈ 2⁻⁸ ≈ 3.9e-3. Inputs are `randn`
(magnitude ~1); a row summing ~8 terms has magnitude up to a few units. Two error models:

- **fp32 accumulation, cast once at the end** (our preferred path): total error ≈ single
  bf16 round of the true sum ≈ ε·|value|. This is *more accurate* than the reference. The
  gap between our fp32 result and the reference bf16-atomic sample is bounded by the
  reference's accumulated rounding, i.e. roughly `k·ε·max_partial`. For a typical value
  `v`, the combined test `|a-b| ≤ atol + rtol·|v|` with rtol=0.05 (5%) leaves a very wide
  margin (5%·|v| ≫ k·ε·|v| for k≲30). **fp32 accumulation is expected to pass comfortably
  and is the safe default.**
- **bf16 accumulation matching the reference**: closest to reference in expectation but
  reproduces its rounding noise; still within tolerance, but no accuracy advantage and it
  depends on bf16 atomic support/quality.

**Decision for the draft:** prefer **fp32 accumulation internally** wherever the design
allows (atomic scatter into an fp32 scratch, or fp32 accumulators in a fused gather), then
cast to bf16 on the final store. Keep bf16-atomic-direct as an alternative only if it is
faster and still passes.

### Overflow / addressing

Destination offset `= idx * 3072 + col`, with `idx < batch_seq_len ≤ 8192` →
max offset ≈ 25.2M `< 2³¹`. So int32 offset arithmetic is safe (avoid int64 pointer math
cost); still, load `token_indices` as int64 (its true dtype) and cast to int32.

### Other numerical risks

- **Poisson tail rows** (20+ contributions): larger sums → larger absolute error, but rtol
  scales with the (larger) magnitude, so still covered.
- **Empty rows** (0 contributions): must yield `output[r] = final[r]` exactly. Atomic
  designs get this for free (row untouched after copy); a fused-gather design must handle a
  zero-length segment (loop runs 0 times, store `final[r]`).
- **No NaN/Inf** in `randn` inputs; no special-value handling needed.

---

## 4. Triton design space

All designs must (a) produce a fresh output buffer, (b) do the accumulation in Triton, and
(c) tile the hidden axis with a block that divides 3072 (no hidden-mask needed).

### Design A — copy + atomic scatter (simple, unambiguously compliant)

1. `output = final_hidden_states.clone()` (torch memcpy = allocation/plumbing; same base
   buffer the reference builds).
2. Triton scatter kernel, grid over `(num_selected, hidden_blocks)`:
   load `expert_outputs[i, cblk]`, read `dst = token_indices[i]`, and
   `tl.atomic_add(output_ptr + dst*3072 + cblk, val)`.

- Variant **A-bf16**: atomic add directly on the bf16 `output` (needs bf16/`bf16x2` atomic
  support; sm_90 has hardware bf16 atomics, and Triton can lower `tl.atomic_add` for
  bf16 — to be confirmed empirically).
- Variant **A-fp32**: allocate an fp32 scratch = `final.float()`, scatter with fp32 atomics
  (fast, universally supported), then a Triton cast kernel writes bf16 `output`. More
  traffic and an extra pass, but accuracy-safe and avoids any bf16-atomic uncertainty.
- Pros: trivial, matches how PyTorch itself implements index_add, all arithmetic in Triton,
  handles empty/duplicate rows for free.
- Cons: atomic RMW amplifies output-side traffic; for the 50 MB cases output spills L2 and
  atomics hit HBM.

Tuning knobs: hidden `BLOCK_H ∈ {256,512,768,1024}`; rows-per-program (amortize the tiny
index load; index cost is negligible vs a 3072-wide row, so likely 1 row/prog is fine);
`num_warps ∈ {2,4,8}`; `num_stages`; grid shape (1D `num_selected` with internal hidden
loop vs 2D `(num_selected, hidden_blocks)`); atomic `sem="relaxed"`; vectorized loads.

### Design B — fp32 scratch (accuracy anchor)

Same as A-fp32 but treated as an explicit correctness anchor: if A-bf16 ever fails
tolerance, B is the guaranteed-correct scatter. Cost model below shows it is the
traffic-heaviest but simplest-to-trust option.

### Design C — sort-based fused gather (no atomics, minimal traffic)

1. Preprocess indices (plumbing): `perm = argsort(token_indices)` and per-row segment
   offsets via histogram + cumulative sum (or `searchsorted`). This orders contributions by
   destination row.
2. Single fused Triton kernel, grid over `(batch_seq_len, hidden_blocks)`: for output row
   `r`, load `final[r, cblk]` into an fp32 accumulator, loop `j` over `[off[r], off[r+1])`
   adding `expert_outputs[perm[j], cblk]`, then store bf16. No atomics; each expert row read
   once; each output row written once; **no separate copy** (the base is folded into the
   accumulator init).

- Pros: lowest HBM traffic (see §5), no atomic contention, exact fp32 accumulation, fuses
  the mandatory copy away.
- Cons: needs a device sort + offset build; the expert reads are in permuted (non-
  contiguous) row order (still coalesced *within* a 3072-wide row, but poorer L2 locality);
  data-dependent inner-loop length in Triton; **compliance gray area** — `argsort`/
  `bincount`/`cumsum` are index preprocessing, not the accumulation itself, but a strict
  reviewer might view them as offloading work. The *accumulation* stays in Triton, which is
  the operation being optimized. Will keep A as the compliant baseline and treat C as an
  upside experiment, documenting the reasoning in candidate records.

Tuning knobs: `BLOCK_H`; whether to physically gather `expert_outputs[perm]` first (extra
8S copy — likely not worth it) vs gather-in-kernel via `perm`; offset-build method
(`bincount+cumsum` vs `sort`-derived `searchsorted`); handling of long tail segments
(cap/loop strategy, `num_stages` for the inner loop).

### Design D — not applicable

There is no per-token block structure to exploit (indices are i.i.d. random), so a
structured/grouped-gather that skips the sort is not available.

---

## 5. Performance model (why speed is even possible)

Let `S = batch_seq_len · 3072 · 2` bytes (one bf16 copy of the output; expert data is
`8S`). Rough HBM traffic:

- **Design A / B (copy + atomic scatter):** read `final` (S) + write `output` (S) for the
  copy, read `expert` (8S) for the scatter, plus atomic RMW on the output. If `output`
  fits L2 the RMW stays on-chip and HBM ≈ **10S**; if `output` spills L2 (the 25–50 MB
  cases, #4/#7/#12/#15/#16 near/over the 50 MB L2) the RMW leaks toward HBM, pushing HBM
  toward **~16–18S**. fp32 scratch (B) further inflates output-side bytes 2×.
- **Design C (sort fused):** sort/offset traffic is negligible (`token_indices` is
  `num_selected·8` bytes ≈ `0.01S`). Kernel HBM = read `final` (S) + read `expert` (8S,
  each row once) + write `output` (S) ≈ **10S** with *no* atomic amplification and *no*
  separate copy.

So all designs are dominated by the mandatory `8S` expert read; the differentiator is the
output-side atomic amplification. Design C's advantage grows on the large (L2-spilling)
shapes and shrinks on the small ones. The baseline (`clone` + `index_add_`) is itself
copy + atomic, so:

- The realistic wins vs baseline are: (1) removing atomic RMW amplification (Design C),
  (2) fusing away the separate clone launch (Design C), (3) better-tuned vectorized atomics
  and fewer launches (Design A). At ~3.35 TB/s HBM, e.g. shape #4 (`8S ≈ 402 MB`) is
  ~0.12 ms of pure expert read; the achievable speedup is the ratio of removed/avoided
  traffic and launch overhead. Small shapes (#3/#10/#14) are launch-overhead bound —
  minimizing kernel launches matters more than bandwidth there.

Profiling plan (via `ncu-report-skill` + `./scripts/ncu_profile.sh`, never during an eval):
confirm memory-bound status, HBM vs L2 hit rate, atomic throughput, and achieved BW; use
that to pick `BLOCK_H`/`num_warps` and decide whether Design C's complexity pays off.

---

## 6. Validation strategy

Isolation forbids running CUDA / the evaluator / any alternate correctness harness
directly, so I cannot diff against torch locally. Validation is therefore:

1. **Static reasoning first** (this draft): confirm semantics, in-place hazard, dtype/
   overflow, empty/duplicate-row handling, and the fp32-accumulation tolerance argument
   *before* spending an evaluation.
2. **Correctness anchor candidate**: make `c001` the simplest highly-likely-correct design
   (Design A, fp32 internal accumulation) to establish that the tolerance passes on all 16
   shapes. Only then optimize.
3. **Per-candidate evaluation** via `./scripts/evaluate_candidate.sh feedback cNNN`, which
   returns per-workload correctness + timing and the geomean. Every one of the 16 workloads
   must pass correctness for the candidate to be valid; geomean speedup is the ranking
   metric.
4. **Boundary coverage** is automatic because feedback runs the full 16-set, but I will
   specifically watch: non-power-of-2 rows (#2 541, #10 131, #12 1879), smallest
   (#10 131 rows, launch-bound), and the three largest L2-spilling shapes (#4/#15/#16).
5. **Profiling** only between evaluations, never concurrent, to guide (not replace)
   tuning; record ncu findings and skill usage in `candidates.jsonl`.
6. **Convergence / stop**: stop at the eval or token budget, or when geomean improvement
   has genuinely converged, then write `SEARCH_COMPLETE`. Never run `final` without
   operator approval.

Record for each candidate: parent, source hash, hypothesis, validation, per-workload
result, geomean, decision, cumulative eval count, and skill usage.

---

## 7. Open questions to resolve during the search

- Does `tl.atomic_add` on bf16 (A-bf16) work and beat fp32-scratch (B) on H100? If bf16
  atomics are unsupported/slow, default to fp32 internal accumulation.
- Is Design C's sort + offset build cheap enough that fused no-atomic gather beats tuned
  atomics on the large shapes — and is the non-contiguous expert gather a real L2 penalty?
- Best `BLOCK_H` and grid shape (1D vs 2D) across the wide row-count range (131 → 8192);
  likely different sweet spots for tiny vs huge shapes.
- Can the mandatory base copy be fused (only Design C fuses it) or must it remain a
  separate torch `clone` / Triton copy for the atomic designs?
- Launch-overhead minimization for the tiny shapes (#3/#10/#14): single-kernel designs and
  fewer launches likely dominate bandwidth concerns there.

---

## 8. Planned candidate progression (to be detailed in docs/plan.md, not now)

1. `c001` — Design A with **fp32 internal accumulation** (copy + fp32-scratch atomic
   scatter, or bf16-output atomic if it proves correct): correctness anchor + first speedup
   vs baseline.
2. Tune `c001` design: `BLOCK_H`, `num_warps`, grid shape, rows-per-program.
3. `A-bf16` direct bf16 atomics if faster and within tolerance.
4. Design C sort-based fused gather as the main speed experiment on large shapes.
5. Iterate with ncu-guided tuning until geomean converges.

No code or `docs/plan.md` is created in this turn per the task instruction; this draft is
the deliverable.
