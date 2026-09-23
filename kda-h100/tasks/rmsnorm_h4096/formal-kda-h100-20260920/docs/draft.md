# Draft — `rmsnorm_h4096`

## 0. Purpose & scope of this document

This is the analysis draft that must precede any code or executable plan. It covers:
the operation semantics, the environment/contract constraints, numerical risks, the
Triton design space, and the validation strategy. It does **not** contain code, an
executable plan (`docs/plan.md`), or candidate sources — those come later, per the
KDA workflow.

---

## 1. Operation semantics

RMSNorm as defined in `task/definition.json` (reference, authoritative):

```python
@torch.no_grad()
def run(hidden_states, weight):
    batch_size, hidden_size = hidden_states.shape   # hidden_size == 4096 (asserted)
    EPS = 1e-5
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

Per row `r` (length `N = 4096`):

1. Promote `x = hidden_states[r]` from bf16 → fp32.
2. `ms = mean(x_i^2)  = (1/N) * Σ x_i^2`   (accumulate in fp32).
3. `inv_rms = rsqrt(ms + 1e-5)`.
4. `y_i = (x_i * inv_rms) * w_i`, with `w` promoted to fp32.
5. Cast `y_i` back to bf16 (round-to-nearest-even).

Key structural facts:
- **No cross-row interaction.** Every output row depends only on its own input row plus
  the shared `weight` vector. Rows are embarrassingly parallel.
- **`hidden_size` is a compile-time constant (4096).** It is a power of two, so a
  per-row tile of `BLOCK = 4096` needs **no masking** on the hidden axis.
- **`weight` is shared** across all rows (shape `[4096]`, bf16). Ideal candidate for L2
  reuse / a single load broadcast.

### Arithmetic / memory intensity (roofline)
For one row: read `x` = 4096·2 B = 8 KiB, read `w` = 8 KiB (amortized/cached across
rows), write `y` = 8 KiB. FLOPs per row ≈ N mul (x²) + N-1 add + 1 rsqrt + 2N mul
≈ 4·N ≈ 16 k flops. Bytes moved (excluding cached weight) ≈ 16 KiB.
Intensity ≈ 1 flop/byte ⇒ **strongly memory-bandwidth bound**. The whole problem is a
fused, single-read/single-write streaming kernel; the optimization target is *effective
HBM bandwidth* for large batches and *launch/latency overhead* for small batches.

Large-batch traffic (batch 14509): ≈ 14509·4096·2 B read + same write ≈ 238 MB read +
238 MB write ≈ 0.48 GB. On a ~2 TB/s-class HBM this is ≈ 0.24 ms lower bound — the big
workloads set the achievable ceiling; the small ones are overhead-bound.

### Why a fused Triton kernel should beat the reference
The reference is eager PyTorch and materializes several full `[batch,4096]` temporaries:
`x` (fp32 copy — doubles the read), `x.pow(2)`, the reduction, `x*inv_rms`, `*weight`,
and the bf16 cast. That is on the order of 5–6 memory passes plus multiple kernel
launches. A single fused Triton kernel does **one bf16 read + one bf16 write** (weight
cached), so:
- Large batches: ~3–6× less DRAM traffic ⇒ large speedup, bandwidth-limited.
- Small batches: 1 launch vs ~5–6 launches ⇒ speedup driven by launch/overhead ratio.

---

## 2. Workloads & what the metric rewards

`task/feedback_workloads.jsonl` contains **14** workloads (all `hidden_size=4096`,
`inputs: random`), batch sizes:

```
small/latency-bound : 1, 7, 15, 16, 34, 63, 64, 79, 170      (9 workloads)
large/bandwidth-bound: 8804, 10827, 11832, 14418, 14509      (5 workloads)
```

> **Documentation discrepancy to keep in mind.** `CLAUDE.md` and `README.md` say
> "five fixed feedback workloads" and "five … count as one candidate evaluation," while
> `TASK.md` line 8 says feedback runs the *FULL official workload set … every shape …
> from `feedback_workloads.jsonl`* and line 12 mentions a *14-workload* final. The file
> itself has 14 entries. I will treat the feedback stage as running **all 14** listed
> workloads and count **one full pass = one candidate evaluation** either way. The "five"
> wording is treated as stale boilerplate; it does not change the immutable-candidate
> accounting.

**Metric = geometric mean speedup** over the selected workloads, with a hard correctness
gate on every workload. Two consequences:

1. **The metric is dominated by the 9 small batches** (they are 9/14 of the geomean
   factors). A big absolute win on the 5 large batches only helps `5/14` of the product.
   Therefore *small-batch speedup is at least as important as large-batch bandwidth.*
2. Small-batch speedup is essentially `torch_overhead / our_overhead`. That means
   **host-side launch overhead in `run()` must be minimized**: no redundant
   `.contiguous()`, no extra dtype casts on the host, a single pre-allocated output,
   one kernel launch, minimal Python. Correctness must never regress on any shape or the
   candidate is invalid.

---

## 3. Environment & contract constraints

- **Target GPU:** `TASK.md` says A800 (`sm_80`). The run-id/README string contains
  `h100`, and the available skills are Hopper/Blackwell-oriented. Actual silicon is
  ambiguous; I will **design a portable memory-bound kernel** (correct and fast on any
  recent NVIDIA arch) and confirm the real device characteristics via profiling before
  arch-specific tuning. The kernel structure (fused, streaming, coalesced) is optimal on
  either sm_80 or sm_90.
- **Triton-only compute.** PyTorch permitted solely for tensor metadata and launch
  plumbing. **No** Torch/CPU/NumPy/CUDA-extension fallback — a failing Triton kernel is
  invalid and must be fixed, not bypassed.
- **Entry point:** `solution/solution.py` exposing `run(hidden_states, weight)` returning
  the bf16 output tensor.
- **Immutability:** any meaningful source/config/launch change ⇒ a new candidate id
  (`c001`, `c002`, …), evaluated sequentially, one source version at a time. Records
  appended to `candidates.jsonl`, never rewritten.
- **Budget:** 100 candidate evaluations; token soft limit 1.0 M / normal 1.5 M /
  absolute 1.65 M. Evaluations are the scarce resource for a kernel this simple — expect
  convergence in a modest number of candidates; spend evaluations only on hypotheses that
  static reasoning cannot resolve.
- **Execution rules:** the **only** sanctioned GPU execution for correctness+timing is
  `./scripts/evaluate_candidate.sh feedback <id>`. Direct CUDA / `nvidia-smi` / the
  external evaluator / any alternate correctness harness are forbidden. Profiling is
  allowed **only** via the `ncu-report-skill` workflow against a profiling harness built
  inside this workspace, and **must never overlap an evaluation** (a foreign process on
  the locked GPU ⇒ return code 3, a wasted evaluation). Note: in this session even plain
  `Bash`/`python -c` is denied ("don't ask mode"), so I cannot freelance local runs — all
  local verification is static reasoning; the evaluator is the correctness oracle.
- **`final`** is operator-approval-only; never run it unprompted.

---

## 4. Numerical analysis & risks

The output is **bf16** (≈8 mantissa bits, ~2–3 significant decimals). The evaluator's
exact tolerance is not published in `definition.json`, so the safe policy is to
**reproduce the reference math faithfully in fp32** and let bf16 rounding be the only
source of difference. Points of care:

1. **fp32 accumulation of the sum of squares.** Load bf16, cast to fp32, square in fp32,
   reduce in fp32 (`tl.sum` over the 4096-tile). Never accumulate in bf16/fp16.
2. **Mean, not raw sum.** Reference divides by `N=4096` *before* adding eps and taking
   rsqrt: `rsqrt(sum_sq/N + 1e-5)`. Order matters for exactness — replicate exactly
   (divide by N then `+eps` then `rsqrt`). Using `1/sqrt(...)` vs `tl.rsqrt` is
   numerically equivalent to within bf16 tolerance; prefer `tl.rsqrt` (`libdevice
   rsqrtf`) for speed and closeness to `torch.rsqrt`.
3. **Reduction order.** `tl.sum` uses a tree reduction; `torch.mean` uses its own order.
   For 4096 well-scaled fp32 addends the relative difference is ~1e-6, far below bf16
   resolution — not a correctness risk.
4. **Weight promotion.** Multiply by `weight.to(fp32)` (cast bf16→fp32 in-kernel) before
   the final cast, matching `(x*inv_rms)*w_fp32`. Do not multiply in bf16.
5. **Final cast rounding.** `.to(tl.bfloat16)` uses round-to-nearest-even, matching
   `tensor.to(torch.bfloat16)`. Keep the whole `(x*inv_rms)*w` product in fp32 and cast
   once at the end.
6. **Scale / overflow.** Random inputs (~unit variance) give `sum_sq ≈ N`, `ms ≈ 1`,
   `inv_rms ≈ 1`. Even with larger magnitudes fp32 has ample range; `eps=1e-5` is
   negligible for these but must still be included for exactness (and guards the
   degenerate all-zero row).
7. **Determinism.** No atomics / no cross-block reduction in the baseline design ⇒ fully
   deterministic output, identical run-to-run. If a split-reduction variant is tried
   later (small batch), its cross-block combine must also stay deterministic and fp32.

**Conclusion:** with fp32 accumulation and a single final RNE cast, matching the
reference is low-risk. The dominant numerical concern is *not* accuracy but avoiding
accidental bf16-domain arithmetic.

---

## 5. Triton design space

### 5.1 Baseline mapping (starting point)
- **One program per row.** `grid = (batch,)`. `pid = tl.program_id(0)`; row base =
  `pid * 4096`; `offs = base + tl.arange(0, 4096)`.
- **`BLOCK = 4096` = `hidden_size` constexpr, no masking** on the hidden axis; `grid`
  exactly equals `batch`, so no masking on the batch axis either. Clean, branch-free.
- **Single load, reuse in registers.** Load `x` once (bf16→fp32), compute `sum_sq`,
  `inv_rms`, then reuse the *same* register tile for `y = x*inv_rms*w`. Avoid a second
  global load of `x`. With `BLOCK=4096` split over `num_warps` warps, each thread holds a
  handful of fp32 elements (e.g. 16 elems/thread at 256 threads) — fits in registers.
- **Coalesced access.** Row-major contiguous layout ⇒ the `arange(0,4096)` load/store is
  naturally coalesced; Triton vectorizes contiguous bf16 accesses.
- **Weight load once per program**, cast to fp32; rely on L2 to keep the 8 KiB weight hot
  across the many concurrent programs. Consider an eviction-policy / cache hint later.

### 5.2 Tuning axes (each distinct config = a new immutable candidate; prefer manual
sweep over `@triton.autotune` so behavior is deterministic and matches the
immutable-candidate model, and to avoid autotune's compile/benchmark cost interacting
with the coarse warmup=2 feedback runs):
- **`num_warps`** ∈ {4, 8, 16}: controls elements/thread and per-SM parallelism for the
  4096-wide row. Larger `num_warps` reduces latency per row (helps small batch, where a
  row may sit alone on an SM) but can lower occupancy for large batch. Prime lever.
- **`num_stages`** (software pipelining of the load): may help hide latency for the
  large-batch streaming case; limited benefit for a single-load kernel.
- **Rows-per-program (`ROWS_PER_BLOCK`)** for large batch: each program strides over
  several rows, amortizing scheduling and improving weight/L2 reuse; reduces grid size
  (14509 → ~a few thousand). Trade-off: fewer programs hurts the *small* batches, so this
  is a large-batch-only or batch-adaptive choice.
- **`eviction_policy` / cache hints:** keep `weight` in L2 (`evict_last`), stream `x`/`y`
  (`evict_first`) to reduce cache pollution on large batches.

### 5.3 Small-batch (latency-bound) ideas — evaluate only if the baseline underperforms
- Batches 1–170 give 1–170 programs on a ~100-SM device: heavy under-utilization but
  tiny absolute work; the kernel is launch/latency-bound, and the win vs torch is the
  launch-count ratio. **Minimizing host overhead in `run()` is the primary lever here.**
- **More warps per row** to bring more threads onto a lone row's 4096 elements (lower
  single-row latency) — cheap to try via the `num_warps` axis.
- **Split-hidden reduction** (multiple programs per row + a combine) could raise SM usage
  for batch=1, but 4096 elements is small; a second pass / atomics likely add more
  latency than they remove. Low priority, only if profiling shows a lone row starving.
- A **batch-adaptive launch** (choose `num_warps`/`ROWS_PER_BLOCK` from `batch` inside
  `run()`) can serve both regimes in one candidate and is deterministic.

### 5.4 Host-side (`run()`) discipline (matters most for the 9 small workloads)
- Assume inputs are contiguous bf16 on CUDA; **do not** call `.contiguous()`,
  `.float()`, `.cuda()`, or reshape unless strictly required (guard cheaply if needed).
- Allocate output with `torch.empty_like(hidden_states)` (bf16) once.
- Single kernel launch; pass `hidden_size` as a `tl.constexpr` (4096) so the compiler
  specializes and drops masking.
- Keep the Python path free of per-call allocations/branches beyond a small
  batch-regime selection.

---

## 6. Candidate roadmap (high level — details go in `docs/plan.md` next turn)

1. **c001 — correctness-first baseline:** one row/program, `BLOCK=4096` constexpr, single
   load+reuse, fp32 accumulation, `tl.rsqrt`, minimal `run()`, default `num_warps`.
   Purpose: lock in correctness on all shapes and get a real perf/geomean baseline.
2. **num_warps sweep** (e.g. 4/8/16) as separate candidates to find the latency/occupancy
   sweet spot; expect small batches to favor more warps, large batches fewer.
3. **Large-batch amortization**: `ROWS_PER_BLOCK` and/or `num_stages`, possibly
   batch-adaptive so small batches keep one-row-per-program.
4. **Cache/eviction hints** for weight reuse if profiling shows L2 pressure.
5. Stop when the geomean converges (diminishing returns across ~2–3 successive
   candidates) and write `SEARCH_COMPLETE`.

Guiding principle: this is a simple memory-bound op with a narrow optimal region; a
small number of well-reasoned candidates should converge. Do not burn evaluations on
speculative micro-variants that static reasoning already rules out.

---

## 7. Validation strategy

Because local execution is unavailable (Bash/python denied) and an "alternate correctness
harness" is prohibited, validation is **static reasoning + the sanctioned evaluator**:

1. **Static numerical review** before each evaluation: confirm fp32 accumulation, exact
   `mean→+eps→rsqrt` order, fp32 weight promotion, single final RNE cast, and no bf16
   arithmetic — per §4.
2. **Static shape/masking review:** confirm no out-of-bounds for all 14 batch sizes
   (grid = batch, BLOCK = 4096 = N, no masking needed); confirm contiguity assumptions
   hold for the given inputs.
3. **Evaluator as the correctness oracle:** `./scripts/evaluate_candidate.sh feedback
   <id>` runs all feedback workloads with a correctness gate; a pass certifies numerical
   equivalence within tolerance across every shape (small and large). This is the single
   trusted correctness signal.
4. **Performance signal:** the evaluator returns per-workload timings/speedups and the
   geomean; record every workload's result, not just the geomean, to see which regime
   (small vs large) each change moves.
5. **Profiling (optional, diagnostic only):** if a candidate underperforms and the cause
   is unclear, use the `ncu-report-skill` workflow on a workspace-local profiling harness
   to check achieved DRAM bandwidth / occupancy / stalls — **never** while an evaluation
   is running, and mindful that even this may be gated by the current execution
   restrictions.
6. **Budget discipline:** treat each evaluation as costly; batch reasoning so each new
   candidate tests a distinct, motivated hypothesis. Append one complete JSON record per
   evaluated candidate (parent, source hash, hypothesis, validation, per-workload result,
   geomean, decision, cumulative eval count, skill usage).

### Risks & mitigations summary
| Risk | Likelihood | Mitigation |
|---|---|---|
| bf16-domain arithmetic breaks tolerance | low (if disciplined) | fp32 accumulate + single final cast; static review |
| eps/mean order mismatch | low | replicate `sum/N + eps → rsqrt` exactly |
| Small-batch speedup weak (dominates geomean) | medium | minimize `run()` overhead; num_warps; batch-adaptive launch |
| Large-batch not bandwidth-saturated | medium | rows-per-block, num_stages, eviction hints; verify via ncu |
| Wrong device assumption (A800 vs H100) | medium | portable memory-bound design; confirm via profiling before arch tuning |
| Autotune vs coarse warmup interaction | low-med | use fixed configs / manual candidate sweep, not `@triton.autotune` |
| Wasted evaluation from profiler overlap | low | never overlap profiling and evaluation |

---

## 8. Open questions to resolve during the search
- Actual GPU and its HBM bandwidth / SM count (affects occupancy tuning, not structure).
- Exact evaluator tolerance (design assumes faithful fp32 reproduction is sufficient).
- Whether the reference/torch baseline is eager (assumed) or `torch.compile`d — affects
  the expected speedup magnitude, not the kernel design.
- Best `num_warps` and whether a batch-adaptive launch beats a single fixed config on the
  combined geomean.

**Next step (separate turn):** write `docs/plan.md` with the concrete, executable
candidate plan (c001 baseline spec + sweep order + decision rules). No code before the
plan.
