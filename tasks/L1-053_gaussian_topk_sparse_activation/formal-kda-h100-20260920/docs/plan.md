# Plan — L1/053 Gaussian Top-K Sparse Activation (H100, sm_90)

Executable, sequential KDA optimization plan. Builds directly on `docs/draft.md`.
This turn produces the plan only; no candidate is implemented or evaluated.

---

## 1. Objective & ranking

- **Goal:** maximize geometric-mean speedup over the reference across the full
  12-workload feedback set, with **every** selected workload passing correctness
  (`max_atol = 1e-5`, `max_rtol = 0.05`).
- A candidate is *valid* only if all 12 workloads pass correctness; an invalid
  candidate cannot be selected regardless of speed.
- Ranking metric = geomean speedup of valid candidates.

## 2. Fixed decisions (locked before coding)

These are settled by the draft's analysis and will not be re-litigated per
candidate unless the evaluator forces it:

1. **Fused single-kernel Triton** reading bf16 directly, computing in fp32,
   writing bf16 once (RNE). PyTorch used only for: reshape `[B,S,H]→[M,H]`,
   `torch.empty_like` output, host scalar math, kernel launch.
2. **Host-side `z = ndtri(target_sparsity)`** computed once per `run()` call by
   porting the reference's own Abramowitz–Stegun 26.2.23 constants/branches,
   evaluated in fp64 then narrowed to fp32 and passed as a kernel scalar. No
   polynomial/branch logic on the GPU.
3. **Population variance** (`unbiased=False`, divide by `H`), matching the
   reference. `var = max(var, 0.0)` clamp before `sqrt` always.
4. **fp32 accumulators everywhere** (never bf16/fp16 accumulation).
5. **Early return** the untouched input tensor when `target_sparsity == 0.0`
   (parity/safety; not exercised by the feedback set).
6. **Row = contiguous last dim `H`**, treat tensor as `[M,H]`, `M = B*S`. Pass row
   stride to the kernel; do not force `.contiguous()` by default (avoids an extra
   full-tensor copy).
7. Keep an `H`-tail mask in the tile loop for generality even though every
   feedback `H` is a multiple of 4096 (mask cost is negligible).

## 3. Candidate lineage strategy

Immutable, sequential IDs. Each node names its parent. A new ID is minted for any
meaningful source/config/launch change. Later branches are **conditional** on
evidence from earlier evals/profiles; only the c001 baseline is unconditional.

- **c001 (root):** §6.1 fused one-program-per-row, **two-pass** over `H` (pass 1
  stats, pass 2 apply), **naive one-pass variance** (`sum`, `sum²` in one read),
  host `z`, `var` clamp, bf16 RNE store, early-return. Conservative starting
  launch config: `BLOCK_H = 1024`, `num_warps = 4`, `num_stages = 2`.
  Purpose: establish correctness + baseline geomean.

Branching from c001 (choose based on evidence, one change per candidate):

- **Correctness branch (only if c001 fails any workload):**
  - **c00x-A:** replace naive variance with a stable **two-pass mean/variance**
    (mean pass, then `sum((x−mean)²)` pass, then apply = 3 reads) — fixes
    cancellation.
  - **c00x-B:** if still failing, Welford accumulation, or investigate stride /
    mask / z-value / rounding as diagnosed. Never a torch fallback.
- **Performance branch (c001 valid but slow), pursued in this rough order, one
  knob per candidate, each guided by ncu evidence not guesswork:**
  1. **Launch-config sweep** on the §6.1 kernel: `BLOCK_H ∈ {512,1024,2048,4096}`,
     `num_warps ∈ {4,8,16}`, `num_stages ∈ {2,3,4}`. May be realized as a Triton
     `@autotune` over a small keyed set (key on `H`) — but note autotune's config
     set is itself "source"; changing it ⇒ new candidate ID.
  2. **Reduce pass-2 HBM re-reads** if ncu shows the re-read misses L2:
     - **c-single-read:** whole-row-in-registers single-read variant
       (`BLOCK_H = next_pow2(H)`) — only for shapes where it does not spill.
     - **c-two-kernel:** §6.3 split (reduction kernel → `cutoff[M]`; elementwise
       apply kernel). 2R+1W but perfect parallelism in the apply kernel.
  3. **Small-M occupancy** (#11/#4/#1/#12) if ncu shows they are occupancy/latency
     bound (not launch-overhead bound):
     - **c-split-row:** §6.4 split each row across `P` programs with atomic/
       two-stage reduction into scratch `[M,2]`, then finalize+apply.
  4. **Hopper-specific** (consult **KernelWiki** first): TMA bulk loads,
     `cp.async` pipelining, L2 residency hints — only if simple tiling leaves
     bandwidth on the table per the roofline (§6.6 of draft).
  5. **Shape-dispatch heuristic:** pick design (§6.1 vs single-read vs two-kernel
     vs split-row) at launch from `(M, H)` thresholds, if no single design wins
     across all shapes.

At most one meaningful change per candidate so each eval attributes cause→effect.

## 4. Per-candidate procedure (executable loop)

For each candidate `cNNN`:

1. **Implement** `solution/solution.py` (single immutable source version).
2. **Analytical pre-check (no GPU, conserves budget):**
   - Element-wise math matches reference: fp32 upcast, `/H` variance,
     `cutoff = mean + std·z`, relu, bf16 RNE.
   - Host `ndtri` reproduces A&S constants/branches; 0.1/0.2/0.3 hit the central
     branch → `z ≈ −1.2816 / −0.8416 / −0.5244`.
   - `var = max(var,0)` clamp present; fp32 accumulators; grid covers all `M`
     rows; `BLOCK_H | H` for {4096,8192,12288,16384}; tail mask correct.
   - `target_sparsity == 0.0` early return returns the input unchanged.
   - Confirm only PyTorch metadata/launch usage, no computational fallback.
3. **Record source hash** of `solution/solution.py` (e.g. `sha256`).
4. **Evaluate exactly once:** `./scripts/evaluate_candidate.sh feedback cNNN`.
   Ensure no profiling job is running on the GPU first.
5. **Append** one JSON record to `candidates.jsonl` (never rewrite prior lines) in
   the format of §6.
6. **Decide:** keep / discard / branch per §3 and stopping criteria §7.
7. **Optional profiling** (only when no eval is running), via the workspace
   launcher only: `./scripts/ncu_profile.sh --set <set> -o profile/<id> python
   harness.py`. Build an in-workspace `harness.py` that imports `solution.run` and
   drives the shape(s) of interest. Never invoke `ncu` directly; never profile and
   evaluate concurrently (foreign process ⇒ return code 3, wasted budget slot).

## 5. Profiling protocol (ncu-report-skill)

- Trigger only for a **valid** candidate that is slower than the roofline
  expectation, to choose among performance branches.
- Build `profile/harness.py` inside the workspace that calls `solution.run` on a
  target shape (start with the two extremes: #10 large / #11 tiny).
- Use the `ncu-report-skill` workflow through `./scripts/ncu_profile.sh`.
- Metrics of interest: achieved **DRAM throughput** vs 3.35 TB/s roofline; **L2
  hit rate** on the pass-2 re-read (decides §6.2/§6.3); **occupancy** and **warp
  stall reasons** (decides §6.4 for small-M); registers/spills (decides
  single-read viability).
- Serialize strictly against evaluation: finish one before starting the other.

## 6. Evidence format (candidates.jsonl)

One JSON object per evaluated candidate, appended (append-only), with fields:

```json
{
  "id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py>",
  "hypothesis": "fused one-program-per-row two-pass naive-variance baseline; expect correctness + 2-4x geomean vs reference from fused traffic + fewer launches",
  "config": {"design": "6.1", "BLOCK_H": 1024, "num_warps": 4, "num_stages": 2, "variance": "naive_one_pass"},
  "validation": {
    "analytical_precheck": "pass",
    "all_workloads_correct": true,
    "failures": []
  },
  "per_workload": [
    {"uuid": "151250c8-...", "B":1,"S":512,"H":12288,"sparsity":0.2,"correct": true, "speedup": 0.0}
    /* ... one entry per of the 12 workloads: correct + speedup ... */
  ],
  "geomean_speedup": 0.0,
  "decision": "keep|discard|branch->cNNN",
  "cumulative_evaluations": 1,
  "skills_used": ["<KernelWiki|ncu-report-skill|none>"],
  "notes": "return code, anomalies, ncu findings pointer"
}
```

Rules: `per_workload` lists all 12 with correctness + speedup; `geomean_speedup`
computed over the 12; `cumulative_evaluations` is the running total against the
100 budget; never edit a previously written line.

## 7. Stopping criteria

Stop the search and write `SEARCH_COMPLETE` (with reason) when any holds:

1. **Convergence:** the best valid geomean fails to improve by ≥ ~2% over the
   last 2–3 attempted candidates, and no untried evidence-backed lever remains.
2. **Roofline saturation:** ncu shows the dominant large shapes (#10/#7/#2) at
   ≳ 85% of achievable DRAM bandwidth — further tuning cannot materially help the
   geomean.
3. **Budget:** approaching the 100-evaluation cap, or the token soft limit (5.0M);
   hard-stop before the normal (6.0M) / absolute (6.5M) limits.
4. **No valid candidate producible:** exhausted correctness branch without a
   passing Triton kernel — report the failure honestly (no torch fallback).

`SEARCH_COMPLETE` records: chosen best candidate ID, its geomean, the reason,
cumulative evaluations, and remaining budget. **Never run `final` without explicit
operator approval.**

## 8. Risk register & mitigations (execution-time)

| Risk | Trigger | Mitigation |
|---|---|---|
| Naive variance cancellation | c001 correctness fail on offset data | c00x-A stable two-pass → c00x-B Welford |
| sqrt(negative var) → NaN | any | `var = max(var,0)` clamp (fixed decision) |
| bf16 store mismatch | rtol fail on near-knee values | fp32 compute + RNE store already matches ref; investigate ordering, not dtype |
| Non-contiguous input | eval fail on stride | pass row stride; only then consider `.contiguous()` copy |
| Wasted eval (rc 3) | profiler + eval overlap | strict serialization; never background-profile during eval |
| Register spill in single-read | large H | restrict single-read design to small H; verify via ncu regs/spills |
| Autotune config = source change | changing keyed set | mint new candidate ID each time the config set changes |

## 9. Immediate next actions (subsequent turns, not now)

1. Implement `c001` (§3 root) as `solution/solution.py`.
2. Run analytical pre-check (§4.2), record source hash.
3. `./scripts/evaluate_candidate.sh feedback c001`; append record (§6).
4. Branch per §3 using evidence; profile via §5 only when warranted and never
   concurrent with an eval.

## 10. Decision log

- **c001** — root design §6.1 (fused one-program-per-row, two-pass, naive
  variance, `BLOCK_H=1024/warps=4/stages=2`). First eval hit **rc=3**: a foreign
  process (pid 90762, 910 MiB) appeared on the locked GPU during timing, so the
  controller discarded the measurement; the retry was refused (`already
  completed`). Not a kernel fault. Consumed 1 budget slot. → reissue identical
  source as `c002`.
- **c002** — identical source to c001. **rc=0, 12/12 PASSED**, geomean
  **23.198×** (arith mean 31.47×). All correct within tol (max_rel ≤ 0.167 on
  #2/#7 near the relu knee, still within the 0.99 matched-ratio tolerance).
  Baseline kept. Cumulative evaluations = 2.
  - Speedup spread: tiny/launch-bound shapes dominate (#11 =103×, #4 =57×,
    #9 =48×, #1 =38×, #12 =32×) as predicted; **large bandwidth-bound shapes lag**
    — #10 (B64/S1024/H8192) =8.07×, #7 =9.09×, #2 =9.74×, #3 =12.3×. These four
    set the low end of the geomean and are the primary optimization target.
  - **Next candidate (c003):** performance branch §3.1 — sweep
    `BLOCK_H`/`num_warps`/`num_stages` (favoring more warps + deeper pipelining)
    to raise achieved DRAM bandwidth on the big shapes. Consider ncu on #10 to
    confirm whether pass-2 re-reads miss L2 (→ single-read / two-kernel §3.2)
    before investing there. Profiling only when no eval is running, via
    `./scripts/ncu_profile.sh`.

- **c003** — perf §3.1, `num_warps` 4→8 (only change vs c002). **rc=0, 12/12
  PASSED**, geomean **23.283×** (arith mean 30.65×). New best, but only **+0.37%**
  over c002. Effect is a redistribution, not a uniform win:
  - Big bandwidth-bound shapes improved: #10 8.07→**9.24×**, #7 9.09→**10.29×**,
    #8 19.75→21.01×, #3 12.32→12.43×, #2 9.74→9.83×.
  - Small/launch-bound shapes regressed: #9 47.68→40.03×, #5 19.81→18.67×,
    #11 103.1→99.6×, #1 38.30→37.52×.
  - Kept as best. The tiny net gain shows blind config sweeps are near their
    ceiling; the geomean is now gated by the big shapes (#2/#7/#10 still
    8–10×) whose reference latency is large but our kernel is likely at/near HBM
    roofline.
  - **Next (c004):** before more blind sweeps, profile #10 (B64/S1024/H8192, the
    2.1 GB shape at 9.24×) with `ncu --set basic` via `./scripts/ncu_profile.sh`
    on an in-workspace harness (never during an eval) to measure achieved DRAM
    throughput and the L2 hit rate on the pass-2 re-read. If the pass-2 re-read
    misses L2, move to a single-read (§3.2) or two-kernel design; if we are
    already near roofline, the search has effectively converged.

- **c004** — single-read whole-row design (§3.2 / draft §6.2): one tile
  `BLOCK_H = next_pow2(H)` up to 16384. **rc=137 (SIGKILL)** — the giant
  compile-time tile (16384-wide for H=12288/16384) OOM-killed the Triton/LLVM
  compiler before any workload ran; `runs/candidates/c004/` empty. Exactly the
  register-spill/compile risk the draft flagged for §6.2. Slot consumed.
  → abandon giant single-tile; make it compile-safe by capping single-read to a
  width that compiles.
- **c005** — compile-safe **hybrid**: single-read whole-row tile for `H ≤ 8192`
  (`BLOCK_H = next_pow2(H)`, 1R+1W), proven c003 two-pass loop for `H > 8192`.
  **rc=0, 12/12 PASSED**, geomean **27.936×** (arith mean 40.71×) — **new best,
  +20%** over c003. Single-read is a big win on every `H ≤ 8192` shape:
  #4 56→**93×**, #11 99.6→**156×**, #8 21→**29.4×**, #5 18.7→**22.6×**,
  #7 10.3→**13.2×**, #10 9.24→**11.87×**. The five `H > 8192` shapes
  (H=12288/16384: #1/#2/#3/#6/#12) are unchanged (still two-pass).
  - **Now the geomean is gated by the H=12288/16384 shapes** still on the
    two-pass path (#2 9.78×, #3 12.4×, #6 20.3×). A 16384-wide single tile OOMs
    the compiler (c004), so the lever is the **two-pass loop's `BLOCK_H`**: try
    2048/4096 to shorten the loop and increase in-flight loads, and/or a
    mid-width single-read (e.g. process the row in 2 halves) for H=12288/16384.
  - **Next (c006):** raise the two-pass `BLOCK_H` (H>8192 path) from 1024 to
    2048 or 4096 and re-measure the H=12288/16384 shapes.

- **c006** — two-pass `BLOCK_H` 1024→4096 on the `H > 8192` path (single-read
  path unchanged). **rc=0, 12/12 PASSED**, geomean **31.450×** (arith mean
  45.08×) — **new best, +12.6%** over c005. Every H=12288/16384 shape improved:
  #1 37.9→**60.1×**, #12 32.2→**50.8×**, #6 20.3→**24.5×**, #3 12.4→**14.5×**,
  #2 9.78→**11.07×**. Single-read (H≤8192) shapes unchanged as expected.
  - **Roofline check (H100 HBM3 ≈ 3.35 TB/s):** the near-roofline shapes are the
    big-M ones. #10 (M=65536,H=8192, single-read 1R+1W ≈ 2.1 GB) 0.706 ms vs
    ~0.63 ms ideal → ~89%. #7 (610 MB) 0.202 ms vs ~0.182 ms → ~90%. These are
    essentially done. But the two-pass **H=12288** shapes are at their *2R+1W*
    roofline, not 1R+1W: #2 0.178 ms ≈ 2R+1W ideal (0.180 ms); #3 0.088 ms ≈
    2R+1W ideal (0.090 ms). Moving them to single-read (1R+1W) would cut ~1/3 of
    traffic → est. #2 ~16×, #3 ~21×.
  - **Lever:** extend single-read to H=12288. Blocker: `next_pow2(12288)=16384`,
    and a 16384-wide tile OOM-killed the compiler in c004. Mitigation: use more
    warps (num_warps=16) for the wide tile to cut per-thread register pressure,
    and keep H=16384 on the two-pass path (widest, highest OOM risk).
  - **Next (c007):** raise `single_read_max_H` to 12288 with num_warps=16 on the
    wide single-read tile; H=16384 stays two-pass. If it OOMs (rc 137) record and
    revert; if it compiles, big win on #2/#3/#1/#12.

- **c007** — chunked-resident single-read for `8192 < H` with `≤3` chunks of
  `CHUNK=4096` (so H=12288 → 3 resident tiles, 1R+1W without the monolithic
  16384 tile), `num_warps=16`; H=16384 stays two-pass. **rc=1 FAIL, 8/12** —
  **RUNTIME_ERROR on all four H=12288 shapes** (#1/#2/#3/#12, the chunked path);
  the 8 non-chunked shapes passed. Root cause: the Python-list-of-resident-tiles
  pattern (`resident.append(v)` … `resident[i]`) produced a kernel that faults at
  runtime on the H=12288 path (out-of-resources / illegal access from the
  resident-list codegen; **not** interference — `foreign_process_detected=false`
  and failures are deterministic on exactly the chunked-path shapes). Note the
  controller ran on gpu5 in a **shared-count** lease (a co-tenant present), but
  since the failures are the deterministic runtime errors of a buggy kernel, the
  measurement is a genuine correctness FAIL, not an rc=3 invalidation.
  → **REJECT**; `solution.py` reverted to c006 (best valid). Slot consumed.
  - **Lesson:** avoid Python-list residency across the reduce/apply boundary.
  - **Next (c008):** to still cut H=12288 to 1R+1W, keep residency in
    **explicitly-named** tiles (`v0,v1,v2 = load…`; no list) so codegen is a fixed
    unrolled sequence — or accept c006's two-pass on H=12288 and instead probe
    remaining levers (num_warps/num_stages on the two-pass path, or accept
    convergence). The four remaining laggards (#2 11.07×, #10 11.87×, #7 13.23×,
    #3 14.52×) are all high-M and near their respective roofline, so headroom is
    limited: H=12288 two-pass is at 2R+1W roofline (single-read would give ~1.5×
    on those), the H=8192 single-read shapes are already ~89–90% of 1R+1W.
