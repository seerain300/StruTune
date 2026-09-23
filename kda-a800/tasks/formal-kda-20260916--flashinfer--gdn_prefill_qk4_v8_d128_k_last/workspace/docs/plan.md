# Plan — `gdn_prefill_qk4_v8_d128_k_last`

Executable, sequential KDA optimization plan. Builds on `docs/draft.md`. Target: **A800 `sm_80`
(Ampere)**, **Triton‑only** compute, submission `solution/solution.py::run(q,k,v,state,A_log,a,
dt_bias,b,cu_seqlens,scale) -> (output, new_state)`. Budget: **100 evaluations**, tokens 1.0M soft /
1.2M hard. Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN`; `final` is
operator‑only. This turn writes the plan only — **no implementation, no evaluation**.

---

## 0. Guiding principles

1. **Correctness before speed.** The first candidate is the highest‑confidence, order‑preserving
   f32 implementation. Do not trade precision for speed until a green baseline exists on all five
   workloads.
2. **One immutable idea per candidate.** Each `cNNN` is a single, frozen source version encoding one
   hypothesis. Never mutate an evaluated candidate; never reuse an ID for changed source.
3. **Spend evaluations deliberately.** Each `feedback` run consumes 1 of 100 evaluations and covers
   all five workloads. Reason on paper first; only evaluate when the hypothesis is concrete and the
   guardrail checklist (§4) passes on inspection.
4. **Change one variable at a time** once past the baseline, so each evaluation attributes a delta to
   a cause (precision knob, chunk size, block size, launch structure).
5. **Triton‑only.** No Torch/CPU/NumPy/CUDA‑extension compute fallback. A failing Triton kernel is
   invalid and must be fixed or abandoned, never silently backed by Torch.

---

## 1. Solution scaffold (fixed across candidates)

`solution/solution.py` structure that stays stable so only kernel/config changes drive new IDs:

- `run(...)`: metadata/launch plumbing only (allowed PyTorch use).
  - Validate/normalize `scale` (`None`/`0.0 → 1/sqrt(128)`).
  - Derive `N = cu_seqlens.numel() - 1`, `T`, dtypes; allocate `output [T,8,128] bf16` and
    `new_state [N,8,128,128] f32` (zero‑initialized so empty sequences are correct by default).
  - Compute grids on device from `cu_seqlens` **without per‑sequence `.item()` host syncs** where
    possible (pass `cu_seqlens` into kernels; precompute per‑program `(seq_id, start, len)` mapping
    on device or with a single cheap host pass).
  - Handle `state is None` (zero entry state) even though all feedback workloads pass a state.
  - Launch the Triton kernel(s); return `(output, new_state)`.
- Triton kernel(s): the actual compute; this is what varies per candidate.
- Keep any autotune configs and precision constants as explicit compile‑time args so a change to
  them is a clear new candidate.

---

## 2. Candidate lineage strategy

Lineage is a tree rooted at the correctness anchor. Each node names parent, the single change, and
the hypothesis. IDs are assigned strictly in evaluation order; a rejected branch does not free its ID.

### Phase A — Correctness anchor
- **c001 (parent: none) — A0 order‑preserving f32 recurrence.**
  Grid over `(seq, v_head)` = `N×8`. State `S=[K=128,V=128]` f32 in registers/SMEM. Loop tokens of
  the sequence performing exact rank‑1 gated delta updates and `o=scale·q@S` on the inclusive state,
  in f32. GVA `head//2` indexing; V↔K transpose on load/store; empty‑seq `new_state=0`.
  **Goal:** establish that the pipeline wires up and passes tolerance on all five workloads, and
  reveal the speedup baseline scale. Expected already ≫1× vs the Python‑loop reference (no host loop,
  no per‑token launch), but low arithmetic intensity.

### Phase B — Chunked throughput path
- **c002 (parent: c001) — A1 chunked gated delta rule (UT/WY), conservative & high‑precision.**
  Chunk `C=32`. Per (seq,head): f32 gate precompute + log‑space cumulative decay; decayed intra
  matrices; `[C,C]` UT transform solved in f32; sequential inter‑chunk state scan; intra+inter
  output. State‑touching matmuls use high precision (`input_precision="ieee"` or `"tf32x3"`).
  **Hypothesis:** large speedup on medium/large workloads (15856e8c, 5d3fc66a) while staying within
  tolerance because summation‑order change is bounded by high‑precision matmuls. If correctness
  fails → diagnose (mask/transpose/UT) before any perf tuning; if a numerical fail, branch to smaller
  `C` (c003a) or stricter precision.

### Phase C — Precision tuning (only after B passes)
- **c00x — precision knob sweep** on state‑touching matmuls: `ieee` → `tf32x3` → default TF32.
  One candidate per setting. **Hypothesis:** relax toward TF32 for speed as long as `new_state`
  stays within tolerance. Keep the strictest setting that still passes as the new baseline.

### Phase D — Block/chunk tuning
- **c00x — chunk size** `C ∈ {16, 32, 64}` (one per candidate).
- **c00x — V/K blocking** `BV ∈ {64,128}` (occupancy vs tile‑size tradeoff, esp. for 5d3fc66a).
- **c00x — `num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`.**
  **Hypothesis:** 64 KB f32 state tile limits occupancy; smaller `BV` raises program count and helps
  the large workload; larger tiles help medium. Tune for geomean, re‑validating correctness each step.

### Phase E — Launch structure & overhead (protect the tiny workloads)
- **c00x — fused vs pipelined** kernels (gate precompute inline vs separate pass).
- **c00x — low‑overhead path for tiny sequences** (A2 hybrid): host‑side shape logic routes
  T≤threshold to a lightweight kernel; still one immutable candidate.
  **Hypothesis:** two of five workloads (T=35, 42) are launch/latency bound; cutting launches and
  host syncs lifts their contribution to the geomean without touching the large‑workload path.

Branching rule: if a candidate regresses geomean or fails correctness, revert to the last accepted
node as parent and try the next single change; record the negative result rather than deleting it.

---

## 3. Per‑candidate execution loop

For each `cNNN`, in order:
1. Freeze the intended single change and write it as the source version (one file version).
2. Run the **static guardrail checklist (§4)** by inspection.
3. `./scripts/evaluate_candidate.sh feedback cNNN` (consumes 1 evaluation, all five workloads).
4. Read per‑workload pass/fail + speedup and geomean.
5. Append one JSON record to `candidates.jsonl` (§6). Never rewrite prior records.
6. Decide keep / reject / branch per §5; set the parent for the next candidate accordingly.

---

## 4. Correctness checks (static, before every evaluation)

1. **State transpose**: k‑last `[N,H,V,K]` loaded as internal `[K,V]` and stored back transposed.
2. **GVA indexing**: `q_head = k_head = v_head // 2`, `v_head ∈ [0,8)`; no DRAM expansion of q/k.
3. **Inclusive causal mask**: output uses post‑update state; intra mask includes the diagonal.
4. **Gate math in f32**: `x=a.f32()+dt_bias`, `g=exp(-exp(A_log)·softplus(x))`, `beta=sigmoid(b)`;
   numerically safe `softplus` (`x` large → `x`); cumulative decay in log space (f32).
5. **Scale** applied to the output only (`scale·q@S`), never to the state update.
6. **Empty sequences** (`seq_len ≤ 0`): `new_state[seq]=0` (matches reference `continue` before
   write), no output writes.
7. **Partial‑chunk masking**: padded tokens contribute 0 to matmuls, UT solve, and state update.
8. **Dtypes/shapes**: `output` bf16 `[T,8,128]`, `new_state` f32 `[N,8,128,128]`.
9. **`state is None`** path returns correct zero‑entry‑state behavior.
10. **Variable‑length iteration** driven by `cu_seqlens`; no reliance on uniform seq length.

If any check fails on inspection, fix before spending an evaluation.

---

## 5. Decision & stopping criteria

**Per‑candidate decision:**
- **Reject/invalid** if any workload fails correctness (candidate cannot be selected regardless of
  speed). Diagnose root cause; branch a fix.
- **Keep as new baseline** if all five pass and geomean improves over the current best by a
  meaningful margin (target ≳ 3% geomean, or a clear win on a workload class without regressing
  others).
- **Keep‑but‑not‑baseline / record** if all pass but no improvement — log as evidence, revert parent.

**Global stopping (write `SEARCH_COMPLETE` with reason) when any holds:**
- Evaluation budget (100) reached, or token budget approached (stop well before 1.2M hard).
- Geomean improvement has converged: e.g. ≥3 consecutive accepted‑or‑tried candidates yield <1%
  geomean gain, and remaining design axes (precision, chunk, block, launch) are exhausted.
- No valid candidate exists after exhausting the correctness‑fix branches (report failure honestly;
  never substitute a non‑Triton fallback).

**`final`** is run only after explicit operator approval, on the best valid candidate.

---

## 6. Evidence format (`candidates.jsonl`, one JSON object per evaluated candidate)

Append‑only; never rewrite earlier records. Each record contains:

```json
{
  "candidate_id": "c001",
  "parent": null,
  "source_sha256": "<hash of solution.py at evaluation time>",
  "phase": "A|B|C|D|E",
  "hypothesis": "one-sentence claim being tested",
  "change_from_parent": "the single variable changed",
  "static_checks": {"passed": true, "notes": "..."},
  "workloads": [
    {"uuid": "1efaf2a9...", "T": 42,   "N": 2,  "correct": true, "speedup": 0.0},
    {"uuid": "d3dc3577...", "T": 35,   "N": 1,  "correct": true, "speedup": 0.0},
    {"uuid": "1d0cc342...", "T": 294,  "N": 3,  "correct": true, "speedup": 0.0},
    {"uuid": "15856e8c...", "T": 3028, "N": 5,  "correct": true, "speedup": 0.0},
    {"uuid": "5d3fc66a...", "T": 8192, "N": 34, "correct": true, "speedup": 0.0}
  ],
  "all_correct": true,
  "geomean_speedup": 0.0,
  "decision": "keep-baseline|reject|record-only|branch",
  "cumulative_evaluations": 1,
  "skills_used": [],
  "notes": "diagnosis / next step"
}
```

Rules: `all_correct` must be true for a candidate to be selectable; `geomean_speedup` is the primary
ranking metric; `cumulative_evaluations` increments by exactly 1 per record; `skills_used` records
any KernelWiki/ncu usage (expected empty — see draft §6, Ampere/no‑profiler). Keep hypothesis and
`change_from_parent` to one clear statement each so the lineage is auditable.

---

## 7. Risk register → mitigation mapping

| Risk (draft §3) | First mitigation | Fallback branch |
|---|---|---|
| TF32 drift in `new_state` | c002 uses `ieee`/`tf32x3` on state matmuls | Phase C relaxes only if tolerance holds |
| UT‑transform conditioning | solve in f32, `C=32` | smaller `C=16` branch |
| Gate overflow (softplus/exp) | safe softplus, log‑space cumsum in f32 | — |
| Partial/empty chunks | explicit masking; `new_state=0` for empty seq | dedicated static check #6,#7 |
| Order‑change error | c001 anchors exact order first | keep c001 as correctness reference |
| Tiny‑workload overhead | minimize launches/host syncs; Phase E hybrid | route by shape in `run` |

---

## 8. Immediate next actions (subsequent turns, not this one)

1. Implement **c001** (A0 order‑preserving f32 recurrence) per §1 scaffold and §4 checks.
2. Evaluate c001; record evidence (§6); confirm all five pass and note baseline geomean.
3. Proceed to **c002** (A1 chunked) only after c001 is green.

No implementation or evaluation is performed in this turn.

---

## 9. Decision log

- **c001 (Phase A, eval #1) — KEEP-BASELINE.** Order-preserving pure-f32 per-token recurrence.
  All five workloads PASS (atol=0.01, rtol=0.01, matched_ratio=0.99). Geomean **127.88x**
  (arith mean 140.06x). Per-workload speedups: 15856e8c=130.9x, 1efaf2a9=107.5x, d3dc3577=69.5x
  (tiny single-seq, only 16 programs — the geomean floor), 5d3fc66a=255.7x (large, best
  parallelism), 1d0cc342=136.6x. Correctness is comfortable; the large workload is already fast,
  and the small/tiny ones are launch/occupancy bound rather than compute bound.
  **Next (c002):** Phase B chunked gated-delta-rule (UT/WY) to raise throughput on medium/large
  via tensor cores, keeping state-touching matmuls high-precision, while not regressing the tiny
  single-seq case. Because the whole op is latency/occupancy-bound (not compute-bound) and c001 is
  already ~70–256x, the expected upside is moderate; will re-validate correctness and keep the
  strictest precision that passes.

- **c002 (Phase D, eval #2) — KEEP-BASELINE.** Occupancy tuning: `BV=64 -> BV=32` (V-blocks per
  (seq,head) 2 -> 4, program count `N*Hv*2 -> N*Hv*4`). Semantics-preserving (V columns are
  independent), confirmed by max_abs/max_rel identical to c001. All five PASS. Geomean
  **127.88x -> 174.75x (+36.7%)**; every workload improved: 15856e8c 130.9->184.3x,
  1efaf2a9 107.5->131.8x, d3dc3577 (floor) 69.5->89.1x, 5d3fc66a 255.7->373.8x,
  1d0cc342 136.6->201.5x. This is strong evidence the op is **occupancy/latency-bound**: adding
  independent programs helps uniformly, including the large workload (which was assumed near-saturated
  at 544 programs). Deferred the chunked/UT path (Phase B) because the cheap occupancy lever is
  paying off and carries no numerical risk.
  **Next (c003):** continue the occupancy sweep — `BV=16` (8 V-blocks/head, program count `N*Hv*8`).
  Hypothesis: the tiny single-seq floor (d3dc3577, still only 32 programs at BV=32) and other small
  workloads keep improving until per-program launch/register/DRAM-traffic overhead offsets the
  occupancy gain. Keep the best-performing BV as baseline; if BV=16 regresses, revert to c002 and pivot
  to the chunked path or num_warps tuning.
