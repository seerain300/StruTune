# Plan — L2/043 Mamba-2 Chunk Scan with Segment Sum (H100 / sm_90)

Executable, sequential KDA optimization plan derived from `docs/draft.md`.
This turn produces the plan only; no candidate is implemented or evaluated here.

Authoritative constraints (from `CLAUDE.md` / `TASK.md`):
- Triton-only compute; PyTorch allowed for metadata/launch plumbing only; **no**
  Torch/CPU/NumPy/CUDA-extension/alternate fallback. A failing Triton kernel is
  invalid and must be fixed, never replaced by a fallback.
- Submission: `solution/solution.py` exposing `run(hidden_states, A, B, C, D, initial_states)`.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback <cid>` (full 16-workload
  feedback set = one evaluation). Budget: 100 evaluations. Token soft/normal/absolute
  limits: 9M / 10M / 11M.
- Immutable candidate IDs `c001, c002, …`; any meaningful source/config/launch change
  ⇒ new ID; never reuse an ID for changed source; never rewrite earlier records.
- Profiling only via `./scripts/ncu_profile.sh` (never raw `ncu`), **never** concurrent
  with an evaluation (concurrency ⇒ return code 3, wasted eval).
- `final` only with explicit operator approval. Write `SEARCH_COMPLETE` on convergence.

---

## 1. Objective and target restatement

Reproduce the reference `run` (see §2 of the draft for the verified math) as a fused
Triton implementation that is numerically within tolerance on all 16 feedback
workloads and maximizes geometric-mean speedup. The reference is memory-bandwidth
bound on ~0.5 GB fp32 `[B,H,NC,Q,Q]` `L` and `[B,NC,Q,Q,H]` `CB` intermediates and
many separate einsums; the win comes from keeping `Q×Q` blocks on-chip, exploiting
`G=1` (B/C shared across the 16 heads), and never materializing the giant tensors.

Fixed constants: `H=16, P=64, N=256, G=1, Q(chunk)=256`. Variable: `B`, `S`.

---

## 2. Algorithm decomposition (target implementation)

The recurrence over chunks is the only sequential dependency. Adopt the standard
`mamba_ssm`/FLA four-stage split (draft §4, Option A → hybrid Option C):

1. **cumsum** `acs[b,h,c,t] = cumsum_t A` (padded 0), and chunk totals
   `dchunk[b,h,c] = acs[...,Q-1]`. Fuse into consumers via `tl.cumsum` where cheap.
2. **chunk_state** (parallel over `b,c,h`): `state[b,c,h,d,s] = Σ_t x·B·decay_states`,
   `decay_states[t]=exp(acs[Q-1]-acs[t])`. Shape `[B,NC,H,P,N]`.
3. **state_passing** (sequential over `c`, parallel over `b,h` and `P·N` tiles):
   `S_c = exp(dchunk_{c-1})·S_{c-1} + state_{c-1}`, `S_0=initial_states`. Emit
   `states_in[b,c,h,d,s]` (state entering chunk `c`) and `final_state = S_{NC}`.
4. **chunk_scan** (parallel over `b,c,h` + row tiles of `Q`): `CB[i,j]=Σ_s C[i,s]B[j,s]`
   (shared across heads), `L[i,j]=exp(acs[i]-acs[j])` for `i≥j` else 0,
   `Y_diag=Σ_{j≤i} CB·L·x`; `Y_off[t,d]=(Σ_s C[t,s]·S_c[d,s])·exp(acs[t])`;
   `y=Y_diag+Y_off+D[h]·x`; de-pad, reshape `[B,S,H*P]` head-major, cast bf16.

Correctness-critical invariants (from draft §5): all cumsum/exp/decay math and matmul
accumulation in **fp32**; `L` derived from the same `acs` everywhere; causal mask
`i≥j` with `L[i,i]=1`; padded positions load 0 via masked loads; initial-state path
never special-cased away; output head-major `out[..., h*64+d]`; `D`-residual added in
fp32 pre-cast.

---

## 3. Candidate lineage strategy

Each candidate is one immutable source version of `solution/solution.py`. Lineage is a
DAG rooted at the first correct kernel; each node states parent, a single-variable
hypothesis, and a keep/revert decision. Only proceed to optimization candidates after
correctness is locked.

### Phase 0 — Correctness (must pass all 16 before any perf work)
- **c001 — correctness-first fused baseline.** Full 4-stage decomposition, simplest
  safe tiling (`BLOCK_M=64`, full `N=256`/`Q=256` contraction blocks as fits),
  fp32 or tf32 `tl.dot`, straightforward grids. Goal: pass all 16 workloads; also
  probes toolchain (`tl.cumsum`, `tl.dot` accumulate/tf32, `tl.associative_scan`
  availability). If any workload fails correctness, the next candidate fixes the bug
  (still Phase 0), not performance.

### Phase 1 — Precision/perf of the dominant kernel
Profile after c001 to confirm which kernel dominates (expected: `chunk_scan`, the
`Q×Q` term). Then, one variable per candidate:
- **Precision ladder** on the two contractions (`CB` over `N`, `Y_diag`/`Y_off`):
  bf16 dot → tf32 dot → fp32 dot. Pick the fastest that keeps ≥98% match with margin.
- **Head reuse:** loop `H` inside a `(b,chunk)` program to compute `CB` once vs one
  program per `(b,chunk,h)`. Measure occupancy vs reuse tradeoff (matters on small `B`).
- **Causal block skipping:** skip `j`-blocks entirely above the diagonal.

### Phase 2 — Tiling / occupancy / launch
- Autotune-style sweeps (each config = new candidate if it changes shipped source):
  `BLOCK_M ∈ {32,64,128}`, contraction `BLOCK_K ∈ {64,128,256}`, `num_warps ∈ {4,8}`,
  `num_stages ∈ {2,3,4}`. Prefer a single autotuned kernel (config list is part of the
  immutable source) over many hand-picked candidates to conserve the eval budget.
- Grid layout for small-batch shapes (`B=1`): more parallelism (split `Q` rows, split
  `P`/`N`) to fill 132 SMs.

### Phase 3 — Fusion / HBM reduction
- Fuse `chunk_cumsum` into consumers (drop a kernel + HBM round-trip).
- Consider fusing `state_passing` into `chunk_scan` or storing `chunk_state`/`states_in`
  in bf16/fp16 to cut the `[B,NC,H,P,N]` HBM traffic (measure vs precision cost).
- Per-`(b,h)` sequential-over-chunks fused variant (draft §4.2) as an alternative for
  large `S` / small `B`; keep only if it beats the parallel design.

Each phase gates the next: do not start Phase N+1 until Phase N's best candidate is
recorded and shows the expected trend. Revert any candidate that regresses geomean or
breaks correctness; branch the next hypothesis from the last kept candidate.

---

## 4. Correctness checks

1. **Pre-eval logic review (no GPU):** before spending an evaluation, statically
   verify against draft §2 math: cumsum/decay signs, causal mask direction
   (`i≥j`), initial-state prepend, head-major output index, fp32 accumulation, masked
   padded loads.
2. **Authoritative check:** `./scripts/evaluate_candidate.sh feedback <cid>` over all
   16 workloads. Correctness is per-workload (`atol` 0.014–0.046, `rtol` 0.05,
   match-ratio ≥ 0.98). A candidate "passes" only if **all 16** pass.
3. **Optional local sanity harness (never shipped as fallback):** a separate local
   script may compare the Triton `run` against a torch copy of the reference on a
   couple of shapes to catch gross bugs before an eval; the shipped `solution.py`
   stays Triton-only with no torch compute path.
4. **Boundary coverage** is automatic: the feedback set spans exact multiples
   (256/512/1024/2048/4096), pad cases (293/541/997/1321/1879), single-chunk (256),
   and batch 1→32. Any pad/single-chunk/layout bug fails immediately.
5. **Margin tracking:** where the evaluator surfaces max error / match ratio, log it to
   judge how much tolerance headroom each precision choice leaves before committing to
   the fastest safe variant.

---

## 5. Performance hypotheses (each testable, one variable)

- **H1 (fusion):** Eliminating the giant `L`/`CB` HBM intermediates via a fused
  on-chip `Q×Q` kernel yields a large geomean speedup, largest on big-`Q×Q` shapes
  (`B=1,S=4096`; `B=4,S=2048`; `B=8,S=1024`). *Test:* c001 geomean vs reference.
- **H2 (head reuse):** Computing `CB` once per `(b,chunk)` and looping the 16 heads
  reduces redundant `N=256` contraction work ~16× for that term. *Test:* head-loop
  candidate vs per-`(b,chunk,h)` candidate.
- **H3 (precision):** bf16/tf32 `tl.dot` is markedly faster than fp32 dot and stays
  within the 0.98 match ratio given the loose tolerances. *Test:* precision-ladder
  candidates; keep fastest with correctness margin.
- **H4 (causal skip):** Skipping above-diagonal `j`-blocks roughly halves `chunk_scan`
  matmul work. *Test:* skip vs no-skip candidate.
- **H5 (occupancy):** For small `B`, splitting `Q` rows / `P`,`N` across more programs
  fills SMs and improves those shapes without hurting large `B`. *Test:* grid-split
  candidate on `B=1` shapes.
- **H6 (state HBM):** Storing `chunk_state`/`states_in` in bf16 or fusing state-passing
  cuts HBM traffic with acceptable precision cost. *Test:* bf16-intermediate / fused
  candidate vs fp32-intermediate.

Profiling (via `./scripts/ncu_profile.sh`, never concurrent with an eval) confirms the
mechanism behind each accepted candidate (bandwidth reduction, occupancy, dot dtype).

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- Improvement has converged: best geomean gains < ~2% over the last 2–3 accepted
  candidates and no untested hypothesis is expected to help materially.
- Kernel is near a hardware bound (profiler shows the dominant kernel bandwidth- or
  compute-bound near roofline with high occupancy).
- Evaluation budget nearly exhausted (reserve ≥ a few evals; do not exceed 100).
- Token soft limit (9M) approached: wind down to recording the best valid candidate.

Never run `final` without explicit operator approval; when approved, run `final` once
on the best valid candidate.

---

## 7. Evidence format (append one JSON object per evaluated candidate to `candidates.jsonl`)

Never rewrite earlier records. Each record includes:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "correctness-first fused 4-stage baseline; probe tl.cumsum/tl.dot",
  "changes_vs_parent": "initial implementation",
  "validation": {
    "stage": "feedback",
    "all_pass": true,
    "num_workloads": 16,
    "per_workload": [
      {"uuid": "6dc47920-…", "axes": {"batch_size": 1, "seq_len": 1024},
       "pass": true, "speedup": 0.0, "max_err": null, "match_ratio": null}
      /* one entry per workload */
    ]
  },
  "geomean_speedup": 0.0,
  "decision": "keep|revert|bugfix-next",
  "cumulative_evaluations": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill?"],
  "notes": "profiler findings, precision margin, next hypothesis"
}
```

Rules: `source_sha256` recomputed each eval; `cumulative_evaluations` monotonic;
`decision` justifies the next lineage step; record every workload's pass/fail even when
all pass; note skill usage per record.

---

## 8. Immediate next actions (next turn, not this one)

1. Implement **c001** (`solution/solution.py`): 4-stage Triton decomposition per §2,
   correctness-first tiling, fp32/tf32 dots, Triton-only.
2. Static logic review against draft §2 invariants.
3. Evaluate `./scripts/evaluate_candidate.sh feedback c001`; append the record.
4. If all pass → profile the dominant kernel (non-concurrent) → begin Phase 1.
   If any fail → Phase 0 bugfix candidate.

---

## 9. Decision log

### c001 — evaluated (feedback), 1 cumulative evaluation — REVERT
- **Toolchain confirmed:** Triton 3.5.0 (SOL-ExecBench venv). `tl.dot` supports
  `input_precision ∈ {"tf32","tf32x3","ieee"}`; default is `tf32`.
- **Result:** 0/16 passed — **every** workload `RUNTIME_ERROR` (`max_abs=max_rel=0.0`),
  including the simplest `B=1,S=256,NC=1`. Clean run: `foreign_process_detected=false`,
  return code 1 (a real failure, not code-3 interference). The evaluator surfaces only
  per-workload status, not the Python/Triton traceback.
- **Diagnosis:** uniform failure across all shapes ⇒ a compile/launch/resource error,
  not a shape-edge/pad or numerical bug. Prime suspect is register/resource pressure in
  `_chunk_scan_kernel`/`_chunk_state_kernel`, which hold full state-dim `N=256` fp32
  tiles live (`C_i[64,256]`, `R_c[64,256]`, `B_j[·,256]`) and use fp32 `ieee` dots —
  likely a PTXAS register-allocation / "too many resources" launch failure that fails
  identically everywhere. Secondary suspects: the `ieee` fp32 dot path and the dynamic
  loop bound `range(0, mblock+1)`.
- **Phase 0 still active** (no correct baseline yet).

### c002 — planned (bugfix, Phase 0)
- **Change (single theme = shrink live footprint + cheaper dots):** tile the state dim
  `N=256` with an explicit `BLOCK_K` loop (accumulate `CB`/`Y_off`/`chunk_state` over
  K-blocks) so no `[·,256]` fp32 tile is fully resident; keep `P` tiled where helpful.
  Move accuracy-critical `tl.dot`s to bf16/tf32 inputs with fp32 accumulate (loose
  tolerances — atol ≥ 0.014, match ≥ 0.98 — give headroom), while keeping cumsum, exp,
  decays, and the `D` residual in fp32.
- **Fallback if still RUNTIME_ERROR:** obtain the concrete error via a minimal
  single-shape probe / `./scripts/ncu_profile.sh` run (never concurrent with an eval)
  before spending another evaluation.

### c002 — evaluated (feedback), 2 cumulative evaluations — REVERT
- **Change made:** N contraction tiled to ≤64×64 `BLOCK_K=64` in `chunk_scan` (Y_off
  and CB); `chunk_state` output tiled over N (`BLOCK_S=64`); `tl.dot` → `tf32`.
- **Result:** identical 0/16, **all RUNTIME_ERROR**, `max_abs=max_rel=0.0`, clean run
  (`foreign_process_detected=false`, return code 1).
- **Resource-pressure hypothesis REJECTED.** c002 is far lighter than c001 yet fails
  byte-for-byte the same, including `B=1,S=256,NC=1`. Since the kernels are compiled
  once per constexpr signature (all of `H,P,N,Q,BLOCK_*` are constant across every
  workload; only `S,NC` are runtime args), a uniform all-shape `RUNTIME_ERROR` points
  to a **compile-time error in the shared compiled kernel** that is reused (and
  re-fails) on every workload — not a resource/shape issue.
- **New prime suspect:** the `program_id`-dependent **dynamic loop bound**
  `for jblock in range(0, mblock + 1)` in `_chunk_scan_kernel` (with `mblock =
  tl.program_id(1)`) wrapping `tl.dot` and a nested static k-loop — a likely
  Triton pipelining/lowering failure at compile time. This construct is present
  identically in both c001 and c002. Secondary: a shared-code API issue
  (`tl.cumsum` on a 1-D tensor, or the `state_passing` `range(0,NC)` carrying a
  register tile across iterations).

### c003 — planned (bugfix, Phase 0)
- **Change (single theme = remove the dynamic loop bound):** make `chunk_scan`'s
  column loop fully static — `for jblock in tl.static_range(0, Q // BLOCK_N)` over
  **all** column blocks, relying on the existing causal `tl.where(rows>=cols, …, 0)`
  mask to zero the strictly-upper-triangular blocks (numerically identical; the cost
  is redundant masked matmuls, acceptable for a Phase-0 correctness baseline). Keep
  every other kernel unchanged.
- **Escalation if c003 still RUNTIME_ERRORs:** stop guessing and capture the concrete
  Python/Triton traceback via a minimal single-shape probe under
  `./scripts/ncu_profile.sh` (never concurrent with an evaluation) before spending
  another evaluation on a blind fix.
