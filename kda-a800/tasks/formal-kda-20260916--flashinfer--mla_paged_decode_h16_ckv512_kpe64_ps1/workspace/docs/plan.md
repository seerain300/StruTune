# Executable Optimization Plan — `mla_paged_decode_h16_ckv512_kpe64_ps1`

Companion to `docs/draft.md`. This plan is the **operational contract** for the KDA search: it fixes
the candidate lineage, the exact `run(...)` interface, the per-candidate correctness gate, the
performance hypotheses each candidate tests, the stopping criteria, and the evidence schema written to
`candidates.jsonl`. No code is written or evaluated in this turn.

Target: **A800 / sm_80 (Ampere)**. Primary impl **Triton**; PyTorch only for metadata/launch. No
Torch/CPU/NumPy/CUDA-ext computational fallback. A failed Triton candidate is invalid and is **not**
replaced by a fallback — it is recorded as failed and the next candidate iterates.

---

## 0. Ground rules & interface contract

### 0.1 Entry point (`solution/solution.py`)
```python
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # returns (output, lse)
    #   output: bf16 [B, 16, 512]
    #   lse:    fp32 [B, 16]   (base-2 log-sum-exp)
```
- Signature/positional order must match `definition.json` exactly.
- All compute tensors already on GPU (`q_nope.device`). Allocate outputs on that device.
- Read constants from tensor shapes; assert `num_qo_heads==16, ckv==512, kpe==64, page_size==1` (mirrors
  reference; cheap and catches launch-plumbing mistakes).
- `sm_scale` is a runtime **python/scalar float32** — pass as a kernel scalar arg, never bake a constant.
- Host code may use torch for: output allocation, reading `kv_indptr`/max-length for grid/split sizing,
  computing `num_splits`, allocating scratch. It must **not** perform the attention math.

### 0.2 Immutability / lineage rules (from CLAUDE.md, TASK.md)
- Candidates are `c001, c002, …`, implemented **one source version at a time**, each immutable once
  evaluated.
- Any meaningful change to source, config (block sizes, num_warps, num_stages, split policy), or launch
  ⇒ **new candidate ID**. Never mutate an evaluated candidate's source.
- Evaluate **only** with `./scripts/evaluate_candidate.sh feedback cNNN`. The 5 fixed feedback workloads
  together = **one** candidate evaluation.
- Append exactly **one** JSON record per evaluated candidate to `candidates.jsonl`; never rewrite prior
  records.
- Budget: 100 candidate evaluations; token soft 1.0M / hard 1.2M. Stop at budget or convergence.
- `final` (47-workload) only on explicit operator approval — never self-initiated.

### 0.3 Source layout convention
- Keep the *current* candidate's implementation at `solution/solution.py` (the harness hashes this).
- Preserve each evaluated candidate's exact source under `runs/candidates/cNNN/solution.py` (copy) so
  lineage and source hashes are auditable without rewriting history. (If the harness snapshots source
  itself, this is a redundant local audit copy only; it never overrides the evaluated file.)
- Record the source hash reported by the harness (or `sha256(solution.py)`) in each evidence record.

---

## 1. Reference-equivalence specification (the correctness oracle in words)

Every candidate MUST reproduce, per batch element `b` with token list
`tok = kv_indices[kv_indptr[b]:kv_indptr[b+1]]`, `L=len(tok)`:

1. `S = (q_nope[b] @ Kc.T + q_pe[b] @ Kp.T) * sm_scale`  where `Kc=ckv[tok]`, `Kp=kpe[tok]`.
   - Scores accumulated in fp32; QK dot uses bf16 inputs (exact) → fp32 acc.
2. `lse[b] = log2(sum_j exp2(S_j * log2e))`  = `logsumexp(S)/ln2` (base-2), fp32.
3. `attn = softmax(S)` (fp32), `output[b] = (attn @ Kc).to(bf16)`  (value == `Kc`, the 512-dim latent).
4. **Empty** (`L==0`): `output[b]=0`, `lse[b]=-inf`.
5. `sm_scale` applied exactly once, before softmax/lse.

Implementation identities used (must hold in every candidate, incl. split-KV):
- Base-2 online softmax: `m,l` tracked in fp32; `p=exp2(S*log2e - m)`; `lse=m+log2(l)`.
- Split merge is associative: for partials `(m_i,l_i,acc_i)`,
  `M=max_i m_i`, `l=Σ l_i·2^{m_i-M}`, `acc=Σ acc_i·2^{m_i-M}`, `out=acc/l`, `lse=M+log2(l)`.
- Masked tail lanes (n ≥ split_end) get `S=-inf` pre-`exp2` ⇒ `p=0`, never NaN.

This spec is the checklist against which each candidate is read before evaluation (§4).

---

## 2. Candidate lineage strategy

Sequential, one hypothesis per ID, smallest attributable change. Planned spine (later IDs may adapt
based on evidence — the plan is a decision tree, not a fixed list):

| ID | Change vs parent | Primary hypothesis (H) | Success signal |
|----|------------------|------------------------|----------------|
| **c001** | Baseline: single fused kernel, `grid=(B,)`, 16 heads together, fp32 online base-2 softmax, bf16 QK & PV dots, fp32 acc, empty-seq guard, direct write (no split). | Fusing the whole op into one Triton kernel that streams KV once beats the eager Python-loop reference on all 5 workloads. | All 5 pass correctness; geomean > 1.0 (expect ≫). |
| **c002** | Add **split-KV** (partial + combine kernels), adaptive `num_splits`; `num_splits==1` fast path writes directly (≡ c001 behavior). | Filling the GPU at low batch (esp. wl3 B=1; also wl1/2/4 with 16 CTAs) removes SM under-utilization. | wl3 speedup ↑ markedly vs c001; others ≥ c001; geomean ↑. |
| **c003** | Tune `BLOCK_N` (try 64→{32,128}) at fixed split policy. | KV-streaming throughput is sensitive to token-block size / SMEM footprint. | geomean ↑ or identify best BLOCK_N. |
| **c004** | Tune `num_warps`/`num_stages` (pipelining of gather+K load). | Deeper pipelining hides gather latency; right warp count balances acc register pressure vs occupancy. | geomean ↑. |
| **c005** | Refine `num_splits` policy (target-CTA multiplier, min tokens/split, per-batch max-L). | Better occupancy match across the batch spectrum (1/16/64) than the c002 heuristic. | wl3 & wl5 both ≥ prior best; geomean ↑. |
| **c006+** | Structural options as evidence dictates: SMEM-resident `Kc` reuse, output-dim tiling if register spill observed, fused single-pass combine, or `num_splits` LUT per (B,max_L). | Address the specific bottleneck the evidence points to. | geomean ↑ without correctness regression. |
| **cNNN (contingency)** | **Numerical hardening** — only if any candidate fails correctness: PV in TF32/fp32, or hi/lo bf16 split of `P`. | Reference PV is fp32; bf16 PV lost too much precision for the evaluator tolerance. | Failing workload(s) now pass; minimal perf cost. |

Branching policy:
- Advance the "best valid candidate" pointer only to a candidate that (a) passes all 5 and (b) has
  geomean ≥ current best (ties broken toward simpler/lower-variance source).
- If a tuning candidate regresses, **do not** revert by editing an old ID; branch a new ID from the last
  best. Keep exactly one change dimension per ID so regressions are attributable.
- Contingency (numerical) candidates preempt tuning: correctness first, then speed.

---

## 3. Per-candidate execution procedure (repeat for each cNNN)

1. **Design note** (in the evidence record's `hypothesis`): state the single change and expected effect.
2. **Implement** `solution/solution.py` (Triton kernel(s) + host launcher). One source version only.
3. **Static self-review** against §1 spec and §4 checklist (reasoning-level; no alternate numeric
   harness, no direct CUDA/profiler run).
4. **Snapshot** source to `runs/candidates/cNNN/solution.py`; compute/record source hash.
5. **Evaluate once**: `./scripts/evaluate_candidate.sh feedback cNNN`.
6. **Record** one JSON line in `candidates.jsonl` per §5 (parent, hash, hypothesis, validation, per-
   workload result, geomean, decision, cumulative eval count, skill usage).
7. **Decide**: keep/branch/revert-pointer/harden per §2 branching policy and §6 stopping check.

Never run step 5 more than once for the same source. If a code change is needed after evaluation, it is
a new ID.

---

## 4. Correctness checks (gate before every evaluation)

Static checklist read against the source (must all be ✔ before invoking the evaluator):

- **C1 Shapes/dtypes:** returns `(output bf16 [B,16,512], lse fp32 [B,16])`; outputs allocated on input
  device; positional signature matches `definition.json`.
- **C2 Score math:** `S = qn·Kcᵀ + qp·Kpᵀ`, both terms; fp32 accumulation; `sm_scale` applied exactly
  once.
- **C3 Value math:** output uses `Kc` (ckv, 512-dim) as V — **not** `Kp`.
- **C4 LSE base-2:** `lse = m + log2(l)` with base-2 online softmax (`exp2`, `log2e` scaling); fp32.
- **C5 Empty sequence:** `L==0` ⇒ `output[b]=0`, `lse[b]=-inf`; `acc/l` division guarded when `l==0`.
- **C6 Masking:** tail lanes `n≥end` get `S=-inf` before `exp2` ⇒ `p=0`; no NaN/Inf leakage into acc.
- **C7 Indexing:** gather uses `kv_indices[kv_indptr[b]+..]`; int64 base-pointer arithmetic for the
  989k-row cache; per-batch `L` read from indptr (no equal-split assumption).
- **C8 Split correctness (c002+):** partials use consistent `sm_scale` (applied once, in partial kernel);
  combine uses the associative merge in §1; `num_splits==1` path is bit-for-bit the single-kernel path;
  empty batch handled in both kernels; scratch fully covered (no uninitialized partial read).
- **C9 No fallback:** no Torch/CPU/NumPy/CUDA-ext compute path anywhere; Triton is the only compute.
- **C10 Launch integrity:** grid covers all `(b, split)`; scratch buffers sized for `B·num_splits`;
  `num_splits` recomputed per workload from that workload's shapes.

If the evaluator reports a correctness failure, the candidate is recorded **failed** (decision =
`reject`), and the next candidate targets the specific failure (numerical hardening or masking/empty-seq
fix), not a re-run of the same source.

---

## 5. Evidence format (one JSON object per evaluated candidate, appended to `candidates.jsonl`)

```json
{
  "candidate": "c001",
  "parent": null,
  "source_hash": "<sha256 of solution.py at eval time>",
  "hypothesis": "Single fused Triton kernel streaming KV once beats the eager Python-loop reference.",
  "change_summary": "Baseline; grid=(B,), 16 heads/CTA, fp32 online base-2 softmax, bf16 QK+PV, fp32 acc.",
  "config": {"BLOCK_N": 64, "num_warps": 4, "num_stages": 3, "split_kv": false, "num_splits": 1},
  "validation": {
    "static_checks": {"C1": true, "C2": true, "C3": true, "C4": true, "C5": true,
                       "C6": true, "C7": true, "C8": "n/a", "C9": true, "C10": true},
    "all_correct": true
  },
  "workloads": [
    {"uuid": "220b10b5-...", "B": 16, "num_kv_indices": 10857, "correct": true, "speedup": 0.0},
    {"uuid": "e417264f-...", "B": 16, "num_kv_indices": 12857, "correct": true, "speedup": 0.0},
    {"uuid": "deb5f26c-...", "B": 1,  "num_kv_indices": 208,   "correct": true, "speedup": 0.0},
    {"uuid": "34026642-...", "B": 16, "num_kv_indices": 1857,  "correct": true, "speedup": 0.0},
    {"uuid": "7e083fc8-...", "B": 64, "num_kv_indices": 30745, "correct": true, "speedup": 0.0}
  ],
  "geomean_speedup": 0.0,
  "decision": "accept|reject|keep-as-best|superseded",
  "cumulative_evaluations": 1,
  "skills_used": [],
  "notes": "Observations, regressions, next-step rationale (which hypothesis to test next)."
}
```
Field rules:
- `speedup`, `geomean_speedup`: copied verbatim from the evaluator output (do not recompute/round away
  the evaluator's numbers). If a workload fails correctness, mark `correct:false` and set its speedup as
  reported (or `null`) and `all_correct:false`, `geomean_speedup` per evaluator.
- `parent`: the candidate this one was derived from (`null` for c001).
- `decision`: `accept` (valid, becomes/keeps a lineage node), `reject` (failed correctness or clear
  regression), `keep-as-best` (new best-valid pointer), `superseded` (later candidate beat it).
- `cumulative_evaluations`: running count of feedback evaluations used (budget tracking vs 100).
- `skills_used`: expected `[]` for all candidates (see §8) unless a skill is actually consulted.
- Never edit a prior line; corrections/observations about an old candidate go in the *new* candidate's
  `notes`.

A short human-readable summary table (best-so-far geomean, per-workload trend) will be kept in
`docs/plan.md` §7 progress log or a `docs/progress.md` appended after each eval — never by rewriting
`candidates.jsonl`.

---

## 6. Performance hypotheses & expected evidence

Ranked, each mapped to the candidate that tests it and the workload where it should show:

- **H1 (fusion).** Replacing the eager per-batch Python loop with one fused kernel removes launch and
  intermediate-traffic overhead → large geomean gain, biggest on **low-batch/short-seq** where per-op
  overhead dominates (wl3, wl4). Tested by **c001**.
- **H2 (KV read once, reuse across 16 heads + K/V).** Batching heads and reusing `Kc` for score+value
  keeps the kernel near the bandwidth roofline (~30 FLOP/byte). Present from **c001**; validated by
  near-bandwidth scaling on the highest-traffic workload (**wl5**, ~35 MB).
- **H3 (occupancy via split-KV).** At B∈{1,16} the batch-only grid underfills 108 SMs; split-KV raises
  concurrent CTAs to ~2–4× SM count without adding KV traffic. Tested by **c002**; expect **wl3 (B=1)**
  to jump most, wl1/2/4 to improve, wl5 (already B=64) roughly flat.
- **H4 (block size).** `BLOCK_N` trades SMEM footprint / gather efficiency vs pipeline depth. Tested by
  **c003**; expect a modest, workload-dependent optimum around 64.
- **H5 (pipelining / warps).** `num_stages`/`num_warps` hide gather+K-load latency but fight the
  `[16,512]` fp32 acc register pressure. Tested by **c004**; watch for spill regressions.
- **H6 (split policy).** A shape-aware `num_splits(B, max_L)` beats a flat heuristic across the 1→64
  batch span. Tested by **c005**.
- **H7 (structural, contingent).** SMEM-resident `Kc`, output-dim tiling, or fused combine help only if
  evidence shows the specific bottleneck (spill, combine cost, re-read). Tested by **c006+** as needed.

Interpretation guide when reading evaluator output:
- Compare per-workload speedups **and** geomean across candidates. Attribute wl3 movement to occupancy
  (H3/H6), wl5 movement to bandwidth/config (H2/H4/H5), wl4 to overhead (H1).
- A candidate that raises geomean but regresses one workload below the best-valid is `accept` but not
  `keep-as-best` unless net-better and no workload fails correctness.

---

## 7. Stopping criteria (convergence)

Stop the search and write `SEARCH_COMPLETE` (with reason) when **any** of:

- **S1 Convergence:** the last **3** accepted candidates improve best-valid geomean by **< 2%**
  cumulatively, and no untested hypothesis in §6 has a plausible >2% upside.
- **S2 Roofline:** measured behavior on the bandwidth-dominated workloads (esp. wl5) is within a small
  margin of the streaming-KV bandwidth bound (further gains would require reducing irreducible KV
  traffic, which correctness forbids).
- **S3 Budget:** cumulative feedback evaluations approach the 100 cap, or token usage approaches the
  1.0M soft limit (leave headroom before 1.2M hard).
- **S4 Diminishing design space:** all planned candidates (c001–c006+) evaluated and no evidence-backed
  new hypothesis remains.

On stop: record the best-valid candidate ID, its geomean and per-workload speedups, and the stop reason
in `SEARCH_COMPLETE`. Ensure `solution/solution.py` holds the best-valid source. Do **not** run `final`;
await explicit operator approval.

Never declare convergence while a correctness failure is unresolved — a failing best candidate is not a
valid stopping state.

---

## 8. Skill usage policy (recorded per candidate)

- **KernelWiki**: scoped to Blackwell/Hopper (tcgen05/TMEM/WGMMA/CLC/NVFP4, FA-4, DeepGEMM, 2-SM). This
  is Ampere (sm_80); those primitives don't exist here → **not invoked**; techniques don't transfer.
- **ncu-report-skill**: profiles on B200/sm_100 and CLAUDE.md forbids running a profiler directly →
  **not invoked**.
- Every evidence record sets `skills_used: []` unless a skill is genuinely consulted (which is not
  expected for this task). This is stated explicitly to satisfy the "record skill usage" requirement.

---

## 9. Risk register & mitigations (carried from draft §4/§10)

| Risk | Trigger to watch | Mitigation | Candidate |
|------|------------------|-----------|-----------|
| PV bf16 precision fails evaluator tol | any workload `correct:false` w/ small error | TF32/fp32 PV, then hi/lo `P` split | contingency cNNN |
| `[16,512]` fp32 acc register spill | perf regression w/ high num_warps | num_warps=4, output-dim tiling, SMEM staging | c004/c006 |
| Split policy mismatch to batch span | wl3 or wl5 not improving | shape-aware `num_splits`, min-tokens/split floor | c005 |
| Empty batch row present in a workload | any correctness fail on masking | unconditional 0/−inf guard (C5/C6) already required | c001 |
| Index/pointer overflow | wrong values on large-cache gather | int64 base-pointer arithmetic (C7) | c001 |
| Uninitialized split scratch | intermittent wrong combine | zero/size scratch for full `B·num_splits`; cover all splits | c002 |

---

## 10. Immediate next action
Proceed down the lineage per §2/§6/§7. See §11 progress log for current state.

---

## 11. Progress log

### c001 — evaluated, REJECT (compile failure, 0/5)
- **Config:** grid=(B,), BLOCK_N=64, num_warps=4, num_stages=2, no split-KV; fp32 online softmax; bf16
  QK+PV dots; base-2 lse = (m + ln l)·log2(e); empty-seq guard.
- **Result:** `CompilationError` — a module-global python float `_LOG2E` was referenced inside the
  `@triton.jit` kernel. This Triton version forbids accessing non-`constexpr` module globals from a
  kernel. Kernel never launched; all 5 workloads `RUNTIME_ERROR`. `valid_run=False`.
- **Diagnosis:** purely mechanical/plumbing bug, not algorithmic. Static checks C1–C10 for the intended
  math are unaffected; the fusion/structure of the baseline is intact.
- **Evaluator facts learned:** Tolerance is `atol=0.01 rtol=0.01 matched_ratio=0.99` (previously
  unstated — this loosens the PV-precision risk in draft §4.2: bf16 PV very likely passes). Device is
  `A800-SXM4-80GB` as expected. `trials=1 warmup=3 iters=100`.

### c002 — evaluated, VALID, keep-as-best (5/5, geomean 63.63x)
- **Change (single, mechanical):** inlined the `log2(e)` literal `1.4426950408889634` at the `lse`
  computation; removed the module-global `_LOG2E` reference that made c001 fail to compile. No
  algorithmic or config change vs c001's intent.
- **Result:** all 5 pass. Per-workload speedups: wl1 46.22x, wl2 43.58x, wl3 (B=1) 45.93x, wl4 93.68x,
  wl5 (B=64) 120.33x. Geomean 63.63x; arith mean 69.95x. Abs err ~1.56e-2 (near atol=1e-2 but passes
  matched_ratio=0.99) — bf16 PV precision is fine, numerical hardening not needed.
- **Diagnosis:** the batch-only grid (`grid=(B,)`) underfills the 108-SM A800 at low batch. wl2/wl1/wl3
  (16 or 1 CTAs) are the weakest at ~44-46x; wl5 (64 CTAs) is already 120x. Occupancy is the bottleneck
  at low/mid batch. Best-valid pointer = c002.

### c003 — planned next (immediate): split-KV (FlashDecoding)
- **Change:** add split-KV. Partial kernel `grid=(B, num_splits)` streams a contiguous token sub-range
  per CTA with online base-2 softmax, writing partial `(acc[16,512], m[16], l[16])` to scratch; combine
  kernel `grid=(B,)` merges via the associative log-sum-exp rule, divides by `l`, casts bf16 output and
  writes base-2 lse. `num_splits==1` still routes through combine (or a direct path) with identical math.
  Adaptive `num_splits` chosen on host from `B`, target CTA count (~2-4x 108 SMs), and max per-batch L.
- **Hypothesis (H3):** raising concurrent CTA count at low/mid batch removes SM under-utilization without
  adding KV traffic. Expect wl1/2/3/4 to improve markedly (esp. wl3 B=1), wl5 (already B=64) roughly flat.
- **Correctness focus:** C8 (split merge, consistent single sm_scale application, empty-batch handling in
  both kernels, full scratch coverage). Keep base-2 online softmax identical to c002 per split.

