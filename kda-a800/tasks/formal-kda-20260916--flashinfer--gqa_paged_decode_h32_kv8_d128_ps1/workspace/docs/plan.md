# Optimization Plan — `gqa_paged_decode_h32_kv8_d128_ps1`

Executable, sequential KDA plan derived from `docs/draft.md`. Target: **A800 / sm_80**, Triton primary,
PyTorch only for metadata/launch. One immutable candidate = one source version evaluated over all five
fixed feedback workloads via `./scripts/evaluate_candidate.sh feedback cNNN`.

Guiding fact (from draft §2): the kernel is **HBM-bandwidth-bound** (AI ≈ 4 FLOP/byte ≪ ridge ≈ 156).
Therefore the optimization objective is: **(1) load each `(token, kv_head)` K/V slice exactly once**
(GQA reuse), **(2) keep all SMs busy** (occupancy via split-K, especially for W3 and long-sequence
W1/W5), and **(3) minimize launch/overhead** for tiny W3. Tensor-core throughput is *not* the lever.

---

## 0. Ground rules (apply to every candidate)

- **Entry point:** `solution/solution.py` exposing
  `run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale) -> (output, lse)`.
- **Immutability:** once `cNNN` is evaluated, its `solution.py` bytes are frozen. Any meaningful source /
  config / launch change ⇒ new ID `c(N+1)`. Never reuse an ID for changed source, never rewrite earlier
  `candidates.jsonl` records.
- **Sequential only:** implement, self-audit, evaluate, record — one candidate at a time. Do not
  pre-write multiple candidates.
- **No fallback:** Triton must carry the computation. A failed Triton kernel is an invalid candidate;
  do **not** substitute Torch/CPU/NumPy/CUDA-extension math. PyTorch limited to allocation, stride
  extraction, dtype/shape metadata, `num_splits` heuristic computation, and grid launch.
- **Dtype discipline:** fp32 accumulation for QK / softmax / PV; store `output` bf16, `lse` f32.
- **Determinism of decision:** a candidate is *kept* only if it (a) passes correctness on all five
  workloads and (b) improves geomean vs the current best (or is a required correctness fix). Otherwise
  it is *rejected* and lineage continues from the last kept candidate.
- **Skill note:** `KernelWiki` / `ncu-report-skill` are Blackwell/Hopper-scoped; sm_80 is out of scope,
  so they are not invoked. Record `"skill_usage": "none (A800/sm_80 out of KernelWiki+ncu scope)"`.

---

## 1. Candidate lineage strategy

Tree, not a straight line: each new candidate names its `parent` (the source it was edited from) and a
single-variable hypothesis, so a regression can be abandoned without losing the best branch.

```
c001 (correctness baseline, single-pass, fp32 FMA)
  └─ c002 (split-K / flash-decoding + combine pass)      [occupancy: W3, W1, W5]
       ├─ c003 (num_splits heuristic tuning)
       ├─ c004 (BLOCK_N sweep)
       ├─ c005 (num_warps / num_stages sweep)
       ├─ c006 (gather pipelining / vectorization)
       └─ c007 (optional tl.dot QK/PV path) — only if compute shows up
  └─ (fallback branch) if c002 regresses vs c001, keep c001 and tune single-pass instead
```

Rules:
- **One variable per candidate.** Change either structure *or* a tuning knob, never several at once —
  otherwise attribution is impossible and evaluations are wasted.
- **Parent = best kept ancestor** unless deliberately exploring an alternative branch (note it in the
  hypothesis).
- **Autotune is allowed but must be deterministic and bounded** (small explicit config list). If Triton
  `@autotune` is used, the config search space is part of the immutable source; changing it ⇒ new ID.
- **`final` is operator-only** and never run without explicit approval.

---

## 2. Sequential candidate specifications

### c001 — Correctness-first single-pass baseline  *(parent: none)*
**Structure:** grid `(B, num_kv_heads=8)`. Each program handles one `(b, kv)` and computes **all G=4
query heads** `h ∈ {4·kv … 4·kv+3}` together (GQA reuse). Single pass over that sequence's tokens.

- Load `q_rows = q[b, 4kv:4kv+4, :]` → `[4,128]`, upcast fp32, pre-scale by `sm_scale · log2(e)`.
- Registers: `acc[4,128]=0`, `m[4]=-inf`, `l[4]=0` (fp32).
- Loop over token tiles `BLOCK_N` from `start=kv_indptr[b]` to `end=kv_indptr[b+1]`:
  - `page = kv_indices[start + n0 + arange(BLOCK_N)]`, mask `start+... < end`.
  - Gather `k_tile[BLOCK_N,128]`, `v_tile[BLOCK_N,128]` at `page·1024 + kv·128 + arange(128)` (int64 addr).
  - `p = q_scaled @ k_tile.T` → `[4,BLOCK_N]` (fp32 FMA, draft §5.5 Option A); masked lanes → `-inf`.
  - Online softmax: `m_new=max(m, rowmax(p))`; `alpha=exp2(m-m_new)`; `pexp=exp2(p-m_new)`;
    `l = l*alpha + rowsum(pexp)`; `acc = acc*alpha[:,None] + pexp @ v_tile`.
- Epilogue: guard `l==0` (empty/no-token) → `out=0`, `lse=-inf`; else `out = acc / l[:,None]`,
  `lse = m + log2(l)`. Store `out` bf16, `lse` f32.
- **Fixed knobs:** `BLOCK_N=64`, `num_warps=4`, `num_stages=2` (conservative, revisit later).
- **Purpose:** establish correctness + reference timing. **Must pass before any optimization.**

### c002 — Split-K / flash-decoding + combine  *(parent: c001)*
**Hypothesis:** W3 (8 base CTAs) and long-sequence W1/W5 leave SMs idle / CTAs long-running; splitting
the KV axis raises occupancy and hides latency → geomean up.

- Pass 1 grid `(B, 8, S)`: program `(b, kv, j)` processes token sub-range
  `[start + j·chunk, min(start + (j+1)·chunk, end))`, producing partial `(m_j, l_j, acc_j[4,128])`
  into buffers `partial_acc[B,8,4,S,128]` (f32), `partial_m[B,8,4,S]`, `partial_l[B,8,4,S]`.
- Pass 2 combine grid `(B, 8)` (or `(B,32)`): reduce over `S` with the log-sum-exp merge
  (draft §4): `M=max_j m_j`; `L=Σ l_j·exp2(m_j-M)`; `A=Σ acc_j·exp2(m_j-M)`; `out=A/L`; `lse=M+log2(L)`;
  skip `l_j==0` splits; empty guard as in c001.
- **`num_splits` = 1 fast-path:** if the heuristic (see c003) yields `S==1`, skip pass 2 and let pass 1
  write final output directly (avoids the extra launch on short workloads / W2).
- **Initial heuristic (frozen in c002):** `S = clamp(ceil(target_ctas / (B·8)), 1, S_max)` with
  `target_ctas ≈ 108·2`, `S_max=8`, and a floor requiring each split ≥ `BLOCK_N` tokens
  (`S = min(S, max(1, ceil(max_tokens / BLOCK_N)))` using host-side `max_tokens` from `kv_indptr` diffs).
- **Purpose:** the main performance lever. Compare geomean vs c001; keep whichever wins per workload
  is captured by the geomean.

### c003 — `num_splits` heuristic tuning  *(parent: c002)*
Adjust `target_ctas`, `S_max`, and the min-tokens-per-split floor to better balance W1/W3/W5 without
hurting W2/W4. Single-variable: only the heuristic constants change.

### c004 — `BLOCK_N` sweep  *(parent: best of c002/c003)*
Try `BLOCK_N ∈ {32, 64, 128}` (via bounded `@autotune` or a chosen constant). Larger amortizes index
loads on long sequences (W1/W5); smaller reduces tail waste on short (W2/W3). Pick geomean winner.

### c005 — `num_warps` / `num_stages` sweep  *(parent: best so far)*
Small tiles (`BLOCK_M=4`) may favor `num_warps ∈ {1,2}`; gather pipelining may favor `num_stages ∈ {2,3}`.
Bounded search.

### c006 — Gather / vectorization refinement  *(parent: best so far)*
Load the `page` id vector once per tile and reuse for K and V; ensure 128-wide contiguous inner loads
are vectorized; consider loading K and V tiles with software pipelining. Only if evidence shows
memory-stall headroom.

### c007 — Optional tensor-core `tl.dot` path  *(parent: best so far, conditional)*
Only pursue if timings suggest FMA compute (not memory) is limiting — unlikely given the roofline.
bf16 inputs, fp32 accumulate; `M=4` padded to 16. Compare against the FMA branch; keep only if faster
*and* still correct.

> Later candidate numbers are provisional; the *actual* next candidate is always chosen by the evidence
> from the previous evaluation (e.g., skip c007 if compute never dominates; add extra heuristic tweaks if
> W3 stays launch-bound).

---

## 3. Correctness checks (self-audit before every evaluation)

Static review of the frozen source against this checklist (no local CUDA/torch run is permitted):

1. **LSE = base-2:** `lse = m + log2(l)` in the pre-scaled base-2 domain (q pre-multiplied by
   `sm_scale·log2(e)`); *not* an extra `/ln2`. Empty ⇒ `-inf`.
2. **Empty / `l==0` guard:** `output=0`, `lse=-inf`, no `NaN`/`inf` leaks. Applies in both single-pass
   epilogue and split-K combine (all-empty splits).
3. **Tail mask:** `start + offset < end` ⇒ masked logits `-inf` (→`exp2`=0), masked V-contribution 0,
   running max `m` unaffected by masked lanes.
4. **GQA mapping:** `kv = h//4`; the 4 heads sharing `kv` are computed from a single K/V load.
5. **Dtype:** fp32 acc; `output` stored bf16; `lse` stored f32; partial buffers f32.
6. **Addressing:** int64 offset arithmetic for gathers; correct strides for `q`, `k/v_cache`
   (`page·1024 + kv·128`), `output`, `lse`.
7. **Split-K combine:** each `acc_j` and `l_j` rescaled by `exp2(m_j − M)`; empty splits skipped;
   result identical to single-pass within fp32/bf16 tolerance.
8. **Shapes:** `output [B,32,128]` bf16, `lse [B,32]` f32; `run` returns the tuple in this order.
9. **No fallback path** anywhere; Triton kernels do all math.

The evaluator is the sole source of truth for numerical pass/fail and timing. Coverage rationale
(draft §7): W3 → tiny/low-occupancy + empty edge; W1/W5 → long-sequence + split-K; W2/W4 → short/medium
+ tail masking. Passing all five validates the single immutable kernel.

---

## 4. Performance hypotheses (each falsifiable by one evaluation)

| ID | Hypothesis | Predicted effect | Falsified if |
|----|-----------|------------------|--------------|
| H1 | Fused GQA-reuse Triton kernel ≫ Python double-loop reference | Large geomean speedup at c001 | c001 not markedly faster than reference |
| H2 | Split-K raises occupancy for W3 + long W1/W5 | c002 geomean > c001, esp. W3/W1/W5 | c002 ≤ c001 on those workloads |
| H3 | `num_splits` heuristic tuning further balances load | c003 > c002 | no workload improves |
| H4 | Larger `BLOCK_N` helps long seqs, smaller helps short | some `BLOCK_N` beats 64 | 64 already optimal everywhere |
| H5 | Fewer warps / more stages fit the small `BLOCK_M=4` tile | c005 > parent | no config beats default |
| H6 | Memory-bound ⇒ `tl.dot` gives no benefit | c007 ≈ FMA (skip) | `tl.dot` markedly faster |

Primary metric: **geometric mean speedup over the 5 workloads**, with the hard gate that *every*
workload must pass correctness (a fail invalidates the candidate regardless of speed).

---

## 5. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
1. **Convergence:** two consecutive kept candidates improve geomean by **< 2%**, and the remaining
   ideas are low-expected-value tuning.
2. **Roofline saturation:** measured time on the bandwidth-bound workloads (W1, W5) approaches the HBM
   floor (draft §2: ~43 µs / ~104 µs), leaving little headroom.
3. **Budget:** approaching the **100-evaluation** cap or the **token soft limit (1.0M)**; hard-stop
   before 1.2M tokens. Reserve margin for recording.
4. **Idea exhaustion:** the candidate ladder (§2) is exhausted and no new evidence-backed hypothesis
   remains.

At stop, the best *valid* candidate is the `final` recommendation — but `final` is run **only** after
explicit operator approval.

---

## 6. Evidence format (append one JSON object per evaluated candidate to `candidates.jsonl`)

Append-only; never edit prior lines. Schema per record:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "single-pass GQA-grouped fp32-FMA baseline; establish correctness + timing",
  "change_summary": "grid (B,8); 4 shared qo-heads/program; online base-2 softmax; BLOCK_N=64",
  "validation": {
    "all_pass": true,
    "per_workload": {
      "W1": {"uuid": "ccdc67b6...", "pass": true,  "speedup": 0.0, "time_us": 0.0},
      "W2": {"uuid": "d2c89e1d...", "pass": true,  "speedup": 0.0, "time_us": 0.0},
      "W3": {"uuid": "aa937325...", "pass": true,  "speedup": 0.0, "time_us": 0.0},
      "W4": {"uuid": "39ca5ed6...", "pass": true,  "speedup": 0.0, "time_us": 0.0},
      "W5": {"uuid": "91552da7...", "pass": true,  "speedup": 0.0, "time_us": 0.0}
    }
  },
  "geomean_speedup": 0.0,
  "decision": "keep|reject",
  "decision_reason": "baseline correct; sets reference for split-K",
  "cumulative_evaluations": 1,
  "skill_usage": "none (A800/sm_80 out of KernelWiki+ncu scope)"
}
```

Notes:
- Fill `speedup`/`time_us`/`pass` from the evaluator's actual output; `0.0` placeholders above are
  schema illustrations only.
- `source_sha256` ties the record to immutable bytes; recompute on every evaluation.
- `decision` = **keep** iff `all_pass && geomean_speedup > best_so_far` (or a required correctness fix);
  else **reject**. `cumulative_evaluations` counts all evaluations to date (budget tracking).
- On any correctness failure, record `all_pass:false`, `decision:"reject"`, and a `failure_note`
  describing the suspected cause and the corrective next candidate.

---

## 7. Progress log

- **c001 — KEPT.** Single-pass GQA-grouped baseline, 5/5 pass, geomean **633.20x**
  (W1 416x, W2 1425x, W3 99x, W4 1345x, W5 1285x). Correctness confirmed including
  base-2 LSE, tail mask, and empty guard. Bottlenecks that bound the geomean:
  - **W3** (batch=1, 65 tokens): only 99x. Grid = (1, 8) = 8 CTAs on 108 SMs →
    severely under-occupied + launch/latency-bound.
  - **W1** (batch=16, ~1307 tok/seq): 416x, sol=227µs vs ~43µs HBM floor. Grid =
    (16, 8) = 128 CTAs, each long-running over ~1307 tokens → poor latency hiding /
    load imbalance across variable sequence lengths.
  Both point to the same fix: **split the KV axis (flash-decoding / split-K)** to
  raise CTA count and bound per-CTA work.

### Next action
Implement **c002** per §2 (split-K / flash-decoding, two-pass with combine, S==1
fast-path, host-side `num_splits` heuristic from `kv_indptr` diffs). Parent = c001.
Single structural change; evaluate once and compare geomean, watching W3/W1/W5.
