# Plan — `gqa_paged_prefill_causal_h32_kv8_d128_ps1`

Executable, sequential KDA optimization plan. Derived from `docs/draft.md`, `task/definition.json`,
and the five fixed feedback workloads. Target: NVIDIA A800 (`sm_80`). Compute is Triton‑only;
PyTorch only for metadata/launch plumbing. Validation = reasoning + the official feedback evaluator
(`./scripts/evaluate_candidate.sh feedback cNNN`) only — no local harness/profiler/CUDA/nvidia‑smi.

Recall the decisive finding from the draft: **every feedback workload has `num_kv << num_q`**, so the
op is **memset‑bound on the mostly‑zero `output`** with a negligible attention tail. The plan is built
around: (1) get an *exactly correct* fused Triton kernel first, then (2) drive toward the `output`
memset bandwidth floor by eliminating wasted work, not by chasing FLOPs.

---

## 0. Ground rules & guardrails (apply to every candidate)

- One immutable kernel version over all 5 feedback WLs = one evaluation. Budget: 100 evals; token soft
  1.0M / hard 1.2M. Stop at budget or convergence.
- Candidates are sequential and immutable: `c001, c002, …`. Never reuse an id for changed source. Any
  meaningful source/config/launch change ⇒ new id.
- Triton‑only compute. **No** Torch/CPU/NumPy/CUDA‑extension/alternate computational fallback. A failing
  Triton kernel is *invalid* — fix or abandon the branch; never substitute a Torch compute path.
- Do not read/modify the evaluator, dataset, controller, launcher, or shared config. Work only in this
  workspace. Only external knowledge source: the `KernelWiki` skill.
- Do not run raw CUDA, profilers, `nvidia-smi`, the external evaluator directly, or any alternate
  correctness harness. Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN`.
- Never run `final` without explicit operator approval.
- Append exactly one JSON record per evaluated candidate to `candidates.jsonl`; never rewrite earlier
  records.

---

## 1. Solution contract (all candidates)

`solution/solution.py` exposes:

```python
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # returns (output, lse)
```

Invariants every candidate must satisfy:
- `output`: `[total_q, 32, 128]` bf16, initialized to 0; only active query rows are written.
- `lse`: `[total_q, 32]` f32, initialized to `-inf`; only active query rows are written; base‑2.
- Constants asserted / assumed: `num_qo_heads=32`, `num_kv_heads=8`, `head_dim=128`, `page_size=1`,
  `gqa_ratio=4`.
- `page_size==1` ⇒ page id == KV token id; gather `k_cache[page_ids,0,:,:]`, `v_cache` likewise.
- Skipped work preserved exactly:
  - whole sequence with `num_q==0` or `num_kv==0` → all its rows stay `0` / `-inf`.
  - query with `max_kv_idx = min(q_idx+1+delta, num_kv) <= 0` (i.e. local `q_idx < num_q-num_kv`) →
    row stays `0` / `-inf`.
- Base‑2 LSE math (draft §4): `qk_scale = sm_scale·LOG2E`, `M'=max_j s'_j`, `p_j=exp2(s'_j-M')`,
  `L'=Σp_j`, `lse = M'+log2(L')`, `output = (Σ p_j v_j)/L'` cast to bf16.
- Accumulation in f32 to mirror the reference.
- Allowed plumbing: `torch.zeros(output)`, `torch.full(lse,-inf)`, reading `indptr` to build a launch
  schedule, dtype/stride bookkeeping. All attention math lives in the Triton kernel.

---

## 2. Derived per‑workload quantities (for correctness reasoning)

Batch = `len_indptr-1`. Per sequence: `num_q`, `num_kv`, `delta=num_kv-num_q`,
`active_start=max(0,num_q-num_kv)`, `n_active=min(num_q,num_kv)` (0 if `num_kv==0`).

| WL | seqs | total_q | Σkv indices | num_pages | regime notes |
|----|-----:|--------:|------------:|----------:|--------------|
| W1 | 38   | 7,140   | 97          | 399,096   | many seqs, kv/seq≈2–3, big output (~58 MB) |
| W2 | 3    | 123     | 12          | 135,698   | tiny; launch/overhead‑bound |
| W3 | 28   | 13,515  | 28          | 2         | **duplicate pages** (only 2 pages, 28 indices); kv/seq≈1; ~111 MB |
| W4 | 1    | 16,384  | 3           | 25,634    | single seq, 3 active of 16,384; extreme tail; ~134 MB |
| W5 | 10   | 4,950   | 25          | 296,392   | kv/seq≈2–3; ~41 MB |

Edge cases that MUST be exercised by c001 correctness (draft §4): single‑KV row (`L'=1`,
`lse=M'`); `max_kv_idx<=0` skip; `num_kv==0` skip; duplicate page ids (W3); last query = full `num_kv`
window; shrinking causal prefix for the few active queries.

---

## 3. Candidate lineage (sequential, executable)

Each candidate lists: **hypothesis**, **change from parent**, **kernel/launch design**, **correctness
checks to reason through before submit**, **expected outcome**, **decision rule**. Later ids are
provisional and may be re‑prioritized based on evidence, but ids are never reused for changed source.

### c001 — Correctness baseline (dense grid + in‑kernel skip)
- **Parent:** none.
- **Hypothesis:** a straightforward fused Triton kernel with exact math beats the Python‑loop reference
  by a large margin and, more importantly, is *provably correct*. Prioritize simplicity of the logic
  that can be wrong (scheduling) over peak perf.
- **Design:**
  - `output=torch.zeros`, `lse=torch.full(-inf)` (memset supplies the zero/`-inf` bulk).
  - Grid `(batch, max_q_tiles, num_kv_heads)` where `max_q_tiles = ceil(max_seq_q / BLOCK_Q)` computed
    host‑side from `qo_indptr`. Each program:
    - loads `q_start,q_end` (and `kv_start,kv_end`) for its `batch`; computes `num_q,num_kv,delta,
      active_start`.
    - early‑exit if `num_kv==0`, or if the tile's query range lies entirely below `active_start`, or
      entirely `>= num_q`.
    - for its `BLOCK_Q` queries × 4 packed qo heads (one kv head): gather K/V for pages
      `kv_indices[kv_start:kv_end]` (≤ full `num_kv`, tiny ⇒ single `BLOCK_KV` covering `num_kv`),
      compute QKᵀ in f32 (`tl.dot(..., input_precision="ieee")` or explicit f32 reduce), apply causal
      prefix mask `j <= q_idx+delta` and `j < num_kv`, exp2 softmax, PV, write `output` (bf16) + `lse`.
  - `BLOCK_D=128`, `BLOCK_Q=16` (small; active tails short), `BLOCK_KV` ≥ max `num_kv` in feedback
    (≤ ~97 → e.g. 128), f32 accumulator, `num_warps=4`, `num_stages=2`.
  - K/V gather masked by `j < num_kv`; duplicate page ids handled by plain gather.
- **Correctness checks (reason before submit):** all of §2 edge cases; `-inf`/zero preserved for
  inactive rows; base‑2 LSE identity on single‑KV; per‑row shrinking prefix (not flat window);
  int32 indptr arithmetic; no OOB gather; `page_size=1` squeeze handled via stride indexing.
- **Expected:** correct on all 5 WLs; speedup ≫1 (reference is a Python quadruple loop). Many dead
  blocks (W4) but each is a cheap early‑exit.
- **Decision:** if all 5 pass correctness → adopt as baseline, proceed to c002. If any fail → next
  candidate is a *correctness fix* (not a perf change), diagnosing from the evaluator's error signal.

### c002 — Compact tail‑only schedule (Option S)
- **Parent:** c001 (assuming pass).
- **Hypothesis:** eliminating dead blocks (W4: ~128×8 mostly no‑op tiles) removes launch/scheduling
  overhead and moves large‑WL time toward the `output` memset floor.
- **Change:** host‑side build a compact tile list covering only `[active_start, num_q)` per sequence
  (single `.cpu()` sync of `qo_indptr`/`kv_indptr`, which the reference also syncs). Grid =
  `(num_active_tiles, num_kv_heads)`; each tile carries `(batch_id, q_tile_start)`. Kernel body
  unchanged from c001 except tile→sequence mapping comes from the schedule arrays.
- **Correctness checks:** schedule reproduces exactly the active row set of c001; no sequence straddling;
  `num_kv==0` sequences contribute zero tiles; determinism.
- **Expected:** neutral→better on W1/W3/W5 (fewer launches), clearly better on W4; W2 dominated by fixed
  overhead. Correctness identical to c001.
- **Decision:** adopt if geomean ≥ c001 and all pass; else keep c001 and branch.

### c003 — Block‑size / warp / stage tuning
- **Parent:** best of {c001,c002}.
- **Hypothesis:** for a memset‑bound problem the kernel's own cost is small, but `BLOCK_Q`, `BLOCK_KV`,
  `num_warps`, `num_stages` still affect occupancy/overlap on the active tail and the write pattern.
- **Change:** sweep a *small* fixed set (e.g. `BLOCK_Q∈{8,16,32}`, `num_warps∈{2,4}`) — but **each
  distinct config is its own candidate id** (c003, c004, …); do not use runtime autotune that changes
  source semantics between evals without a new id. Pick one config per candidate; compare.
- **Correctness checks:** unchanged math; verify masks still valid for new `BLOCK_KV`.
- **Expected:** marginal gains; identifies the best static config.
- **Decision:** keep the winning config as the new baseline for subsequent candidates.

### c004+ — Output‑write / memset optimization
- **Parent:** best so far.
- **Hypotheses to test (one per candidate):**
  1. Is the `torch.zeros` allocation the dominant cost? If so, confirm it is unavoidable (must produce
     zeros for inactive rows) and ensure we do not additionally pass over the zero region in‑kernel.
  2. Fallback‑if‑needed: if the evaluator rejects `torch.zeros` as non‑Triton, replace with a tiny
     Triton memset kernel or have the main kernel also write zeros/`-inf` to inactive rows (same
     bandwidth). This is a correctness/compliance branch, not a perf win.
  3. Ensure `lse` and `output` writes are coalesced (contiguous `head_dim` innermost).
- **Correctness checks:** zero/`-inf` bulk still exactly produced.
- **Decision:** adopt only if geomean improves and all pass.

### c005+ — Compute‑path refinement (only if warranted)
- **Parent:** best so far.
- **Hypothesis:** the attention compute is negligible in feedback, so `ieee` f32 dot is fine. Only if a
  candidate fails tolerance would we revisit; only if the final 38‑set is expected to have larger KV
  would compute throughput matter. Keep an online‑softmax `BLOCK_KV`‑streaming path so correctness holds
  if KV grows.
- **Change:** e.g. `tf32x3` or tensor‑core dot *only* if a tolerance‑safe speedup is plausible on a
  larger‑KV case; otherwise skip. Each variant = new id.
- **Decision:** adopt only with strict correctness + geomean gain.

### Contingency branches
- **Correctness failure on any WL** → immediate next candidate is a targeted fix (diagnose from
  evaluator output: which WL, output vs lse, magnitude). Common suspects: causal prefix off‑by‑one,
  `active_start` boundary, `num_kv==0`/`max_kv_idx<=0` handling, base‑2 vs base‑e LSE, bf16 accumulation
  drift (switch to `ieee`).
- **Compilation/launch failure** → fix Triton API usage; never fall back to Torch compute.

---

## 4. Performance hypotheses (ranked, falsifiable)

1. **H1 (primary):** Runtime of large WLs (W3/W4) is dominated by the `output` bf16 write (~55–70 µs
   floor); the attention tail is <1 µs. ⇒ Once correct + no dead blocks, we are near the floor.
   *Falsify:* if c002 (compact schedule) gives large gains beyond expected launch savings, dead‑block
   overhead was larger than modeled.
2. **H2:** Reference (Python quadruple loop + `q.to(f32)` + `torch.zeros`) is milliseconds–seconds ⇒
   speedups ≫10× on W3/W4/W5, large on W1, smaller on W2 (tiny, overhead‑bound).
3. **H3:** Never reading full `k_cache`/`v_cache` (up to ~817 MB) is essential; gathering only ≤97 pages
   keeps KV traffic negligible.
4. **H4:** Block/warp tuning yields marginal (<10%) improvement because compute isn't the bottleneck;
   convergence expected within a handful of candidates.
5. **H5:** W2 speedup is bounded by fixed kernel‑launch + host‑sync overhead; minimizing host syncs
   (single `.cpu()`) matters most there.

---

## 5. Correctness checklist (run mentally before every submission)

- [ ] `output` dtype bf16 `[total_q,32,128]`, `lse` f32 `[total_q,32]`; init `0` / `-inf`.
- [ ] Inactive rows/sequences (`num_kv==0`, `max_kv_idx<=0`) left untouched.
- [ ] `active_start=max(0,num_q-num_kv)`, `max_kv_idx=min(q_idx+1+delta,num_kv)` reproduced exactly.
- [ ] Causal prefix per row (shrinking triangle), `j<num_kv` mask.
- [ ] Base‑2 LSE via `M'+log2(L')` with `qk_scale=sm_scale·LOG2E`; single‑KV ⇒ `lse=M'`.
- [ ] f32 accumulation (`ieee` dot or explicit f32); no premature bf16 rounding except final store.
- [ ] Paged gather `k_cache[page_ids,0,kv_head,:]`; page id==token id; duplicates fine (W3).
- [ ] GQA mapping `kv_head=h//4`; 4 qo heads share K/V.
- [ ] No OOB loads (mask `j<num_kv`, row `<q_end`); int32 arithmetic safe.
- [ ] No full `k_cache`/`v_cache` read; no global `q.to(f32)`.
- [ ] Only `torch` plumbing (alloc, indptr read, strides); all math in Triton.

---

## 6. Stopping criteria

Stop and write `SEARCH_COMPLETE` when ANY of:
- Geomean speedup improvement over the previous best is <3% across two consecutive adopted candidates
  (convergence near the memset floor), AND all 5 WLs pass correctness.
- The measured large‑WL times are within ~1.2× of the modeled `output`‑write floor (H1 satisfied) with
  no further structural lever available.
- Evaluation budget (100) or token budget (soft 1.0M / hard 1.2M) is reached.
- No correctness‑passing candidate can be produced after a bounded number of fix attempts on a branch
  (record the failure and stop that branch).

`SEARCH_COMPLETE` must state: best candidate id, its geomean, why further search is unwarranted,
remaining budget. Never trigger `final` without operator approval.

---

## 7. Evidence format (`candidates.jsonl`, one JSON object per evaluated candidate)

Append‑only; never rewrite earlier records. Each record:

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "Exactly-correct fused Triton baseline; dense grid + in-kernel skip.",
  "change_from_parent": "initial implementation",
  "design_notes": "torch.zeros/full memset; grid (batch,max_q_tiles,kv_heads); f32 ieee dot; exp2 base-2 LSE; write only active tail",
  "validation": {
    "correctness_reasoning": "checklist §5 items verified; edge cases W3 dup-pages, W4 tail, num_kv==0",
    "compiled": true
  },
  "per_workload": [
    {"wl": "W1", "uuid": "d1b0b14d...", "passed": true,  "speedup": null, "ref_time_ms": null, "cand_time_ms": null, "notes": ""},
    {"wl": "W2", "uuid": "7e6843d3...", "passed": true,  "speedup": null},
    {"wl": "W3", "uuid": "75ab4c21...", "passed": true,  "speedup": null},
    {"wl": "W4", "uuid": "a94c44ab...", "passed": true,  "speedup": null},
    {"wl": "W5", "uuid": "a1638bf3...", "passed": true,  "speedup": null}
  ],
  "geomean_speedup": null,
  "all_passed": true,
  "decision": "adopt|reject|fix-next",
  "decision_reason": "...",
  "cumulative_evaluations": 1,
  "skill_usage": ["KernelWiki: <if used, what for>"],
  "timestamp": "..."
}
```

Rules:
- Fill `speedup`/`time` fields from the evaluator's returned metrics; use `null` only if the evaluator
  does not report a field.
- `geomean_speedup` = geometric mean over the 5 WLs (only meaningful if `all_passed`).
- `decision` ∈ {`adopt` (new best), `reject` (worse/equal), `fix-next` (correctness failed, next
  candidate fixes it)}.
- `cumulative_evaluations` increments by 1 per evaluated candidate.
- Record `KernelWiki` skill usage whenever consulted (topic + how it informed the change).

---

## 8. Immediate next actions (next turn, not this one)

1. Implement `c001` per §3 (correctness baseline). One source version.
2. Reason through the §5 checklist and §2 edge cases before evaluating.
3. Run `./scripts/evaluate_candidate.sh feedback c001`; append the §7 record.
4. Branch per the c001 decision rule (fix vs c002 compact schedule).

Guiding principle: **correctness first (validation is reasoning + the official evaluator only), then
drive toward the `output` memset bandwidth floor by removing wasted work — not by chasing FLOPs.**

---

## 9. Progress log

- **c001 (adopted, eval #1).** Correctness baseline as designed in §3. All 5 feedback WLs PASSED
  (atol=0.01/rtol=0.01/0.99). Geomean **92.36x** (W1 181.93x, W2 114.14x, W3 69.12x, W4 31.97x,
  W5 146.46x). Confirms H2 (huge speedups vs Python-loop reference) and the memset-bound picture:
  the two large-`output` WLs are the *lowest* speedups — W4 (~134 MB out, single seq, 3 active) at
  32x and W3 (~111 MB, dup pages) at 69x — i.e. their absolute times (0.63 ms / 2.32 ms) are near
  the output-write floor, while small-`output` WLs score high. Candidate times: W1 3.09 ms,
  W2 0.62 ms, W3 2.32 ms, W4 0.63 ms, W5 1.00 ms.
  - Observation refining next steps: W1 (only 58 MB out) is 3.09 ms — *slower* than W3's 2.32 ms
    despite less output. W1 has 38 seqs × 97 kv indices and max_q_tiles driven by its longest seq;
    the dense `(batch, max_q_tiles, kv_heads)` grid launches many dead blocks (every seq padded to
    the global max tile count). So c002 (compact tail-only schedule, Option S) is well-motivated:
    remove dead blocks, especially for W1/W3/W4. Expected biggest wins on W1 and W4.
  - **Next candidate: c002** — host-built compact tile list over only `[active_start, num_q)` per
    sequence; grid `(num_active_tiles, num_kv_heads)`. Same kernel math, tile→sequence mapping from
    schedule arrays; single `.cpu()` sync of indptr. Adopt if geomean ≥ 92.36x and all pass.
