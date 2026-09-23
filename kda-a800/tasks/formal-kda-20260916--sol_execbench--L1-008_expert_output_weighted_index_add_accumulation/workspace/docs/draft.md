# Draft — L1/008 Expert Output Weighted Index-Add Accumulation

Task ID: `sol_execbench :: L1/008_expert_output_weighted_index_add_accumulation`
Target GPU: NVIDIA A800 (`sm_80`, Ampere), HBM2e ~2.0 TB/s, L2 = 40 MB.
Primary implementation must be **Triton**; PyTorch only for metadata / launch plumbing.
No Torch/CPU/NumPy/CUDA-extension computational fallback.

---

## 1. What the operation actually computes

Reference (`task/definition.json`):

```python
@torch.no_grad()
def run(final_hidden_states, expert_outputs, token_indices):
    output = final_hidden_states.clone()
    output.index_add_(dim=0, index=token_indices, source=expert_outputs)
    return output
```

Semantically, for `i in [0, num_selected_tokens)`:

```
output[token_indices[i], :] += expert_outputs[i, :]
```

with the base value `final_hidden_states` copied first, and **atomic add** semantics
because the destination indices contain duplicates (top-k routing → many source rows
map to the same token row).

### Tensors / dtypes / shapes

| name                 | shape                          | dtype    | role                          |
|----------------------|--------------------------------|----------|-------------------------------|
| `final_hidden_states`| `[M, H]`                       | bf16     | base accumulation buffer      |
| `expert_outputs`     | `[N, H]`                       | bf16     | per-selected-token weighted output (already weighted) |
| `token_indices`      | `[N]`                          | int64    | destination row for each source row |
| `output` (return)    | `[M, H]`                       | bf16     | `clone(base)` + scatter-add   |

Constants: `H = hidden_size = 3072`, `num_experts_per_tok = 8`.
Relations: `M = batch_seq_len = batch_size * seq_len`, `N = num_selected_tokens = 8 * M`.

Important: the routing weight multiply has **already been folded into** `expert_outputs`
(the description says "weighted outputs"; `run` does a plain `index_add_`). So this kernel
is a pure **scatter-add**, not a fused multiply-scatter. No extra weight tensor is provided.

### Feedback workload sizes

| # | uuid | batch | seq | M (rows) | N=8M | src read (bf16) | out (bf16) | fp32 scratch | atol |
|---|------|-------|-----|----------|------|-----------------|------------|--------------|------|
| 1 | ada14c03 | 2  | 256  | 512  | 4096  | 25.2 MB  | 3.1 MB  | 6.3 MB   | 0.087 |
| 2 | 8c5c94b5 | 32 | 256  | 8192 | 65536 | 402.7 MB | 50.3 MB | 100.7 MB | 0.100 |
| 3 | b61ee9a7 | 16 | 256  | 4096 | 32768 | 201.3 MB | 25.2 MB | 50.3 MB  | 0.094 |
| 4 | 2c1d8396 | 2  | 1024 | 2048 | 16384 | 100.7 MB | 12.6 MB | 25.2 MB  | 0.093 |
| 5 | 8b677344 | 4  | 512  | 2048 | 16384 | 100.7 MB | 12.6 MB | 25.2 MB  | 0.093 |

`rtol = 0.05` for all. Workload 2 dominates cost; workload 1 is launch-overhead bound.

---

## 2. Roofline / cost model

Zero real FLOPs (one add per element). This is **purely memory-bandwidth + atomic-contention
bound**. The unavoidable HBM floor per invocation:

- read `expert_outputs`: `N*H*2 = 8*M*H*2` bytes (dominant — 8× the base),
- read base once + write output once: `~2 * M*H*2` bytes.

So `src` read is ~8/10 of the essential traffic and is identical for **any** correct
implementation (torch or ours). The reference also streams `src` exactly once. Therefore the
only place we can win is:

1. the **atomic accumulation cost** (torch `index_add_` on bf16 uses a bit-level CAS loop for
   `gpuAtomicAdd(bfloat16)` on `sm_80` — bf16 has no native global atomic add before Hopper),
   and contention amplifies that; and
2. **L2 residency** of the accumulator (`out`/scratch is 3–100 MB vs 40 MB L2), which decides
   whether the RMW traffic stays on-chip or spills to HBM.

Estimated lower bound for the dominant W2: `~453 MB / 2 TB/s ≈ 0.23 ms` if atomics are cheap.
The opportunity vs torch: replace slow bf16-CAS atomics with **native fp32 global atomics**
(single-instruction `red.global.add.f32` on Ampere) and better block/vectorization.

---

## 3. Numerical analysis & risk

### 3.1 How the reference accumulates
`index_add_` on CUDA uses non-deterministic **atomic** ordering. For bf16 it is a CAS loop that
reads bf16 → widens → adds → **rounds back to bf16 after every single add**. So the reference is:

```
acc = round_bf16( ... round_bf16(round_bf16(base + s_{p0}) + s_{p1}) ... )
```

for a *random permutation* `p` of the ~8 source rows hitting a given output row. The reference
is therefore itself only an approximation of the true sum, it is **run-to-run non-deterministic
in the low bits**, and its error grows with the number of collisions and the magnitude.

### 3.2 Our choice: accumulate in fp32, round once
If we accumulate all contributions in **fp32** and round to bf16 exactly once at the end, our
result is essentially the *true* sum rounded to bf16 — strictly **more accurate** than the
reference. The deviation `|ours − ref|` is bounded by the reference's own bf16 accumulation
error, i.e. we inherit the reference's error budget, not add to it.

Magnitude sanity check: base ~ N(0,1); each output row gets ~8 additions of N(0,1) values, so
typical row magnitude ~ `sqrt(1 + 8) ≈ 3`, occasional |value| up to ~6–8. bf16 ulp at |x|≈4 is
`2^(2-8) = 0.0156`; ~8 sequential roundings → worst-case reference error ~0.05–0.12 in absolute
terms, but where that occurs `|ref|` is large so `rtol*|ref| = 0.05*4 = 0.2` dominates. Under the
standard `allclose` rule `|a-b| <= atol + rtol*|b|`, the given `atol≈0.09..0.10` + `rtol=0.05`
comfortably covers the difference between a once-rounded fp32 sum and a repeatedly-rounded bf16
sum. **Risk: LOW**, and fp32 accumulation is the safe direction (we err toward more accuracy).

### 3.3 Tolerance-semantics caveat
I will assume the correctness check is `torch.allclose(actual, ref, atol=max_atol, rtol=max_rtol)`
= `|actual-ref| <= atol + rtol*|ref|`. If instead the harness checks *max absolute* AND *max
relative* error **separately** (stricter, no additive coupling), the near-zero rows are the risk:
where `|ref|≈0`, `rtol*|ref|≈0`, so only `atol` protects us. Our fp32 result at those rows is
still within ~a few bf16 ulps of the reference (both are small sums), well under `atol≈0.09`.
Either interpretation looks safe; I will treat the near-zero, high-collision rows as the
correctness stress case during validation reasoning.

### 3.4 Correctness edge cases to guard
- **Rows with zero contributions** must equal `base` exactly (init must copy base for *all* M
  rows, not only touched ones). With 8× oversampling most rows are hit, but some are not.
- **Fresh output tensor**: reference `clone`s; we must **not** mutate `final_hidden_states`
  in place (the harness may reuse inputs across timing repeats). Allocate a new output.
- **int64 indices**: destination offset `idx*H` can reach `8191*3072 ≈ 25.2M` — fits int32 for
  these workloads, but I will use int64 offset arithmetic in pointers to be safe/general.
- **No out-of-range indices** (generator uses `randint(0, M)`), so no masking of indices needed;
  only the hidden-dim tail needs masking if `H` is not a multiple of the block.
- `H = 3072 = 3*1024` — clean multiples exist (512, 768, 1024, 1536, 3072), so hidden-tail
  masking may be avoidable with the right block, but I will keep a mask for safety.

---

## 4. Triton design space

Common floor: `expert_outputs` must be streamed once (~8/10 of traffic) in every variant. The
variants differ only in the accumulator representation, atomic type, and number of passes.

### Option A — Direct bf16 atomic scatter (mimics reference)
One program per `(source row, hidden block)`: load `idx`, load `src` block, `tl.atomic_add`
into a bf16 `output` pre-filled with `base`.
- Pros: minimal passes (clone base + scatter); output is only 3–50 MB → best L2 residency;
  numerically identical style to reference.
- Cons: bf16 global atomics on `sm_80` compile to a **CAS loop** (Triton emits a `atom.cas`
  retry loop for bf16) — slow, and contention (avg 8 hits/row) amplifies retries. Likely no
  better than torch, which does the same thing.
- Verdict: baseline/reference-parity candidate; probably **not** the winner but cheap to test.

### Option B — fp32-scratch native atomic scatter  ← primary candidate
1. Allocate an fp32 scratch `buf[M, H]`; **init** `buf = base` (Triton cast-copy kernel).
2. **Scatter**: one program per `(source row, hidden block)` → `tl.atomic_add` (native
   `red.global.add.f32`) `src.to(fp32)` into `buf[idx]`.
3. **Finalize**: Triton kernel casts `buf` → bf16 `output` (fresh tensor).
- Pros: **native fp32 atomics** (single instruction, fast even under contention); fp32
  accumulation → most accurate → safest vs tolerance; entirely Triton-native (rule-clean).
- Cons: extra passes (init read 50 + write 100; finalize read 100 + write 50 for W2 ≈ +300 MB)
  and the fp32 scratch (50–100 MB) exceeds L2 for W2/W3 → atomic RMW spills to HBM.
- Micro-variants to try:
  - **B0**: init `buf=base`, scatter, cast. (base read once in init.)
  - **B1**: init `buf=0` (memset, write-only), scatter, finalize `out = round(base + buf)`
    (fused: read base + read buf, write out). Slightly cheaper init.
- Verdict: **strongest, safest first candidate (c001).** Expected win comes from native fp32
  atomics replacing torch's bf16 CAS loop.

### Option C — fp32 atomics, fused init via "seed" trick
Avoid a separate init pass by seeding: run scatter into a zeroed buf, then in finalize add base.
This is exactly B1. Keeps only 2 kernels beyond memset. Worth measuring against B0.

### Option D — Sort / argsort → segmented gather (no atomics)
Precompute `perm = argsort(token_indices)` and per-row offsets, then a Triton kernel gathers each
row's contiguous run of contributions and reduces in fp32 (no atomics, deterministic).
- Pros: no atomic contention; perfectly deterministic; single accurate reduction.
- Cons: **`argsort`/`sort`/`bincount`/`cumsum` are non-trivial Torch computation** — likely a
  rule violation ("PyTorch only for metadata and launch plumbing"; "no alternate implementation
  fallback"). Also adds a sort (~N log N) whose overhead can dominate small workloads.
- Verdict: **avoid** for rule-compliance reasons; keep only as a last-resort idea and only if a
  Triton-native sort/segmentation is feasible (it is not, cheaply). **Not pursued.**

### Option E — Load-balanced / larger-tile scatter (tuning layer on B)
On top of Option B: tune the hidden block (`H=3072` → BLOCK ∈ {512,768,1024,1536,3072}),
`num_warps`, `num_stages`; consider **one program per full source row** (BLOCK=3072, single
`idx` load per row, 6 KB streamed load) vs multiple hidden blocks per row (more parallelism,
more redundant `idx` loads). Also consider vectorized loads and `.to(tl.float32)` on load.
- Verdict: the real performance search happens here once B is correct.

### Chosen ordering for candidates (to be finalized in plan.md)
1. **c001 = Option B0** (fp32 scratch, native atomics, 3 Triton kernels) — correctness + baseline win.
2. **c002 = Option B1** (memset init + fused base-add finalize) — fewer bytes.
3. **c003+** = block/warps/stages autotune, full-row vs tiled, dtype-on-load tweaks (Option E).
4. Optionally **Option A** (bf16 direct atomics) as a comparison point for L2-residency effect on
   small workloads (W1) where the accumulator fits in cache.

Note: `tcgen05`/TMA/warp-specialization/CLC etc. are Hopper/Blackwell features — **not applicable
to `sm_80`**; the KernelWiki skill is out of scope for this task and will not be used.

---

## 5. Key implementation details / pitfalls

- **Grid**: scatter grid `(N, cdiv(H, BLOCK))` (or `(N,)` for full-row). N up to 65536, so up to
  ~393k programs — fine.
- **Index load**: load `token_indices[pid_n]` as int64 scalar; compute row base offset
  `idx * H` in int64; add hidden offsets; mask hidden tail (`offs_h < H`).
- **Atomic**: `tl.atomic_add(buf_ptr + row*H + offs_h, val_f32, mask=...)`. Confirm Triton
  version emits native `red/atom.global.add.f32` on `sm_80` (expected). For Option A confirm
  whether bf16 `atomic_add` is even supported / how it lowers.
- **No in-place mutation** of inputs; allocate `output` and `buf` fresh via `torch.empty`.
- **Init correctness**: every one of the M rows must be initialized (cover full `[M,H]` in the
  init/finalize grids, mask tails).
- **Determinism**: our fp32 path is deterministic; reference is not — expect tiny, tolerance-safe
  disagreement, and do not chase bit-exactness.
- **Contention hotspots**: random duplicates → some rows hit >8×; fp32 native atomics serialize
  cheaply. Acceptable. Avoid designs whose per-element retry cost blows up (bf16 CAS).
- **Small-workload (W1) regime**: everything L2-resident; kernel-launch overhead dominates.
  Minimize number of kernel launches (favor B1's 3 launches; watch that extra passes don't
  regress W1). This is a per-workload tension to watch in evidence.

---

## 6. Validation strategy

1. **Correctness (per candidate, all 5 workloads)** via `./scripts/evaluate_candidate.sh feedback cNNN`
   only — never a private harness, never direct CUDA/profiler/nvidia-smi. The evaluator applies
   the per-workload `max_atol`/`max_rtol`.
2. **Numerical pre-reasoning** before each eval: confirm (a) all M rows initialized to base,
   (b) fresh output, (c) fp32 accumulation → deviation ≤ reference's own bf16 error ≤ tolerance,
   (d) hidden-tail masking correct. Stress-focus on near-zero and high-collision rows (§3.3).
3. **Performance**: geomean speedup across the 5 workloads is the ranking metric; every workload
   must pass correctness. Record per-workload speedup, watch the small-workload (W1) launch-overhead
   regime and the large-workload (W2) L2-spill regime separately, since they favor different
   variants (bf16-cached vs fp32-native-atomic).
4. **A/B discipline**: change exactly one meaningful thing per candidate ID; never reuse an ID for
   changed source; append one JSON record per eval to `candidates.jsonl` (parent, source hash,
   hypothesis, per-workload result, geomean, decision, cumulative eval count, skill usage).
5. **Convergence / stop**: stop at budget (100 evals / 1.0M token soft / 1.2M hard) or when the
   geomean plateaus; then write `SEARCH_COMPLETE`. `final` only with explicit operator approval.

### Open questions to resolve empirically (via candidate evals)
- Does native fp32-scratch (B) actually beat torch given +300 MB of extra passes? (Expected yes,
  from atomic-cost savings — confirm on W2.)
- B0 vs B1 (init-copy vs memset+fused-add): which wins net bytes/launches?
- Does bf16-direct atomics (A) win on small/L2-resident W1 despite CAS loops?
- Best hidden BLOCK / full-row vs tiled / num_warps / num_stages for the scatter kernel.
