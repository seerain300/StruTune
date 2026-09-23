# Draft — L2/043 Mamba-2 Chunk Scan with Segment Sum

Task: optimize the official SOL-ExecBench task `L2/043_mamba_chunk_scan_with_segsum`
on NVIDIA **A800 (sm_80)**. Primary implementation must be **Triton**; PyTorch only
for metadata/launch plumbing. No Torch/CPU/NumPy/CUDA-extension computational fallback.
Ranking: geometric-mean speedup vs. the reference, subject to every selected workload
passing correctness.

---

## 1. Operation summary

This is the standard **Mamba-2 SSD (State-Space Duality) chunked parallel scan**. The
sequence is split into chunks of `chunk_size = 256`; within each chunk an
attention-like "diagonal" block is computed, and across chunks a state recurrence with
exponential decay is run. Output = intra-chunk (diagonal) + inter-chunk (off-diagonal)
+ `D` skip connection.

### Fixed dimensions (from `task/definition.json`)
| symbol | name | value |
|--------|------|-------|
| H | num_heads | 16 (const) |
| P | head_dim | 64 (const) |
| N | state_size | 256 (const) |
| G_grp | n_groups | 1 (const) |
| Q | chunk_size | 256 (const) |
| B | batch_size | var |
| L | seq_len | var |
| — | hidden_out | H·P = 1024 |

Note `N == Q == 256`, `P == 64`. Everything is bf16 on input/output; the reference does
all math in **float32** and casts back to bf16 at the very end.

### Inputs / outputs
- `hidden_states` (X): `[B, L, H, P]` bf16 — the "values".
- `A`: `[B, H, L]` bf16 — per-head log-decay (note the H,L order, transposed vs X).
- `B` param: `[B, L, G_grp=1, N]` bf16 — shared across heads.
- `C` param: `[B, L, G_grp=1, N]` bf16 — shared across heads.
- `D`: `[H]` bf16 — skip/residual scale.
- `initial_states`: `[B, H, P, N]` bf16 — carry-in SSM state.
- Output `output`: `[B, L, H·P]` bf16.
- Output `final_state`: `[B, H, P, N]` bf16.

### Padding / chunking
`pad = (Q - L % Q) % Q`; seq is zero-padded to `L_pad = L + pad`, `NC = L_pad / Q`
chunks. Padding value 0 flows through cleanly: A-pad=0 keeps cumsum flat, X-pad=0/B-pad=0
contribute nothing, and the tail is cropped back to `L` at the end. So partial chunks need
only **row masking on loads (other=0)**, no special-case math.

### Per-workload metadata (derived)
| B | L | pad | L_pad | NC | last-chunk valid rows | tiles (B·NC·H) |
|---|-----|-----|-------|----|----------------------|----------------|
| 2 | 293 | 219 | 512 | 2 | 37 | 64 |
| 4 | 1024| 0   | 1024 | 4 | 256 | 256 |
| 4 | 256 | 0   | 256  | 1 | 256 | 64 |
| 1 | 1024| 0   | 1024 | 4 | 256 | 64 |
| 4 | 541 | 227 | 768  | 3 | 29 | 192 |

Parallel work (independent `(b, chunk, head)` tiles) ranges 64–256. A800 has 108 SMs, so
for the 64-tile cases occupancy is a first-order concern — the kernel decomposition should
expose extra parallelism (e.g. split over head_dim / state, or map heads×batch×chunk to a
large grid) rather than one heavy block per tile.

---

## 2. Algebraic structure (what we actually must compute)

Let, per `(b, chunk, head)`:
- `a[t] = cumsum_t A[t]` over the chunk (t = 0..Q-1), fp32. `a_last = a[Q-1]`.

The reference's `segment_sum` is exactly the pairwise cumulative difference:
`segsum(A)[i,j] = Σ_{k=j+1..i} A[k] = a[i] - a[j]` for `i ≥ j`, else `-inf`.
Therefore the intra-chunk decay matrix is
```
L[i,j] = exp(a[i] - a[j])   for i ≥ j,   0 otherwise   (lower-triangular)
```
This is the crucial simplification: the reference materializes `L` as
`[B,H,NC,Q,Q]` and `G,M` as `[B,NC,Q,Q,H]` (multi-GB tensors — see §5), but we never
need to materialize the `[Q,Q]` (let alone `[Q,Q,H]`) matrices to global memory.

**(1) Intra-chunk / diagonal output**
- `G[i,j] = C_i · B_j = Σ_s C[i,s] B[j,s]` — **head-independent** because n_groups=1
  (B, C are broadcast from the single group across all 16 heads). So `G` is computed
  **once per `(b, chunk)`** and reused by all H heads → up to 16× less CB^T matmul work.
- `M[i,j] = G[i,j] · L[i,j]` (lower-tri; head-dependent via L).
- `Y_diag[i,:] = Σ_{j≤i} M[i,j] · X[j,:]`  → `[Q, P]`.

**(2) Per-chunk state (right factor)**
- `decay_state[t] = exp(a_last - a[t])`.
- `states[d, s] = Σ_t decay_state[t] · X[t,d] · B[t,s]`  → `[P, N]` per `(b,chunk,head)`.

**(3) Inter-chunk recurrence (middle factor)**
- `A_chunk_end[c] = a_last(c)`, padded with a leading 0 → length NC+1.
- `decay_chunk[i,j] = exp(segsum(A_chunk_end_padded))[i,j]`  → `[NC+1, NC+1]` per `(b,head)`.
- `new_states[i] = Σ_j decay_chunk[i,j] · states_with_init[j]`, where index 0 is
  `initial_states`. `states_out = new_states[:-1]`, `final_state = new_states[-1]`.
  With NC ≤ 4 this is tiny — effectively a short prefix scan over chunks.

**(4) Off-diagonal / state→output (left factor)**
- `state_decay_out[t] = exp(a[t])`.
- `Y_off[t,:] = state_decay_out[t] · (C_t · states_out) = exp(a[t]) · Σ_s C[t,s] states_out[s,:]`
  → `[Q, P]`.

**(5) Combine**
- `y = Y_diag + Y_off`, reshape to `[B, L_pad, H, P]`, add `D_residual = D[h]·X`,
  crop to `L`, reshape to `[B, L, H·P]`, cast bf16.
- `final_state` cast bf16.

### Exploitable properties (optimization levers)
1. **Head-shared B, C, G** (n_groups=1): compute `G = C Bᵀ`, and the `C·states` /
   `X·B` contractions, only once per group where possible. G is genuinely head-independent.
2. **No `[Q,Q]` materialization**: fuse `L`, `M`, and `Y_diag = MX` inside one kernel
   using `tl.dot` on `[Q,Q]·[Q,P]` with the mask/decay applied in registers/SRAM.
3. **Tiny inter-chunk stage**: NC ≤ 4 ⇒ the recurrence is cheap; do it with a small
   kernel (or even a sequential per-chunk scan) rather than a general bmm.
4. **N=Q=256, P=64**: convenient tile shapes; `[256,256]`, `[256,64]` fit A800 SRAM
   with fp32 accumulation.

---

## 3. Constraints

- **Isolation**: work only in this workspace; only `KernelWiki` / `ncu-report-skill`
  skills as external knowledge. No web, subagents, MCP, other agents.
- **Triton-only compute**; PyTorch allowed only for shapes/allocation/launch. A failed
  Triton kernel is invalid — must NOT be replaced by a Torch/CPU/NumPy path.
- **Immutable candidates**: `solution/solution.py` versions `c001`, `c002`, … each with
  a unique ID + source hash; never reuse an ID for changed source; append one JSON record
  per evaluated candidate to `candidates.jsonl` (never rewrite earlier records).
- **Evaluation**: only via `./scripts/evaluate_candidate.sh feedback <id>`. The five fixed
  workloads together = one candidate evaluation. Budget 100 evals; token soft 1.0M / hard
  1.2M. `final` (16 workloads) is operator-only.
- **Forbidden**: running CUDA/profilers/`nvidia-smi`/the external evaluator/any alternate
  correctness harness directly; modifying evaluator/dataset/controller/launcher/config.
- Correctness gate per workload: `max_atol` (0.02–0.029), `max_rtol = 0.05`,
  `required_match_ratio = 0.98`. So up to 2% of elements may exceed tolerance — some slack
  for a few blown-up exp positions, but the bulk must match the fp32 reference.

---

## 4. Numerical risks & fidelity plan

The reference casts bf16→fp32, computes everything in fp32, then casts back. To stay in
tolerance we must **mirror the fp32 math**, especially around the exponentials.

1. **exp overflow / random-walk `a`.** Inputs are `type: random` bf16; `A` is *not*
   guaranteed negative here. `a[t] = cumsum` of up to 256 values; if A ~ O(1), partial
   sums random-walk to ±tens, and `exp(a[i]-a[j])` / `exp(a[t])` can reach `exp(±50)`
   (within fp32 range) or occasionally overflow to `inf`. Whatever the reference produces
   (including inf at extreme positions) is ground truth, so we replicate exactly:
   - Always form the **difference first, then exp**: `exp(a[i]-a[j])`, `exp(a_last-a[t])`,
     matching `segsum` (never `exp(a[i])/exp(a[j])`, which overflows differently).
   - Keep `a`, cumsum, and all exp arguments in **fp32**.
   - The 0.98 match ratio absorbs a small number of extreme/inf positions if they occur.
2. **tf32 vs ieee in `tl.dot`.** A800 `tl.dot` defaults to tf32 (10-bit mantissa) for
   fp32 inputs. The `C·B`, `M·X`, `X·B`, `C·states` matmuls feed into exp-amplified paths;
   tf32 error (~1e-3 rel) could be magnified. Plan: start with **fp32 accumulation**;
   evaluate tf32 (`input_precision="tf32"`) for the big matmuls first (rtol=0.05 is loose),
   and fall back to `"ieee"` selectively if a workload fails match ratio. Treat precision
   mode as a tunable across candidates.
3. **bf16 inputs to `tl.dot`.** Could load X/B/C as bf16 and use tf32/bf16 matmul with
   fp32 accumulate (fast, and the reference itself starts from bf16 data). Risk: extra
   rounding vs. the reference's explicit fp32 upcast. Validate empirically; bf16 inputs
   likely fine for the value matmul (`M·X`) but riskier where results are exponentiated.
4. **cumsum correctness.** Need an in-chunk prefix sum over Q=256 (`tl.cumsum` along the
   sequence axis) matching `torch.cumsum`. Padding rows contribute A=0 so cumsum stays
   flat — must load A with `other=0.0` beyond valid length.
5. **Padding & masking.** Last chunk of L=293/541 has 37/29 valid rows. All seq-indexed
   loads (X, A, B, C) must mask invalid rows to 0. `state_decay_out`/`decay_state` on
   padded rows are harmless (multiplied by zeroed data) and those rows are cropped anyway.
   Output store must be masked to `< L` on the seq axis.
6. **Accumulation order in states/inter-chunk.** The chunk recurrence sums a decayed
   `initial_states` (bf16→fp32) with per-chunk states. Keep fp32 throughout; cast to bf16
   only when writing `final_state` and `output`.
7. **D residual** added in fp32 before final bf16 cast (`y + D[h]·X`), matching reference.
8. **Layout / transpose hazards.** `A` is `[B,H,L]` (head-major) while X/B/C are seq-major
   `[B,L,·]`; strides must be handled explicitly to avoid mis-indexing when chunking.

Fidelity acceptance heuristic: aim comfortably inside atol/rtol on the bulk; use match
ratio only as a safety valve for rare exp extremes, not as a crutch for systematic error.

---

## 5. Why the reference is slow (headroom)

The reference materializes, in fp32 global memory, `L: [B,H,NC,Q,Q]` and
`G,M: [B,NC,Q,Q,H]`. For (B=4, L=1024): `4·4·256·256·16·4B ≈ 2.1 GB` **each**, plus the
`einsum('bcihs,bcjhs->bcijh')` over expanded (broadcast) B/C. This is dominated by DRAM
traffic on multi-GB intermediates and redundant per-head recomputation of the
head-independent `G`. A fused Triton kernel that (a) never writes `[Q,Q]` matrices to
DRAM, (b) computes `G`/CB once per group, and (c) keeps decays in registers should give a
large speedup — the memory-bound reference is the main source of headroom, not raw FLOPs.

---

## 6. Triton design space

### 6a. Kernel decomposition options
- **Option A — Faithful 4-stage SSD (mamba_ssm-style):**
  1. `chunk_cumsum`: compute `a = cumsum(A)` per `(b,head,chunk)` (+ `a_last`).
  2. `chunk_state`: `states[p,n] = Σ_t exp(a_last-a[t])·X[t,p]·B[t,n]` → `[B,NC,H,P,N]`.
  3. `state_passing / inter-chunk`: apply `decay_chunk` recurrence over NC (+ initial),
     produce `states_out` and `final_state`.
  4. `chunk_scan`: fuse `Y_diag = (G∘L)·X` and `Y_off = exp(a)·(C·states_out)`, add D,
     write output. This mirrors well-known, well-tuned kernels; good default.
- **Option B — Two big fused kernels:** a "state" kernel (stages 1–2) and a "scan" kernel
  (stages 1,3,4 fused with recomputed cumsum). Fewer launches / less intermediate DRAM,
  more register pressure.
- **Option C — Aggressive single mega-kernel per `(b,head)`** looping chunks internally to
  carry the state in registers/SRAM (exploits NC ≤ 4). Removes the inter-chunk kernel and
  the `states` round-trip entirely, but serializes chunks within a block and raises
  register/SRAM pressure; attractive because NC is tiny.

Plan: **start with Option A** (clear, matches a proven structure, easy to validate stage
by stage via the harness), then explore B/C for launch-overhead and DRAM reduction. The
first landed candidate `c001` should prioritize *correctness within tolerance* over peak
speed.

### 6b. Tiling & mapping
- Diagonal (`chunk_scan`) block: program per `(b, chunk, head)`; tile the `i` (query) axis
  in blocks of e.g. 64/128 rows over Q=256, inner-loop `j` over `≤ i` key blocks; do
  `tl.dot([BLK_i, Q_kv], [Q_kv, P])` with the lower-tri mask + `exp(a_i-a_j)` decay applied
  to the CB score tile in SRAM. P=64 fits a single dot N-dim.
- `chunk_state`: tile over `(p, n)` = `(P=64, N=256)`, reduce over t (Q=256) with decayed
  X and B. `[P,N]` fp32 accumulator per `(b,chunk,head)` fits SRAM.
- Reuse head-independent `G`/CB and `C·states` across the 16 heads where the decomposition
  allows (esp. CB in the diagonal stage).
- Expose enough grid parallelism for the 64-tile workloads (split head_dim / query blocks
  to fill 108 SMs).

### 6c. Numeric/dtype knobs (candidate axes)
- matmul precision: `ieee` vs `tf32`; input dtype bf16 vs fp32 per matmul.
- `BLOCK` sizes for query/key/state tiling; `num_warps`, `num_stages`.
- fuse vs. split stages; recompute cumsum vs. reload.

### 6d. Inter-chunk stage
NC ≤ 4 ⇒ implement as a tiny per-`(b,head)` kernel that (a) builds `decay_chunk` from
`a_last` values, (b) does the length-(NC+1) recurrence starting from `initial_states`.
Cheap; correctness-critical for `final_state`. Could also be a sequential in-kernel scan.

---

## 7. Validation strategy

- **Harness-only correctness.** Per project rules I will not run CUDA/profilers/an
  alternate correctness harness locally; the only correctness signal is
  `./scripts/evaluate_candidate.sh feedback <id>` over the five fixed workloads (atol/rtol/
  match-ratio per workload). Each candidate = one evaluation over all five.
- **Coverage across the five workloads** already exercises the important regimes:
  - partial last chunk with heavy padding (L=293 → 37 valid; L=541 → 29 valid),
  - exact multiples (L=256 single chunk; L=1024 four chunks),
  - batch scaling (B=1,2,4) and thus occupancy variety (64–256 tiles).
  So passing all five is a strong signal for both correctness (padding/masking, cumsum,
  decay) and speed generality. Both outputs (`output` and `final_state`) are checked.
- **Incremental, reasoned candidates.** Build up: `c001` = correct faithful Option-A
  kernel with conservative precision (fp32/ieee where risky) to lock correctness and get a
  baseline geomean; subsequent candidates tune precision (tf32/bf16 inputs), tiling,
  fusion, and grid parallelism, one immutable change per ID.
- **Regression discipline.** Record per candidate: parent, source hash, hypothesis,
  per-workload pass/fail + speedup, geomean, decision, cumulative eval count, skill usage.
  Never rewrite earlier records. Keep any candidate that regresses correctness reverted.
- **Static self-checks before each eval** (no GPU): re-derive shapes/strides, mask
  boundaries, cumsum axis, exp-difference ordering, and dtype casts against §2/§4 to avoid
  wasting evaluations on avoidable bugs.
- **Skills.** Consult `KernelWiki` for A800/Hopper SSD/chunk-scan kernel patterns and
  `ncu-report-skill` conceptually for bottleneck reasoning (without running the profiler
  here), and log their usage.

---

## 8. Open questions / risks to resolve during implementation

1. Distribution/scale of `random` inputs — governs how often `exp` blows up and whether
   tf32 is safe. Resolve empirically via the first evaluations.
2. Best precision setting per matmul (tf32 vs ieee vs bf16-input) to maximize speed while
   holding the 0.98 match ratio.
3. Whether to reuse head-independent `G`/CB explicitly (needs a group-major loop) vs.
   recompute per head for simplicity in `c001`.
4. Option A vs. C for the inter-chunk state given NC ≤ 4 (register-carry mega-kernel could
   remove a whole DRAM round-trip).
5. Grid/occupancy tuning for the 64-tile workloads on 108 SMs.
6. Exact stride handling for `A: [B,H,L]` vs seq-major X/B/C during chunking.

---

## 9. Next step

Proceed to `docs/plan.md` with an executable, staged plan: `c001` = correct Option-A
Triton implementation (conservative precision) to establish a passing baseline geomean,
followed by tuned candidates (precision, head-shared reuse, fusion, tiling, occupancy).
No solution code before the plan is written.
