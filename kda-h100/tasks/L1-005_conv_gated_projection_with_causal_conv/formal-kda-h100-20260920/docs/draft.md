# Draft — L1/005 `conv_gated_projection_with_causal_conv`

Target: NVIDIA H100 (`sm_90`). Submission: `solution/solution.py` exposing `run(...)`.
Primary compute must be **Triton**; PyTorch only for metadata / launch plumbing; **no** Torch/CPU/NumPy/CUDA-extension computational fallback.

This document is analysis only. No `docs/plan.md` and no solution code are produced in this turn.

---

## 1. Operation summary

This is the dominant non-attention "short-conv gated projection" block of the LFM2 architecture. It fuses:

1. a triple linear projection (`in_proj`, output width `3H`),
2. element-wise input gating,
3. a **depthwise causal 1-D convolution** (`kernel_size=4`, `groups=H`),
4. an output gating,
5. a final linear output projection (`out_proj`).

### 1.1 Signature (from `task/definition.json`)

| Tensor | Shape | Dtype | Role |
|---|---|---|---|
| `x` | `(B, S, H)` | bf16 | input hidden states |
| `in_proj_weight` | `(3H, H)` | bf16 | triple projection weight |
| `in_proj_bias` | `(3H,)` | bf16 | triple projection bias |
| `conv_weight` | `(H, 1, K)` | bf16 | depthwise conv weight (`K=4`) |
| `conv_bias` | `(H,)` | bf16 | depthwise conv bias |
| `out_proj_weight` | `(H, H)` | bf16 | output projection weight |
| `out_proj_bias` | `(H,)` | bf16 | output projection bias |
| **output** | `(B, S, H)` | bf16 | result |

Constants: `H = hidden_size = 2048`, `K = conv_kernel_size = 4`, `3H = triple_hidden = 6144`.
Variables: `B = batch_size`, `S = seq_len`.

### 1.2 Exact math (as implemented by the reference)

Let `N = B·S` be the token count. The reference does, in bf16 at every stage (each PyTorch op rounds its result back to bf16):

```
BCx = F.linear(x, in_proj_weight, in_proj_bias)      # (B, S, 3H)
BCx = BCx.transpose(-1,-2)                            # (B, 3H, S)
Bg, Cg, V = BCx.chunk(3, dim=1)                       # each (B, H, S)
Bx  = Bg * V                                          # (B, H, S)  input gating
Bxp = F.pad(Bx, (K-1, 0))                             # left-pad seq by 3
conv = F.conv1d(Bxp, conv_weight, conv_bias, groups=H)# (B, H, S)  depthwise causal
y   = Cg * conv                                        # (B, H, S)  output gating
y   = y.transpose(-1,-2).contiguous()                 # (B, S, H)
output = F.linear(y, out_proj_weight, out_proj_bias)  # (B, S, H)
```

**Chunk semantics (verified by re-deriving the transpose+chunk indexing).** After the
`transpose` the channel split maps directly onto the natural `(B,S,3H)` layout of `BCx`:

- `Bg[b,s,h]  = BCx[b, s,   h]`     — projection rows `[0 : H]`   (the **input gate**)
- `Cg[b,s,h]  = BCx[b, s, H+h]`     — projection rows `[H : 2H]`  (the **output gate**)
- `V [b,s,h]  = BCx[b, s, 2H+h]`    — projection rows `[2H : 3H]` (the **value**)

So per hidden channel `h`:
`Bx[b,s,h] = Bg·V`, using `in_proj_weight` rows `h` and `2H+h`; the output gate `Cg` uses rows `H+h`.

**Causal conv (verified as cross-correlation with left padding `K-1=3`).** For each `(b,h)`:
```
conv[b,s,h] = conv_bias[h] + Σ_{k=0..3} conv_weight[h,0,k] · Bx[b, s-3+k, h]
```
with `Bx[b, s', h] = 0` for `s' < 0`. i.e. taps are
`w0·Bx[s-3] + w1·Bx[s-2] + w2·Bx[s-1] + w3·Bx[s] + bias`.
The convolution and its left-padding are **per-sequence (per-batch)**: position `s=0,1,2`
sees zeros for the out-of-range taps — halo must **never** bleed across batch boundaries.

Final:
```
y[b,s,h]   = Cg[b,s,h] · conv[b,s,h]
output[b,s,o] = out_proj_bias[o] + Σ_{h=0..H-1} y[b,s,h] · out_proj_weight[o,h]
```
Both linears follow `F.linear` convention `out[n,o] = Σ_k in[n,k]·W[o,k]` (weight is `[out,in]`).

### 1.3 Compute characterization

Per token the FLOPs are dominated by the two matmuls:

- `in_proj`:  `2·N·H·(3H) = 6·N·H²`
- `out_proj`: `2·N·H·H   = 2·N·H²`
- conv + gating: `O(N·H·K)` — negligible (memory-bound elementwise).

Total ≈ `8·N·H²`, of which **`in_proj` is 75%**. The conv/gating steps carry almost no
arithmetic but in the reference they cost several full `(N·H)` / `(N·3H)` memory round-trips
(transpose, pad, contiguous), which is the real inefficiency (see §4).

---

## 2. Workload set (16 feedback workloads)

`H=2048`, `K=4` fixed. Ordered by `N = B·S`:

| uuid (short) | B | S | N=B·S | atol | rtol | regime |
|---|---|---|---|---|---|---|
| dcc98535 | 2 | 128 | 256 | 0.0082 | 0.05 | tiny |
| 66fd2ad8 | 1 | 256 | 256 | 0.0082 | 0.05 | tiny |
| b0a9e3f0 | 1 | 1024 | 1024 | 0.013 | 0.05 | small |
| 8b678b13 | 4 | 256 | 1024 | 0.013 | 0.05 | small |
| bb262a64 | 1 | 2048 | 2048 | 0.0098 | 0.05 | small/med |
| 3ac091d6 | 4 | 512 | 2048 | 0.0098 | 0.05 | small/med |
| 1979d4cb | 2 | 1024 | 2048 | 0.0098 | 0.05 | small/med |
| b662596d | 4 | 541 | 2164 | 0.0098 | 0.05 | med, ragged S |
| 36e77f89 | 2 | 1879 | 3758 | 0.028 | 0.05 | med, ragged S |
| 121c8e28 | 16 | 256 | 4096 | 0.028 | 0.05 | med, large B |
| 1408082c | 2 | 2048 | 4096 | 0.028 | 0.05 | med |
| 0a859b3a | 1 | 4096 | 4096 | 0.028 | 0.05 | med |
| d9432839 | 8 | 997 | 7976 | 0.021 | 0.05 | large, ragged S |
| 6033fd0f | 2 | 4096 | 8192 | 0.021 | 0.05 | large |
| 58e3ac47 | 32 | 256 | 8192 | 0.021 | 0.05 | large, large B |
| 01719394 | 1 | 8192 | 8192 | 0.021 | 0.05 | large |

Observations that drive design:
- `N` spans **256 → 8192** (32×). Both launch-overhead-bound (tiny `N`) and
  throughput-bound (large `N`) regimes are present; a single fixed tiling will not be optimal
  across all — **autotuning over `N`/shape** is warranted.
- Several `S` are **non-power-of-two / prime-ish** (1879, 997, 541, 256, 128). Seq masking
  and correct per-batch causal boundaries are mandatory; the tiling must handle ragged `S`.
- Large `B` with small `S` (`16×256`, `32×256`) stresses the per-batch causal-halo logic:
  many independent short sequences → many batch boundaries where the conv halo must be masked.
- `atol` is fairly generous (0.008–0.028) and `rtol=0.05` everywhere — this is bf16-level
  tolerance, consistent with the reference itself rounding every intermediate to bf16.
- Feedback evaluation runs **all 16** as one candidate evaluation (warmup 2 / 10 iters, coarse).
  Correctness must pass on **every** workload for the candidate to be valid.

---

## 3. Constraints & rules (operational)

- **Triton-only compute.** `torch.matmul`/`F.linear`/`F.conv1d` are *not* allowed as the
  implementation — they'd be a Torch fallback. Every FLOP (both GEMMs, the conv, both gatings)
  must be produced by `@triton.jit` kernels. Torch is allowed only for `.shape`, `.stride`,
  allocating outputs, `.view`/`.reshape` metadata, and kernel launch.
- **No fallback of any kind** (Torch/CPU/NumPy/CUDA-ext/alternate). A failing Triton kernel is
  an invalid candidate — it must be fixed, not routed around.
- **Isolation.** Work only in this workspace. External knowledge only via the installed
  `KernelWiki` and `ncu-report-skill` skills. Do not touch evaluator/controller/dataset.
- **Evaluation.** Only via `./scripts/evaluate_candidate.sh feedback <cid>`. One full-set run =
  one evaluation. Budget 100 evaluations; token soft-limit 9M. `final` is operator-only.
- **Immutability.** Each meaningful source/config/launch change ⇒ a new `cNNN` id; never
  reuse an id for changed source; never rewrite prior `candidates.jsonl` records.
- **Profiling.** Only through `./scripts/ncu_profile.sh` (ncu-report-skill workflow), and
  **never** concurrently with an evaluation (co-resident process ⇒ discarded measurement,
  return code 3, wasted eval). Serialize profiling and evaluation strictly.
- **Environment note.** Direct local Python execution is not available to me in this session
  (sandboxed Bash denies it), so I cannot run an offline numeric check. Correctness must be
  argued on paper before spending an evaluation, and the evaluator is the only ground truth.
  This raises the cost of each eval and argues for careful, conservative first candidates.

---

## 4. Where the speedup comes from (reference inefficiency)

The reference is a chain of ~8 CUDA kernels/launches with heavy layout churn:

1. cuBLAS `in_proj` GEMM → `(B,S,3H)` bf16 (`3H=6144` wide).
2. `transpose(-1,-2)` → non-contiguous `(B,3H,S)`.
3. `chunk` (views), `Bg*V` elementwise, `F.pad` (materializes padded copy).
4. cuDNN/depthwise `conv1d`.
5. `Cg*conv` elementwise, `transpose` + **`.contiguous()`** (full copy of `N·H`).
6. cuBLAS `out_proj` GEMM.

The arithmetic is only ~`8·N·H²`, but steps 2–5 add several full `(N·H)`–`(N·3H)` global
memory passes plus multiple launches. Opportunities:

- **Kill the transposes / pad / contiguous.** Emit `in_proj` results already in the natural
  `(B,S,H)` layout and do gating in the epilogue; do the conv in-place over the seq axis;
  fuse the output gate. Nothing needs to be physically transposed.
- **Fuse launches.** Collapse the ~8 launches into 2–3 Triton kernels; big win at small `N`
  where launch/overhead dominates (tiny/small workloads, ~half the set).
- **Reduce intermediate traffic.** Only `Bx` and `Cg` need to leave the first kernel; `V` is
  consumed immediately. Fusing conv+gate into the `out_proj` prologue further removes a full
  `y` round-trip.
- **Competitive GEMM.** At large `N` the two GEMMs dominate and must approach cuBLAS bf16.
  This is the main *risk* (see §6): a poorly-tuned Triton GEMM could erase the fusion win at
  `N=8192`. Autotuning + wgmma-friendly tiles are required there.

---

## 5. Triton design space

### 5.1 Kernel decomposition options

Numbered from simplest/safest to most fused.

**Option A — three kernels (robust baseline).**
- **K1 `triple_gemm_gate`**: for a tile of `BLOCK_M` tokens × `BLOCK_H` hidden channels,
  contract over `K=H` and accumulate **three** results simultaneously (`Bg`, `Cg`, `V`) from the
  three weight row-blocks (rows `h`, `H+h`, `2H+h`) sharing the *same* `x` tile. Epilogue:
  `Bx = Bg*V` (+ store), store `Cg`. Grouping by hidden channel `h` is essential because
  `Bx` pairs columns `h` and `2H+h`, which lie `2H` apart in the `3H` output — a plain
  column-tiled `3H` GEMM cannot pair them within one tile. Reuses the `x` tile across the three
  weight tiles ⇒ good arithmetic intensity; total FLOPs identical to a single `N×3H` GEMM.
- **K2 `causal_conv_gate`**: memory-bound. For each `(b, seq-tile, hidden-tile)` load `Bx`
  for the tile plus the 3-row causal halo (masked at seq/batch boundaries), do the 4-tap FMA
  with per-channel `conv_weight`/`conv_bias`, multiply by `Cg`, store `y`.
- **K3 `out_gemm`**: standard `(N×H)·(H×H)ᵀ + bias` bf16 GEMM.

**Option B — two kernels (fuse conv+gate into out_proj prologue).**
K1 as above; then a fused **`conv_gate_gemm`**: for a `BLOCK_M` token tile, load `Bx`
(`BLOCK_M+3` rows halo) and `Cg`, materialize the `y` tile on the fly (conv+gate) in
registers/SMEM, and immediately feed it as the `A` operand of the `out_proj` matmul (contract
over `H`). Saves a full `y` global write+read (`2·N·H·2` bytes). Slightly trickier because the
halo/causal masking must be exact per batch and the token tile must not straddle batch
boundaries in a way that corrupts the halo.

**Option C — single mega-kernel.** Rejected for now: the conv couples adjacent seq positions
across `in_proj` output tiles, and the two GEMMs contract over different dims (`H` then `H`
again but on a produced intermediate). Full fusion would force recomputation or complex
cross-tile synchronization for marginal gain. Keep as a stretch idea only.

**Initial choice bias:** start from **Option A** (easiest to reason about and validate on
paper, since I cannot run a local check), confirm correctness + baseline speedup, then move to
**Option B** for the traffic saving once the numerics are trusted.

### 5.2 Tiling / launch parameters (H100 / wgmma)

- `in_proj` (K1): contraction `K=H=2048`. Candidate tiles `BLOCK_M ∈ {64,128}`,
  `BLOCK_H ∈ {64,128,256}`, `BLOCK_K ∈ {32,64,128}`; `num_stages ∈ {3,4,5}`,
  `num_warps ∈ {4,8}`. `tl.dot` with bf16 operands and **fp32 accumulate** (maps to wgmma on
  sm90). Weight `B`-operand loaded transposed (via strides) so `dot(x[M,K], Wᵀ[K,N])` matches
  `out[n,o]=Σ_k x[n,k]·W[o,k]`.
- `out_proj` (K3/Option B): same GEMM template, `N`-dim `=H=2048`, `K=H=2048`.
- conv (K2): pick `BLOCK_M` seq × `BLOCK_H` channels sized for coalesced bf16 loads; the 4 taps
  as 4 masked shifted loads or one `BLOCK_M+3` load then register shifts.
- **Autotune** over the shape/`N` regimes (tiny vs large). Use `triton.autotune` keyed on
  `N` (and maybe `S`) so tiny workloads get small tiles (low launch/occupancy overhead) and
  large workloads get throughput tiles. KernelWiki confirms Triton is a good fit for
  memory-bound fused kernels and the persistent/warp-specialized matmul tutorial pattern is the
  reference structure; on sm90 the standard `tl.dot` path lowers to wgmma with fp32 accumulate
  (`lang-triton`, `technique-kernel-fusion`, `technique-epilogue-fusion`).
- Consider a **persistent** grid for the GEMMs at large `N` to cut tail effects
  (`technique-tile-scheduling`/`pattern-tail-effect`), but only after a correct baseline.
- Flatten `(B,S)`→`N` for the GEMMs (they are per-token independent); keep explicit `(b,s)`
  indexing (or `row = b·S + s` with a `local-s` check) in the conv kernel so causal masking is
  applied per batch.

### 5.3 Data types & epilogue

- Loads bf16; `tl.dot` fp32 accumulate; add bias in fp32 before the bf16 store — this matches
  cuBLAS `F.linear` semantics.
- Store `Bx`, `Cg`, `y` as **bf16** (matches the reference, which rounds each intermediate to
  bf16, and halves intermediate traffic). Conv accumulates in fp32 then rounds to bf16 before
  the output gate — mirroring `F.conv1d`'s bf16 output.

---

## 6. Numerical risks & mitigations

1. **Accumulation vs cuBLAS/cuDNN.** Both use fp32 accumulate over `K=2048`; `tl.dot` fp32
   accumulate matches. Different summation order gives only last-bit differences ≪ the bf16
   `rtol=0.05`/`atol≥0.008`. Low risk.
2. **Intermediate rounding placement.** The reference rounds `Bx`, `conv`, `y` to bf16. If we
   keep them in fp32 we are *more* accurate than the reference but the metric is vs the
   reference. To minimize divergence and match the target we will round `Bx`/`conv`/`y` to bf16
   at the same points. (We can revisit keeping conv in fp32 if a candidate fails only on a
   cancellation-sensitive workload — but bf16 rounding is the conservative default.)
3. **Causal-boundary correctness (highest-value risk).** Off-by-one in the tap alignment or
   halo leaking across batch boundaries would fail correctness. Pin the mapping:
   `conv[s] = Σ_{k=0..3} w[k]·Bx[s-3+k]`, `Bx[<0]=0`, boundaries **per (b)** not per flattened
   `N`. Large-`B`/short-`S` workloads (`16×256`, `32×256`, `2×128`) are the acid test.
4. **Chunk mapping.** `Bg=rows[0:H]`, `Cg=rows[H:2H]`, `V=rows[2H:3H]`, `Bx=Bg·V`,
   `y=Cg·conv`. A swap here would fail everywhere; re-verified in §1.2.
5. **Ragged `S` masking.** `S ∈ {128,256,512,541,997,1024,1879,2048,4096,8192}` — seq/token
   tiles must mask the `N%BLOCK_M`/`S%BLOCK_S` remainder. Non-multiples (1879, 997, 541) are the
   tests.
6. **Bias dtype.** `in_proj_bias`/`conv_bias`/`out_proj_bias` are bf16; add in fp32 accumulator
   then round — matches `F.linear`/`F.conv1d`.
7. **`atol` scaling with magnitude.** Larger-`N`/`S` workloads have larger `atol`, tracking the
   larger accumulated magnitudes; keeping fp32 accumulation avoids catastrophic bf16
   accumulation error inside the dot products.

---

## 7. Validation strategy

Because I cannot execute Python/torch locally in this session, validation is:

1. **Paper correctness first.** Fully pin indexing (chunk map, tap alignment, per-batch causal
   masking, weight-transpose in `tl.dot`, bias placement) before the first evaluation. §1.2 and
   §6 are that pin.
2. **Conservative c001 = Option A**, correctness-first tiling with safe masks (no fancy
   persistence/autotune yet), so the first eval primarily proves *correctness on all 16*. A
   correct-but-modest baseline that already removes transposes/launches should beat the
   reference on the small/medium half even if the big-`N` GEMM is untuned.
3. **Read per-workload evaluator output** (pass/fail + timing) as the validation signal; the
   ragged-`S` and large-`B` workloads are the correctness canaries, the `N=8192` workloads are
   the throughput canaries.
4. **Profile with `ncu-report-skill`** (via `./scripts/ncu_profile.sh`) only after a correct
   baseline, and never concurrently with an evaluation. Use it to decide whether K1 is
   compute-bound (tune tiles/stages) or the conv/traffic is the bottleneck (motivating Option B).
5. **One change ⇒ one candidate id.** Change tiling/autotune config, kernel fusion level, or
   dtype policy only as a new `cNNN`, logging parent, source hash, hypothesis, per-workload
   result, geomean, decision, cumulative eval count, and skill usage to `candidates.jsonl`.
6. **Budget discipline.** 100 evals / 9M-token soft cap. Prefer autotune *inside* one candidate
   (many configs, one eval) over many single-config candidates. Stop and write `SEARCH_COMPLETE`
   when geomean improvement genuinely converges.

---

## 8. Candidate roadmap (sketch — full plan deferred to `docs/plan.md`)

- **c001** — Option A, correctness-first fused 3-kernel pipeline (triple-GEMM+gate → conv+gate
  → out-GEMM), modest fixed tiles. Goal: pass all 16, establish baseline geomean.
- **c002+** — add `triton.autotune` over `N`/shape regimes for both GEMMs (tiny vs large tiles).
- **cNNN** — Option B: fuse conv+gate into the `out_proj` prologue to remove the `y` round-trip.
- **cNNN** — persistent / tile-scheduling for large-`N` GEMM tail effects; tune `num_stages`,
  swizzle, vectorized loads for the conv kernel.
- **cNNN** — revisit intermediate dtype policy only if a specific workload's numerics require it.

This ordering front-loads correctness (cheap to reason, expensive to get wrong given no local
run) and back-loads throughput tuning (guided by ncu), converging when the geomean plateaus.

---

## 9. Skill usage

- **KernelWiki** consulted: `lang-triton` (Triton on Hopper/Blackwell — `tl.dot`→wgmma fp32
  accumulate on sm90, good fit for memory-bound fused kernels, autotune/persistent matmul
  tutorial pattern), `technique-kernel-fusion` (collapse multi-launch chains, reduce
  intermediate traffic), `technique-epilogue-fusion` (fuse scale/bias/activation/gating into the
  GEMM epilogue), plus primer references to `pattern-tail-effect`/`technique-tile-scheduling`
  for large-`N` GEMM scheduling. These inform the Option A/B decomposition, the epilogue-gating
  design, and the autotune/persistence roadmap.
- **ncu-report-skill** will be used (via `./scripts/ncu_profile.sh`, serialized against
  evaluations) once a correct baseline exists, to attribute time between the two GEMMs and the
  conv/traffic and to guide tiling and the Option B fusion decision.
