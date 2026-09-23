# Plan — L2/051 `seqlen-finetuned-reconstructed_hyena_complete_forward_block`

Executable, sequential KDA optimization plan. Builds on `docs/draft.md` (dataflow §1,
risks §4, design space §5). Target H100 `sm_90`, Triton-only compute, fp32 I/O.

## 0. Operating rules (self-binding)

- Implement candidates `c001, c002, …` **sequentially**, one immutable source version at a
  time. Never reuse an ID for changed source.
- Evaluate only via `./scripts/evaluate_candidate.sh feedback cNNN` (full 16-workload set =
  one eval). Budget: 100 evals; token soft 9M / normal 10M / absolute 11M.
- Append exactly one JSON record per evaluated candidate to `candidates.jsonl`; never
  rewrite earlier records.
- No computational fallback: `torch.fft`, `F.linear/conv1d`, `torch.matmul`, `F.gelu`,
  `F.layer_norm` are disallowed as the implementation. Torch only for `.shape`, allocation,
  strides, dtype, and kernel launch.
- Profiling only via `./scripts/ncu_profile.sh` (ncu-report-skill), never overlapping an
  eval. Never run `ncu`/`nvidia-smi`/CUDA directly. Local ad-hoc `python`/numpy is blocked
  → correctness is established by construction + the evaluator only.
- Never run `final` without explicit operator approval. Create `SEARCH_COMPLETE` only on
  genuine convergence.

## 1. Reference-equivalence contract (must hold for every candidate)

The kernels must reproduce, within tolerance, exactly this math (see draft §1). Frozen
constants: `d_model=256, d_inner=1024, inner_width=768, filter_order=64, emb_dim=5,
order=2, short_filter_order=3`. `l_filter = L` for all feedback/final shapes (pad-back
branch is dead; a passthrough assert is acceptable).

1. `residual = hidden_states` (fp32, captured before any op).
2. `n1 = LN(hidden_states; norm1_w, norm1_b, eps)`, var `unbiased=False` over 256.
3. `u = n1 @ in_proj_weight^T + in_proj_bias` → `[B,L,768]`, viewed channel-major `[B,768,L]`.
4. Causal short conv (per channel, taps at offsets {−2,−1,0}, zero left-pad):
   `uc[b,c,t] = scb[c] + Σ_{j=0..2} scw[c,0,j]·u[b,c, t-(2-j)]` (u[...,<0]=0).
5. Split channels: `x0=uc[:,0:256]`, `x1=uc[:,256:512]`, `v0=uc[:,512:768]`.
6. Filter gen (batch-independent, depends only on L): build `z[L,5] =
   [t, cos(-f0 w), cos(-f1 w), sin(-f0 w), sin(-f1 w)]`, `t=linspace(0,1,L)`,
   `w=2π·[0..L-1]/L`, `f=[1e-4, 1.0]`. Then
   `h=sin(sin_freq·(z@fl1^T+b1))`; `h=sin(sin_freq·(h@fl2^T+b2))`;
   `h=sin(sin_freq·(h@fl3^T+b3))`; `h=h@fl_final^T` (no bias) → `[L,256]`.
   `decay[l,c]=exp(-t[l]·|exp_mod_deltas[c]|)`; `h = h·(decay+exp_mod_shift) + filter_bias[c]`.
7. **Scramble** into `k[256,L]`: flatten `h[L,256]` row-major to `B[l*256+c]`, re-view as
   `[256,L]` → `k[a,b] = B[a*L+b] = h[(a*L+b)//256, (a*L+b)%256]`. **Not** a clean transpose.
8. Gate + causal FIR (order-2 loop runs once, `x_i=x1`):
   `g = v0·x1`; `y[b,c,t] = Σ_{s=0..t} k[c,s]·g[b,c,t-s]`; `vc = y + g·filter_bias[c]`.
9. `y2 = vc·x0` → token-major `[B,L,256]`;
   `hyena_out = y2 @ out_proj_weight^T + out_proj_bias`.
10. `r = hyena_out + residual`.
11. `n2 = LN(r; norm2_w, norm2_b, eps)`.
12. `m = gelu_tanh(n2 @ fc1^T + fc1_b)`; `mlp_out = m @ fc2^T + fc2_b`.
13. `output = mlp_out + r`.

Equivalently `output = mlp_out + hyena_out + hidden_states`.

## 2. Candidate lineage strategy

Tree, not a chain: keep the last **known-correct** candidate as a stable parent; branch
optimizations from it; abandon a branch that regresses correctness or speed. Each node
records its parent. Naming stays sequential (`c001, c002, …`) regardless of tree position;
lineage is captured in the `parent` field.

Phase A — correctness (get all 16 passing):
- **c001**: reference-faithful, moderately fused, fp32 IEEE, conservative blocks. Goal:
  pass all 16 within tolerance and establish the baseline geomean. Kernels per draft §5:
  - K1a: LN1 + in_proj → `u[B,768,L]` (channel-major).
  - K1b: short causal conv + split + gate `g=v0·x1`; keep `x0,x1,g` channel-major.
  - K2: filter gen → scrambled `k[256,L]` (verify §1.7 mapping here first).
  - K3: causal FIR + `vc=y+g·fb` + final gate `y2=vc·x0` → token-major `[B,L,256]`.
  - K4a: out_proj + residual1 → `r`; K4b: LN2 + fc1 + gelu_tanh; K4c: fc2 + residual2.
  - If c001 misses, bisect by draft §4 risk order (scramble → conv boundary → LN var mode →
    GELU flavor → positional-embedding column order → `sin_freq`/bias placement). Each fix
    is a **new** candidate (c002, c003 …), changing one hypothesis at a time.

Phase B — fusion (reduce launches / DRAM traffic; dominates small-L, overhead-bound
shapes):
- Fuse LN into the following GEMM epilogue's prologue (LN1→in_proj, LN2→fc1).
- Fuse the two residual adds into GEMM epilogues (out_proj, fc2).
- Fuse K1a+K1b (project then conv within one program using neighbor recompute or a
  two-stage shared-memory pass), and fuse K3's epilogue chain (already planned in c001).
- Collapse the K4 tail (out_proj→LN2→fc1→gelu→fc2) toward fewer kernels where register/SMEM
  budget allows.

Phase C — tuning (dominates large-L, compute-bound shapes):
- Autotune BLOCK_M/N/K, num_warps, num_stages for the GEMMs (K1a, out_proj, fc1, fc2) and
  BLOCK_T / reduction tiling for the FIR.
- Size-regime specialization: a "small" config (L≤256, big fused launches, minimize kernel
  count) vs "large" config (L≥2048, tuned tiles). Selection by cheap host-side branch on
  `B,L` (allowed — plumbing only).
- FIR optimization: block the triangular `s`-reduction; skip fully-below-diagonal tiles;
  consider a banded fast path. Only pursue if profiling shows the FIR is a hotspot.

Phase D — precision levers (guarded):
- Try tf32 tensor-core matmuls on the dense GEMMs (in_proj/out_proj/fc1/fc2). Because the
  output is dominated by the O(1) `hidden_states` passthrough plus small (~0.01–0.1)
  corrections, tf32 on the corrections *may* stay within `atol≈3e-4`. This is a **measured**
  experiment: adopt only if the eval still passes all 16 with margin. Keep an fp32 parent to
  fall back to. Do not tf32 the LN/variance or the `sin`/`exp` filter math.

Only one meaningful change per candidate so each eval attributes cause cleanly.

## 3. Correctness checks (per candidate, before spending an eval)

Static self-review checklist (no local execution available):
1. Scramble address math matches `k[a,b]=hflat[a*L+b]` (draft §1.7).
2. Short conv: left zero-pad of 2, taps map `{scw[:,0,0]→t-2, scw[:,0,1]→t-1,
   scw[:,0,2]→t}`, output truncated to L; add `scb`.
3. FIR: sum `s=0..t` only (causal, zero beyond); fp32 accumulate; `vc` uses gated `g` in the
   bias term, final gate uses `x0`.
4. LN: mean & var over the 256 feature dim, `unbiased=False` (÷256), affine after.
5. GELU is tanh approximation with `0.044715` and `√(2/π)`.
6. Positional embedding: column order `[t, cos(-f0 w), cos(-f1 w), sin(-f0 w), sin(-f1 w)]`,
   `f=[1e-4,1.0]`; `sin_freq` multiplies each linear output inside `sin`.
7. Filter epilogue order: `·(decay+shift)` then `+filter_bias`; bias broadcast per channel.
8. Shapes/strides: `u,x0,x1,g` channel-major `[B,256/768,L]`; `k` `[256,L]`; output
   token-major `[B,L,256]` matching the contract.
9. No disallowed torch compute op remains in the path (grep the source before eval).

Eval-driven verification: c001's eval is the first ground truth. Read the per-workload
pass/fail + max-abs/rel error the controller returns; localize a failing stage via the §4
risk order; smallest shapes (2×128, 1×131, 1×256) isolate boundary/scramble bugs cheapest.

## 4. Performance hypotheses (to test with evals + ncu)

- **H1**: Fusing LN into the adjacent GEMM removes two `[B,L,256]` DRAM round-trips → wins
  most on small/overhead-bound shapes (32×256, 2×128, 1×131, 1×256, 4×256).
- **H2**: The K4 MLP (fc1 1024-wide + fc2) is the dense hotspot on large shapes
  (2×4096, 1×4096, 2×2048/2053); tuned GEMM tiles + fused GELU epilogue give the biggest
  large-L gain.
- **H3**: The `O(L²)` FIR is the risk hotspot at large L; blocked/triangular tiling and
  below-diagonal skipping cut it. Verify hotspot rank via ncu before investing.
- **H4**: Reducing kernel-launch count (fewer, more-fused kernels) improves the geomean
  because the set is dominated by small shapes where launch overhead is a large fraction.
- **H5**: tf32 matmuls (Phase D) improve large-L throughput; adopt only if all 16 still pass
  with error margin. Each hypothesis validated by comparing geomean across consecutive
  candidates and by ncu on the specific kernel; profiling never overlaps an eval.

## 5. Stopping criteria

Stop and write `SEARCH_COMPLETE` (with reason) when any holds:
- No correct candidate improves geomean by >~1% over the best for ~3 consecutive candidates
  (convergence plateau).
- Token budget approaches the 9M soft limit, or evaluation count nears 100.
- Profiling shows the dominant kernels are near hardware roofline (memory- or
  tensor-core-bound) with no structural lever left.
Always retain the best **valid** (all-16-passing) candidate as the final-eval nominee;
never run `final` without operator approval.

## 6. Evidence format (one JSON object per evaluated candidate → `candidates.jsonl`)

```json
{
  "candidate": "c001",
  "parent": null,
  "source_sha256": "<hash of solution/solution.py at eval time>",
  "hypothesis": "reference-faithful fused Triton baseline; establish correctness+geomean",
  "change_from_parent": "initial implementation",
  "validation": {
    "all_pass": true,
    "per_workload": [
      {"uuid": "ecdc9dd6-…", "B": 1, "L": 1024, "pass": true,
       "max_atol": 0.0, "max_rtol": 0.0, "match_ratio": 1.0, "speedup": 0.0}
    ],
    "num_workloads": 16
  },
  "geomean_speedup": 0.0,
  "decision": "keep|revert|branch-parent-for-next",
  "cumulative_evals": 1,
  "skills_used": ["KernelWiki", "ncu-report-skill"],
  "notes": "observations, failing-stage diagnosis, next hypothesis"
}
```

Rules: append-only; `source_sha256` computed from the exact evaluated file; `speedup` and
`geomean_speedup` from the controller output; `decision` states whether this becomes the
parent for the next candidate. Record skill usage (KernelWiki for H100 GEMM/fusion/
warp-specialization guidance; ncu-report-skill for profiling) whenever used.

## 7. First actions (next turn, not this one)

1. Implement `solution/solution.py` for **c001** per §2 Phase A (Triton kernels K1a–K4c,
   fp32 IEEE, conservative blocks), self-review against §3 checklist.
2. Evaluate: `./scripts/evaluate_candidate.sh feedback c001`.
3. Append the c001 record to `candidates.jsonl`; decide keep/branch; proceed to Phase A
   fixes or Phase B fusion accordingly.
