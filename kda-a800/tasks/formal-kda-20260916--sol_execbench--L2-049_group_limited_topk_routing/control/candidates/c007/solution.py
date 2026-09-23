"""
Solution for SOL-ExecBench L2/049 group_limited_topk_routing (A800 / sm_80).

Candidate: c007  (parent c006)  --  remove python-list companion capture
================================================================================
Six structurally distinct kernels now return 0/5 RUNTIME_ERROR *identically*, with
no traceback:
  c001 3D-fused / c002 2D-fused / c003 argmax-free-fused / c004 two-kernel split /
  c005 minimal-resource split / c006 sigmoid-via-exp.
Direct experiment has ruled out: fused-epilogue logic, register pressure, 3D ops,
tl.trans, 3-arg tl.dot, tl.argmax, int64 kernel stores, strided weight operand,
the -inf sentinel, the num_stages>=2 dynamic-shared-memory opt-in (c005 ~17KB
single-buffered still failed), AND tl.sigmoid/tl.exp (c006 used exp, still failed).

Elimination-by-necessity: this is a Triton kernel-optimization benchmark with a
matmul-heavy reference, so tl.dot and the absolute-basic ops
(tl.load/store/where/max/min/sum/arange/zeros, integer //, .to() casts) MUST work
on this stack. After removing sigmoid/exp, the one remaining NON-idiomatic
construct present in 100% of c001-c006 is the "python-list companion capture":
    sel_vals = []
    for i in tl.static_range(TOPK):
        ...
        sel_vals.append(val_i)          # storing traced values in a python list
    for i in tl.static_range(TOPK):
        w_i = sel_vals[i] / denom * ... # indexing that list in a 2nd unrolled loop
and the accompanying per-iteration scalar tl.store of a single index.

c007 changes EXACTLY this: the top-8 loop now accumulates the selected index and
the bias-free companion score into two [TOPK] REGISTER vectors via masked-where
(no python list, no second unrolled loop), then does ONE vectorized store of all 8
indices and ONE of all 8 weights. This is the standard idiomatic Triton pattern.
Everything else (minimal GEMM Kernel A, exp sigmoid, finite -1e30 sentinel,
contiguous [D,E] weight, int32 store + torch int64 cast) is held identical to c006.

Clean attribution:
  - if c007 PASSES  -> the python-list companion pattern was THE universal blocker.
  - if c007 ERRORS  -> the fault is not in any kernel construct I can vary; it is a
                       harness/environment/import-level condition (every possible
                       Triton kernel fails), and no justified kernel-level next
                       candidate remains.

Two score arrays per token: scores = sigmoid(logits) (returned-weight source, NO
bias); scores_routing = scores + bias (drives ALL three selections). All epilogue
math is fp32; only tl.dot uses tensor cores. Triton owns all computation; PyTorch
is used only for allocation / launch plumbing, a layout-only weight transpose, and
a trivial int32->int64 result cast. No Torch/CPU/NumPy/CUDA-extension
computational fallback exists.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: gate GEMM  logits[T, E] = hidden_states[T, D] @ weightT[D, E]
#   (bf16 tensor-core inputs, fp32 accumulate). weightT is contiguous [D, E].
# ---------------------------------------------------------------------------
@triton.jit
def _gemm_kernel(
    hs_ptr, wt_ptr, logits_ptr,
    T, D,
    stride_hs_m, stride_hs_k,
    stride_wt_k, stride_wt_e,
    stride_lo_m, stride_lo_e,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T
    offs_e = tl.arange(0, E)                       # whole expert dim (256) in one block

    acc = tl.zeros((BLOCK_M, E), dtype=tl.float32)
    for k in range(0, D, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < D
        a = tl.load(
            hs_ptr + offs_m[:, None] * stride_hs_m + offs_k[None, :] * stride_hs_k,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )  # [BLOCK_M, BLOCK_K] bf16
        b = tl.load(
            wt_ptr + offs_k[:, None] * stride_wt_k + offs_e[None, :] * stride_wt_e,
            mask=k_mask[:, None], other=0.0,
        )  # [BLOCK_K, E] bf16 (contiguous along E)
        acc += tl.dot(a, b)                        # fp32 accumulate -> [BLOCK_M, E]

    tl.store(
        logits_ptr + offs_m[:, None] * stride_lo_m + offs_e[None, :] * stride_lo_e,
        acc, mask=m_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Kernel B: per-token routing over logits[E].  grid = (T,).
#   Companion capture uses [TOPK] register vectors (no python list).
# ---------------------------------------------------------------------------
@triton.jit
def _route_kernel(
    logits_ptr, bias_ptr, idx_ptr, wt_ptr,
    scaling,
    stride_lo_m, stride_lo_e,
    stride_idx_m, stride_idx_n,
    stride_wt_m, stride_wt_n,
    E: tl.constexpr, NG: tl.constexpr, EPG: tl.constexpr,
    TOPK: tl.constexpr, TOPG: tl.constexpr,
):
    pid = tl.program_id(0)                         # one token per program
    offs_e = tl.arange(0, E)                       # [E]
    offs_g = tl.arange(0, NG)                      # [NG]
    offs_k = tl.arange(0, TOPK)                    # [TOPK]
    expert_group = offs_e // EPG                   # [E] group id per expert (0..7)

    NEG = -1e30                                    # finite sentinel (< any sigmoid+bias)
    BIG = E                                        # sentinel index for min-index reduce

    logit = tl.load(logits_ptr + pid * stride_lo_m + offs_e * stride_lo_e)  # [E] fp32
    scores = 1.0 / (1.0 + tl.exp(-logit))          # [E]  (returned-weight source, no bias)
    bias = tl.load(bias_ptr + offs_e).to(tl.float32)
    sr = scores + bias                             # [E]  (drives selection)

    # ---- group stage: top-2 sum within each group ----
    group_scores = tl.zeros((NG,), dtype=tl.float32)
    for gg in tl.static_range(NG):
        in_g = expert_group == gg                  # [E]
        gv = tl.where(in_g, sr, NEG)               # [E]
        m1 = tl.max(gv)                            # scalar largest in group
        gv2 = tl.where(gv == m1, NEG, gv)          # suppress the (unique) max
        m2 = tl.max(gv2)                           # scalar second largest
        gsum = m1 + m2                             # scalar
        group_scores += tl.where(offs_g == gg, gsum, 0.0)

    # ---- top-TOPG groups -> per-expert selection mask ----
    sel_e = tl.zeros((E,), dtype=tl.float32)
    gcur = group_scores
    for _ in tl.static_range(TOPG):
        gmax = tl.max(gcur)                         # scalar best group score
        gcand = tl.where(gcur == gmax, offs_g, NG)  # [NG]
        gi = tl.min(gcand)                          # scalar smallest-index best group
        sel_e = tl.where(expert_group == gi, 1.0, sel_e)
        gcur = tl.where(offs_g == gi, NEG, gcur)
    masked = tl.where(sel_e > 0.5, sr, NEG)         # [E]

    # ---- top-TOPK experts; accumulate into [TOPK] register vectors (no python list) ----
    sel_idx = tl.zeros((TOPK,), dtype=tl.int32)     # [TOPK] chosen expert indices
    sel_val = tl.zeros((TOPK,), dtype=tl.float32)   # [TOPK] bias-free companion scores
    for i in tl.static_range(TOPK):
        emax = tl.max(masked)                       # scalar best masked score
        ecand = tl.where(masked == emax, offs_e, BIG)  # [E]
        ei = tl.min(ecand)                          # scalar smallest-index best expert
        val_i = tl.sum(tl.where(offs_e == ei, scores, 0.0))  # scalar bias-free score
        masked = tl.where(offs_e == ei, NEG, masked)
        at_i = offs_k == i                          # [TOPK] one-hot for slot i
        sel_idx = tl.where(at_i, ei.to(tl.int32), sel_idx)
        sel_val = tl.where(at_i, val_i, sel_val)

    denom = tl.sum(sel_val) + 1e-20                 # scalar
    w_vec = sel_val / denom * scaling               # [TOPK]

    tl.store(idx_ptr + pid * stride_idx_m + offs_k * stride_idx_n, sel_idx)
    tl.store(wt_ptr + pid * stride_wt_m + offs_k * stride_wt_n, w_vec)


def run(hidden_states, weight, expert_bias, routed_scaling_factor):
    # Constants (fixed by the task definition).
    E = 256          # num_experts
    NG = 8           # n_group
    EPG = E // NG    # experts_per_group = 32
    TOPK = 8         # num_experts_per_tok
    TOPG = 4         # topk_group

    assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda
    T, D = hidden_states.shape
    assert weight.shape[0] == E and weight.shape[1] == D
    assert expert_bias.shape[0] == E

    # Layout-only plumbing: contiguous [D, E] weight so the GEMM B-operand load is
    # contiguous. Pure data movement, not the routing computation.
    weight_t = weight.t().contiguous()             # [D, E] bf16

    logits = torch.empty((T, E), dtype=torch.float32, device=hidden_states.device)

    # ---- Kernel A: gate GEMM -> logits[T, E]  (minimal shared-memory footprint) ----
    BLOCK_M_A = 16
    BLOCK_K = 32
    grid_a = (triton.cdiv(T, BLOCK_M_A),)
    _gemm_kernel[grid_a](
        hidden_states, weight_t, logits,
        T, D,
        hidden_states.stride(0), hidden_states.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        logits.stride(0), logits.stride(1),
        BLOCK_M=BLOCK_M_A, BLOCK_K=BLOCK_K, E=E,
        num_warps=4, num_stages=1,
    )

    # ---- Kernel B: per-token routing ----
    topk_idx32 = torch.empty((T, TOPK), dtype=torch.int32, device=hidden_states.device)
    topk_weight = torch.empty((T, TOPK), dtype=torch.float32, device=hidden_states.device)
    grid_b = (T,)
    _route_kernel[grid_b](
        logits, expert_bias, topk_idx32, topk_weight,
        float(routed_scaling_factor),
        logits.stride(0), logits.stride(1),
        topk_idx32.stride(0), topk_idx32.stride(1),
        topk_weight.stride(0), topk_weight.stride(1),
        E=E, NG=NG, EPG=EPG, TOPK=TOPK, TOPG=TOPG,
        num_warps=4, num_stages=1,
    )

    topk_idx = topk_idx32.to(torch.int64)
    return topk_idx, topk_weight
