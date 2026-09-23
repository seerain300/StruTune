"""
Solution for SOL-ExecBench L2/049 group_limited_topk_routing (A800 / sm_80).

Candidate: c003  (parent c002)
Architecture: single fully-fused Triton kernel.
  - gate GEMM  [T,4096] x [4096,256]  (bf16 tensor-core inputs, fp32 accumulate)
  - fused routing epilogue entirely in-register (no logits DRAM round-trip):
      sigmoid -> +bias -> group top-2 sum -> top-4 groups -> mask -> top-8 experts
      -> bias-free companion scores -> normalize (+1e-20) -> * routed_scaling_factor

Two score arrays are tracked per token:
  scores          = sigmoid(logits)                (returned-weight source, NO bias)
  scores_routing  = scores + expert_bias           (drives ALL selection stages)

--- Why c003 is a broad de-risking rewrite ---
c001 (3D-reshape epilogue) and c002 (fully-2D epilogue) both returned 0/5
RUNTIME_ERROR with NO traceback exposed by the evaluator. Two structurally very
different epilogues failing IDENTICALLY means the fault is a construct COMMON to
both, not the epilogue-shape logic. A blind RUNTIME_ERROR cannot be bisected one
construct per candidate, so c003 removes every version-fragile Triton surface the
two shared, at once:
  1. 3-arg tl.dot(a,b,acc)  -> manual `acc += tl.dot(a,b)` (older Triton reads the
     3rd positional as trans_a, a hard type error).
  2. tl.argmax(axis=1)  -> portable max-value + min-matching-index reduction
     (tl.max then tl.min over where(val==max, iota, BIG)). Deterministic tie-break
     to the smallest index; identical selection SET on the tie-free random inputs.
  3. int64 tl.store from the kernel  -> kernel stores int32 indices; the
     int32->int64 cast is done in torch plumbing (a result dtype conversion, NOT
     computation).
  4. num_stages 3 -> 2 (reduce shared-memory / launch-resource pressure from
     pipelining the w[BLOCK_K,256] bf16 tile).
Epilogue stays fully 2D (no 3D reshape/broadcast), weight is loaded pre-transposed
(no tl.trans), BLOCK_M=16. The surviving op set is then minimal: tl.load/store,
tl.dot, tl.sigmoid, tl.where, tl.max, tl.min, tl.sum, integer arange/compare.

All epilogue math is fp32; only tl.dot uses tensor cores. Triton owns all
computation; PyTorch is used only for allocation / launch plumbing and a trivial
int32->int64 result cast. No Torch/CPU/NumPy/CUDA-extension computational
fallback exists.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _routing_kernel(
    hs_ptr, w_ptr, bias_ptr,          # inputs: bf16, bf16, bf16
    idx_ptr, wt_ptr,                  # outputs: int32, fp32
    scaling,                          # fp32 scalar (routed_scaling_factor)
    T, D,                             # runtime dims
    stride_hs_m, stride_hs_k,
    stride_w_e, stride_w_k,
    stride_idx_m, stride_idx_n,
    stride_wt_m, stride_wt_n,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    E: tl.constexpr, NG: tl.constexpr, EPG: tl.constexpr,
    TOPK: tl.constexpr, TOPG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T

    offs_e = tl.arange(0, E)              # [E] full expert dimension (256 fits one block)
    offs_g = tl.arange(0, NG)            # [NG] group axis (8)
    expert_group = offs_e // EPG         # [E] group index per expert (0..7)

    # ---- gate GEMM: acc[BLOCK_M, E] = hs[BLOCK_M, D] @ weight[E, D].T ----
    # weight loaded pre-transposed as [BLOCK_K, E]; manual accumulate (no 3-arg dot).
    acc = tl.zeros((BLOCK_M, E), dtype=tl.float32)
    for k in range(0, D, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < D
        hs = tl.load(
            hs_ptr + offs_m[:, None] * stride_hs_m + offs_k[None, :] * stride_hs_k,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )  # [BLOCK_M, BLOCK_K] bf16
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_w_k + offs_e[None, :] * stride_w_e,
            mask=k_mask[:, None], other=0.0,
        )  # [BLOCK_K, E] bf16
        acc += tl.dot(hs, w)                  # bf16 inputs, fp32 accumulate -> [BLOCK_M, E]

    NEG_INF = float("-inf")
    BIG = E                                    # sentinel index for min-index reduction

    # ---- scores (bias-free) and scores_routing (+bias) ----
    scores = tl.sigmoid(acc)                                  # [BLOCK_M, E] fp32
    bias = tl.load(bias_ptr + offs_e).to(tl.float32)          # [E]
    scores_routing = scores + bias[None, :]                   # [BLOCK_M, E]

    # ---- group stage: top-2 sum within each group (fully 2D, static-unrolled) ----
    group_scores = tl.zeros((BLOCK_M, NG), dtype=tl.float32)
    for gg in tl.static_range(NG):
        in_g = expert_group[None, :] == gg                   # [1, E] -> broadcast
        gv = tl.where(in_g, scores_routing, NEG_INF)         # [BLOCK_M, E]
        m1 = tl.max(gv, axis=1)                               # [BLOCK_M] largest in group
        gv2 = tl.where(gv == m1[:, None], NEG_INF, gv)        # suppress the (unique) max
        m2 = tl.max(gv2, axis=1)                              # [BLOCK_M] second largest
        gsum = m1 + m2                                        # [BLOCK_M]
        group_scores += tl.where(offs_g[None, :] == gg, gsum[:, None], 0.0)

    # ---- select top-TOPG groups -> expert-level selection mask (argmax-free) ----
    sel_e = tl.zeros((BLOCK_M, E), dtype=tl.float32)
    gs = group_scores
    for _ in tl.static_range(TOPG):
        gmax = tl.max(gs, axis=1)                             # [BLOCK_M] best group score
        gcand = tl.where(gs == gmax[:, None], offs_g[None, :], NG)
        gi = tl.min(gcand, axis=1)                            # [BLOCK_M] smallest-index best group
        hit_e = expert_group[None, :] == gi[:, None]         # [BLOCK_M, E]
        sel_e = tl.where(hit_e, 1.0, sel_e)
        gs = tl.where(offs_g[None, :] == gi[:, None], NEG_INF, gs)
    masked = tl.where(sel_e > 0.5, scores_routing, NEG_INF)   # [BLOCK_M, E]

    # ---- select top-TOPK experts (argmax-free); capture bias-free companion scores ----
    denom = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sel_vals = []
    for i in tl.static_range(TOPK):
        emax = tl.max(masked, axis=1)                         # [BLOCK_M] best masked score
        ecand = tl.where(masked == emax[:, None], offs_e[None, :], BIG)
        ei = tl.min(ecand, axis=1)                            # [BLOCK_M] smallest-index best expert
        hit_e = offs_e[None, :] == ei[:, None]                # [BLOCK_M, E]
        val_i = tl.sum(tl.where(hit_e, scores, 0.0), axis=1)  # bias-free score at ei
        masked = tl.where(hit_e, NEG_INF, masked)
        denom += val_i
        sel_vals.append(val_i)
        tl.store(
            idx_ptr + offs_m * stride_idx_m + i * stride_idx_n,
            ei.to(tl.int32), mask=m_mask,
        )

    denom = denom + 1e-20
    for i in tl.static_range(TOPK):
        w_i = sel_vals[i] / denom * scaling
        tl.store(
            wt_ptr + offs_m * stride_wt_m + i * stride_wt_n,
            w_i, mask=m_mask,
        )


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

    # Kernel writes int32 indices; convert to the int64 contract in torch plumbing.
    topk_idx32 = torch.empty((T, TOPK), dtype=torch.int32, device=hidden_states.device)
    topk_weight = torch.empty((T, TOPK), dtype=torch.float32, device=hidden_states.device)

    BLOCK_M = 16
    BLOCK_K = 64
    grid = (triton.cdiv(T, BLOCK_M),)

    _routing_kernel[grid](
        hidden_states, weight, expert_bias,
        topk_idx32, topk_weight,
        float(routed_scaling_factor),
        T, D,
        hidden_states.stride(0), hidden_states.stride(1),
        weight.stride(0), weight.stride(1),
        topk_idx32.stride(0), topk_idx32.stride(1),
        topk_weight.stride(0), topk_weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        E=E, NG=NG, EPG=EPG, TOPK=TOPK, TOPG=TOPG,
        num_warps=4, num_stages=2,
    )

    topk_idx = topk_idx32.to(torch.int64)
    return topk_idx, topk_weight
