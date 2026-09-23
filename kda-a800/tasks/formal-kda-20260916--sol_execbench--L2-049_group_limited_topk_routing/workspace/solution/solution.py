"""
Solution for SOL-ExecBench L2/049 group_limited_topk_routing (A800 / sm_80).

Candidate: c013  (parent c012)  --  deeper GEMM tiling (BLOCK_M=64, stages=3)
================================================================================
Champion progression (all 5/5 correct):
  c010  0.8592x  (exact fp32 GEMM, correct order established)
  c011  1.7525x  (bf16 HMMA GEMM revert -- precision proven comparison-irrelevant)
  c012  3.1957x  (GEMM tiling BLOCK_M=32, BLOCK_K=64, num_stages=2, num_warps=8;
                  also proved the >48KB dynamic-shared-memory opt-in works here)

Runtime is GEMM-bound and c012 still had headroom. c013 pushes the GEMM tiling one
more step (perf-only, correctness-neutral: bf16 logits identical up to fp32
summation order):
    BLOCK_M=64, BLOCK_K=64, num_stages=3, num_warps=8.
  * BLOCK_M 32->64: 64 tokens/program -> larger M tiles, higher tensor-core
    utilization, fewer programs.
  * num_stages 2->3: deeper software pipeline to better overlap global loads with
    HMMA compute across the 4096-long K loop.
  * BLOCK_K=64, num_warps=8 unchanged from c012.
Shared memory: B-tile [64,256] bf16 = 32KB; x3 stages ~= 96-100KB -> well into the
dynamic-shared-memory opt-in regime that c012 already proved works on this A800.
Risk: BLOCK_M=64 raises the [64,256] fp32 accumulator (64KB regs) + smem pressure,
which could cut occupancy or spill. If c013 is slower than c012, that is the signal
to stop tiling and pivot to the structural win (re-fusion / routing-kernel tuning);
if faster, continue.

The routing kernel B is byte-identical to c011/c012 (exact fp32 epilogue, torch.topk
output order). Only Kernel A's launch parameters differ.

Two score arrays per token: scores = sigmoid(logits) (returned-weight source, NO
bias); scores_routing = scores + bias (drives ALL three selections). All epilogue
math is fp32; the GEMM multiplies in bf16 tensor cores with fp32 accumulate. Triton
owns all computation; PyTorch is used only for allocation / launch plumbing, a
layout-only weight transpose, and a trivial int32->int64 result cast. No Torch/CPU/
NumPy/CUDA-extension computational fallback exists.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A: gate GEMM  logits[T, E] = hidden_states[T, D] @ weightT[D, E]
#   bf16 HMMA: bf16 operands, fp32 accumulate (fast tensor-core path).
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
        acc += tl.dot(a, b)                        # bf16 HMMA, fp32 accumulate -> [BLOCK_M, E]

    tl.store(
        logits_ptr + offs_m[:, None] * stride_lo_m + offs_e[None, :] * stride_lo_e,
        acc, mask=m_mask[:, None],
    )


# ---------------------------------------------------------------------------
# Kernel B: per-token routing over logits[E].  grid = (T,).
#   Output order reproduces torch.topk(sorted=False): the 7 strictly-largest
#   experts by ascending index, then the min-score selected expert last.
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

    # ---- top-TOPK experts (DESCENDING by masked score); [TOPK] register vectors ----
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

    # ---- reorder to torch.topk(sorted=False) gatherTopK order ----
    # picks i=0..TOPK-2 (strictly-larger) -> slots 0..TOPK-2 by ASCENDING index.
    # pick  i=TOPK-1    (min score = kth) -> slot  TOPK-1 (last).
    j_first = offs_k[None, :] < (TOPK - 1)                          # [1,TOPK] j in first 7
    lt = sel_idx[None, :] < sel_idx[:, None]                        # [i,j] = idx[j] < idx[i]
    cmp = (lt & j_first).to(tl.int32)                               # restrict j to first 7
    rank_first = tl.sum(cmp, axis=1)                               # [TOPK] asc-index rank among first 7
    is_last = offs_k == (TOPK - 1)                                  # [TOPK]
    rank = tl.where(is_last, TOPK - 1, rank_first)                  # [TOPK] destination slot

    tl.store(idx_ptr + pid * stride_idx_m + rank * stride_idx_n, sel_idx)
    tl.store(wt_ptr + pid * stride_wt_m + rank * stride_wt_n, w_vec)


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

    # ---- Kernel A: bf16 HMMA gate GEMM -> logits[T, E]  (deeper-pipelined tiles) ----
    BLOCK_M_A = 64
    BLOCK_K = 64
    grid_a = (triton.cdiv(T, BLOCK_M_A),)
    _gemm_kernel[grid_a](
        hidden_states, weight_t, logits,
        T, D,
        hidden_states.stride(0), hidden_states.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        logits.stride(0), logits.stride(1),
        BLOCK_M=BLOCK_M_A, BLOCK_K=BLOCK_K, E=E,
        num_warps=8, num_stages=3,
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
