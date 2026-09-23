"""L2/049 group_limited_topk_routing — Triton solution.

Candidate c001: faithful fused correctness anchor.

Semantics (see docs/draft.md / docs/plan.md), all constants fixed by task/definition.json:
  E=256 experts, n_group=8, experts_per_group=32, topk_group=4, top_k=8, K=4096.

Pipeline, fused into one Triton kernel per block of tokens:
  1. logits = hidden @ weight.T            (bf16 tl.dot, fp32 accumulate)
  2. scores = sigmoid(logits)              (fp32; bias-free, used for weights)
  3. scores_for_routing = scores + bias    (fp32; drives ALL selection)
  4. group score = (top-2 within group).sum()          -> [BM, 8]
  5. select top-4 groups (set) -> group mask -> expert mask [BM, 256]
  6. mask non-selected experts to fp32-min, take top-8 experts
  7. gather bias-free scores at the 8 indices, normalize, * routed_scaling_factor

Numerical note: hidden_states and weight are bf16 (8-bit mantissa), so each product
is exactly representable in fp32 and a bf16 tensor-core dot with fp32 accumulation
matches the reference fp32 matmul to within accumulation-order noise. All top-k ties
are broken by lowest index to mirror torch.topk / CUDA argmax.
"""

import torch
import triton
import triton.language as tl

# fp32 min sentinel (== torch.finfo(torch.float32).min), used to mask out candidates.
_NEG = -3.4028234663852886e38


@triton.jit
def _route_kernel(
    hidden_ptr,          # [T, K] bf16
    weight_ptr,          # [E, K] bf16
    bias_ptr,            # [E]    bf16
    out_idx_ptr,         # [T, TOP_K] int64
    out_w_ptr,           # [T, TOP_K] float32
    scaling,             # fp32 scalar (routed_scaling_factor)
    T,
    stride_hm, stride_hk,
    stride_we, stride_wk,
    K: tl.constexpr,
    N: tl.constexpr,               # num_experts = 256
    N_GROUP: tl.constexpr,         # 8
    GROUP_SZ: tl.constexpr,        # experts_per_group = 32
    TOPK_GROUP: tl.constexpr,      # 4
    TOP_K: tl.constexpr,           # 8
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)      # [BM]
    m_mask = offs_m < T
    offs_e = tl.arange(0, N)                            # [256]

    # ---- projection GEMM: logits[BM, N] = hidden[BM, K] @ weight[E, K].T ----
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptr = hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk
        a = tl.load(a_ptr, mask=m_mask[:, None], other=0.0)          # [BM, BK] bf16
        b_ptr = weight_ptr + offs_e[None, :] * stride_we + offs_k[:, None] * stride_wk
        b = tl.load(b_ptr)                                            # [BK, N] bf16
        acc += tl.dot(a, b)                                           # fp32 accumulate

    # ---- scores ----
    scores = tl.sigmoid(acc)                                          # [BM, N] fp32, bias-free
    bias = tl.load(bias_ptr + offs_e).to(tl.float32)                  # [N]
    sr = scores + bias[None, :]                                       # scores_for_routing

    # ---- group top-2 sum -> group_scores[BM, N_GROUP] ----
    sr3 = tl.reshape(sr, (BLOCK_M, N_GROUP, GROUP_SZ))
    gcol = tl.arange(0, GROUP_SZ)[None, None, :]                      # [1,1,32]
    max1 = tl.max(sr3, axis=2, keep_dims=True)                        # [BM,8,1]
    is1 = sr3 == max1
    idx1 = tl.min(tl.where(is1, gcol, GROUP_SZ), axis=2, keep_dims=True)  # lowest-index max
    sr3_wo1 = tl.where(gcol == idx1, _NEG, sr3)                       # remove exactly one
    max2 = tl.max(sr3_wo1, axis=2, keep_dims=True)                    # [BM,8,1]
    group_scores = tl.reshape(max1 + max2, (BLOCK_M, N_GROUP))        # [BM,8]

    # ---- select top-4 groups (as a set) -> group_mask[BM, N_GROUP] ----
    gcol8 = tl.arange(0, N_GROUP)[None, :]                            # [1,8]
    gs = group_scores
    group_mask = tl.zeros((BLOCK_M, N_GROUP), dtype=tl.float32)
    for _ in range(TOPK_GROUP):
        gm = tl.max(gs, axis=1, keep_dims=True)                       # [BM,1]
        isg = gs == gm
        gidx = tl.min(tl.where(isg, gcol8, N_GROUP), axis=1, keep_dims=True)
        selg = gcol8 == gidx                                         # [BM,8]
        group_mask = tl.where(selg, 1.0, group_mask)
        gs = tl.where(selg, _NEG, gs)

    # ---- expand group mask to expert level, mask out non-selected experts ----
    emask3 = tl.broadcast_to(
        tl.reshape(group_mask, (BLOCK_M, N_GROUP, 1)),
        (BLOCK_M, N_GROUP, GROUP_SZ),
    )
    emask = tl.reshape(emask3, (BLOCK_M, N))                          # [BM,256]
    masked = tl.where(emask > 0.5, sr, _NEG)                         # [BM,256]

    # ---- expert top-8 (iterative argmax, lowest-index tie-break) ----
    ecol = tl.arange(0, N)[None, :]                                  # [1,256]
    ms = masked
    idx_list = []
    w_list = []
    wsum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for _ in range(TOP_K):
        em = tl.max(ms, axis=1, keep_dims=True)                      # [BM,1]
        ise = ms == em
        eidx = tl.min(tl.where(ise, ecol, N), axis=1)                # [BM] lowest-index max
        sele = ecol == eidx[:, None]                                 # [BM,256]
        wj = tl.sum(tl.where(sele, scores, 0.0), axis=1)             # [BM] bias-free score
        idx_list.append(eidx)
        w_list.append(wj)
        wsum += wj
        ms = tl.where(sele, _NEG, ms)

    inv = scaling / (wsum + 1e-20)                                   # [BM]
    for j in range(TOP_K):
        out_off = offs_m * TOP_K + j
        tl.store(out_idx_ptr + out_off, idx_list[j].to(tl.int64), mask=m_mask)
        tl.store(out_w_ptr + out_off, w_list[j] * inv, mask=m_mask)


def run(hidden_states, weight, expert_bias, routed_scaling_factor):
    """Group-limited top-k expert routing. Triton-only compute; torch for plumbing."""
    assert hidden_states.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16

    hidden_states = hidden_states.contiguous()
    weight = weight.contiguous()
    expert_bias = expert_bias.contiguous()

    T, K = hidden_states.shape
    E = weight.shape[0]

    # Task-fixed routing constants.
    N_GROUP = 8
    GROUP_SZ = E // N_GROUP        # 32
    TOPK_GROUP = 4
    TOP_K = 8

    device = hidden_states.device
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int64, device=device)
    topk_weight = torch.empty((T, TOP_K), dtype=torch.float32, device=device)

    BLOCK_M = 32
    BLOCK_K = 64

    grid = (triton.cdiv(T, BLOCK_M),)
    _route_kernel[grid](
        hidden_states, weight, expert_bias,
        topk_idx, topk_weight,
        float(routed_scaling_factor),
        T,
        hidden_states.stride(0), hidden_states.stride(1),
        weight.stride(0), weight.stride(1),
        K=K,
        N=E,
        N_GROUP=N_GROUP,
        GROUP_SZ=GROUP_SZ,
        TOPK_GROUP=TOPK_GROUP,
        TOP_K=TOP_K,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return topk_idx, topk_weight
