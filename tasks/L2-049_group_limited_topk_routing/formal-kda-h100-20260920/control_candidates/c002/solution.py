"""L2/049 group_limited_topk_routing — Triton solution.

Candidate c002: 2D-only rewrite of the fused correctness anchor.

c001 (single fused kernel, same semantics) failed to compile on all 16 shapes
(uniform RUNTIME_ERROR, no traceback) because its routing epilogue used
version-sensitive constructs: 3D tl.reshape([BM,256]->[BM,8,32]),
tl.max(axis=2, keep_dims=True) over a 3D tile, and tl.broadcast_to for the
group->expert mask. c002 removes all 3D tiles and expresses group top-2, top-4
group selection, and group->expert mask expansion with pure 2D ops using
group-id arithmetic (gid = ecol // 32) and softmax-style [:, None] broadcasts.

Semantics (task/definition.json), constants fixed:
  E=256 experts, n_group=8, experts_per_group=32, topk_group=4, top_k=8, K=4096.
Pipeline per block of tokens:
  1. logits = hidden @ weight.T     (fp32 tl.dot, input_precision="tf32", fp32 accum)
  2. scores = sigmoid(logits)       (fp32; bias-free, used for returned weights)
  3. scores_for_routing = scores + bias   (fp32; drives ALL selection)
  4. group score = (top-2 within group).sum()          -> [BM, 8]
  5. select top-4 groups (set) -> expert keep mask [BM, 256]
  6. mask non-kept experts to fp32-min, take top-8 experts (lowest-index ties)
  7. gather bias-free scores at the 8 indices, normalize, * routed_scaling_factor

Precision: inputs are bf16; cast to fp32 and use tf32 tensor-core dot. tf32 (10
mantissa bits) is more accurate than a bf16 dot (8 bits) while still using
tensor cores, giving a safe-but-fast anchor. Lower precision (bf16) is a later
tuning axis if this passes.
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

    # ---- projection GEMM: logits[BM, N] = hidden[BM, K] @ weight[E, K].T ----
    offs_e = tl.arange(0, N)                            # [256]
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptr = hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk
        a = tl.load(a_ptr, mask=m_mask[:, None], other=0.0).to(tl.float32)   # [BM, BK]
        b_ptr = weight_ptr + offs_e[None, :] * stride_we + offs_k[:, None] * stride_wk
        b = tl.load(b_ptr).to(tl.float32)                                    # [BK, N]
        acc += tl.dot(a, b, input_precision="tf32")                          # fp32 accumulate

    # ---- scores ----
    scores = tl.sigmoid(acc)                                # [BM, N] fp32, bias-free
    bias = tl.load(bias_ptr + offs_e).to(tl.float32)        # [N]
    sr = scores + bias[None, :]                             # scores_for_routing [BM, N]

    ecol = tl.arange(0, N)[None, :]                         # [1, 256]
    gid = ecol // GROUP_SZ                                  # [1, 256] group id per expert
    col8 = tl.arange(0, N_GROUP)[None, :]                   # [1, 8]

    # ---- group top-2 sum -> group_scores[BM, N_GROUP] (2D only) ----
    group_scores = tl.zeros((BLOCK_M, N_GROUP), dtype=tl.float32)
    for g in range(N_GROUP):
        in_g = gid == g                                     # [1, 256]
        mg = tl.where(in_g, sr, _NEG)                       # [BM, 256]
        m1 = tl.max(mg, axis=1)                             # [BM]
        is1 = mg == m1[:, None]
        idx1 = tl.min(tl.where(is1, ecol, N), axis=1)       # [BM] lowest-index max
        mg2 = tl.where(ecol == idx1[:, None], _NEG, mg)     # drop exactly one element
        m2 = tl.max(mg2, axis=1)                            # [BM]
        gsum = (m1 + m2)[:, None]                           # [BM, 1]
        group_scores = tl.where(col8 == g, gsum, group_scores)

    # ---- select top-4 groups (set) -> group_keep[BM, N_GROUP] ----
    gs = group_scores
    group_keep = tl.zeros((BLOCK_M, N_GROUP), dtype=tl.float32)
    for _ in range(TOPK_GROUP):
        gm = tl.max(gs, axis=1)                             # [BM]
        isg = gs == gm[:, None]
        gidx = tl.min(tl.where(isg, col8, N_GROUP), axis=1) # [BM] lowest-index max group
        selg = col8 == gidx[:, None]                        # [BM, 8]
        group_keep = tl.where(selg, 1.0, group_keep)
        gs = tl.where(selg, _NEG, gs)

    # ---- expand group_keep -> expert keep mask [BM, 256] (2D only) ----
    keep_expert = tl.zeros((BLOCK_M, N), dtype=tl.float32)
    for g in range(N_GROUP):
        gk_g = tl.max(tl.where(col8 == g, group_keep, 0.0), axis=1)   # [BM] keep of group g
        keep_expert = tl.where(gid == g, gk_g[:, None], keep_expert)

    masked = tl.where(keep_expert > 0.5, sr, _NEG)          # [BM, 256]

    # ---- expert top-8 (iterative argmax, lowest-index tie-break) ----
    ms = masked
    idx_list = []
    w_list = []
    wsum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for _ in range(TOP_K):
        em = tl.max(ms, axis=1)                             # [BM]
        ise = ms == em[:, None]
        eidx = tl.min(tl.where(ise, ecol, N), axis=1)       # [BM] lowest-index max
        sele = ecol == eidx[:, None]                        # [BM, 256]
        wj = tl.sum(tl.where(sele, scores, 0.0), axis=1)    # [BM] bias-free score
        idx_list.append(eidx)
        w_list.append(wj)
        wsum += wj
        ms = tl.where(sele, _NEG, ms)

    inv = scaling / (wsum + 1e-20)                          # [BM]
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
