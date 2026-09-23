import torch
import torch.nn as nn

import triton
import triton.language as tl

# 1) GEMM: logits = hidden @ weight_T (weight_T has shape [K, N], i.e., [hidden_dim, num_experts])
@triton.jit
def _gemm_linear_kernel(
    hidden_ptr,      # [M, K]
    weightT_ptr,     # [K, N]
    logits_ptr,      # [M, N]
    M, K, N,
    stride_hm, stride_hk,
    stride_wk, stride_wn,
    stride_lm, stride_ln,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    m_mask = offs_m < M
    n_mask = offs_n < N
    k_mask = offs_k < K

    # Initialize accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    # Note: dynamic loop over K is okay since offs_k and k_mask are computed per program
    while True:
        k_mask_curr = k_mask & (offs_k < K)
        # Load A tile: hidden[m, k]
        a = tl.load(hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
                    mask=m_mask[:, None] & k_mask_curr[None, :], other=0.0)
        # Load B tile: weightT[k, n]
        b = tl.load(weightT_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                    mask=k_mask_curr[:, None] & n_mask[None, :], other=0.0)
        # Accumulate
        acc += tl.dot(a, b)
        # Advance k
        offs_k += TILE_K
        k_mask = offs_k < K
        # Break if no more K
        if not tl.any(k_mask):
            break

    # Store results
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln, acc, mask=store_mask)


# 2) Elementwise: scores = sigmoid(logits) + expert_bias
@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr,      # [M, N], float32
    bias_ptr,        # [N], float32
    scores_ptr,      # [M, N], float32
    M, N,
    stride_lm, stride_ln,
    stride_bn,
    stride_sm, stride_sn,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Load logits tile
    logits = tl.load(logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
                     mask=m_mask[:, None] & n_mask[None, :], other=0.0)
    # Load bias per column
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)  # [64]
    # Broadcast bias over rows
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias[None, :]
    # Store
    tl.store(scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn,
             scores, mask=m_mask[:, None] & n_mask[None, :])


# 3) Group top-2 sum per token: [M, 8] = sum(top2) per group of 32 experts
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,      # [M, N], float32
    group_scores_ptr,  # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    base = t * N  # row base for scores_ptr
    # Iterate over 8 groups, each of size 32
    for g in range(8):
        start = g * 32
        # Load 32 values
        idx = tl.arange(0, 32)
        vals = tl.load(scores_ptr + base + start + idx, mask=idx < 32, other=-float('inf'))
        # Compute top-2 (unsorted)
        max0 = tl.max(vals, axis=0)
        vals2 = tl.where(vals == max0, -float('inf'), vals)
        max1 = tl.max(vals2, axis=0)
        sum2 = max0 + max1
        tl.store(group_scores_ptr + t * stride_gm + g * stride_gn, sum2)


# 4) Select top-4 groups per token: write indices [M, 4], int32
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # [M, 8], float32
    top4_groups_ptr,   # [M, 4], int32
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    # Bubble selection: pick 4 maxima
    for r in range(4):
        maxv = -float('inf')
        max_idx = -1
        for g in range(8):
            v = tl.load(group_scores_ptr + t * stride_gm + g * stride_gn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, g, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top4_groups_ptr + t * stride_tm + r * stride_tn, max_idx)


# 5) Mask non-selected groups to -inf in scores_ptr (i.e., set scores for non-selected groups to -inf)
@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr,          # [M, N], float32
    top4_groups_ptr,     # [M, 4], int32
    M, N,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    # Iterate groups 0..7; set to -inf if not in top4
    for g in range(8):
        is_selected = 0
        for r in range(4):
            idx = tl.load(top4_groups_ptr + t * stride_tm + r * stride_tn)
            if idx == g:
                is_selected = 1
                break
        start = g * 32
        idx32 = tl.arange(0, 32)
        # For each selected, leave as is; for non-selected, set to -inf
        # Compare is_selected (scalar 0/1) with scalar; vectorized over 32 columns
        neg = -float('inf')
        vals = tl.load(scores_ptr + t * stride_sm + start + idx32, mask=idx32 < 32, other=0.0)
        new_vals = tl.where(is_selected == 1, vals, neg)
        tl.store(scores_ptr + t * stride_sm + start + idx32, new_vals, mask=idx32 < 32)


# 6) Select top-8 from masked scores: write indices [M, 8], int32
@triton.jit
def _select_top8_masked_kernel(
    scores_ptr,          # [M, N], float32 (after masking)
    top8_idx_ptr,        # [M, 8], int32
    M, N,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    # Iteratively select maxima 8 times
    for r in range(8):
        maxv = -float('inf')
        max_idx = -1
        for n in range(N):
            v = tl.load(scores_ptr + t * stride_sm + n * stride_sn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, n, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top8_idx_ptr + t * stride_tm + r * stride_tn, max_idx)


# 7) Normalize selected scores and apply routed_scaling_factor: output [M, 8], float32
@triton.jit
def _normalize_and_scale_kernel(
    scores_ptr,          # [M, N], float32 (masked and post-selected)
    top8_idx_ptr,        # [M, 8], int32
    out_ptr,             # [M, 8], float32
    M, N,
    routed_scale,        # float32
    stride_sm, stride_sn,
    stride_tm, stride_tn,
    out_stride_m, out_stride_n,
):
    t = tl.program_id(0)
    if t >= M:
        return
    total = 0.0
    for r in range(8):
        idx = tl.load(top8_idx_ptr + t * stride_tm + r * stride_tn)
        v = tl.load(scores_ptr + t * stride_sm + idx * stride_sn)
        total += v
    inv = 1.0 / (total + 1e-20)
    for r in range(8):
        idx = tl.load(top8_idx_ptr + t * stride_tm + r * stride_tn)
        v = tl.load(scores_ptr + t * stride_sm + idx * stride_sn)
        scaled = v * inv * routed_scale
        tl.store(out_ptr + t * out_stride_m + r * out_stride_n, scaled)


# ModelNew forward: Triton-only implementation
class ModelNew(nn.Module):
    def __init__(self, routed_scaling_factor: float):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Shapes
        M, K = hidden_states.shape  # M = num_tokens, K = hidden_dim
        N = weight.shape[0]         # num_experts

        device = hidden_states.device
        dtype = torch.float32

        # 1) Compute logits = hidden @ weight.T using Triton
        hidden_contig = hidden_states.contiguous().to(dtype)
        weightT = weight.contiguous().to(dtype).transpose(0, 1)  # [K, N]
        logits = torch.empty((M, N), device=device, dtype=dtype)
        stride_hm, stride_hk = hidden_contig.stride()
        stride_wk, stride_wn = weightT.stride()  # note: weightT is [K, N]
        stride_lm, stride_ln = logits.stride()
        TILE_M, TILE_N, TILE_K = 64, 64, 64
        grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N), triton.cdiv(K, TILE_K))
        _gemm_linear_kernel[grid](
            hidden_contig, weightT, logits,
            M, K, N,
            stride_hm, stride_hk,
            stride_wk, stride_wn,
            stride_lm, stride_ln,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) scores = sigmoid(logits) + expert_bias (Triton)
        scores = torch.empty_like(logits)
        stride_sm, stride_sn = logits.stride()
        bias_f32 = expert_bias.contiguous().to(dtype)
        stride_bn = bias_f32.stride(0)
        grid_elem = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _sigmoid_add_bias_kernel[grid_elem](
            logits, bias_f32, scores,
            M, N,
            stride_lm=stride_sm, stride_ln=stride_sn,
            stride_bn=stride_bn,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            num_warps=2, num_stages=1,
        )

        # 3) Group top-2 sum per token: [M, 8]
        group_scores = torch.empty((M, 8), device=device, dtype=dtype)
        stride_gm, stride_gn = group_scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token: [M, 4], int32
        top4_groups = torch.empty((M, 4), device=device, dtype=torch.int32)
        stride_tm, stride_tn = top4_groups.stride()
        _select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M,
            stride_gm=stride_gm, stride_gn=stride_gn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups: set scores of non-selected groups to -inf
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups,
            M, N,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1, num_stages=1,
        )

        # 6) Select top-8 from masked scores: [M, 8], int32
        top8_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        out_stride_m, out_stride_n = top8_idx.stride()
        _select_top8_masked_kernel[(M,)](
            scores, top8_idx,
            M, N,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_tm=out_stride_m, stride_tn=out_stride_n,
            num_warps=1, num_stages=1,
        )

        # 7) Normalize and scale: [M, 8], float32
        final_weights = torch.empty((M, 8), device=device, dtype=dtype)
        _normalize_and_scale_kernel[(M,)](
            scores, top8_idx, final_weights,
            M, N,
            self.routed_scaling_factor,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_tm=out_stride_m, stride_tn=out_stride_n,
            out_stride_m=final_weights.stride(0), out_stride_n=final_weights.stride(1),
            num_warps=1, num_stages=1,
        )

        # Prepare indices output: cast to int64
        topk_idx = top8_idx.to(torch.int64)

        return topk_idx, final_weights


def run(*args):
    return ModelNew()(*args)
