import torch
import triton
import triton.language as tl


# Kernel 1: GEMM logits = hidden @ weight.T, hidden[M, K], weight[N, K], logits[M, N]
@triton.jit
def _linear_proj_kernel(
    hidden, weight, logits,
    M, K, N,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_lm, stride_ln,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile along M
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, TILE_K):
        offs_k = k0 + tl.arange(0, TILE_K)
        # For N, we don't have grid coverage, but we can loop N in chunks inside the kernel.
        for n0 in range(0, N, TILE_N):
            offs_n = n0 + tl.arange(0, TILE_N)

            # A tile: hidden[offs_m, offs_k]
            a_ptrs = hidden + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk)
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [TILE_M, TILE_K]

            # B tile: weight[offs_n, offs_k]
            b_ptrs = weight + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
            b_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [TILE_N, TILE_K]

            # acc += A @ B^T -> [TILE_M, TILE_N]
            acc += tl.dot(a, tl.trans(b))

        # After processing all N tiles, store results for current M tile
        l_ptrs = logits + (offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln)
        l_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(l_ptrs, acc, mask=l_mask)


# Kernel 2: scores = sigmoid(logits) + expert_bias, elementwise
@triton.jit
def _sigmoid_add_bias_kernel(
    logits, bias, scores,
    M, N,
    stride_lm, stride_ln,
    stride_bm, stride_bn,
    stride_sm, stride_sn,
):
    pid = tl.program_id(0)
    total = M * N
    idx = pid * 1024 + tl.arange(0, 1024)
    mask = idx < total
    # map linear idx to (m, n)
    m = idx // N
    n = idx % N
    # pointers
    log_ptr = logits + m * stride_lm + n * stride_ln
    bias_ptr = bias + n * stride_bn
    score_ptr = scores + m * stride_sm + n * stride_sn
    # load
    log_val = tl.load(log_ptr, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptr, mask=mask, other=0.0)
    # compute
    sig = 1.0 / (1.0 + tl.exp(-log_val))
    val = sig + bias_val
    # store
    tl.store(score_ptr, val, mask=mask)


# Kernel 3: group_top2_sum: from scores[M, N], compute group_scores[M, 8] where each group has 32 experts
@triton.jit
def _group_top2_sum_kernel(
    scores, group_scores,
    M, N,
    EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    t = tl.program_id(0)  # one program per token
    # For each group g in 0..7, compute top-2 within that group and sum
    for g in range(8):
        start = g * EXP_PER_GROUP
        offs = start + tl.arange(0, EXP_PER_GROUP)
        mask = (offs < N) & (t < M)
        ptrs = scores + t * stride_sm + offs * stride_sn
        vals = tl.load(ptrs, mask=mask, other=-1e30)
        # compute top-2
        # First max
        max1 = tl.max(vals, axis=0)
        idx1 = start + tl.argmax(vals, axis=0)
        # Second max (exclude idx1)
        vals2 = tl.where(offs == idx1, -1e30, vals)
        max2 = tl.max(vals2, axis=0)
        group_scores[t, g] = max1 + max2


# Kernel 4: select top-4 groups per token using iterative max-finding (sorted=False)
@triton.jit
def _select_top4_groups_bubble_kernel(
    group_scores, top4_groups,
    M, NUM_GROUPS: tl.constexpr,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)  # one program per token
    # positions 0..NUM_GROUPS-1
    for pos in range(4):
        best_val = -1.0
        best_idx = 0
        for g in range(NUM_GROUPS):
            val = group_scores[t, g]
            if val > best_val:
                best_val = val
                best_idx = g
        # write selected group index
        tl.store(top4_groups + t * stride_tm + pos * stride_tn, best_idx)
        # remove it by setting to -inf
        group_scores[t, best_idx] = -1.0


# Kernel 5: mask non-selected groups: masked_scores[t, g*32:g*32+32] = scores[t, g*32:g*32+32] for selected g; else -inf
@triton.jit
def _mask_nonselected_groups_kernel(
    scores, top4_groups, masked_scores,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
    stride_mm, stride_mn,
):
    t = tl.program_id(0)  # one program per token
    # initialize with -inf
    for n in range(N):
        ptr = masked_scores + t * stride_mm + n * stride_mn
        tl.store(ptr, -1.0)
    # restore selected groups
    for pos in range(4):
        g = tl.load(top4_groups + t * stride_tm + pos * stride_tn)
        start = g * EXP_PER_GROUP
        for n in range(EXP_PER_GROUP):
            src = scores + t * stride_sm + (start + n) * stride_sn
            dst = masked_scores + t * stride_mm + (start + n) * stride_mn
            val = tl.load(src)
            tl.store(dst, val)


# Kernel 6: select top-8 from masked_scores per token, sorted=False
@triton.jit
def _select_top8_masked_kernel(
    masked_scores, top8_indices,
    M, N,
    stride_mm, stride_mn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)  # one program per token
    for p in range(8):
        max_val = -1.0
        max_idx = 0
        for n in range(N):
            ptr = masked_scores + t * stride_mm + n * stride_mn
            val = tl.load(ptr)
            take = val > max_val
            max_idx = tl.where(take, n, max_idx)
            max_val = tl.where(take, val, max_val)
        tl.store(top8_indices + t * stride_tm + p * stride_tn, max_idx)
        # remove selected by setting to -inf
        ptr = masked_scores + t * stride_mm + max_idx * stride_mn
        tl.store(ptr, -1.0)


# Kernel 7: normalize and scale: topk_weight = (selected_scores / sum(selected)+eps) * routed_scaling_factor
@triton.jit
def _normalize_and_scale_kernel(
    masked_scores, top8_indices, routed_scaling_factor, topk_weight,
    M, N,
    stride_mm, stride_mn,
    stride_tm, stride_tn,
    eps: tl.constexpr,
):
    t = tl.program_id(0)  # one program per token
    denom = 0.0
    for p in range(8):
        idx = tl.load(top8_indices + t * stride_tm + p * stride_tn)
        val = tl.load(masked_scores + t * stride_mm + idx * stride_mn)
        denom += val
    denom = denom + eps
    for p in range(8):
        idx = tl.load(top8_indices + t * stride_tm + p * stride_tn)
        val = tl.load(masked_scores + t * stride_mm + idx * stride_mn)
        w = val / denom
        w = w * routed_scaling_factor
        ptr = topk_weight + t * stride_tm + p * stride_tn
        tl.store(ptr, w)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only forward. All computation is done in Triton kernels.
        Inputs:
          hidden_states: [num_tokens, 256]
          weight: [256, 256]
          expert_bias: [256]
          routed_scaling_factor: float
        Outputs:
          topk_idx: [num_tokens, 8], int32
          topk_weight: [num_tokens, 8], float32
        """
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]

        # Ensure dtypes and contiguity for Triton
        hidden = hidden_states.contiguous().to(torch.float32)
        weight_t = weight.contiguous().to(torch.float32)  # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)

        # 1) GEMM logits = hidden @ weight.T, logits[M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        TILE_M, TILE_N, TILE_K = 128, 32, 64
        grid = (triton.cdiv(M, TILE_M),)  # one program per tile along M; loop over N inside kernel
        _linear_proj_kernel[grid](
            hidden, weight_t, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            logits.stride(0), logits.stride(1),
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) scores = sigmoid(logits) + expert_bias
        scores = torch.empty_like(logits, dtype=torch.float32, device=device)
        grid_sigmoid = (M * N,)
        _sigmoid_add_bias_kernel[grid_sigmoid](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            bias.stride(0), bias.stride(0),
            scores.stride(0), scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 3) group_scores per token
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_gm=group_scores.stride(0), stride_gn=group_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) select top-4 groups per token
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        _select_top4_groups_bubble_kernel[(M,)](
            group_scores, top4_groups,
            M, NUM_GROUPS=8,
            stride_gm=group_scores.stride(0), stride_gn=group_scores.stride(1),
            stride_tm=top4_groups.stride(0), stride_tn=top4_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) mask non-selected groups
        masked_scores = torch.empty_like(scores)
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_tm=top4_groups.stride(0), stride_tn=top4_groups.stride(1),
            stride_mm=masked_scores.stride(0), stride_mn=masked_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) select top-8 from masked_scores
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_mm=masked_scores.stride(0), stride_mn=masked_scores.stride(1),
            stride_tm=top8_indices.stride(0), stride_tn=top8_indices.stride(1),
            num_warps=1, num_stages=1,
        )

        # 7) normalize and scale
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        eps = 1e-20
        _normalize_and_scale_kernel[(M,)](
            masked_scores, top8_indices, routed_scaling_factor, topk_weight,
            M, N,
            stride_mm=masked_scores.stride(0), stride_mn=masked_scores.stride(1),
            stride_tm=top8_indices.stride(0), stride_tn=top8_indices.stride(1),
            eps=eps,
            num_warps=1, num_stages=1,
        )

        # Return indices and weights (indices are int32 in kernels, cast to int64 for parity with original)
        topk_idx = top8_indices.to(torch.int64)
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
