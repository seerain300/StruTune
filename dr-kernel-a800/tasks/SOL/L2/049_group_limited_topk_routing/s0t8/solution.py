import torch
import triton
import triton.language as tl


# 1) Triton GEMM: logits = hidden @ weight.T
# hidden: [M, K], weight: [N, K] (PyTorch weight is [num_experts, hidden_dim])
@triton.jit
def _linear_proj_kernel(
    hidden_ptr, weight_ptr, logits_ptr,
    M, K, N,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_lm, stride_ln,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # Accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, TILE_K):
        a_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + (k + offs_k[None, :]) * stride_hk
        b_ptrs = weight_ptr + offs_n[None, :] * stride_wn + (k + offs_k[:, None]) * stride_wk
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (k + offs_k[:, None] < K), other=0.0)
        # acc += a @ b
        acc += tl.dot(a, b)

    # Write back
    out_ptrs = logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 2) Sigmoid + expert bias
@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_lm, stride_ln,
    stride_bn,
    stride_om, stride_on,
):
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    if m >= M:
        return
    l = tl.load(logits_ptr + m * stride_lm + n * stride_ln)
    b = tl.load(bias_ptr + n * stride_bn)
    s = 1.0 / (1.0 + tl.exp(-l)) + b
    tl.store(scores_ptr + m * stride_om + n * stride_on, s)


# 3) Group top-2 sum per token
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N,
    EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    for g in range(8):
        start = g * EXP_PER_GROUP
        idx = start + tl.arange(0, EXP_PER_GROUP)
        vals = tl.load(scores_ptr + t * stride_sm + idx * stride_sn, mask=(idx < N), other=-float('inf'))
        # top-2 via reductions
        top1 = tl.max(vals, axis=0)
        mask1 = vals == top1
        # exclude top1 by setting it to -inf then take max again
        vals2 = tl.where(mask1, -float('inf'), vals)
        top2 = tl.max(vals2, axis=0)
        tl.store(group_scores_ptr + t * stride_gm + g * stride_gn, top1 + top2)


# 4) Select top-4 groups per token (bubble-like: find max, mark, repeat 4 times)
@triton.jit
def _select_top4_groups_bubble_kernel(
    group_scores_ptr, top4_groups_ptr,
    M,
    stride_gm, stride_gn,
    num_groups: tl.constexpr,  # 8
):
    t = tl.program_id(0)
    if t >= M:
        return
    best_vals = tl.zeros((num_groups,), dtype=tl.float32) - float('inf')
    best_idxs = tl.zeros((num_groups,), dtype=tl.int32)
    # Perform selection of 4 groups
    for r in range(4):
        # find max among remaining
        for g in range(num_groups):
            val = tl.load(group_scores_ptr + t * stride_gm + g * stride_gn)
            if val > best_vals[0]:
                best_vals = [val] + list(best_vals[:-1])
                best_idxs = [g] + list(best_idxs[:-1])
        # mark as used by setting to -inf (so not selected again)
        tl.store(group_scores_ptr + t * stride_gm + best_idxs[0] * stride_gn, -float('inf'))
    # store top 4 indices
    for r in range(4):
        tl.store(top4_groups_ptr + t * 4 + r, best_idxs[r])


# 5) Mask non-selected groups: set masked_scores to -inf for non-selected groups
@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr, top4_groups_ptr, masked_scores_ptr,
    M, N,
    EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_mm, stride_mn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    for g in range(8):
        start = g * EXP_PER_GROUP
        # check if this group is in top4 for token t
        for r in range(4):
            group_idx = tl.load(top4_groups_ptr + t * 4 + r)
            if g == group_idx:
                # keep scores in this group, do nothing
                break
        else:
            # not selected, set to -inf
            idx = start + tl.arange(0, EXP_PER_GROUP)
            vals = tl.load(scores_ptr + t * stride_sm + idx * stride_sn, mask=(idx < N), other=0.0)
            tl.store(masked_scores_ptr + t * stride_mm + idx * stride_mn, -float('inf'), mask=(idx < N))


# 6) Select top-8 from masked scores (iteratively select max 8 times)
@triton.jit
def _select_top8_masked_kernel(
    masked_scores_ptr, top8_indices_ptr,
    M, N,
    stride_mm, stride_mn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    for r in range(8):
        best_val = -float('inf')
        best_idx = 0
        for n in range(N):
            val = tl.load(masked_scores_ptr + t * stride_mm + n * stride_mn)
            if val > best_val:
                best_val = val
                best_idx = n
        # mark as used by setting to -inf
        tl.store(masked_scores_ptr + t * stride_mm + best_idx * stride_mn, -float('inf'))
        tl.store(top8_indices_ptr + t * 8 + r, best_idx)


# 7) Normalize and scale selected scores (write normalized weight)
@triton.jit
def _normalize_and_scale_kernel(
    masked_scores_ptr, top8_indices_ptr, selected_weight_ptr,
    M, N, routed_scaling_factor,
    stride_mm, stride_mn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    total = 0.0
    for r in range(8):
        idx = tl.load(top8_indices_ptr + t * 8 + r)
        val = tl.load(masked_scores_ptr + t * stride_mm + idx * stride_mn)
        total += val
    for r in range(8):
        idx = tl.load(top8_indices_ptr + t * 8 + r)
        val = tl.load(masked_scores_ptr + t * stride_mm + idx * stride_mn)
        norm = val / (total + 1e-20) * routed_scaling_factor
        tl.store(selected_weight_ptr + t * 8 + r, norm)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # hidden_states: [M, K], weight: [N, K], expert_bias: [N]
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]
        assert weight.shape[1] == K, "weight second dim must equal hidden_dim"
        assert expert_bias.shape[0] == N, "expert_bias must match num_experts"

        # Ensure dtype float32 for kernels
        hidden_contig = hidden_states.contiguous().to(torch.float32)
        weight_contig = weight.contiguous().to(torch.float32)
        expert_bias_f32 = expert_bias.contiguous().to(torch.float32)

        # 1) Triton GEMM: logits = hidden @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_hm, stride_hk = hidden_contig.stride()
        stride_wn, stride_wk = weight_contig.stride()
        stride_lm, stride_ln = logits.stride()
        TILE_M = 64
        TILE_N = 32
        TILE_K = 64
        grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
        _linear_proj_kernel[grid](
            hidden_contig, weight_contig, logits,
            M, K, N,
            stride_hm, stride_hk,
            stride_wn, stride_wk,
            stride_lm, stride_ln,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + expert bias (Triton)
        scores = torch.empty_like(logits)
        stride_sm, stride_sn = logits.stride()
        stride_om, stride_on = scores.stride()
        _sigmoid_add_bias_kernel[(M * N,)](
            logits, expert_bias_f32, scores,
            M, N,
            stride_sm, stride_sn,
            expert_bias_f32.stride(0),
            stride_om, stride_on,
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 sum per token
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_gm, stride_gn = group_scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        _select_top4_groups_bubble_kernel[(M,)](
            group_scores, top4_groups,
            M,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_groups=8,
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups
        masked_scores = torch.empty_like(scores)
        stride_mm, stride_mn = masked_scores.stride()
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=1, num_stages=1,
        )

        # 6) Select top-8 from masked scores
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=1, num_stages=1,
        )

        # 7) Normalize and scale
        selected_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        _normalize_and_scale_kernel[(M,)](
            masked_scores, top8_indices, selected_weight,
            M, N, routed_scaling_factor,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=1, num_stages=1,
        )

        # Return topk_idx and topk_weight. Note: original code returns
        # topk_idx as indices and topk_weight as normalized and scaled selected scores.
        # Here, top8_indices are the selected expert indices; selected_weight are the
        # normalized and scaled values. However, original code returns topk_idx and
        # topk_weight as two outputs. We will return top8_indices as topk_idx and
        # selected_weight as topk_weight.
        # Cast top8_indices to long for consistency
        topk_idx = top8_indices.to(torch.long)
        # selected_weight already computed in Triton; ensure dtype torch.float32
        topk_weight = selected_weight

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
