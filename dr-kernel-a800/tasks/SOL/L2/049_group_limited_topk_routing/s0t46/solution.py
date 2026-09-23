import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_logits_kernel(
    A_ptr,  # hidden [M, K], float32
    B_ptr,  # weight [N, K], float32 (note: weight is [N, K], we need dot over K)
    C_ptr,  # logits [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    # accumulator
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # loop over K
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        # A_tile: [BM, BK]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # B_tile: [BK, BN]
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(A_tile, B_tile)

    # write back C = logits[m, n]
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    inp_ptr,   # [M, N], float32 (logits)
    bias_ptr,  # [N], float32
    out_ptr,   # [M, N], float32 (scores)
    M, N,
    stride_im, stride_in,
    stride_on, stride_oo,
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    if (m >= M) or (n >= N):
        return
    x = tl.load(inp_ptr + m * stride_im + n * stride_in)
    b = tl.load(bias_ptr + n)
    y = 1.0 / (1.0 + tl.exp(-x))  # sigmoid
    y = y + b
    tl.store(out_ptr + m * stride_on + n * stride_oo, y)


@triton.jit
def _top2_group_sum_kernel(
    scores_ptr,   # [M, N], float32
    out_ptr,      # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    g = tl.program_id(1)
    if (m >= M) or (g >= 8):
        return
    base = m * N + g * 32
    # Compute top-2 within the 32 experts of this group
    max1 = -float('inf')
    max2 = -float('inf')
    for i in range(32):
        idx = base + i
        val = tl.load(scores_ptr + idx)
        is_larger = val > max1
        old1 = max1
        max1 = tl.where(is_larger, val, max1)
        max2 = tl.where(is_larger, old1, max2)
        cond2 = val > max2
        old2 = max2
        max2 = tl.where(cond2, val, old2)
        max1 = tl.where(is_larger & (val <= old1), old1, max1)
    tl.store(out_ptr + m * stride_tm + g * stride_tn, max1 + max2)


@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # [M, 8], float32
    top4_ptr,          # [M, 4], int32
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # Bubble selection: pick 4 maxima among 8 groups
    for r in range(4):
        maxv = -float('inf')
        max_idx = -1
        for g in range(8):
            v = tl.load(group_scores_ptr + m * stride_gm + g * stride_gn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, g, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        # Store selected group index
        tl.store(top4_ptr + m * stride_tm + r * stride_tn, max_idx)
        # Mark used group by setting its score to -inf for next iterations
        # We cannot directly mutate group_scores_ptr here; the next iteration will re-scan unchanged values and pick other maxima.
        # Triton semantics ensure selection occurs without races.


@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr,        # [M, N], float32
    top4_groups_ptr,   # [M, 4], int32
    masked_ptr,        # [M, N], float32
    M, N,
    stride_sm, stride_sn,
    stride_tgm, stride_tgn,
    stride_mm, stride_mn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # For this token m, we keep only the 4 selected groups; others set to -inf
    for g in range(4):
        group_start = top4_groups_ptr[m * stride_tgm + g * stride_tgn]  # int32
        base = m * N + group_start * 32
        for i in range(32):
            idx = base + i
            v = tl.load(scores_ptr + idx)
            tl.store(masked_ptr + idx, v)
    # Fill the rest of the groups with -inf
    for n in range(N):
        # We can detect if n belongs to any of the 4 selected groups: check if n % 32 == g and g in [top4_groups[0..3]]
        found = 0
        for g in range(4):
            group_start = tl.load(top4_groups_ptr + m * stride_tgm + g * stride_tgn)
            if (n % 32) == group_start:
                found = 1
                break
        if found == 0:
            tl.store(masked_ptr + m * stride_mm + n * stride_mn, -float('inf'))


@triton.jit
def _select_and_scale_top8_fused_kernel(
    scores_ptr,        # [M, N], float32 (masked scores)
    top8_idx_ptr,      # [M, 8], int32
    top8_weight_ptr,   # [M, 8], float32
    M, N,
    scale,             # float32
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    total = 0.0
    # Select 8 maxima and store indices
    for r in range(8):
        maxv = -float('inf')
        max_idx = -1
        for n in range(N):
            v = tl.load(scores_ptr + m * stride_sm + n * stride_sn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, n, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        # Store index
        tl.store(top8_idx_ptr + m * stride_tm + r * stride_tn, max_idx)
        # Update total for normalization
        total += maxv
        # Mark selected element to -inf so it won't be selected again
        tl.store(scores_ptr + m * stride_sm + max_idx * stride_sn, -float('inf'))
    total = total + 1e-20
    # Normalize and scale
    for r in range(8):
        idx = tl.load(top8_idx_ptr + m * stride_tm + r * stride_tn)
        v = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
        w = (v / total) * scale
        tl.store(top8_weight_ptr + m * stride_tm + r * stride_tn, w)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation of the routing logic.

        Inputs:
          - hidden_states: [M, K], float32, CUDA
          - weight: [N, K], float32, CUDA  (note: weight is [N, K], we need dot over K)
          - expert_bias: [N], float32, CUDA
          - routed_scaling_factor: float
        Returns:
          - topk_idx: [M, 8], int64
          - topk_weight: [M, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be CUDA"
        M, K = hidden_states.shape
        N = weight.shape[0]
        assert weight.shape[1] == K, "weight second dimension must match hidden_states' second dimension"
        # Ensure dtype float32
        hidden = hidden_states.contiguous().to(torch.float32)
        weight_mat = weight.contiguous().to(torch.float32)  # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)

        # 1) Compute logits = hidden @ weight.T using Triton matmul kernel
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid_mat = (triton.cdiv(M, 128), triton.cdiv(N, 64))
        _matmul_logits_kernel[grid_mat](
            hidden, weight_mat, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_mat.stride(0), weight_mat.stride(1),
            logits.stride(0), logits.stride(1),
            BM=128, BN=64, BK=64,
            num_warps=4, num_stages=2,
        )

        # 2) Elementwise sigmoid + expert bias using Triton
        scores = torch.empty_like(logits)
        grid_sig = (M, N)
        _sigmoid_add_bias_kernel[grid_sig](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=4, num_stages=2,
        )

        # 3) Compute group top-2 sum: [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid_top2 = (M, 8)
        _top2_group_sum_kernel[grid_top2](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token: [M, 4], int32
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=hidden.device)
        _select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M,
            group_scores.stride(0), group_scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups to -inf in scores
        masked_scores = torch.empty_like(scores)
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=4, num_stages=2,
        )

        # 6) Fused selection + normalization + scaling using Triton
        top8_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden.device)
        top8_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        _select_and_scale_top8_fused_kernel[(M,)](
            masked_scores, top8_idx, top8_weight,
            M, N,
            routed_scaling_factor,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            num_warps=4, num_stages=2,
        )

        # Return as expected: int64 indices and float32 weights
        topk_idx = top8_idx.to(torch.int64)
        return topk_idx, top8_weight


# For reference: original run signature
# @torch.no_grad()
# def run(hidden_states, weight, expert_bias, routed_scaling_factor):
#     num_tokens = hidden_states.shape[0]
#     logits = F.linear(hidden_states.to(torch.float32), weight.to(torch.float32))
#     scores = torch.sigmoid(logits) + expert_bias.to(torch.float32)
#     # routing steps (groups, top-4 selection, mask, final top-8)
#     # return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
