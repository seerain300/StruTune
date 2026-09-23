import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sigmoid_bias_kernel(
    logits_ptr,       # *float32, [M, N]
    expert_bias_ptr,  # *float32, [N]
    scores_ptr,       # *float32, [M, N], output
    M, N,
    stride_lm, stride_ln,
    stride_sm, stride_sn,
):
    t = tl.program_id(0)
    n = tl.program_id(1)
    if t >= M or n >= N:
        return
    val = tl.load(logits_ptr + t * stride_lm + n * stride_ln)
    bias = tl.load(expert_bias_ptr + n)
    s = 1.0 / (1.0 + tl.exp(-val))
    out = s + bias
    tl.store(scores_ptr + t * stride_sm + n * stride_sn, out)


@triton.jit
def _compute_group_scores_kernel(
    scores_ptr,     # *float32, [M, N], input scores after sigmoid + bias
    group_scores_ptr,  # *float32, [M, 8], output
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    # one program per token
    t = tl.program_id(0)
    if t >= M:
        return
    # loop over groups
    for g in range(8):
        base = g * 32
        # maintain top-2 within this group of 32
        top1 = -float('inf')
        top2 = -float('inf')
        for j in range(32):
            n = base + j
            if n < N:
                v = tl.load(scores_ptr + t * stride_sm + n * stride_sn)
                cond1 = v > top1
                old1 = top1
                top1 = tl.where(cond1, v, top1)
                top2 = tl.where(cond1, old1, tl.maximum(top2, old1))
                cond2 = v > top2
                top2 = tl.where(cond2, v, top2)
                # keep top1 >= top2
                top1 = tl.maximum(top1, top2)
        tl.store(group_scores_ptr + t * stride_gm + g * stride_gn, top1 + top2)


@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # *float32, [M, 8]
    top4_groups_ptr,   # *int32, [M, 4]
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    for r in range(4):
        maxv = -float('inf')
        max_idx = -1
        for g in range(8):
            v = tl.load(group_scores_ptr + t * stride_gm + g * stride_gn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, g, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top4_groups_ptr + t * stride_tm + r * stride_tn, max_idx)


@triton.jit
def _mask_nonselected_groups_kernel(
    logits_ptr,          # *float32, [M, N], original logits
    group_idx_ptr,       # *int32, [M, 4], top-4 group indices per token
    masked_ptr,          # *float32, [M, N], output masked logits
    M, N,
    stride_lm, stride_ln,
    stride_gm, stride_gn,
    stride_mm, stride_mn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    for r in range(4):
        g = tl.load(group_idx_ptr + t * stride_gm + r * stride_gn)
        base = g * 32
        for j in range(32):
            n = base + j
            if n < N:
                old = tl.load(logits_ptr + t * stride_lm + n * stride_ln)
                new = tl.where(r == 0, -float('inf'), old)  # all selected groups get -inf
                tl.store(masked_ptr + t * stride_mm + n * stride_mn, new)


@triton.jit
def _select_top8_kernel(
    masked_ptr,     # *float32, [M, N], masked logits (we set to -inf non-selected per token)
    top8_idx_ptr,   # *int32, [M, 8]
    M, N,
    stride_mm, stride_mn,
    stride_im, stride_in,
):
    t = tl.program_id(0)
    if t >= M:
        return
    for r in range(8):
        maxv = -float('inf')
        max_idx = -1
        for n in range(N):
            v = tl.load(masked_ptr + t * stride_mm + n * stride_mn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, n, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top8_idx_ptr + t * stride_im + r * stride_in, max_idx)
        # mark selected column to -inf for next iterations
        for n in range(N):
            v = tl.load(masked_ptr + t * stride_mm + n * stride_mn)
            new = tl.where(n == max_idx, -float('inf'), v)
            tl.store(masked_ptr + t * stride_mm + n * stride_mn, new)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure dtype and contiguity; original uses float32
        hidden_states = hidden_states.to(torch.float32).contiguous()
        weight = weight.to(torch.float32).contiguous()  # [N, K]
        expert_bias = expert_bias.to(torch.float32).contiguous()

        M, K = hidden_states.shape
        N = weight.shape[0]  # num_experts

        # 1) Compute logits = F.linear(hidden, weight, None) -> [M, N]
        logits = F.linear(hidden_states, weight)  # (M, K) @ (K, N) => (M, N), float32

        device = logits.device

        # 2) Sigmoid and expert bias addition via Triton elementwise kernel
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_sigmoid = (M, N)
        _sigmoid_bias_kernel[grid_sigmoid](
            logits,
            expert_bias,
            scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 3) Compute group scores per token (sum of top-2 per group)
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        grid_groups = (M,)
        _compute_group_scores_kernel[grid_groups](
            scores,
            group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        grid_select4 = (M,)
        _select_top4_groups_kernel[grid_select4](
            group_scores,
            top4_groups,
            M,
            group_scores.stride(0), group_scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Mask out non-selected groups on original logits: set selected groups to -inf per token
        masked_logits = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_mask = (M,)
        _mask_nonselected_groups_kernel[grid_mask](
            logits,
            top4_groups,
            masked_logits,
            M, N,
            logits.stride(0), logits.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            masked_logits.stride(0), masked_logits.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Select top-8 indices from masked logits using Triton
        top8_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        grid_top8 = (M,)
        _select_top8_kernel[grid_top8](
            masked_logits,
            top8_idx,
            M, N,
            masked_logits.stride(0), masked_logits.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # 7) Gather selected logits, normalize, and scale
        # Gather selected values from original logits using indices
        top8_idx_long = top8_idx.to(torch.int64)  # gather expects int64
        selected_logits = logits.gather(1, top8_idx_long)  # [M, 8], float32

        # Normalize and scale
        eps = 1e-20
        denom = selected_logits.sum(dim=1, keepdim=True) + eps  # [M, 1]
        topk_weight = (selected_logits / denom) * self.routed_scaling_factor  # [M, 8], float32

        # Return indices and weights as in original
        topk_idx = top8_idx_long  # [M, 8], int64

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
