import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_proj_kernel(
    hidden_ptr,  # [M, K]
    weight_ptr,  # [N, K] but we load as (K, N) via stride
    logits_ptr,  # [M, N] output
    M, K, N,
    stride_hm, stride_hk,  # strides for hidden: (M, K)
    stride_wk, stride_wn,  # strides for weight: (K, N)
    stride_lm, stride_ln,  # strides for logits: (M, N)
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    # Accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    # Loop over K
    for k0 in range(0, K, TILE_K):
        offs_k = k0 + tl.arange(0, TILE_K)
        # Load A tile: A[offs_m, offs_k]
        a_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # Load B tile: B[offs_k, offs_n] where weight is [N, K], so we index (K, N)
        b_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    # Store acc to logits
    out_ptrs = logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,      # [M, N], float32
    bias_ptr,        # [N], float32
    scores_ptr,      # [M, N], float32
    M, N,
    stride_sm, stride_sn,
    stride_bm,
    stride_om, stride_on,
):
    pid = tl.program_id(0)
    offs = pid * 1 + tl.arange(0, 1)  # process one element per program, grid size M*N
    total = M * N
    if offs >= total:
        return
    m = offs // N
    n = offs % N
    x = tl.load(logits_ptr + m * stride_sm + n * stride_sn)
    b = tl.load(bias_ptr + n * stride_bm)
    y = 1.0 / (1.0 + tl.exp(-x))
    y = y + b
    tl.store(scores_ptr + m * stride_om + n * stride_on, y)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,      # [M, N], float32
    group_scores_ptr,  # [M, 8], float32
    M, N,
    EXP_PER_GROUP: tl.constexpr,  # 32
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Process 8 groups per token
    for g in range(0, 8):
        base = g * EXP_PER_GROUP
        # Load 32 experts for this group
        idx = base + tl.arange(0, EXP_PER_GROUP)
        ptrs = scores_ptr + pid * stride_sm + idx * stride_sn
        vals = tl.load(ptrs, mask=idx < N, other=-1e30)
        # Find top-2 via two reductions
        max1 = tl.max(vals, axis=0)
        vals2 = vals
        vals2[vals2 == max1] = -1e30
        max2 = tl.max(vals2, axis=0)
        group_scores_ptr[pid, g] = max1 + max2


@triton.jit
def select_top4_groups_kernel(
    group_scores_ptr,  # [M, 8], float32
    top4_groups_ptr,   # [M, 4], int32
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Per-token selection of top-4 groups: iterative selection
    selected = tl.zeros((4,), dtype=tl.int32)  # placeholder, but we'll store using loops
    for t in range(0, 4):
        # Find max score among remaining 8
        max_val = -1e30
        for g in range(0, 8):
            score = tl.load(group_scores_ptr + pid * stride_gm + g * stride_gn)
            if score > max_val:
                max_val = score
                max_idx = g
        # Store index
        tl.store(top4_groups_ptr + pid * stride_tm + t * stride_tn, max_idx)


@triton.jit
def mask_nonselected_groups_kernel(
    scores_ptr,         # [M, N], float32
    top4_groups_ptr,    # [M, 4], int32
    masked_ptr,         # [M, N], float32, will be filled with -inf for non-selected groups
    M, N,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
    stride_mm, stride_mn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(0, 4):
        g = tl.load(top4_groups_ptr + pid * stride_tm + t * stride_tn).to(tl.int32)
        base = g * 32
        for i in range(0, 32):
            idx = base + i
            ptr = scores_ptr + pid * stride_sm + idx * stride_sn
            val = tl.load(ptr)
            # Write -inf for non-selected groups elsewhere
            # We'll fill masked with -inf for all and then overwrite selected
            # but here we only need to mark non-selected groups. Easiest is to write -inf to masked for non-selected idx.
            # However, masked is input to this kernel as output; we'll do it via store logic.
            # Since we don't have direct way to "non-selected", we can implement: overwrite masked for idx in selected groups only by copying from scores; others set to -inf.
            # To do that, we need to know which groups are selected. We'll assume we only need to set -inf for non-selected idx based on g. So for each t, we set -inf for all idx not in this group.
            pass
    # Simple implementation: fill masked with -inf then overwrite selected groups. But Triton doesn't support multi-branch vector writes; so we fill entire masked with -inf per token row, then we cannot overwrite without a separate kernel. To keep it simple and correct, we can use torch to pre-fill -inf and then use Triton to copy selected group scores. However, the requirement is to do everything in Triton. So we'll implement a full write:
    # First fill masked with -inf
    for i in range(0, N):
        neg_inf = -1e30
        tl.store(masked_ptr + pid * stride_mm + i * stride_mn, neg_inf)
    # Then for each t, copy selected group scores
    for t in range(0, 4):
        g = tl.load(top4_groups_ptr + pid * stride_tm + t * stride_tn).to(tl.int32)
        base = g * 32
        for i in range(0, 32):
            idx = base + i
            src = scores_ptr + pid * stride_sm + idx * stride_sn
            val = tl.load(src)
            dst = masked_ptr + pid * stride_mm + idx * stride_mn
            tl.store(dst, val)


@triton.jit
def select_top8_masked_kernel(
    masked_ptr,         # [M, N], float32 (after mask_nonselected_groups)
    top8_indices_ptr,   # [M, 8], int32
    M, N,
    stride_mm, stride_mn,
    stride_tmm, stride_tmn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Iteratively select maxima 8 times (without replacement via setting the chosen element to -inf)
    for t in range(0, 8):
        max_val = -1e30
        # Loop over N to find max
        for i in range(0, N):
            ptr = masked_ptr + pid * stride_mm + i * stride_mn
            val = tl.load(ptr)
            if val > max_val:
                max_val = val
                max_idx = i
        # Store index
        tl.store(top8_indices_ptr + pid * stride_tmm + t * stride_tmn, max_idx)
        # Mark this index as selected by setting to -inf for subsequent iterations
        ptr = masked_ptr + pid * stride_mm + max_idx * stride_mn
        tl.store(ptr, -1e30)


@triton.jit
def normalize_and_scale_kernel(
    masked_ptr,           # [M, N], float32
    top8_indices_ptr,     # [M, 8], int32
    topk_weight_ptr,      # [M, 8], float32
    M,
    routed_scale,         # float32
    stride_mm, stride_mn,
    stride_tmm, stride_tmn,
    stride_wm, stride_wn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sum_val = 0.0
    # Compute sum of selected scores
    for t in range(0, 8):
        idx = tl.load(top8_indices_ptr + pid * stride_tmm + t * stride_tmn)
        val = tl.load(masked_ptr + pid * stride_mm + idx * stride_mn)
        sum_val += val
    # Normalize and scale, store
    for t in range(0, 8):
        idx = tl.load(top8_indices_ptr + pid * stride_tmm + t * stride_tmn)
        val = tl.load(masked_ptr + pid * stride_mm + idx * stride_mn)
        norm = val / (sum_val + 1e-20)
        out = norm * routed_scale
        tl.store(topk_weight_ptr + pid * stride_wm + t * stride_wn, out)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be CUDA tensors"
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]  # num_experts, expected 256
        # 1) GEMM logits = hidden @ weight.T using Triton
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        # Make sure strides are element-wise
        hidden_contig = hidden_states.contiguous()
        weight_contig = weight.contiguous()
        TILE_M = 64
        TILE_N = 64
        TILE_K = 64
        grid = lambda meta: (triton.cdiv(M, meta['TILE_M']), triton.cdiv(N, meta['TILE_N']))
        linear_proj_kernel[grid](
            hidden_contig, weight_contig, logits,
            M, K, N,
            hidden_contig.stride(0), hidden_contig.stride(1),
            weight_contig.stride(1), weight_contig.stride(0),  # (K, N) via weight[k, n]
            logits.stride(0), logits.stride(1),
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + bias (Triton)
        scores = torch.empty_like(logits)
        bias_f32 = expert_bias.to(torch.float32).contiguous()
        grid_elem = (M * N,)
        sigmoid_add_bias_kernel[grid_elem](
            logits, bias_f32, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            bias_f32.stride(0),
            scores.stride(0), scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 sum per token (Triton)
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_gm=group_scores.stride(0), stride_gn=group_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token (Triton)
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M,
            group_scores.stride(0), group_scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups (Triton). We need masked scores [M, N].
        masked_scores = torch.empty_like(scores)
        mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Select top-8 from masked scores (Triton)
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_indices.stride(0), top8_indices.stride(1),
            num_warps=1, num_stages=1,
        )

        # 7) Normalize and scale (Triton)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        normalize_and_scale_kernel[(M,)](
            masked_scores, top8_indices, topk_weight,
            M, routed_scaling_factor,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_indices.stride(0), top8_indices.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            num_warps=1, num_stages=1,
        )

        # Return indices and normalized weights as in original: topk_idx (int) and topk_weight (float)
        # topk_idx: [M, 8], int32 from top8_indices
        # topk_weight: [M, 8], float32
        # Note: The original returns (topk_idx, topk_weight). We will return top8_indices (int32) and topk_weight (float32).
        # If you need int, you can convert top8_indices to int64 or int32 accordingly.
        return top8_indices, topk_weight


def run(*args):
    return ModelNew()(*args)
