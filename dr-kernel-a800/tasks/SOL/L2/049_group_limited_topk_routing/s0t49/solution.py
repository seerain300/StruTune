import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_linear_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Grid: (pid_m tiles over M, pid_n tiles over N, pid_k chunks over K)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)

    # Pointers for A (hidden) and B (weight transposed), and C (logits)
    A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Load and accumulate
    # K loop: chunked
    for k in range(0, K, BK):
        k_mask = (k + offs_k) < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        # advance B along K
        B_ptrs += BK * stride_bk

    # Write back to C
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=out_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_lm, stride_ln,
    stride_bn,
    num_tokens,  # just for clarity; not used but can be used to size loops if needed
):
    # This kernel assumes we pass a 2D grid of (M, N) and compute elementwise sigmoid + bias.
    # We'll implement as a 2D grid over tokens and columns; however Triton expects flat indexing.
    # Instead, we use a 1D grid over total elements and compute indices via // and %.
    # But to keep it simple and robust, we'll implement a 2D grid over (M, N) where we broadcast bias per column.
    # For safety and simplicity, we'll restrict to M,N from logits_ptr. Note: Triton doesn't support arbitrary 2D grid size detection,
    # so we assume M,N are provided. We'll use strides to index.
    # Here we provide a simpler elementwise kernel by 1D grid over (M*N):
    total = M * N
    idx = tl.program_id(0)
    if idx >= total:
        return
    m = idx // N
    n = idx % N
    # Load logits and bias
    logits = tl.load(logits_ptr + m * stride_lm + n * stride_ln)
    bias = tl.load(bias_ptr + n * stride_bn)  # bias is per-expert (column)
    # Compute sigmoid
    # Triton provides tl.sigmoid in recent versions; fallback using 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-logits))
    scores = sig + bias
    tl.store(scores_ptr + m * stride_lm + n * stride_ln, scores)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,            # *f32, [M, N]
    group_scores_ptr,      # *f32, [M, 8]
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    # One program per token
    m = tl.program_id(0)
    if m >= M:
        return
    # Each group has 32 experts
    e_per_group = 32
    n_groups = 8
    for g in range(n_groups):
        base = g * e_per_group
        maxv1 = -float('inf')
        maxv2 = -float('inf')
        # Scan 32 experts in this group
        for j in range(e_per_group):
            n_idx = base + j
            v = tl.load(scores_ptr + m * stride_sm + n_idx * stride_sn)
            # Update top-2
            cond1 = v > maxv1
            old1 = maxv1
            maxv1 = tl.where(cond1, v, maxv1)
            maxv2 = tl.where(cond1, old1, maxv2)
            cond2 = (v > maxv2) & (~cond1)
            maxv2 = tl.where(cond2, v, maxv2)
        tl.store(group_scores_ptr + m * stride_gm + g * stride_gn, maxv1 + maxv2)


@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,      # *f32, [M, 8]
    top4_groups_ptr,       # *i32, [M, 4]
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # Iteratively pick 4 maxima
    for r in range(4):
        maxv = -float('inf')
        max_idx = -1
        for g in range(8):
            v = tl.load(group_scores_ptr + m * stride_gm + g * stride_gn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, g, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top4_groups_ptr + m * stride_tm + r * stride_tn, max_idx)


@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr,            # *f32, [M, N]
    selected_groups_ptr,   # *i32, [M, 4]
    M, N,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    neg_inf = -1.0e30
    for r in range(4):
        g = tl.load(selected_groups_ptr + m * stride_tm + r * stride_tn)
        # For this selected group, do nothing (keep original scores)
        pass
    # For all other groups, set scores to -inf
    for n in range(N):
        keep = 0
        for r in range(4):
            g_sel = tl.load(selected_groups_ptr + m * stride_tm + r * stride_tn)
            if g_sel == (n // 32):
                keep = 1
                break
        v = tl.load(scores_ptr + m * stride_sm + n * stride_sn)
        v = tl.where(keep == 1, v, neg_inf)
        tl.store(scores_ptr + m * stride_sm + n * stride_sn, v)


@triton.jit
def _final_top8_and_scale_kernel(
    masked_scores_ptr,     # *f32, [M, N] (masked)
    top8_idx_ptr,          # *i32, [M, 8]
    scaled_weights_ptr,    # *f32, [M, 8]
    M, N,
    scale,                 # f32
    stride_sm, stride_sn,
    stride_tm, stride_tn,
):
    m = tl.program_id(0)
    if m >= M:
        return
    total = 0.0
    for r in range(8):
        maxv = -float('inf')
        max_idx = -1
        for n in range(N):
            v = tl.load(masked_scores_ptr + m * stride_sm + n * stride_sn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, n, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(top8_idx_ptr + m * stride_tm + r * stride_tn, max_idx)
        # Update total with selected value
        idx = tl.load(top8_idx_ptr + m * stride_tm + r * stride_tn)
        v = tl.load(masked_scores_ptr + m * stride_sm + idx * stride_sn)
        total += v
    total = total + 1e-20
    for r in range(8):
        idx = tl.load(top8_idx_ptr + m * stride_tm + r * stride_tn)
        v = tl.load(masked_scores_ptr + m * stride_sm + idx * stride_sn)
        w = (v / total) * scale
        tl.store(scaled_weights_ptr + m * stride_tm + r * stride_tn, w)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Inputs:
        # hidden_states: [M, K] float32, M=num_tokens, K=hidden_dim=256
        # weight: [N, K] float32, N=num_experts=256
        # expert_bias: [N] float32
        # routed_scaling_factor: float
        device = hidden_states.device
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "Use float32 tensors"
        M, K = hidden_states.shape
        N, Kw = weight.shape
        assert Kw == K, "weight second dim must match hidden_dim"
        assert N == 256, "num_experts must be 256"
        assert K == 256, "hidden_dim must be 256"

        # 1) Compute logits = hidden @ weight.T using Triton matmul kernel
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        # Ensure contiguous
        hidden = hidden_states.contiguous()
        weight_T = weight.transpose(0, 1).contiguous()  # [K, N]
        # Grid dims
        BM = 128
        BN = 64
        BK = 64
        grid = (
            triton.cdiv(M, BM),
            triton.cdiv(N, BN),
            triton.cdiv(K, BK),
        )
        _matmul_linear_kernel[grid](
            hidden, weight_T, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            logits.stride(0), logits.stride(1),
            BM=BM, BN=BN, BK=BK,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + expert bias (Triton elementwise)
        scores = torch.empty_like(logits)
        _sigmoid_add_bias_kernel[(M * N,)](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            expert_bias.stride(0),
            num_tokens=M,
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 per token: [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token: [M, 4]
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        _select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M,
            group_scores.stride(0), group_scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups: set others to -inf in scores
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups,
            M, N,
            scores.stride(0), scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Final top-8 selection and scaling: produce topk_idx and scaled_weights
        top8_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        scaled_weights = torch.empty((M, 8), dtype=torch.float32, device=device)
        _final_top8_and_scale_kernel[(M,)](
            scores, top8_idx, scaled_weights,
            M, N,
            routed_scaling_factor,
            scores.stride(0), scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # Cast indices to int64 as original topk_idx dtype
        topk_idx = top8_idx.to(torch.int64)
        # scaled_weights is float32 as per original topk_weight dtype
        return topk_idx, scaled_weights


def run(*args):
    return ModelNew()(*args)
