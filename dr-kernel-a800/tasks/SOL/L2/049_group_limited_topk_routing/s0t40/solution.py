import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32 (we will pass weight as [N, K] and view as B[K, N] via strides)
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Loop over K in chunks of BK
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        # A tile: [BM, BK]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        # B tile: we pass weight as [N, K], but view as B[K, N] via strides
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        # acc += a @ b
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _sigmoid_bias_kernel(
    logits_ptr,   # [M, N], float32
    bias_ptr,     # [N], float32
    scores_ptr,   # [M, N], float32
    M, N,
    stride_l0, stride_l1,
    stride_s0, stride_s1,
):
    t = tl.program_id(0)  # one program per row (token)
    n = tl.program_id(1)  # grid along columns
    if (t >= M) or (n >= N):
        return
    # Compute sigmoid and add bias[n]
    v = tl.load(logits_ptr + t * stride_l0 + n * stride_l1)
    b = tl.load(bias_ptr + n)
    v = 1.0 / (1.0 + tl.exp(-v)) + b
    tl.store(scores_ptr + t * stride_s0 + n * stride_s1, v)


@triton.jit
def _group_top2_and_masked_scores_kernel(
    scores_ptr,         # [M, N], float32
    group_scores_ptr,   # [M, 8], float32
    masked_ptr,         # [M, N], float32
    group_idx_ptr,      # [M, 4], int32
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    stride_mm, stride_mn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)  # one program per token
    if t >= M:
        return

    # Compute group scores: sum of top-2 within each group of 32 experts
    g = 0
    while g < 8:
        start = g * 32
        end = start + 32
        col_ids = start + tl.arange(0, 32)
        # Load 32 columns for this token
        ptrs = scores_ptr + t * stride_sm + col_ids * stride_sn
        vals = tl.load(ptrs)  # shape [32], float32
        # Compute top-2 via sorting is not possible; use iterative find max twice
        m1 = tl.max(vals, axis=0)
        # Remove only one occurrence of m1 by setting its index to -inf (randomly pick one)
        # We can find index of m1: for each i, if vals[i] == m1, set to -inf; otherwise leave. It's fine since we only need top-2 values.
        # Instead, we compute second max by substituting m1 with -inf
        vals2 = tl.where(vals == m1, -float('inf'), vals)
        m2 = tl.max(vals2, axis=0)
        group_scores_ptr[t * stride_gm + g * stride_gn] = m1 + m2
        g += 1

    # Select top-4 group indices by scanning group_scores (4 iterations)
    for r in range(4):
        maxv = -float('inf')
        max_idx = -1
        for g in range(8):
            v = tl.load(group_scores_ptr + t * stride_gm + g * stride_gn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, g, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(group_idx_ptr + t * stride_tm + r * stride_tn, max_idx)
        # Mark selected group as -inf in masked scores by setting that group's 32 columns to -inf
        g = tl.load(group_idx_ptr + t * stride_tm + r * stride_tn)
        start = g * 32
        end = start + 32
        col_ids = start + tl.arange(0, 32)
        ptrs = scores_ptr + t * stride_sm + col_ids * stride_sn
        # Set masked to -inf for selected group
        tl.store(masked_ptr + t * stride_mm + col_ids * stride_mn, -float('inf'),
                 mask=(t < M) & (col_ids < N))
        # For other groups, keep original values
        pass


@triton.jit
def _final_select_top8_kernel(
    masked_ptr,          # [M, N], float32
    selected_idx_ptr,    # [M, 8], int32
    M, N,
    stride_mm, stride_mn,
    stride_im, stride_in,
):
    t = tl.program_id(0)  # one program per token
    if t >= M:
        return
    for i in range(8):
        maxv = -float('inf')
        max_idx = -1
        for n in range(0, N):
            v = tl.load(masked_ptr + t * stride_mm + n * stride_mn)
            is_larger = v > maxv
            max_idx = tl.where(is_larger, n, max_idx)
            maxv = tl.where(is_larger, v, maxv)
        tl.store(selected_idx_ptr + t * stride_im + i * stride_in, max_idx)
        # Zero out the selected position for next iterations
        if i < 7:
            tl.store(masked_ptr + t * stride_mm + max_idx * stride_mn, -float('inf'),
                     mask=(t < M) & (max_idx < N))


@triton.jit
def _gather_and_normalize_kernel(
    masked_ptr,         # [M, N], float32
    selected_idx_ptr,   # [M, 8], int32
    routed_scale,       # float32
    out_ptr,            # [M, 8], float32
    M, N,
    stride_mm, stride_mn,
    stride_ism, stride_isn,
    stride_om, stride_on,
):
    t = tl.program_id(0)  # one program per token
    if t >= M:
        return
    total = 0.0
    # Compute sum of selected scores
    for i in range(8):
        idx = tl.load(selected_idx_ptr + t * stride_ism + i * stride_isn)
        v = tl.load(masked_ptr + t * stride_mm + idx * stride_mn)
        total += v
    total = total + 1e-20
    for i in range(8):
        idx = tl.load(selected_idx_ptr + t * stride_ism + i * stride_isn)
        v = tl.load(masked_ptr + t * stride_mm + idx * stride_mn)
        v = v / total * routed_scale
        tl.store(out_ptr + t * stride_om + i * stride_on, v)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,  # [N, K], N=256, K=hidden_dim=256
                expert_bias: torch.Tensor,  # [N]
                routed_scaling_factor: float):
        """
        Triton-only implementation of the original run function.
        Returns:
          - topk_idx: [num_tokens, 8], int64
          - topk_weight: [num_tokens, 8], float32
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        assert hidden_states.dim() == 2, "hidden_states must be [M, K]"
        assert weight.dim() == 2 and weight.shape[0] == 256 and weight.shape[1] == 256, "weight must be [256, 256]"
        M, K = hidden_states.shape
        N = weight.shape[0]
        assert N == 256, "num_experts must be 256"
        assert expert_bias.shape[0] == N, "expert_bias must match num_experts"

        # 1) GEMM: logits = hidden @ weight.T → [M, N]
        # Prepare B as [K, N] view via strides: we pass weight [N, K], but read as B[k, n] = weight[n, k]
        A = hidden_states.contiguous()  # [M, K]
        W = weight.contiguous()         # [N, K]
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        grid = (triton.cdiv(M, 128), triton.cdiv(N, 256))  # for N=256, second dim is 1
        _matmul_kernel[grid](
            A, W, logits,
            M, N, K,
            A.stride(0), A.stride(1),
            K, N,  # strides for B[K, N]: stride_bk=W.stride(1)=1 (since [N,K]), stride_bn=W.stride(0)=K
            logits.stride(0), logits.stride(1),
            BM=128, BN=256, BK=64,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + expert bias: scores[M, N]
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_sb = (M, N)
        _sigmoid_bias_kernel[grid_sb](
            logits, expert_bias,
            scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 and select top-4 groups per token, then mask
        group_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        masked = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_groups = (M,)
        _group_top2_and_masked_scores_kernel[grid_groups](
            scores, group_scores, masked, group_idx,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            masked.stride(0), masked.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # 4) Final top-8 selection from masked scores
        selected_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        grid_final = (M,)
        _final_select_top8_kernel[grid_final](
            masked, selected_idx,
            M, N,
            masked.stride(0), masked.stride(1),
            selected_idx.stride(0), selected_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Gather selected scores and normalize with routed scaling
        out = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid_gn = (M,)
        _gather_and_normalize_kernel[grid_gn](
            masked, selected_idx, routed_scaling_factor,
            out,
            M, N,
            masked.stride(0), masked.stride(1),
            selected_idx.stride(0), selected_idx.stride(1),
            out.stride(0), out.stride(1),
            num_warps=1, num_stages=1,
        )

        # Return topk_idx and topk_weight with the same shapes/dtypes as original
        # Note: original returns topk_idx as int64, topk_weight as float32
        topk_idx = selected_idx.to(torch.int64)            # [M, 8], int64
        topk_weight = out                                 # [M, 8], float32

        return topk_idx, topk_weight


# Helper function to mirror the original call signature (optional for testing)
def run(hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float):
    model = ModelNew().cuda()
    # Move inputs to CUDA if available
    if hidden_states.device.type != 'cuda':
        hidden_states = hidden_states.cuda()
    if weight.device.type != 'cuda':
        weight = weight.cuda()
    if expert_bias.device.type != 'cuda':
        expert_bias = expert_bias.cuda()
    return model(hidden_states, weight, expert_bias, routed_scaling_factor)


# Quick self-test (optional):
# hidden_states = torch.randn(2048, 256, device='cuda', dtype=torch.float32)
# weight = torch.randn(256, 256, device='cuda', dtype=torch.float32)  # [N, K]
# expert_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# routed_scaling_factor = 1.5
# idx, weights = run(hidden_states, weight, expert_bias, routed_scaling_factor)
# print(idx.shape, idx.dtype, weights.shape, weights.dtype)


def run(*args):
    return ModelNew()(*args)
