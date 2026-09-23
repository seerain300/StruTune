import torch
import triton
import triton.language as tl


# 1) Triton GEMM: logits = hidden @ weight.T
@triton.jit
def linear_proj_kernel(
    A, B, C,
    M, K, N,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid over tiles of (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K], A is [M, K]
        A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N], B is [N, K] and we want weight^T, i.e., accessing B[n, k]
        B_ptrs = B + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        B_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(a, b)

    # Write back C: [M, N]
    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    X, Bias, Y,
    M, N,
    stride_xm, stride_xn,
    stride_b,
    stride_ym, stride_yn,
):
    # grid: (M*N,)
    pid = tl.program_id(0)
    t = pid // N
    col = pid % N
    x = tl.load(X + t * stride_xm + col * stride_xn)
    b = tl.load(Bias + col * stride_b)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(Y + t * stride_ym + col * stride_yn, y)


# 3) Triton: compute group_scores[t, g] = sum of top-2 scores over experts in group g (32 per group)
@triton.jit
def group_top2_sum_kernel(
    Scores, GroupScores,
    M, N, EXP_PER_GROUP: tl.constexpr,
):
    # One program per token
    pid = tl.program_id(0)
    t = pid
    group_scores = tl.zeros(8, dtype=tl.float32)

    for g in range(8):
        base = g * EXP_PER_GROUP
        top1 = -float("inf")
        top2 = -float("inf")
        # loop over 32 elements in the group
        for j in range(EXP_PER_GROUP):
            val = tl.load(Scores + t * N + base + j)
            # update top2
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        group_scores[g] = top1 + top2

    # store [M, 8]
    out_ptrs = GroupScores + t * 8 + tl.arange(0, 8)
    tl.store(out_ptrs, group_scores)


# 4) Triton: select top-4 groups per token
@triton.jit
def select_top4_groups_kernel(
    GroupScores, Top4Groups,
    M,
    stride_gs, stride_tg,
    EXP_PER_GROUP: tl.constexpr,
):
    # One program per token
    pid = tl.program_id(0)
    t = pid

    # Initialize top4 indices with -1
    idxs = tl.full((4,), -1, dtype=tl.int32)
    scores = tl.zeros((8,), dtype=tl.float32)
    # Load group scores
    for g in range(8):
        scores[g] = tl.load(GroupScores + t * 8 + g)

    # Bubble-like selection: for i in 0..3, find max and set its index
    for i in range(4):
        max_val = -float("inf")
        max_idx = -1
        for g in range(8):
            if scores[g] > max_val:
                max_val = scores[g]
                max_idx = g
        # set selected index
        idxs[i] = max_idx
        # set score to -inf for next selection
        scores[max_idx] = -float("inf")

    # Store to Top4Groups[t, :]
    for i in range(4):
        tl.store(Top4Groups + t * 4 + i, idxs[i])


# 5) Triton: mask non-selected groups to -inf in masked_scores
@triton.jit
def mask_nonselected_groups_kernel(
    Scores, Top4Groups, MaskedScores,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_tg,
    stride_mm, stride_mn,
):
    # One program per token
    pid = tl.program_id(0)
    t = pid

    # Load top4 group indices
    # Note: Triton scalar loads from 1D tensors using pointer arithmetic
    g0 = tl.load(Top4Groups + t * 4 + 0)
    g1 = tl.load(Top4Groups + t * 4 + 1)
    g2 = tl.load(Top4Groups + t * 4 + 2)
    g3 = tl.load(Top4Groups + t * 4 + 3)

    # Create masks for selected groups
    selected_cols = (tl.arange(0, N) // EXP_PER_GROUP) == g0 or \
                    (tl.arange(0, N) // EXP_PER_GROUP) == g1 or \
                    (tl.arange(0, N) // EXP_PER_GROUP) == g2 or \
                    (tl.arange(0, N) // EXP_PER_GROUP) == g3

    # Load original scores row
    row = tl.load(Scores + t * N + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    row = tl.where(selected_cols, row, -float("inf"))
    # Store to MaskedScores
    tl.store(MaskedScores + t * N + tl.arange(0, N), row)


# 6) Triton: select top-8 from masked_scores per token (iterative)
@triton.jit
def select_top8_masked_kernel(
    MaskedScores, Top8Indices,
    M, N,
    stride_mm, stride_mn,
):
    # One program per token
    pid = tl.program_id(0)
    t = pid

    idxs = tl.full((8,), -1, dtype=tl.int32)
    vals = tl.zeros((N,), dtype=tl.float32)

    # Load the row
    for j in range(N):
        vals[j] = tl.load(MaskedScores + t * N + j)

    # Iteratively select maxima
    for i in range(8):
        max_val = -float("inf")
        max_idx = -1
        # find max
        for j in range(N):
            if vals[j] > max_val:
                max_val = vals[j]
                max_idx = j
        # store idx
        idxs[i] = max_idx
        # set to -inf
        vals[max_idx] = -float("inf")

    # Store indices
    for i in range(8):
        tl.store(Top8Indices + t * 8 + i, idxs[i])


# 7) Triton: normalize selected scores and scale
@triton.jit
def normalize_and_scale_kernel(
    MaskedScores, Top8Indices, TopkWeight,
    M, N, eps: tl.constexpr, scale: tl.constexpr,
    stride_mm, stride_mn,
):
    # One program per token
    pid = tl.program_id(0)
    t = pid

    # Gather selected values from MaskedScores using indices
    selected = tl.zeros((8,), dtype=tl.float32)
    for i in range(8):
        idx = tl.load(Top8Indices + t * 8 + i)
        selected[i] = tl.load(MaskedScores + t * N + idx)

    # Sum with eps
    s = selected[0]
    for i in range(1, 8):
        s += selected[i]
    s = s + eps

    # Normalize and scale
    scaled = selected / s * scale

    # Store result
    out_ptrs = TopkWeight + t * 8 + tl.arange(0, 8)
    tl.store(out_ptrs, scaled)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype/device
        device = hidden_states.device
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1 and expert_bias.shape[0] == weight.shape[0], "expert_bias must be [num_experts]"
        num_tokens, hidden_dim = hidden_states.shape
        num_experts = weight.shape[0]
        assert hidden_dim == 256 and num_experts == 256, "This implementation expects hidden_dim=256 and num_experts=256"
        assert hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16), "hidden_states must be float16/float32/bfloat16"
        # Use float32 accumulation for GEMM
        A = hidden_states.contiguous().to(torch.float32)  # [M, K]
        W = weight.contiguous().to(torch.float32)        # [N, K]
        M, K, N = num_tokens, hidden_dim, num_experts

        # 1) Triton GEMM: logits[M, N] = A[M, K] @ W[N, K]
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_proj_kernel[grid](
            A, W, logits,
            M, K, N,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Triton sigmoid + bias: scores[M, N]
        scores = torch.empty_like(logits)  # [M, N]
        EXP_PER_GROUP = 32
        sigmoid_add_bias_kernel[(M * N,)](
            logits, expert_bias.to(torch.float32), scores,
            M, N,
            logits.stride(0), logits.stride(1),
            expert_bias.to(torch.float32).stride(0),
            scores.stride(0), scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 3) Triton group top-2 sum: group_scores[M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N, EXP_PER_GROUP=EXP_PER_GROUP,
            stride_gs=scores.stride(0), stride_tg=group_scores.stride(0),
            num_warps=1, num_stages=1,
        )

        # 4) Triton select top-4 groups: top4_groups[M, 4] (int32)
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M,
            group_scores.stride(0), top4_groups.stride(0),
            EXP_PER_GROUP=EXP_PER_GROUP,
            num_warps=1, num_stages=1,
        )

        # 5) Triton mask non-selected groups: masked_scores[M, N]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)
        mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N, EXP_PER_GROUP=EXP_PER_GROUP,
            stride_sm=scores.stride(0), stride_sn=scores.stride(1),
            stride_tg=top4_groups.stride(0),
            stride_mm=masked_scores.stride(0), stride_mn=masked_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Triton select top-8 from masked_scores: top8_indices[M, 8] (int32)
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 7) Triton normalize + scale: topk_weight[M, 8]
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        normalize_and_scale_kernel[(M,)](
            masked_scores, top8_indices, topk_weight,
            M, N, eps=1e-20, scale=routed_scaling_factor,
            stride_mm=masked_scores.stride(0), stride_mn=masked_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # Return indices and weights (indices int32 as in original; weights float32)
        return top8_indices, topk_weight


def run(*args):
    return ModelNew()(*args)
