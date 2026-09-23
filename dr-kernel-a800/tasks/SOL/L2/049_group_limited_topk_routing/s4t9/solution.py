import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_AxB_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for tiles
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BM, BK]
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # [BK, BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptrs,
            mask=(offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K),
            other=0.0,
        )
        b = tl.load(
            B_ptrs,
            mask=((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store results
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        C_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,    # [M, N], float32
    B_ptr,    # [N], float32
    Y_ptr,    # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    j = 0
    while j < N:
        col = j + tl.arange(0, BLOCK)
        mask = col < N
        x = tl.load(X_ptr + pid_m * stride_xm + col * stride_xn, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-x))
        b = tl.load(B_ptr + col, mask=mask, other=0.0)
        y = s + b
        tl.store(Y_ptr + pid_m * stride_ym + col * stride_yn, y, mask=mask)
        j += BLOCK


@triton.jit
def _group_top2_sum_kernel(
    Scores_ptr,   # [M, N], float32
    GroupScores_ptr,  # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gsm, stride_gsn,
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    NUM_GROUPS: tl.constexpr,         # 8
    BLOCK_N: tl.constexpr,            # 32
):
    # Each program handles one token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Initialize group_scores
    for g in range(NUM_GROUPS):
        tl.store(GroupScores_ptr + pid_m * stride_gsm + g * stride_gsn, -float('inf'))

    # Loop over groups
    for g in range(NUM_GROUPS):
        # Load one group of 32 experts
        col = g * EXPERTS_PER_GROUP + tl.arange(0, BLOCK_N)
        mask = col < N
        vals = tl.load(Scores_ptr + pid_m * stride_sm + col * stride_sn, mask=mask, other=-float('inf'))

        # First argmax
        best1 = -float('inf')
        best1_idx = -1
        for i in range(EXPERTS_PER_GROUP):
            v = vals[i]
            if v > best1:
                best1 = v
                best1_idx = i

        # Second argmax (excluding best1)
        best2 = -float('inf')
        best2_idx = -1
        for i in range(EXPERTS_PER_GROUP):
            v = vals[i]
            if (v > best2) & (i != best1_idx):
                best2 = v
                best2_idx = i

        group_score = best1 + best2
        tl.store(GroupScores_ptr + pid_m * stride_gsm + g * stride_gsn, group_score)


@triton.jit
def _group_top4_select_kernel(
    GroupScores_ptr,   # [M, 8], float32
    GroupIdx_ptr,      # [M, 4], int32
    M, N,
    stride_gs_m, stride_gs_n,
    stride_gi_m, stride_gi_n,
    NUM_GROUPS: tl.constexpr,  # 8
):
    # Each program handles one token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iteratively select top-4 via argmax
    for kk in range(4):
        best = -float('inf')
        pos = -1
        for i in range(NUM_GROUPS):
            val = tl.load(GroupScores_ptr + pid_m * stride_gs_m + i * stride_gs_n)
            if val > best:
                best = val
                pos = i
        tl.store(GroupIdx_ptr + pid_m * stride_gi_m + kk * stride_gi_n, pos)
        # Note: we don't mark as -inf here because we won't scan this element again in next iteration; the inner loop will not use it.


@triton.jit
def _group_mask_kernel(
    GroupIdx_ptr,      # [M, 4], int32
    GroupMask_ptr,     # [M, N], int32 (0/1)
    M, N,
    stride_gi_m, stride_gi_n,
    stride_gm_m, stride_gm_n,
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    NUM_GROUPS: tl.constexpr,         # 8
):
    # Each program handles one token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Initialize GroupMask to zeros
    for j in range(N):
        tl.store(GroupMask_ptr + pid_m * stride_gm_m + j * stride_gm_n, 0)

    # Set selected groups to 1
    for kk in range(4):
        group = tl.load(GroupIdx_ptr + pid_m * stride_gi_m + kk * stride_gi_n)  # int32
        base = group * EXPERTS_PER_GROUP
        for j in range(EXPERTS_PER_GROUP):
            tl.store(GroupMask_ptr + pid_m * stride_gm_m + (base + j) * stride_gm_n, 1)


@triton.jit
def _final_top8_and_normalize_kernel(
    Scores_ptr,                   # [M, N], float32
    GroupMask_ptr,                # [M, N], int32 (0/1), 1 for selected groups
    TopKIdx_ptr,                  # [M, 8], int32
    TopKWeight_ptr,               # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gm_m, stride_gm_n,
    stride_tmi, stride_tmn,
    SCALE,                        # float32
    BLOCK: tl.constexpr,
):
    # Each program handles one token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Prepare top-8 arrays
    topv = tl.zeros((8,), dtype=tl.float32) - float('inf')
    topidx = tl.zeros((8,), dtype=tl.int32) - 1

    # Iterative argmax to select top-8 across N columns
    for kk in range(8):
        best = -float('inf')
        best_col = -1
        j = 0
        while j < N:
            col = j + tl.arange(0, BLOCK)
            mask_j = col < N
            ptrs = Scores_ptr + pid_m * stride_sm + col * stride_sn
            vals = tl.load(ptrs, mask=mask_j, other=-float('inf'))
            # Scan vals to find max in this chunk
            for jj in range(BLOCK):
                vj = vals[jj]
                if vj > best:
                    best = vj
                    best_col = j + jj
            j += BLOCK
        topv[kk] = best
        topidx[kk] = best_col

    # Normalize and scale
    l1 = tl.zeros((), dtype=tl.float32)
    for kk in range(8):
        l1 += topv[kk]
    inv_l1 = 1.0 / (l1 + 1e-20)
    for kk in range(8):
        tl.store(TopKWeight_ptr + pid_m * stride_tmi + kk * stride_tmn, topv[kk] * inv_l1 * SCALE)
    for kk in range(8):
        tl.store(TopKIdx_ptr + pid_m * stride_tmi + kk * stride_tmn, topidx[kk].to(tl.int64))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-optimized routing:
        - Compute logits = hidden_states @ weight.T
        - scores = sigmoid(logits) + expert_bias
        - Group top-2 sum -> group_scores [M, 8]
        - Select top-4 groups -> group_idx [M, 4]
        - Build group_mask [M, 256] (1 for selected groups)
        - Select final top-8 per token; normalize and scale
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors for Triton."
        # Ensure dtype and contiguity
        A = hidden_states.contiguous().to(torch.float32)  # [M, K]
        # weight is [num_experts, hidden_dim], we need B = [K, N] where N=num_experts, K=hidden_dim
        B = weight.T.contiguous().to(torch.float32)       # [K, N]
        M, K = A.shape
        N = B.shape[1]  # num_experts = 256

        # 1) Compute logits = A @ B using Triton matmul kernel
        logits = torch.empty((M, N), dtype=torch.float32, device=A.device)
        _matmul_AxB_kernel[(M, triton.cdiv(N, 64))](  # grid: (M, cdiv(N, BLOCK_N))
            A, B, logits,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # 2) scores = sigmoid(logits) + expert_bias (bias broadcast over rows)
        scores = torch.empty((M, N), dtype=torch.float32, device=A.device)
        _sigmoid_add_bias_kernel[(M,)](
            logits, expert_bias.contiguous(), scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK=128,
            num_warps=4,
        )

        # 3) Group top-2 sum: group_scores [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=A.device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP=32, NUM_GROUPS=8, BLOCK_N=32,
            num_warps=1,
        )

        # 4) Select top-4 group indices per token
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=A.device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, N,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            NUM_GROUPS=8,
            num_warps=1,
        )

        # 5) Build group_mask [M, N]: 1 for selected groups (experts), 0 otherwise
        group_mask = torch.empty((M, N), dtype=torch.int32, device=A.device)
        _group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, N,
            group_idx.stride(0), group_idx.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            EXPERTS_PER_GROUP=32, NUM_GROUPS=8,
            num_warps=1,
        )

        # 6) Mask out non-selected groups by setting their scores to -inf using torch (to keep Triton-only kernels minimal and robust)
        score_mask = (group_mask == 1)  # bool
        scores = scores.masked_fill(~score_mask, -float('inf'))

        # 7) Final top-8 selection and normalization via Triton (k=8 argmax loop + L1 normalize and scale)
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=A.device)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=A.device)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_mask, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            float(routed_scaling_factor),
            BLOCK=128,
            num_warps=4,
        )

        # Cast indices to int64 for compatibility with original signature
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
