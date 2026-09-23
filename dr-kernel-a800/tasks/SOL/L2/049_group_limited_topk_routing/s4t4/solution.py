import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_no_bias_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: tile over M and N, loop over K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BM, BK]
    B_tile_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # [BK, BN]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(
            A_tile_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_tile_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        A_tile_ptrs += BLOCK_K * stride_ak
        B_tile_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
    # 1D grid over rows; iterate columns in chunks
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
    Scores_ptr,        # [M, N], float32
    GroupScores_ptr,   # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gsm, stride_gsn,
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    NUM_GROUPS: tl.constexpr,         # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    for group in range(NUM_GROUPS):
        start = group * EXPERTS_PER_GROUP
        # pass 1: find max1
        max1 = -float('inf')
        max1_pos = -1
        for i in range(EXPERTS_PER_GROUP):
            val = tl.load(Scores_ptr + pid_m * stride_sm + (start + i) * stride_sn)
            if val > max1:
                max1 = val
                max1_pos = start + i
        # pass 2: find max2 (excluding max1_pos)
        max2 = -float('inf')
        max2_pos = -1
        for i in range(EXPERTS_PER_GROUP):
            idx = start + i
            if idx != max1_pos:
                val = tl.load(Scores_ptr + pid_m * stride_sm + idx * stride_sn)
                if val > max2:
                    max2 = val
                    max2_pos = idx
        sum2 = max1 + max2
        tl.store(GroupScores_ptr + pid_m * stride_gsm + group * stride_gsn, sum2)


@triton.jit
def _group_top4_select_kernel(
    GroupScores_ptr,    # [M, 8], float32
    GroupIdx_ptr,       # [M, 4], int32
    M, N_COLS,          # N_COLS = 8
    stride_gsm, stride_gsn,
    stride_gim, stride_gin,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iterative argmax to select top-4 group indices
    for kk in range(4):
        best = -float('inf')
        pos = -1
        for i in range(8):
            val = tl.load(GroupScores_ptr + pid_m * stride_gsm + i * stride_gsn)
            if val > best:
                best = val
                pos = i
        tl.store(GroupIdx_ptr + pid_m * stride_gim + kk * stride_gin, pos)


@triton.jit
def _final_top8_and_normalize_kernel(
    Scores_ptr,           # [M, N], float32
    GroupIdx_ptr,         # [M, 4], int32
    TopKIdx_ptr,          # [M, 8], int32
    TopKWeight_ptr,       # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gim, stride_gin,
    stride_tmi, stride_tmn,
    SCALE,                # float32
    CHUNK: tl.constexpr,  # chunk size for scanning columns
):
    # One program per token (row)
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Prepare top-8 arrays
    topv = tl.zeros((8,), dtype=tl.float32) - float('inf')
    topidx = tl.zeros((8,), dtype=tl.int32) - 1

    # Scan all N experts in chunks, ignore groups >=4
    j = 0
    while j < N:
        cols = j + tl.arange(0, CHUNK)
        mask = cols < N
        vals = tl.load(Scores_ptr + pid_m * stride_sm + cols * stride_sn, mask=mask, other=-float('inf'))
        # For each column in chunk, check group membership and update top-8
        for jj in range(CHUNK):
            col_j = j + jj
            if col_j < N:
                # Determine group and validity
                group_j = col_j // 32
                valid = group_j < 4  # selected groups are 0..3
                # If valid, update top-8
                if valid:
                    val_j = vals[jj]
                    # Insert into top-8 (shift larger ones down)
                    for kk in range(8):
                        if val_j > topv[kk]:
                            # shift down
                            for k2 in range(7, kk, -1):
                                topv[k2] = topv[k2 - 1]
                                topidx[k2] = topidx[k2 - 1]
                            topv[kk] = val_j
                            topidx[kk] = col_j
                            break
        j += CHUNK

    # Normalize L1 and apply scale
    l1 = 0.0
    for kk in range(8):
        l1 += topv[kk]
    l1 = l1 + 1e-20
    for kk in range(8):
        w = topv[kk] / l1 * SCALE
        tl.store(TopKWeight_ptr + pid_m * stride_tmi + kk * stride_tmn, w)
        tl.store(TopKIdx_ptr + pid_m * stride_tmi + kk * stride_tmn, topidx[kk])


class ModelNew(torch.nn.Module):
    def forward(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float,
    ):
        """
        Triton-only implementation of the original routing pipeline.
        Returns:
        - topk_idx: [num_tokens, 8], int64
        - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be on CUDA device"
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [hidden_dim, num_experts]"
        assert expert_bias.dim() == 1, "expert_bias must be [num_experts]"
        assert weight.shape[1] == 256, "num_experts must be 256"
        M, K = hidden_states.shape
        N = 256

        # 1) Compute logits = hidden_states @ weight (no bias) via Triton
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_no_bias_kernel[grid](
            hidden_states, weight,
            logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Compute scores = sigmoid(logits) + expert_bias via Triton
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        _sigmoid_add_bias_kernel[(M,)](
            logits, expert_bias,
            scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK=256,
            num_warps=4,
        )

        # 3) Group top-2 sum and write [M, 8] via Triton
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP=32,
            NUM_GROUPS=8,
            num_warps=1,
        )

        # 4) Select top-4 groups via Triton, write [M, 4]
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, 8,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=1,
        )

        # 5) Final top-8 selection and normalization via Triton, write [M, 8] idx and weights
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            float(routed_scaling_factor),
            CHUNK=128,
            num_warps=4,
        )

        # Return as in original: int64 indices and float32 weights
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
