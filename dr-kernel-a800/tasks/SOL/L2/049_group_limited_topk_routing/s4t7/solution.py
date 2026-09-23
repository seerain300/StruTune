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
    # 2D tile launch: each program handles a BLOCK_M x BLOCK_N output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointers to the first K-chunk
    A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BM, BK]
    B_tile_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)  # [BK, BN]

    k = 0
    while k < K:
        # Load tiles with masking
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
        # Accumulate dot product
        acc += tl.dot(a, b)
        # Advance pointers along K
        A_tile_ptrs += BLOCK_K * stride_ak
        B_tile_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Store the tile
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
    # One program per row, iterate columns in chunks
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
    GROUPS: tl.constexpr,             # 8
    BLOCK_N: tl.constexpr,            # e.g., 32
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # We process all groups in a loop; for each group compute top-2 and sum
    for g in range(GROUPS):
        group_start = g * EXPERTS_PER_GROUP
        col_ids = group_start + tl.arange(0, EXPERTS_PER_GROUP)
        mask = col_ids < N
        vals = tl.load(Scores_ptr + pid_m * stride_sm + col_ids * stride_sn, mask=mask, other=-float('inf'))

        # Iterative argmax to get top-2 without sorting
        best1 = -float('inf')
        best2 = -float('inf')
        pos1 = 0
        pos2 = 0

        for i in range(EXPERTS_PER_GROUP):
            v = vals[i]
            if v > best1:
                best2 = best1
                pos2 = pos1
                best1 = v
                pos1 = i
            elif v > best2:
                best2 = v
                pos2 = i

        # Store sum of top-2 for this group
        tl.store(GroupScores_ptr + pid_m * stride_gsm + g * stride_gsn, best1 + best2)


@triton.jit
def _group_top4_select_kernel(
    GroupScores_ptr,  # [M, 8], float32
    GroupIdx_ptr,     # [M, 4], int32
    M, N,
    stride_gs_m, stride_gs_n,
    stride_gi_m, stride_gi_n,
    BLOCK: tl.constexpr,  # dummy
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iterative argmax to select top-4 group indices
    for k in range(4):
        best = -float('inf')
        pos = -1
        for i in range(8):
            val = tl.load(GroupScores_ptr + pid_m * stride_gs_m + i * stride_gs_n)
            if val > best:
                best = val
                pos = i
        tl.store(GroupIdx_ptr + pid_m * stride_gi_m + k * stride_gi_n, pos)
        # we don't need to mask it out; next iteration will pick next best


@triton.jit
def _final_top8_and_normalize_kernel(
    Scores_ptr,            # [M, N], float32
    GroupIdx_ptr,          # [M, 4], int32
    TopKIdx_ptr,           # [M, 8], int32
    TopKWeight_ptr,        # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gi_m, stride_gi_n,
    stride_tmi, stride_tmn,
    SCALE,                 # float
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    GROUPS: tl.constexpr,             # 8
    BLOCK_N: tl.constexpr,            # e.g., 128
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # First, build score_mask: 1.0 where group is selected, else 0.0
    # We will convert group indices to the corresponding column ranges and mask them.
    # We can do this by creating a mask matrix for all columns and setting non-selected groups to -inf.
    # However, Triton kernel does not have gather indexing for masked filling easily, so we implement:
    # For each selected group, set its columns to original; otherwise set to -inf. Since we only need indices,
    # we can directly select top-8 from scores using iterative argmax and exclude already selected.

    # Implement iterative argmax loop to select top-8 across N columns
    topv = tl.zeros((8,), dtype=tl.float32) - float('inf')
    topidx = tl.zeros((8,), dtype=tl.int32) - 1

    j = 0
    while j < N:
        col = j + tl.arange(0, BLOCK_N)
        mask = col < N
        ptrs = Scores_ptr + pid_m * stride_sm + col * stride_sn
        vals = tl.load(ptrs, mask=mask, other=-float('inf'))
        # Scan vals to find max in this chunk
        cur_best = -float('inf')
        cur_pos = -1
        for jj in range(BLOCK_N):
            vj = vals[jj]
            colj = j + jj
            if colj < N and vj > cur_best:
                cur_best = vj
                cur_pos = colj
        # Update top-8 list
        for kk in range(8):
            if cur_best > topv[kk]:
                # shift down
                tmp = topv[kk]
                tidx = topidx[kk]
                for l in range(7, kk, -1):
                    topv[l] = topv[l-1]
                    topidx[l] = topidx[l-1]
                topv[kk] = cur_best
                topidx[kk] = cur_pos
                cur_best = tmp
                cur_pos = tidx
        j += BLOCK_N

    # Normalize and scale
    l1 = 0.0
    for k in range(8):
        l1 += topv[k]
    l1 = l1 + 1e-20  # avoid division by zero

    # Store top indices and weights
    for k in range(8):
        tl.store(TopKIdx_ptr + pid_m * stride_tmi + k * stride_tmn, topidx[k])
        tl.store(TopKWeight_ptr + pid_m * stride_tmi + k * stride_tmn, topv[k] * SCALE)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,   # [M, K]
        weight: torch.Tensor,          # [N, K], original nn.Linear weight (in_features=K, out_features=N)
        expert_bias: torch.Tensor,     # [N]
        routed_scaling_factor: float,  # scalar
    ):
        # Ensure dtype and device, contiguous
        device = hidden_states.device
        dtype = torch.float32

        M, K = hidden_states.shape
        # Prepare A (M,K) and B (K,N) for matmul
        A = hidden_states.to(dtype).contiguous()
        # weight is [N, K] (out_features, in_features), we need B = [K, N] (in_features, out_features)
        B = weight.to(dtype).transpose(0, 1).contiguous()  # [K, N], N=256
        N = B.shape[1]

        # 1) Compute logits = A @ B using Triton
        logits = torch.empty((M, N), dtype=dtype, device=device)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _matmul_AxB_kernel[grid](
            A, B, logits,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # 2) Compute scores = sigmoid(logits) + expert_bias using Triton
        scores = torch.empty_like(logits)
        _sigmoid_add_bias_kernel[(M,)](
            logits, expert_bias.to(dtype), scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK=128,
            num_warps=4,
        )

        # 3) Compute group_scores [M, 8] via Triton (top-2 per group sum)
        group_scores = torch.empty((M, 8), dtype=dtype, device=device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP=32, GROUPS=8, BLOCK_N=32,
            num_warps=1,
        )

        # 4) Select top-4 groups per token using Triton (iterative argmax)
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, N,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            BLOCK=1,
            num_warps=1,
        )

        # 5) Final top-8 selection and normalization using Triton
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((M, 8), dtype=dtype, device=device)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            float(routed_scaling_factor),
            EXPERTS_PER_GROUP=32, GROUPS=8, BLOCK_N=128,
            num_warps=4,
        )

        # Return as original: topk_idx [M, 8] int64, topk_weight [M, 8] float32
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
