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
    # 2D launch: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in C
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in C
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile: shape [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A = tl.load(A_tile_ptrs, mask=A_mask, other=0.0)

        # B tile: shape [BLOCK_K, BLOCK_N]
        B_tile_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        B_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        B = tl.load(B_tile_ptrs, mask=B_mask, other=0.0)

        # Matrix multiply accumulate
        acc += tl.dot(A, B)

    # Store C tile
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,    # [M, N], float32
    B_ptr,    # [N], float32
    Y_ptr,    # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_bn,
    BLOCK: tl.constexpr,
):
    # 1D launch: one program per row
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iterate over columns in chunks
    j = 0
    while j < N:
        cols = j + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X_ptr + pid_m * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-x))
        b = tl.load(B_ptr + cols * stride_bn, mask=mask, other=0.0)
        y = s + b
        tl.store(Y_ptr + pid_m * stride_ym + cols * stride_yn, y, mask=mask)
        j += BLOCK


@triton.jit
def _group_top2_sum_kernel(
    Scores_ptr,   # [M, N], float32
    GroupScores_ptr,  # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gsm, stride_gsn,
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    N_GROUPS: tl.constexpr,           # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # We assume num_experts == N_GROUPS * EXPERTS_PER_GROUP (256 in this task).
    for group in range(N_GROUPS):
        group_start = group * EXPERTS_PER_GROUP
        # top-2 values and positions
        v1 = -float('inf'); p1 = -1
        v2 = -float('inf'); p2 = -1
        for j in range(EXPERTS_PER_GROUP):
            idx = group_start + j
            val = tl.load(Scores_ptr + pid_m * stride_sm + idx * stride_sn)
            # update top-2
            if val > v1:
                v2 = v1
                p2 = p1
                v1 = val
                p1 = idx
            elif val > v2:
                v2 = val
                p2 = idx
        # sum of top-2
        sum2 = v1 + v2
        tl.store(GroupScores_ptr + pid_m * stride_gsm + group * stride_gsn, sum2)


@triton.jit
def _group_top4_select_kernel(
    GroupScores_ptr,  # [M, 8], float32
    GroupIdx_ptr,     # [M, 4], int32
    M, N_GROUPS,
    stride_gs_m, stride_gs_n,
    stride_gi_m, stride_gi_n,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iteratively select top-4 via argmax
    for k in range(4):
        best = -float('inf')
        pos = -1
        for g in range(N_GROUPS):
            val = tl.load(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_n)
            if val > best:
                best = val
                pos = g
        # Write selected index
        tl.store(GroupIdx_ptr + pid_m * stride_gi_m + k * stride_gi_n, pos)
        # Optionally mark as used (not needed since we rescan next iteration)


@triton.jit
def _final_top8_and_normalize_kernel(
    Scores_ptr,           # [M, N], float32
    GroupIdx_ptr,         # [M, 4], int32 (unused except for shape)
    TopKIdx_ptr,          # [M, 8], int32
    TopKWeight_ptr,       # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gi_m, stride_gi_n,
    stride_tki_m, stride_tki_n,
    stride_tkw_m, stride_tkw_n,
    SCALE,                # float
    CHUNK: tl.constexpr,  # chunk size for scanning columns (e.g., 128)
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Prepare top-8 arrays
    topv = tl.zeros((8,), dtype=tl.float32) - float('inf')
    topidx = tl.zeros((8,), dtype=tl.int32) - 1

    # Iterative argmax over N columns to select top-8
    for kk in range(8):
        best = -float('inf')
        best_col = -1
        j = 0
        while j < N:
            cols = j + tl.arange(0, CHUNK)
            mask = cols < N
            vals = tl.load(Scores_ptr + pid_m * stride_sm + cols * stride_sn, mask=mask, other=-float('inf'))
            # find max in this chunk
            for jj in range(CHUNK):
                vj = vals[jj]
                if vj > best:
                    best = vj
                    best_col = j + jj
            j += CHUNK
        topv[kk] = best
        topidx[kk] = best_col

    # Normalize by L1 and scale
    l1 = tl.zeros((), dtype=tl.float32)
    for kk in range(8):
        l1 += topv[kk]
    norm = l1 + 1e-20
    for kk in range(8):
        tl.store(TopKWeight_ptr + pid_m * stride_tkw_m + kk * stride_tkw_n, (topv[kk] / norm) * SCALE)
    # store indices
    for kk in range(8):
        tl.store(TopKIdx_ptr + pid_m * stride_tki_m + kk * stride_tki_n, topidx[kk])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure device/dtype
        device = hidden_states.device
        # hidden_states: [M, K], weight: [N, K] in PyTorch (nn.Linear), but we need [K, N] for matmul
        A = hidden_states.contiguous().to(torch.float32)
        M, K = A.shape
        # Prepare B = weight.T as [K, N] (N=num_experts=256)
        # weight is [N, K] from nn.Linear (row-major), so transpose to [K, N]
        Wt = weight.t().contiguous().to(torch.float32)  # [K, N]
        # Allocate logits C = [M, N]
        C = torch.empty((M, Wt.shape[1]), dtype=torch.float32, device=device)

        # 1) GEMM via Triton
        _matmul_no_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(Wt.shape[1], 64))](
            A, Wt, C,
            M, Wt.shape[1], K,
            A.stride(0), A.stride(1),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # 2) Sigmoid + expert_bias via Triton
        Scores = torch.empty_like(C)  # [M, N]
        _sigmoid_add_bias_kernel[(M,)](
            C, expert_bias.to(torch.float32).contiguous(), Scores,
            M, Scores.shape[1],
            C.stride(0), C.stride(1),
            Scores.stride(0), Scores.stride(1),
            expert_bias.stride(0),
            CHUNK=128,
            num_warps=4,
        )

        # 3) Group top-2 sum (per token, per group)
        num_experts = Wt.shape[1]  # 256
        n_group = 8
        experts_per_group = num_experts // n_group  # 32
        group_scores = torch.empty((M, n_group), dtype=torch.float32, device=device)
        _group_top2_sum_kernel[(M,)](
            Scores, group_scores,
            M, num_experts,
            Scores.stride(0), Scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP=experts_per_group, N_GROUPS=n_group,
        )

        # 4) Select top-4 groups
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=1,
        )

        # 5) Final top-8 selection and normalization via Triton
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        _final_top8_and_normalize_kernel[(M,)](
            Scores, group_idx, topk_idx, topk_weight,
            M, num_experts,
            Scores.stride(0), Scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            float(routed_scaling_factor),
            CHUNK=128,
            num_warps=4,
        )

        # Return int64 indices and float32 weights to match original
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
