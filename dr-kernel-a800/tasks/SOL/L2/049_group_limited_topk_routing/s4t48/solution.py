import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_AxB_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32 (we'll form this via weight.T with appropriate strides)
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile as [BLOCK_K, BLOCK_N] from B[k, n]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk) + (offs_n[None, :] * stride_bn)
        b_mask = (offs_k < K)[:, None] & (offs_n < N)[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # acc += a @ b
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,    # [M, N], float32
    B_ptr,    # [N], float32
    Y_ptr,    # [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_b,
    stride_ym, stride_yn,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)

    # Load tile
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm) + (offs_n[None, :] * stride_xn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Sigmoid
    sig = 1.0 / (1.0 + tl.exp(-x))

    # Load bias for columns
    b = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK]
    b = b[None, :]  # broadcast along rows

    y = sig + b

    # Store
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym) + (offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def _group_top2_sum_kernel(
    S_ptr,   # [M, N], float32
    GS_ptr,  # [M, 8], float32
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    EXP_PER_GRP: tl.constexpr, GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Reshape S to [M, GROUPS, EXP_PER_GRP]
    j = 0
    while j < N:
        offs_n = j + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        s_ptrs = S_ptr + pid_m * stride_sm + offs_n * stride_sn
        s = tl.load(s_ptrs, mask=mask, other=0.0)  # [BLOCK_N]

        # For each group, do top-2
        # Unrolled small loop since GROUPS=8 and EXP_PER_GRP=32 are constexpr here
        for g in range(GROUPS):
            start = g * EXP_PER_GRP
            # Prepare group slice; since BLOCK_N > 32, we can load 32 elements per group by looping
            top1 = -float('inf')
            top2 = -float('inf')
            idx1 = -1
            idx2 = -1
            k = 0
            while k < EXP_PER_GRP:
                col = start + k
                # scalar load from s vector at position col
                val = s[col]
                # argmax logic
                if val > top1:
                    top2 = top1
                    idx2 = idx1
                    top1 = val
                    idx1 = col
                elif val > top2:
                    top2 = val
                    idx2 = col
                k += 1
            # Sum of top-2 for this group
            group_sum = top1 + top2
            # Store
            gs_ptr = GS_ptr + pid_m * stride_gm + g * stride_gn
            tl.store(gs_ptr, group_sum)
            j += BLOCK_N
        # We exit while loop once j >= N
    return


@triton.jit
def _group_top4_select_kernel(
    GS_ptr,   # [M, 8], float32
    IDX_ptr,  # [M, 4], int32 (output: group indices)
    M, N,
    stride_gm, stride_gn,
    stride_im, stride_in,
    TOPK_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # We will select top-4 group indices from GS[M,8]
    # Using iterative argmax
    # Each iteration: find max in GS, record its column index, set that to -inf
    selected_cols = tl.zeros((TOPK_GROUPS,), dtype=tl.int1)
    selected_vals = tl.full((TOPK_GROUPS,), -float('inf'), dtype=tl.float32)
    selected_idx = tl.full((TOPK_GROUPS,), -1, dtype=tl.int32)

    # For groups dimension, handle columns 0..7 only
    g = 0
    while g < TOPK_GROUPS:
        # Scan columns 0..7 to find max
        max_val = -float('inf')
        max_col = -1
        j = 0
        while j < 8:
            gs_ptr = GS_ptr + pid_m * stride_gm + j * stride_gn
            val = tl.load(gs_ptr)
            if val > max_val:
                max_val = val
                max_col = j
            j += 1
        # Store index
        tl.store(IDX_ptr + pid_m * stride_im + g * stride_in, max_col)
        # Mark selected in our selected_cols (we can mark after setting -inf, but keep a separate bool list)
        selected_cols[g] = 1
        g += 1


@triton.jit
def _final_top8_and_normalize_kernel(
    S_ptr,            # [M, N], float32
    GROUP_IDX_ptr,    # [M, 4], int32
    OUT_IDX_ptr,      # [M, 8], int32
    OUT_WT_ptr,       # [M, 8], float32 (normalized and scaled)
    M, N,
    stride_sm, stride_sn,
    stride_gim, stride_gin,
    stride_om, stride_on,
    routed_scaling_factor: tl.float32,
    EXP_PER_GRP: tl.constexpr, GROUPS: tl.constexpr, TOPK_GROUPS: tl.constexpr, TOPK_FINAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Step A: Build score_mask from group_idx
    # score_mask[i, n] = 1 if group(i, n) in selected 4 groups, else 0
    # We need to check each n's group index in GROUPS and see if it matches any in GROUP_IDX_ptr[pid_m, :]
    # Initialize mask
    score_mask = tl.zeros((N,), dtype=tl.int1)
    # Each group covers EXP_PER_GRP=32 experts, groups 0..7
    # For each selected group index g in [0..3]
    g = 0
    while g < TOPK_GROUPS:
        gi = tl.load(GROUP_IDX_ptr + pid_m * stride_gim + g * stride_gin)  # int32
        start = gi * EXP_PER_GRP
        j = start
        while j < start + EXP_PER_GRP:
            score_mask[j] = 1
            j += 1
        g += 1

    # Step B: Mask non-selected -> -inf
    j = 0
    while j < N:
        offs_n = j + tl.arange(0, BLOCK_N)
        mask = offs_n < N
        s_ptrs = S_ptr + pid_m * stride_sm + offs_n * stride_sn
        vals = tl.load(s_ptrs, mask=mask, other=0.0)
        m = score_mask[offs_n]  # broadcast
        vals = tl.where(m, vals, -float('inf'))
        tl.store(S_ptr + pid_m * stride_sm + offs_n * stride_sn, vals, mask=mask)
        j += BLOCK_N

    # Step C: Final top-8 selection via iterative argmax on masked S
    selected = tl.zeros((TOPK_FINAL,), dtype=tl.int1)
    selected_vals = tl.full((TOPK_FINAL,), -float('inf'), dtype=tl.float32)
    selected_idx = tl.full((TOPK_FINAL,), -1, dtype=tl.int32)

    k = 0
    while k < TOPK_FINAL:
        max_val = -float('inf')
        max_pos = -1
        j = 0
        while j < N:
            offs_n = j + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            s_ptrs = S_ptr + pid_m * stride_sm + offs_n * stride_sn
            vals = tl.load(s_ptrs, mask=mask, other=-float('inf'))
            # Find max in this chunk
            # Since vals is vector, we can reduce to scalar
            # Triton scalar reduction pattern
            local_max = -float('inf')
            r = 0
            while r < BLOCK_N:
                v = vals[r]
                if v > local_max:
                    local_max = v
                r += 1
            # Now compare local_max to max_val
            if local_max > max_val:
                # Find exact position of max_val in offs_n
                max_val = local_max
                # Scan again to find exact index
                r = 0
                while r < BLOCK_N:
                    v = vals[r]
                    if v == max_val:
                        max_pos = j + r
                        break
                    r += 1
            j += BLOCK_N
        tl.store(OUT_IDX_ptr + pid_m * stride_om + k * stride_on, max_pos)
        tl.store(OUT_WT_ptr + pid_m * stride_om + k * stride_on, max_val)
        selected[k] = 1
        k += 1

    # Step D: Normalize selected weights and scale
    sum_vals = 0.0
    k = 0
    while k < TOPK_FINAL:
        val = tl.load(OUT_WT_ptr + pid_m * stride_om + k * stride_on)
        sum_vals += val
        k += 1

    k = 0
    while k < TOPK_FINAL:
        val = tl.load(OUT_WT_ptr + pid_m * stride_om + k * stride_on)
        norm = val / sum_vals
        scaled = norm * routed_scaling_factor
        tl.store(OUT_WT_ptr + pid_m * stride_om + k * stride_on, scaled)
        k += 1


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,  # [M, K], float16/float32, will be cast to float32 for compute
        weight: torch.Tensor,          # [N, K] (nn.Linear: [num_experts, hidden_dim])
        expert_bias: torch.Tensor,     # [N], float32
        routed_scaling_factor: float,
    ):
        device = hidden_states.device
        dtype = torch.float32

        # Ensure inputs are float32 for Triton compute
        A = hidden_states.to(dtype).contiguous()
        M, K = A.shape
        # B is weight.T as [K, N]
        B = weight.transpose(0, 1).contiguous()  # [K, N]
        N = B.shape[1]

        # Output C = [M, N]
        C_logits = torch.empty((M, N), device=device, dtype=dtype)

        # Launch matmul kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (_ceil_div(M, BLOCK_M), _ceil_div(N, BLOCK_N))
        _matmul_AxB_kernel[grid](
            A, B, C_logits,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C_logits.stride(0), C_logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 2) Sigmoid + bias
        S_scores = torch.empty_like(C_logits)
        _sigmoid_add_bias_kernel[(M,)](
            C_logits, expert_bias, S_scores,
            M, N,
            C_logits.stride(0), C_logits.stride(1),
            expert_bias.stride(0),
            S_scores.stride(0), S_scores.stride(1),
            BLOCK=1024,
            num_warps=4,
        )

        # 3) Group top-2 sum: [M, 8]
        group_scores = torch.empty((M, 8), device=device, dtype=dtype)
        _group_top2_sum_kernel[(M,)](
            S_scores, group_scores,
            M, N,
            S_scores.stride(0), S_scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXP_PER_GRP=32, GROUPS=8,
            BLOCK_N=128,
            num_warps=2,
        )

        # 4) Group top-4 selection: [M, 4]
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, 8,  # second dim is 8
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            TOPK_GROUPS=4,
            BLOCK=128,
            num_warps=2,
        )

        # 5) Final top-8 and normalize: [M, 8] indices and weights
        OUT_IDX = torch.empty((M, 8), device=device, dtype=torch.int32)
        OUT_WT = torch.empty((M, 8), device=device, dtype=dtype)
        _final_top8_and_normalize_kernel[(M,)](
            S_scores, group_idx, OUT_IDX, OUT_WT,
            M, N,
            S_scores.stride(0), S_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            OUT_IDX.stride(0), OUT_IDX.stride(1),
            OUT_WT.stride(0), OUT_WT.stride(1),
            routed_scaling_factor,
            EXP_PER_GRP=32, GROUPS=8, TOPK_GROUPS=4, TOPK_FINAL=8,
            BLOCK_N=128,
            num_warps=4,
        )

        # Cast types to match original: indices int64, weights float32
        topk_idx = OUT_IDX.to(torch.int64)
        topk_weight = OUT_WT

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
