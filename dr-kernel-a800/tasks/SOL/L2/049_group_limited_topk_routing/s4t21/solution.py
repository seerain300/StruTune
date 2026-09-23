import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_AxB_kernel(
    A_ptr,  # [M, K], float32
    B_ptr,  # [K, N], float32 (weight.T)
    C_ptr,  # [M, N], float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,    # [M, N], float32 (logits)
    B_ptr,    # [N], float32 (expert bias)
    Y_ptr,    # [M, N], float32 (scores)
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    CHUNK: tl.constexpr,
):
    # One program per row, process columns in chunks
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    j = 0
    while j < N:
        offs_n = j + tl.arange(0, CHUNK)
        # Load row chunk
        x_ptrs = X_ptr + pid_m * stride_xm + offs_n * stride_xn
        y_ptrs = Y_ptr + pid_m * stride_ym + offs_n * stride_yn
        mask = offs_n < N
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        # sigmoid
        x = 1.0 / (1.0 + tl.exp(-x))
        b = tl.load(B_ptr + offs_n, mask=mask, other=0.0)
        y = x + b
        tl.store(y_ptrs, y, mask=mask)
        j += CHUNK


@triton.jit
def _group_top2_sum_kernel(
    S_ptr,     # [M, N], float32 (scores)
    GS_ptr,    # [M, 8], float32 (group_scores)
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    EXPERTS_PER_GROUP: tl.constexpr, N_GROUPS: tl.constexpr,
):
    # One program per token (row)
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # Compute group-wise top-2 sum
    # Reshape logic: groups are contiguous blocks of EXPERTS_PER_GROUP
    group_scores = tl.zeros((N_GROUPS,), dtype=tl.float32)
    for g in range(N_GROUPS):
        base = g * EXPERTS_PER_GROUP
        # First argmax
        max_val = -float('inf')
        max_idx = 0
        for i in range(EXPERTS_PER_GROUP):
            idx = base + i
            ptr = S_ptr + pid_m * stride_sm + idx * stride_sn
            val = tl.load(ptr)
            if val > max_val:
                max_val = val
                max_idx = idx
        # Exclude max_idx, second argmax
        second_val = -float('inf')
        for i in range(EXPERTS_PER_GROUP):
            idx = base + i
            if idx != max_idx:
                ptr = S_ptr + pid_m * stride_sm + idx * stride_sn
                val = tl.load(ptr)
                if val > second_val:
                    second_val = val
        group_scores[g] = max_val + second_val
    # Store group_scores to [M, 8]
    for g in range(N_GROUPS):
        out_ptr = GS_ptr + pid_m * stride_gm + g * stride_gn
        tl.store(out_ptr, group_scores[g])


@triton.jit
def _group_top4_select_kernel(
    GS_ptr,    # [M, 8], float32 (group_scores)
    GI_ptr,    # [M, 4], int32 (group_idx)
    M, N_GROUPS,
    stride_gsm, stride_gsn,
    stride_gim, stride_gin,
):
    # One program per token (row), iterative argmax for top-4
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # Work with values as float32; track selected indices as int32
    top_vals = tl.full((4,), -float('inf'), dtype=tl.float32)
    top_idxs = tl.zeros((4,), dtype=tl.int32)
    # First top-1
    best_val = -float('inf')
    best_idx = 0
    for g in range(N_GROUPS):
        ptr = GS_ptr + pid_m * stride_gsm + g * stride_gsn
        val = tl.load(ptr)
        if val > best_val:
            best_val = val
            best_idx = g
    top_vals[0] = best_val
    top_idxs[0] = best_idx
    # Second top-2
    for g in range(N_GROUPS):
        if g != best_idx:
            ptr = GS_ptr + pid_m * stride_gsm + g * stride_gsn
            val = tl.load(ptr)
            if val > top_vals[1]:
                top_vals[1] = val
                top_idxs[1] = g
    # Third top-3
    for g in range(N_GROUPS):
        if (g != best_idx) and (g != top_idxs[1]):
            ptr = GS_ptr + pid_m * stride_gsm + g * stride_gsn
            val = tl.load(ptr)
            if val > top_vals[2]:
                top_vals[2] = val
                top_idxs[2] = g
    # Fourth top-4
    for g in range(N_GROUPS):
        if (g != best_idx) and (g != top_idxs[1]) and (g != top_idxs[2]):
            ptr = GS_ptr + pid_m * stride_gsm + g * stride_gsn
            val = tl.load(ptr)
            if val > top_vals[3]:
                top_vals[3] = val
                top_idxs[3] = g
    # Store indices
    for i in range(4):
        out_ptr = GI_ptr + pid_m * stride_gim + i * stride_gin
        tl.store(out_ptr, top_idxs[i])


@triton.jit
def _final_top8_and_normalize_kernel(
    S_ptr,          # [M, N], float32 (masked scores)
    GI_ptr,         # [M, 4], int32 (group_idx)
    TOPK_IDX_ptr,   # [M, 8], int32 (final selected expert indices)
    TOPK_W_ptr,     # [M, 8], float32 (normalized weights * routed_scaling_factor)
    M, N,
    stride_sm, stride_sn,
    stride_gim, stride_gin,
    stride_tm, stride_tn,
    routed_scale,   # float32
    CHUNK: tl.constexpr,
):
    # One program per token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Build score_mask based on group_idx: selected groups = 1, others = 0
    # group_idx [M, 4], each entry g in [0..7]
    score_mask = tl.zeros((N,), dtype=tl.float32)
    # For each group index, set corresponding 32 experts to 1
    for i in range(4):
        g = tl.load(GI_ptr + pid_m * stride_gim + i * stride_gin)  # int32
        base = g * 32
        for j in range(CHUNK):
            idx = base + j
            # limit to N
            if idx < N:
                score_mask[idx] = 1.0

    # Mask non-selected to -inf
    for j in range(CHUNK):
        idx = j
        if idx < N:
            ptr = S_ptr + pid_m * stride_sm + idx * stride_sn
            val = tl.load(ptr)
            new_val = val if (score_mask[idx] == 1.0) else ( -float('inf') )
            tl.store(ptr, new_val)

    # Now select top-8 via iterative argmax
    selected_vals = tl.zeros((8,), dtype=tl.float32)
    selected_idxs = tl.zeros((8,), dtype=tl.int32)
    # Loop 8 times to pick top-8
    for r in range(8):
        best_val = -float('inf')
        best_idx = 0
        for i in range(N):
            ptr = S_ptr + pid_m * stride_sm + i * stride_sn
            val = tl.load(ptr)
            if val > best_val:
                best_val = val
                best_idx = i
        selected_vals[r] = best_val
        selected_idxs[r] = best_idx
        # Set best_idx to -inf so it won't be selected again
        tl.store(S_ptr + pid_m * stride_sm + best_idx * stride_sn, -float('inf'))

    # Normalize selected_vals (L1), then scale and store
    sum_vals = 0.0
    for r in range(8):
        sum_vals += selected_vals[r]
    inv = 1.0 / (sum_vals + 1e-20)
    for r in range(8):
        selected_vals[r] = selected_vals[r] * inv * routed_scale

    # Store results
    for r in range(8):
        out_idx_ptr = TOPK_IDX_ptr + pid_m * stride_tm + r * stride_tn
        out_w_ptr = TOPK_W_ptr + pid_m * stride_tm + r * stride_tn
        tl.store(out_idx_ptr, selected_idxs[r])
        tl.store(out_w_ptr, selected_vals[r])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        device = hidden_states.device
        dtype = torch.float32

        # 1) Prepare A and B for GEMM
        A = hidden_states.contiguous().to(dtype)
        K = A.shape[1]
        # weight is [num_experts, hidden_dim], we want B = weight.T -> [hidden_dim, num_experts] = [K, N]
        B = weight.T.contiguous().to(dtype)  # [K, N], N=num_experts=256
        M, K = A.shape
        N = B.shape[1]

        # Allocate logits
        logits = torch.empty((M, N), dtype=dtype, device=device)

        # Launch Triton GEMM
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = B.stride(0), B.stride(1)
        stride_cm, stride_cn = logits.stride(0), logits.stride(1)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_AxB_kernel[grid](
            A, B, logits,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 2) Sigmoid + bias via Triton
        scores = torch.empty_like(logits, dtype=dtype, device=device)
        stride_xm, stride_xn = logits.stride(0), logits.stride(1)
        stride_ym, stride_yn = scores.stride(0), scores.stride(1)
        _sigmoid_add_bias_kernel[(M,)](  # process entire row; use CHUNK=N to cover all columns
            logits, expert_bias.to(dtype), scores,
            M, N,
            stride_xm, stride_xn,
            stride_ym, stride_yn,
            CHUNK=N,
            num_warps=4,
        )

        # 3) Group top-2 sum -> [M, 8]
        group_scores = torch.empty((M, 8), dtype=dtype, device=device)
        stride_sm, stride_sn = scores.stride(0), scores.stride(1)
        stride_gm, stride_gn = group_scores.stride(0), group_scores.stride(1)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            stride_sm, stride_sn,
            stride_gm, stride_gn,
            EXPERTS_PER_GROUP=32, N_GROUPS=8,
            num_warps=1, num_stages=1,
        )

        # 4) Group top-4 indices -> [M, 4]
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_gsm, stride_gsn = group_scores.stride(0), group_scores.stride(1)
        stride_gim, stride_gin = group_idx.stride(0), group_idx.stride(1)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, 8,
            stride_gsm, stride_gsn,
            stride_gim, stride_gin,
            num_warps=1, num_stages=1,
        )

        # 5) Final top-8 selection and normalization via Triton
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((M, 8), dtype=dtype, device=device)
        stride_sm_f, stride_sn_f = scores.stride(0), scores.stride(1)
        stride_gim_f, stride_gin_f = group_idx.stride(0), group_idx.stride(1)
        stride_tm, stride_tn = topk_idx.stride(0), topk_idx.stride(1)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            stride_sm_f, stride_sn_f,
            stride_gim_f, stride_gin_f,
            stride_tm, stride_tn,
            float(routed_scaling_factor),
            CHUNK=N,  # process all N=256 columns in one pass
            num_warps=4, num_stages=1,
        )

        # Return as required (int64 indices and float32 weights)
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
