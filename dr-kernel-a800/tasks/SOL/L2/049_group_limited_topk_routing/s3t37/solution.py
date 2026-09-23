import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K] = hidden, float32, contiguous
    B_ptr,  # [K, N] = weight.T, float32, contiguous
    C_ptr,  # [M, N] = logits, float32, contiguous
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: one program per tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _sigmoid_bias_kernel(
    X_ptr,   # [M, N] logits
    Bias_ptr, # [N] float32
    Y_ptr,   # [M, N] sigmoid + bias
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_b,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )
    b = tl.load(Bias_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)  # [N]
    y = 1.0 / (1.0 + tl.exp(-x)) + b  # broadcast b over rows
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _group_top2_kernel(
    Scores_ptr,          # [M, 8, 32], float32
    GroupScores_ptr,     # [M, 8], float32
    SelectedI_ptr,       # [M, 4], int32 (will store indices of selected groups)
    M, N_GROUP, EXP_PER_GROUP,
    stride_sm, stride_sgrp, stride_sexp,
    stride_gsm, stride_gsn,
    stride_si_m, stride_si_n,
):
    # Each program processes one token m
    m = tl.program_id(0)
    if m >= M:
        return

    # Compute per-group top-2, sum, and store group scores and indices
    for g in range(0, N_GROUP):
        base = m * stride_sm + g * stride_sgrp
        group_top = [-1.0e20, -1.0e20]
        group_idx = [0, 0]
        for e in range(0, EXP_PER_GROUP):
            val = tl.load(Scores_ptr + base + e * stride_sexp)
            pos = 0
            while pos < 2 and val > group_top[pos]:
                pos += 1
            if pos < 2:
                tmp = group_top[pos]
                tmp_idx = group_idx[pos]
                for j in range(pos, 1):
                    group_top[j] = group_top[j + 1]
                    group_idx[j] = group_idx[j + 1]
                group_top[1] = tmp
                group_idx[1] = tmp_idx
                group_top[pos] = val
                group_idx[pos] = e
        group_sum = group_top[0] + group_top[1]
        tl.store(GroupScores_ptr + m * stride_gsm + g * stride_gsn, group_sum)
        if g < 4:
            tl.store(SelectedI_ptr + m * stride_si_m + g * stride_si_n, group_idx[0])


@triton.jit
def _select_groups_kernel(
    GroupScores_ptr,    # [M, 8], float32
    SelectedI_ptr,      # [M, 4], int32
    M, N_GROUP,
    stride_gsm, stride_gsn,
    stride_si_m, stride_si_n,
):
    m = tl.program_id(0)
    if m >= M:
        return
    top_vals = [-1.0e20, -1.0e20, -1.0e20, -1.0e20]
    top_idxs = [0, 0, 0, 0]
    for g in range(0, N_GROUP):
        val = tl.load(GroupScores_ptr + m * stride_gsm + g * stride_gsn)
        pos = 0
        while pos < 4 and val > top_vals[pos]:
            pos += 1
        if pos < 4:
            tmp_score = top_vals[pos]
            tmp_idx = top_idxs[pos]
            for j in range(pos, 3):
                top_vals[j] = top_vals[j + 1]
                top_idxs[j] = top_idxs[j + 1]
            top_vals[3] = tmp_score
            top_idxs[3] = tmp_idx
            top_vals[pos] = val
            top_idxs[pos] = g
    for i in range(0, 4):
        tl.store(SelectedI_ptr + m * stride_si_m + i * stride_si_n, top_idxs[i])


@triton.jit
def _build_groupmask_kernel(
    SelectedI_ptr,      # [M, 4], int32
    GroupMask_ptr,      # [M, 8], float32 (one-hot mask)
    M, N_GROUP,
    stride_si_m, stride_si_n,
    stride_gm_m, stride_gm_n,
):
    m = tl.program_id(0)
    if m >= M:
        return
    for i in range(0, 4):
        g = tl.load(SelectedI_ptr + m * stride_si_m + i * stride_si_n)
        # place 1.0 at position g in group_mask
        tl.store(GroupMask_ptr + m * stride_gm_m + g * stride_gm_n, 1.0)


@triton.jit
def _mask_scores_kernel(
    Scores_ptr,           # [M, 8, 32], float32 (sigmoid + bias)
    GroupMask_ptr,        # [M, 8], float32
    MaskedScores_ptr,     # [M, 8, 32], float32
    M, N_GROUP, EXP_PER_GROUP,
    stride_sm, stride_sgrp, stride_sexp,
    stride_gm_m, stride_gm_n,
    stride_msm, stride_msg, stride_mse,
):
    m = tl.program_id(0)
    if m >= M:
        return
    for g in range(0, N_GROUP):
        group_mask_val = tl.load(GroupMask_ptr + m * stride_gm_m + g * stride_gm_n)
        base = m * stride_sm + g * stride_sgrp
        for e in range(0, EXP_PER_GROUP):
            val = tl.load(Scores_ptr + base + e * stride_sexp)
            if group_mask_val == 0.0:
                val = -1.0e20  # set to -inf equivalent
            tl.store(MaskedScores_ptr + m * stride_msm + g * stride_msg + e * stride_mse, val)


@triton.jit
def _select_final_top8_kernel(
    MaskedScores_ptr,    # [M, 8, 32], float32
    SelectedJ_ptr,       # [M, 8], int32
    M, N_GROUP, EXP_PER_GROUP,
    stride_msm, stride_msg, stride_mse,
    stride_sj_m, stride_sj_n,
):
    m = tl.program_id(0)
    if m >= M:
        return
    top_vals = [-1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20]
    top_idxs = [0, 0, 0, 0, 0, 0, 0, 0]
    for g in range(0, N_GROUP):
        base = m * stride_msm + g * stride_msg
        for e in range(0, EXP_PER_GROUP):
            val = tl.load(MaskedScores_ptr + base + e * stride_mse)
            pos = 0
            while pos < 8 and val > top_vals[pos]:
                pos += 1
            if pos < 8:
                tmp_score = top_vals[pos]
                tmp_idx = top_idxs[pos]
                for j in range(pos, 7):
                    top_vals[j] = top_vals[j + 1]
                    top_idxs[j] = top_idxs[j + 1]
                top_vals[7] = tmp_score
                top_idxs[7] = tmp_idx
                top_vals[pos] = val
                top_idxs[pos] = e
    for i in range(0, 8):
        tl.store(SelectedJ_ptr + m * stride_sj_m + i * stride_sj_n, top_idxs[i])


@triton.jit
def _gather_normalize_scale_kernel(
    OriginalScores_ptr,  # [M, 256], float32 (sigmoid + bias)
    SelectedJ_ptr,       # [M, 8], int32
    OutputScores_ptr,    # [M, 8], float32
    Scaling_ptr,         # [M] float32 (routed_scaling_factor)
    M, N_EXP,
    stride_os_m, stride_os_n,
    stride_sj_m, stride_sj_n,
    stride_out_m, stride_out_n,
    stride_scale_m,
):
    m = tl.program_id(0)
    if m >= M:
        return
    total = 0.0
    for i in range(0, 8):
        idx = tl.load(SelectedJ_ptr + m * stride_sj_m + i * stride_sj_n)
        val = tl.load(OriginalScores_ptr + m * stride_os_m + idx * stride_os_n)
        total += val
    inv_total = 1.0 / (total + 1e-20)
    scale = tl.load(Scaling_ptr + m * stride_scale_m)
    for i in range(0, 8):
        idx = tl.load(SelectedJ_ptr + m * stride_sj_m + i * stride_sj_n)
        val = tl.load(OriginalScores_ptr + m * stride_os_m + idx * stride_os_n)
        out_val = val * inv_total * scale
        tl.store(OutputScores_ptr + m * stride_out_m + i * stride_out_n, out_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float,
    ):
        # Ensure contiguity and dtypes
        M = hidden_states.shape[0]
        device = hidden_states.device

        # 1) Compute logits = hidden @ weight.T via Triton
        A = hidden_states.contiguous().to(torch.float32)  # [M, K]
        weight_T = weight.t().contiguous().to(torch.float32)  # [K, 256]
        logits = torch.empty((M, 256), dtype=torch.float32, device=device)
        grid_matmul = (triton.cdiv(M, 128), triton.cdiv(256, 64))
        _matmul_kernel[grid_matmul](
            A, weight_T, logits,
            M, 256, weight.shape[0],
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
        )

        # 2) Apply sigmoid and add expert bias via Triton
        scores = torch.empty_like(logits)
        bias = expert_bias.contiguous().to(torch.float32)
        grid_sigmoid = (triton.cdiv(M, 64), triton.cdiv(256, 64))
        _sigmoid_bias_kernel[grid_sigmoid](
            logits, bias, scores,
            M, 256,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            bias.stride(0)
        )

        # Reshape to groups: [M, 8, 32]
        scores_groups = scores.view(M, 8, 32)

        # 3) Compute group scores (sum of top-2 per group) and selected group indices
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        grid_top2 = (M,)
        _group_top2_kernel[grid_top2](
            scores_groups, group_scores, selected_groups,
            M, 8, 32,
            scores_groups.stride(0), scores_groups.stride(1), scores_groups.stride(2),
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1)
        )

        # 4) Select top-4 groups per token via Triton
        # (selected_groups already computed in kernel)

        # 5) Build group mask [M, 8] one-hot via Triton
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=device)
        grid_groupmask = (M,)
        _build_groupmask_kernel[grid_groupmask](
            selected_groups, group_mask,
            M, 8,
            selected_groups.stride(0), selected_groups.stride(1),
            group_mask.stride(0), group_mask.stride(1)
        )

        # 6) Mask scores: set non-selected groups to -inf via Triton
        masked_scores = torch.empty((M, 8, 32), dtype=torch.float32, device=device)
        grid_mask = (M,)
        _mask_scores_kernel[grid_mask](
            scores_groups, group_mask, masked_scores,
            M, 8, 32,
            scores_groups.stride(0), scores_groups.stride(1), scores_groups.stride(2),
            group_mask.stride(0), group_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2)
        )

        # 7) Select final top-8 experts from masked scores via


def run(*args):
    return ModelNew()(*args)
