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
def _group_top2_and_select_kernel(
    Scores_ptr,  # [M, 8, 32] float32
    GroupScores_ptr,  # [M, 8] float32
    SelectedGroups_ptr,  # [M, 4] int32
    GroupMask_ptr,  # [M, 8] float32 (output one-hot)
    M, EXP_PER_GROUP, N_GROUP,
    stride_s_m, stride_s_group, stride_s_exp,
    stride_gs_m, stride_gs_group,
    stride_sg_m, stride_sg_group,
    stride_gm_m, stride_gm_group,
):
    # one program per token
    m = tl.program_id(0)
    if m >= M:
        return

    # Compute group scores (sum of top-2) and select top-4 groups
    # Also write one-hot group mask
    top2_sum = [-1.0e20, -1.0e20, -1.0e20, -1.0e20,
                -1.0e20, -1.0e20, -1.0e20, -1.0e20]
    selected_group_idx = [-1, -1, -1, -1]
    # Loop over groups
    for g in range(0, N_GROUP):
        base = m * stride_s_m + g * stride_s_group
        # top1
        top1 = [-1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20, -1.0e20]
        for e in range(0, EXP_PER_GROUP):
            val = tl.load(Scores_ptr + base + e * stride_s_exp)
            pos = 0
            while pos < 9 and val > top1[pos]:
                pos += 1
            if pos < 9:
                tmp = top1[pos]
                for j in range(pos, 8):
                    top1[j] = top1[j + 1]
                top1[8] = tmp
                top1[pos] = val
        # now top1[8] is invalid, top1[0..8] is top-9 with 9th largest in 8th. We take top2 as max and second max
        # For a proper top2, we should keep only top2 among 9 slots. Simpler approach: recompute top2 per group using 2-pass (top1 then top2)
        # But we can instead recompute the actual top2 by scanning again and recording best and second best:
        top_val = -1.0e20
        second_val = -1.0e20
        for e in range(0, EXP_PER_GROUP):
            val = tl.load(Scores_ptr + base + e * stride_s_exp)
            if val > top_val:
                second_val = top_val
                top_val = val
            elif val > second_val:
                second_val = val
        top2_sum[g] = top_val + second_val

    # Select top-4 groups (indices in [0..7])
    topk = [-1.0e20, -1.0e20, -1.0e20, -1.0e20]
    selected_group_idx = [-1, -1, -1, -1]
    for g in range(0, N_GROUP):
        score = top2_sum[g]
        if score > topk[0]:
            tmp = topk[0]
            for j in range(0, 3):
                topk[j] = topk[j + 1]
            topk[3] = tmp
            tmp = selected_group_idx[0]
            for j in range(0, 3):
                selected_group_idx[j] = selected_group_idx[j + 1]
            selected_group_idx[3] = tmp
            topk[0] = score
            selected_group_idx[0] = g
        elif score > topk[1]:
            tmp = topk[1]
            topk[1] = score
            score = tmp
            tmp = selected_group_idx[1]
            selected_group_idx[1] = g
            selected_group_idx[1] = tmp
        elif score > topk[2]:
            tmp = topk[2]
            topk[2] = score
            score = tmp
            tmp = selected_group_idx[2]
            selected_group_idx[2] = g
            selected_group_idx[2] = tmp
        elif score > topk[3]:
            topk[3] = score
            selected_group_idx[3] = g

    # Write selected groups
    for i in range(0, 4):
        tl.store(SelectedGroups_ptr + m * stride_sg_m + i * stride_sg_group, selected_group_idx[i])

    # Write one-hot group mask: set GroupMask[m, selected_group_idx[i]] = 1.0
    for i in range(0, 4):
        group = selected_group_idx[i]
        tl.store(GroupMask_ptr + m * stride_gm_m + group * stride_gm_group, 1.0)


@triton.jit
def _mask_scores_kernel(
    Scores_ptr,      # [M, 8, 32] float32
    GroupMask_ptr,   # [M, 8] float32 (1.0 where selected, else 0.0)
    Masked_ptr,      # [M, 8, 32] float32
    M, EXP_PER_GROUP, N_GROUP,
    stride_s_m, stride_s_group, stride_s_exp,
    stride_gm_m, stride_gm_group,
    stride_ms_m, stride_ms_group, stride_ms_exp,
):
    # one program per token
    m = tl.program_id(0)
    if m >= M:
        return
    for g in range(0, N_GROUP):
        group_score = tl.load(GroupMask_ptr + m * stride_gm_m + g * stride_gm_group)
        base = m * stride_s_m + g * stride_s_group
        for e in range(0, EXP_PER_GROUP):
            val = tl.load(Scores_ptr + base + e * stride_s_exp)
            out = tl.where(group_score > 0.0, val, -1.0e20)
            tl.store(Masked_ptr + m * stride_ms_m + g * stride_ms_group + e * stride_ms_exp, out)


@triton.jit
def _select_final_top8_kernel(
    Masked_ptr,      # [M, 8, 32] float32
    SelectedI_ptr,   # [M, 8] int32
    M, EXP_PER_GROUP, N_GROUP,
    stride_ms_m, stride_ms_group, stride_ms_exp,
    stride_si_m, stride_si_idx,
):
    m = tl.program_id(0)
    if m >= M:
        return
    top8 = [-1.0e20, -1.0e20, -1.0e20, -1.0e20,
            -1.0e20, -1.0e20, -1.0e20, -1.0e20]
    sel_idx = [0, 0, 0, 0, 0, 0, 0, 0]
    for g in range(0, N_GROUP):
        base = m * stride_ms_m + g * stride_ms_group
        for e in range(0, EXP_PER_GROUP):
            val = tl.load(Masked_ptr + base + e * stride_ms_exp)
            pos = 0
            while pos < 8 and val < top8[pos]:
                pos += 1
            if pos < 8:
                tmp = top8[pos]
                for j in range(pos, 7):
                    top8[j] = top8[j + 1]
                top8[7] = tmp
                for j in range(pos, 7):
                    sel_idx[j] = sel_idx[j + 1]
                sel_idx[7] = sel_idx[7]  # no change
                top8[pos] = val
                sel_idx[pos] = g * EXP_PER_GROUP + e
    for i in range(0, 8):
        tl.store(SelectedI_ptr + m * stride_si_m + i * stride_si_idx, sel_idx[i])


@triton.jit
def _gather_normalize_scale_kernel(
    OriginalScores_ptr,  # [M, 256] float32 (scores post-sigmoid + bias)
    SelectedI_ptr,       # [M, 8] int32
    OutputScores_ptr,    # [M, 8] float32
    Scaling_ptr,         # [M] float32 (routed_scaling_factor)
    M, N_EXP,
    stride_os_m, stride_os_n,
    stride_si_m, stride_si_idx,
    stride_out_m, stride_out_idx,
    stride_scale_m,
):
    m = tl.program_id(0)
    if m >= M:
        return
    total = 0.0
    for i in range(0, 8):
        idx = tl.load(SelectedI_ptr + m * stride_si_m + i * stride_si_idx)
        val = tl.load(OriginalScores_ptr + m * stride_os_m + idx * stride_os_n)
        total += val
    inv_total = 1.0 / (total + 1e-20)
    scale = tl.load(Scaling_ptr + m * stride_scale_m)
    for i in range(0, 8):
        idx = tl.load(SelectedI_ptr + m * stride_si_m + i * stride_si_idx)
        val = tl.load(OriginalScores_ptr + m * stride_os_m + idx * stride_os_n)
        out_val = val * inv_total * scale
        tl.store(OutputScores_ptr + m * stride_out_m + i * stride_out_idx, out_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float,
    ):
        # Ensure device and dtype
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]  # 256

        # 1) Compute logits = hidden @ weight.T using Triton
        hidden = hidden_states.contiguous().to(torch.float32)  # [M, K]
        weight_t = weight.transpose(0, 1).contiguous().to(torch.float32)  # [K, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_matmul = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _matmul_kernel[grid_matmul](
            hidden, weight_t, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # 2) Sigmoid + bias (Triton kernel)
        bias = expert_bias.contiguous().to(torch.float32)  # [N]
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_sigmoid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _sigmoid_bias_kernel[grid_sigmoid](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            bias.stride(0)
        )

        # 3) Reshape to groups: [M, 8, 32]
        scores_groups = scores.view(M, 8, 32)

        # 4) Compute group scores and selected groups; also build group mask in Triton
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=device)
        grid_group_top2 = (M,)
        _group_top2_and_select_kernel[grid_group_top2](
            scores_groups, group_scores, selected_groups, group_mask,
            M, 32, 8,
            scores_groups.stride(0), scores_groups.stride(1), scores_groups.stride(2),
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            group_mask.stride(0), group_mask.stride(1)
        )

        # 5) Mask scores: set non-selected groups to -inf (Triton)
        masked_scores = torch.empty((M, 8, 32), dtype=torch.float32, device=device)
        grid_mask = (M,)
        _mask_scores_kernel[grid_mask](
            scores_groups, group_mask, masked_scores,
            M, 32, 8,
            scores_groups.stride(0), scores_groups.stride(1), scores_groups.stride(2),
            group_mask.stride(0), group_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2)
        )

        # 6) Select final top-8 indices from masked scores (Triton)
        selected_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        grid_select = (M,)
        _select_final_top8_kernel[grid_select](
            masked_scores, selected_idx,
            M, 32, 8,
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2),
            selected_idx.stride(0), selected_idx.stride(1)
        )

        # 7) Gather, normalize, apply scaling (Triton)
        # We need original scores for normalization (pre-topk), which is scores (post-sigmoid + bias)
        original_scores = scores  # [M, 256], float32
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        scaling = torch.empty((M, 1), dtype=torch.float32, device=device)
        scaling.fill_(routed_scaling_factor)  # [M, 1]
        grid_norm = (M,)
        _gather_normalize_scale_kernel[grid_norm](
            original_scores, selected_idx, topk_weight, scaling,
            M, 256,
            original_scores.stride(0), original_scores.stride(1),
            selected_idx.stride(0), selected_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            scaling.stride(0)
        )

        # Return topk_idx (indices) and topk_weight
        # topk_idx is selected_idx; topk_weight is normalized scaled scores
        return selected_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
