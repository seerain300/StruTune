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
    # 2D launch
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
    S_ptr,              # [M, 8, 32] scores after sigmoid + bias (contiguous)
    GroupScores_ptr,    # [M, 8] float32
    Top2Idx_ptr,        # [M, 8, 2] int32 (indices within group)
    M, G, E,
    stride_sm, stride_sg, stride_se,
    stride_gs_m, stride_gs_g,
    stride_tmi, stride_tmj, stride_tm_k,
):
    # One program per token
    pid_m = tl.program_id(0)
    for g in range(0, G):
        top1_val = -1.0e30
        top2_val = -1.0e30
        top1_idx = 0
        top2_idx = 0
        for e in range(0, E):
            s = tl.load(S_ptr + pid_m * stride_sm + g * stride_sg + e * stride_se)
            if s > top1_val:
                top2_val = top1_val
                top2_idx = top1_idx
                top1_val = s
                top1_idx = e
            elif s > top2_val:
                top2_val = s
                top2_idx = e

        tl.store(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_g, top1_val + top2_val)
        tl.store(Top2Idx_ptr + pid_m * stride_tmi + g * stride_tmj + 0 * stride_tm_k, top1_idx)
        tl.store(Top2Idx_ptr + pid_m * stride_tmi + g * stride_tmj + 1 * stride_tm_k, top2_idx)


@triton.jit
def _select_top4_groups_kernel(
    GroupScores_ptr,    # [M, 8]
    SelectedGroup_ptr,  # [M, 4] int32
    M, G,
    stride_gs_m, stride_gs_g,
    stride_sg_m, stride_sg_k,
):
    pid_m = tl.program_id(0)
    top4_val = tl.full((4,), -1.0e30, dtype=tl.float32)
    top4_idx = tl.zeros((4,), dtype=tl.int32)

    for g in range(0, G):
        gs = tl.load(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_g)
        # update top4 buffer
        if gs > top4_val[0]:
            # shift down
            top4_val[3] = top4_val[2]
            top4_val[2] = top4_val[1]
            top4_val[1] = top4_val[0]
            top4_val[0] = gs
            top4_idx[3] = top4_idx[2]
            top4_idx[2] = top4_idx[1]
            top4_idx[1] = top4_idx[0]
            top4_idx[0] = g
        elif gs > top4_val[1]:
            top4_val[3] = top4_val[2]
            top4_val[2] = top4_val[1]
            top4_val[1] = gs
            top4_idx[3] = top4_idx[2]
            top4_idx[2] = top4_idx[1]
            top4_idx[1] = g
        elif gs > top4_val[2]:
            top4_val[3] = top4_val[2]
            top4_val[2] = gs
            top4_idx[3] = top4_idx[2]
            top4_idx[2] = g
        elif gs > top4_val[3]:
            top4_val[3] = gs
            top4_idx[3] = g

    # store results
    tl.store(SelectedGroup_ptr + pid_m * stride_sg_m + 0 * stride_sg_k, top4_idx[0])
    tl.store(SelectedGroup_ptr + pid_m * stride_sg_m + 1 * stride_sg_k, top4_idx[1])
    tl.store(SelectedGroup_ptr + pid_m * stride_sg_m + 2 * stride_sg_k, top4_idx[2])
    tl.store(SelectedGroup_ptr + pid_m * stride_sg_m + 3 * stride_sg_k, top4_idx[3])


@triton.jit
def _build_group_mask_kernel(
    SelectedGroup_ptr,  # [M, 4] int32
    GroupMask_ptr,      # [M, 8] float32
    M, G,
    stride_sg_m, stride_sg_k,
    stride_gmm, stride_gmn,
):
    pid_m = tl.program_id(0)
    for i in range(0, 4):
        g = tl.load(SelectedGroup_ptr + pid_m * stride_sg_m + i * stride_sg_k)
        # write 1.0 at selected group index
        tl.store(GroupMask_ptr + pid_m * stride_gmm + g * stride_gmn, 1.0)


@triton.jit
def _mask_scores_kernel(
    S_ptr,              # [M, 8, 32] scores after sigmoid + bias
    GroupMask_ptr,      # [M, 8] float32, 1.0 for selected groups
    MaskedS_ptr,        # [M, 8, 32] masked scores
    M, G, E,
    stride_sm, stride_sg, stride_se,
    stride_gmm, stride_gmn,
    stride_ms_m, stride_ms_g, stride_ms_e,
):
    pid_m = tl.program_id(0)
    for g in range(0, G):
        mask_val = tl.load(GroupMask_ptr + pid_m * stride_gmm + g * stride_gmn)
        if mask_val > 0.0:
            for e in range(0, E):
                s = tl.load(S_ptr + pid_m * stride_sm + g * stride_sg + e * stride_se)
                tl.store(MaskedS_ptr + pid_m * stride_ms_m + g * stride_ms_g + e * stride_ms_e, s)
        else:
            for e in range(0, E):
                tl.store(MaskedS_ptr + pid_m * stride_ms_m + g * stride_ms_g + e * stride_ms_e, -1.0e30)


@triton.jit
def _select_top8_kernel(
    MaskedS_ptr,        # [M, 8, 32] masked scores
    TopIdx_ptr,         # [M, 8] int32
    M, G, E,
    stride_ms_m, stride_ms_g, stride_ms_e,
    stride_tm_m, stride_tm_k,
):
    pid_m = tl.program_id(0)
    best_val = tl.full((8,), -1.0e30, dtype=tl.float32)
    best_idx = tl.zeros((8,), dtype=tl.int32)

    for g in range(0, G):
        for e in range(0, E):
            s = tl.load(MaskedS_ptr + pid_m * stride_ms_m + g * stride_ms_g + e * stride_ms_e)
            for j in range(0, 8):
                if s > best_val[j]:
                    # shift down
                    for l in range(7, j, -1):
                        best_val[l] = best_val[l - 1]
                        best_idx[l] = best_idx[l - 1]
                    best_val[j] = s
                    best_idx[j] = g * 32 + e
                    break  # maintain sorted descending for position j

    for j in range(0, 8):
        tl.store(TopIdx_ptr + pid_m * stride_tm_m + j * stride_tm_k, best_idx[j])


@triton.jit
def _normalize_and_scale_kernel(
    S_ptr,              # [M, 256] original logits (we use the top-8 indices to gather)
    TopIdx_ptr,         # [M, 8] int32
    Weighted_ptr,       # [M, 8] float32
    M, N, K,
    stride_sm, stride_sn,
    stride_tmi, stride_tmj,
    scale, eps,
):
    pid_m = tl.program_id(0)
    total = 0.0
    for j in range(0, 8):
        idx = tl.load(TopIdx_ptr + pid_m * stride_tmi + j * stride_tmj)  # int32
        s = tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn)  # gather original score
        total += s
    for j in range(0, 8):
        idx = tl.load(TopIdx_ptr + pid_m * stride_tmi + j * stride_tmj)
        s = tl.load(S_ptr + pid_m * stride_sm + idx * stride_sn)
        w = s / (total + eps) * scale
        tl.store(Weighted_ptr + pid_m * stride_tmi + j * stride_tmj, w)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure inputs are float32 and contiguous
        hidden = hidden_states.to(torch.float32).contiguous()
        weight = weight.to(torch.float32).contiguous()  # [N, K]
        bias = expert_bias.to(torch.float32).contiguous()  # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]  # hidden features
        N = weight.shape[0]  # number of experts, expected 256
        assert N == 256, "This implementation assumes 256 experts."
        E = 32  # experts per group
        G = 8   # number of groups

        # 1) Matmul logits [M, N] using Triton
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight.transpose(0, 1).contiguous(), logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight.transpose(0, 1).contiguous().stride(0), weight.transpose(0, 1).contiguous().stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + expert bias using Triton
        sigmoid_scores = torch.empty_like(logits)
        _sigmoid_bias_kernel[(M, N)](
            logits, bias, sigmoid_scores,
            M, N,
            logits.stride(0), logits.stride(1),
            sigmoid_scores.stride(0), sigmoid_scores.stride(1),
            1,
            num_warps=4, num_stages=2,
        )

        # 3) Group top-2 per group and group scores using Triton
        group_scores = torch.empty((M, G), dtype=torch.float32, device=sigmoid_scores.device)
        top2_idx = torch.empty((M, G, 2), dtype=torch.int32, device=sigmoid_scores.device)

        _group_top2_kernel[(M,)](
            sigmoid_scores.view(M, G, E),
            group_scores, top2_idx,
            M, G, E,
            sigmoid_scores.view(M, G, E).stride(0), sigmoid_scores.view(M, G, E).stride(1), sigmoid_scores.view(M, G, E).stride(2),
            group_scores.stride(0), group_scores.stride(1),
            top2_idx.stride(0), top2_idx.stride(1), top2_idx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token using Triton
        selected_groups = torch.empty((M, 4), dtype=torch.int32, device=sigmoid_scores.device)
        _select_top4_groups_kernel[(M,)](
            group_scores,
            selected_groups,
            M, G,
            group_scores.stride(0), group_scores.stride(1),
            selected_groups.stride(0), selected_groups.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Build group mask [M, 8] in Triton
        group_mask = torch.empty((M, G), dtype=torch.float32, device=sigmoid_scores.device)
        _build_group_mask_kernel[(M,)](
            selected_groups, group_mask,
            M, G,
            selected_groups.stride(0), selected_groups.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Mask scores: set non-selected group scores to -inf using Triton
        masked_scores = torch.empty((M, G, E), dtype=torch.float32, device=sigmoid_scores.device)
        _mask_scores_kernel[(M,)](
            sigmoid_scores.view(M, G, E),
            group_mask,
            masked_scores,
            M, G, E,
            sigmoid_scores.view(M, G, E).stride(0), sigmoid_scores.view(M, G, E).stride(1), sigmoid_scores.view(M, G, E).stride(2),
            group_mask.stride(0), group_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2),
            num_warps=1, num_stages=1,
        )

        # 7) Select top-8 from masked scores using Triton
        top_idx = torch.empty((M, 8), dtype=torch.int32, device=sigmoid_scores.device)
        _select_top8_kernel[(M,)](
            masked_scores,
            top_idx,
            M, G, E,
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2),
            top_idx.stride(0), top_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # 8) Normalize and apply scaling (use original logits to gather selected expert scores)
        # We gather selected scores from original logits (since we did not keep masked scores for final)
        # However, to be consistent with selection, we can gather from sigmoid_scores:
        # But the original code gathers from 'scores' which is sigmoid_scores here. So we use sigmoid_scores.
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        _normalize_and_scale_kernel[(M,)](
            sigmoid_scores,
            top_idx,
            selected_scores,
            M, N, K,
            sigmoid_scores.stride(0), sigmoid_scores.stride(1),
            top_idx.stride(0), top_idx.stride(1),
            routed_scaling_factor, 1e-20,
            num_warps=1, num_stages=1,
        )

        # Return selected indices and weights
        # top_idx shape: [M, 8] with indices into group*32 + e
        # We need to return (topk_idx, topk_weight). Since Triton kernel above wrote normalized weights,
        # we can return selected_scores as topk_weight and top_idx as topk_idx reshaped appropriately.

        # Reshape outputs for compatibility
        topk_idx = top_idx  # already [M, 8]
        topk_weight = selected_scores  # already [M, 8], normalized and scaled

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
