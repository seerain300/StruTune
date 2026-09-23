import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K] = hidden, float32
    B_ptr,  # [K, N] = weight.T, float32
    C_ptr,  # [M, N] = logits, float32
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,  # e.g., 128
    BLOCK_N: tl.constexpr,  # e.g., 64
    BLOCK_K: tl.constexpr,  # e.g., 64
):
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
def _select_top4_kernel(
    GroupScores_ptr,   # [M, 8]
    GroupIdx_ptr,      # [M, 4] int32
    M, G,
    stride_gs_m, stride_gs_g,
    stride_gi_m, stride_gi_k,
):
    pid_m = tl.program_id(0)
    top4_val = tl.full((4,), -1.0e30, dtype=tl.float32)
    top4_idx = tl.zeros((4,), dtype=tl.int32)
    for g in range(0, G):
        gs = tl.load(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_g)
        # update top-4
        if gs > top4_val[0]:
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

    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 0 * stride_gi_k, top4_idx[0])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 1 * stride_gi_k, top4_idx[1])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 2 * stride_gi_k, top4_idx[2])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 3 * stride_gi_k, top4_idx[3])


@triton.jit
def _group_mask_kernel(
    GroupIdx_ptr,       # [M, 4] int32 selected group indices
    GroupMask_ptr,      # [M, 8] float32, one-hot
    M, G,
    stride_gi_m, stride_gi_k,
    stride_gmm, stride_gmn,
):
    pid_m = tl.program_id(0)
    for k in range(0, G):
        # is selected?
        is_sel = 0
        for j in range(0, 4):
            idx_j = tl.load(GroupIdx_ptr + pid_m * stride_gi_m + j * stride_gi_k)
            if idx_j == k:
                is_sel = 1
                break
        tl.store(GroupMask_ptr + pid_m * stride_gmm + k * stride_gmn, is_sel)


@triton.jit
def _apply_group_mask_and_build_S_masked_kernel(
    S_ptr,              # [M, 8, 32] scores after sigmoid + bias (contiguous)
    GroupMask_ptr,      # [M, 8] float32 0/1
    SMasked_ptr,        # [M, 8, 32] masked scores (non-selected -> -inf)
    M, G, E,
    stride_sm, stride_sg, stride_se,
    stride_gmm, stride_gmn,
    stride_smask_m, stride_smask_g, stride_smask_e,
):
    # One program per token
    pid_m = tl.program_id(0)
    for g in range(0, G):
        mask_g = tl.load(GroupMask_ptr + pid_m * stride_gmm + g * stride_gmn)  # 0.0 or 1.0
        if mask_g == 0.0:
            # set all scores in this group to -inf
            for e in range(0, E):
                s = tl.load(S_ptr + pid_m * stride_sm + g * stride_sg + e * stride_se)
                tl.store(SMasked_ptr + pid_m * stride_smask_m + g * stride_smask_g + e * stride_smask_e, -1.0e30)


@triton.jit
def _select_top8_masked_kernel(
    SMasked_ptr,        # [M, 8, 32]
    SelectedIdx_ptr,    # [M, 8] int32
    M, G, E,
    stride_sm, stride_sg, stride_se,
    stride_si_m, stride_si_k,
):
    # One program per token
    pid_m = tl.program_id(0)
    best_val = tl.full((8,), -1.0e30, dtype=tl.float32)
    best_idx = tl.zeros((8,), dtype=tl.int32)

    for g in range(0, G):
        for e in range(0, E):
            s = tl.load(SMasked_ptr + pid_m * stride_sm + g * stride_sg + e * stride_se)
            # update best
            if s > best_val[0]:
                best_val[7] = best_val[6]
                best_val[6] = best_val[5]
                best_val[5] = best_val[4]
                best_val[4] = best_val[3]
                best_val[3] = best_val[2]
                best_val[2] = best_val[1]
                best_val[1] = best_val[0]
                best_val[0] = s
                best_idx[7] = best_idx[6]
                best_idx[6] = best_idx[5]
                best_idx[5] = best_idx[4]
                best_idx[4] = best_idx[3]
                best_idx[3] = best_idx[2]
                best_idx[2] = best_idx[1]
                best_idx[1] = best_idx[0]
                best_idx[0] = g * E + e
            elif s > best_val[1]:
                best_val[7] = best_val[6]
                best_val[6] = best_val[5]
                best_val[5] = best_val[4]
                best_val[4] = best_val[3]
                best_val[3] = best_val[2]
                best_val[2] = best_val[1]
                best_val[1] = s
                best_idx[7] = best_idx[6]
                best_idx[6] = best_idx[5]
                best_idx[5] = best_idx[4]
                best_idx[4] = best_idx[3]
                best_idx[3] = best_idx[2]
                best_idx[2] = best_idx[1]
                best_idx[1] = g * E + e
            elif s > best_val[2]:
                best_val[7] = best_val[6]
                best_val[6] = best_val[5]
                best_val[5] = best_val[4]
                best_val[4] = best_val[3]
                best_val[3] = best_val[2]
                best_val[2] = s
                best_idx[7] = best_idx[6]
                best_idx[6] = best_idx[5]
                best_idx[5] = best_idx[4]
                best_idx[4] = best_idx[3]
                best_idx[3] = best_idx[2]
                best_idx[2] = g * E + e
            elif s > best_val[3]:
                best_val[7] = best_val[6]
                best_val[6] = best_val[5]
                best_val[5] = best_val[4]
                best_val[4] = best_val[3]
                best_val[3] = s
                best_idx[7] = best_idx[6]
                best_idx[6] = best_idx[5]
                best_idx[5] = best_idx[4]
                best_idx[4] = g * E + e
            elif s > best_val[4]:
                best_val[7] = best_val[6]
                best_val[6] = best_val[5]
                best_val[5] = best_val[4]
                best_val[4] = s
                best_idx[7] = best_idx[6]
                best_idx[6] = best_idx[5]
                best_idx[5] = g * E + e
            elif s > best_val[5]:
                best_val[7] = best_val[6]
                best_val[6] = best_val[5]
                best_val[5] = s
                best_idx[7] = best_idx[6]
                best_idx[6] = g * E + e
            elif s > best_val[6]:
                best_val[7] = best_val[6]
                best_val[6] = s
                best_idx[7] = g * E + e
            elif s > best_val[7]:
                best_val[7] = s
                best_idx[7] = g * E + e

    # write out top-8 indices
    for j in range(0, 8):
        idx_j = best_idx[j]
        expert_id = idx_j % E
        group_id = idx_j // E  # since idx_j is within [0, 8*32), group_id is always 0..7
        tl.store(SelectedIdx_ptr + pid_m * stride_si_m + j * stride_si_k, idx_j)


@triton.jit
def _gather_normalize_scale_kernel(
    S_ptr,              # [M, 256] original sigmoid + bias
    SelectedIdx_ptr,    # [M, 8] int32 (indices in [0,255])
    SelectedScores_ptr, # [M, 8] float32
    TopkWeight_ptr,     # [M, 8] float32
    M, N,
    stride_sm, stride_sn,
    stride_si_m, stride_si_k,
    scale, epsilon,
):
    pid_m = tl.program_id(0)
    for j in range(0, 8):
        idx_j = tl.load(SelectedIdx_ptr + pid_m * stride_si_m + j * stride_si_k)
        s = tl.load(S_ptr + pid_m * stride_sm + idx_j * stride_sn)
        tl.store(SelectedScores_ptr + pid_m * stride_sm + j * stride_sn, s)
    # normalize and scale
    sum_scores = 0.0
    for j in range(0, 8):
        s = tl.load(SelectedScores_ptr + pid_m * stride_sm + j * stride_sn)
        sum_scores += s
    for j in range(0, 8):
        s = tl.load(SelectedScores_ptr + pid_m * stride_sm + j * stride_sn)
        w = s / (sum_scores + epsilon) * scale
        tl.store(TopkWeight_ptr + pid_m * stride_sm + j * stride_sn, w)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and contiguity
        hidden = hidden_states.to(torch.float32).contiguous()
        weight_t = weight.to(torch.float32).contiguous()  # [N, K]
        bias = expert_bias.to(torch.float32).contiguous()
        M, K = hidden.shape
        N = weight_t.shape[0]  # num_experts, expected 256
        assert N == 256, "This implementation assumes 256 experts."
        E = 32
        G = 8

        # 1) Matmul logits = hidden @ weight.T -> [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight_t, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Sigmoid + bias
        sigmoid_scores = torch.empty_like(logits)
        _sigmoid_bias_kernel[(M, N)](
            logits, bias, sigmoid_scores,
            M, N,
            logits.stride(0), logits.stride(1),
            sigmoid_scores.stride(0), sigmoid_scores.stride(1),
            bias.stride(0),
            num_warps=4, num_stages=2,
        )

        # 3) Group top-2 and group scores
        S_group = sigmoid_scores.view(M, G, E).contiguous()
        group_scores = torch.empty((M, G), dtype=torch.float32, device=sigmoid_scores.device)
        top2_idx = torch.empty((M, G, 2), dtype=torch.int32, device=sigmoid_scores.device)

        _group_top2_kernel[(M,)](
            S_group,
            group_scores, top2_idx,
            M, G, E,
            S_group.stride(0), S_group.stride(1), S_group.stride(2),
            group_scores.stride(0), group_scores.stride(1),
            top2_idx.stride(0), top2_idx.stride(1), top2_idx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token
        top4_group = torch.empty((M, 4), dtype=torch.int32, device=sigmoid_scores.device)
        _select_top4_kernel[(M,)](
            group_scores,
            top4_group,
            M, G,
            group_scores.stride(0), group_scores.stride(1),
            top4_group.stride(0), top4_group.stride(1),
            num_warps=1, num_stages=1,
        )

        # 5) Build group_mask (one-hot) [M, 8]
        group_mask = torch.empty((M, G), dtype=torch.float32, device=sigmoid_scores.device)
        _group_mask_kernel[(M,)](
            top4_group,
            group_mask,
            M, G,
            top4_group.stride(0), top4_group.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            num_warps=1, num_stages=1,
        )

        # 6) Apply group mask: set non-selected group scores to -inf, store in SMasked [M, 8, 32]
        SMasked = torch.empty((M, G, E), dtype=torch.float32, device=sigmoid_scores.device)
        _apply_group_mask_and_build_S_masked_kernel[(M,)](
            sigmoid_scores.view(M, G, E),  # S_ptr
            group_mask,                     # GroupMask_ptr
            SMasked,                        # SMasked_ptr
            M, G, E,
            sigmoid_scores.view(M, G, E).stride(0), sigmoid_scores.view(M, G, E).stride(1), sigmoid_scores.view(M, G, E).stride(2),
            group_mask.stride(0), group_mask.stride(1),
            SMasked.stride(0), SMasked.stride(1), SMasked.stride(2),
            num_warps=1, num_stages=1,
        )

        # 7) Select final top-8 from masked scores
        selected_idx = torch.empty((M, 8), dtype=torch.int32, device=sigmoid_scores.device)
        _select_top8_masked_kernel[(M,)](
            SMasked,
            selected_idx,
            M, G, E,
            SMasked.stride(0), SMasked.stride(1), SMasked.stride(2),
            selected_idx.stride(0), selected_idx.stride(1),
            num_warps=1, num_stages=1,
        )

        # 8) Gather selected scores from original sigmoid_scores and normalize + scale
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        _gather_normalize_scale_kernel[(M,)](
            sigmoid_scores,
            selected_idx,
            selected_scores,
            topk_weight,
            M, N,
            sigmoid_scores.stride(0), sigmoid_scores.stride(1),
            selected_idx.stride(0), selected_idx.stride(1),
            routed_scaling_factor, 1e-20,
            num_warps=1, num_stages=1,
        )

        # Return token-wise selected expert indices and normalized weights
        # Convert selected_idx from [M, 8] int32 to torch.long for compatibility
        return selected_idx.to(torch.long), topk_weight


def run(*args):
    return ModelNew()(*args)
