import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K] = hidden_states
    B_ptr,  # [K, N] = weight.T (shape [768, 256])
    C_ptr,  # [M, N] = scores [M, 256]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # rows (tokens)
    pid_n = tl.program_id(1)  # column blocks (experts)
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
    X_ptr,   # [M, N] input scores (matmul output)
    Bias_ptr, # [N] expert bias
    Y_ptr,   # [M, N] output (sigmoid(scores) + bias)
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_b,
):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, N)
    x_ptrs = X_ptr + pid_m * stride_xm + offs_n * stride_xn
    y_ptrs = Y_ptr + pid_m * stride_ym + offs_n * stride_yn
    b = tl.load(Bias_ptr + offs_n * stride_b)
    x = tl.load(x_ptrs)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(y_ptrs, y)


@triton.jit
def _group_top2_kernel(
    S_ptr,              # [M, 8, 32] scores after sigmoid + bias
    GroupScores_ptr,    # [M, 8] float32
    Top2Idx_ptr,        # [M, 8, 2] int32
    M, G, E,
    stride_sm, stride_sg, stride_se,
    stride_gs_m, stride_gs_g,
    stride_tmi, stride_tmj, stride_tm_k,
):
    pid_m = tl.program_id(0)
    for g in range(0, G):
        top1_val = -1.0e30
        top1_idx = 0
        top2_val = -1.0e30
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
        gs = top1_val + top2_val
        tl.store(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_g, gs)
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
def _mask_and_select_top8_kernel(
    S_ptr,               # [M, 256] scores after sigmoid + bias
    GroupMask_ptr,       # [M, 8], float32, 1.0 for selected groups, 0 otherwise
    SelectedIdx_ptr,     # [M, 8] int32
    M, N, G, E,
    stride_sm, stride_sn,
    stride_gmm, stride_gmn,
    stride_si_m, stride_si_n,
):
    pid_m = tl.program_id(0)
    neg_inf = -1.0e30
    best_val = tl.full((8,), neg_inf, dtype=tl.float32)
    best_idx = tl.full((8,), -1, dtype=tl.int32)

    for n in range(0, N):
        selected = tl.load(GroupMask_ptr + pid_m * stride_gmm + (n // E) * stride_gmn)
        # If not selected, set score to -inf
        s = tl.load(S_ptr + pid_m * stride_sm + n * stride_sn)
        s = tl.where(selected > 0.5, s, neg_inf)
        # Iterative top-8 selection
        for j in range(0, 8):
            is_better = s > best_val[j]
            idx_j = j
            # Replace j-th slot if better and not already filled (best_val[j] < 0 means not filled)
            best_val = tl.where(is_better, s, best_val)
            best_idx = tl.where(is_better, n, best_idx)
            # Shift down for subsequent checks
            for i in range(7, 0, -1):
                move = best_val[i - 1] < 0
                best_val[i - 1] = tl.where(move, best_val[i], best_val[i - 1])
                best_idx[i - 1] = tl.where(move, best_idx[i], best_idx[i - 1])
            # If s was replaced, subsequent slots must ignore it in next iterations
            # We can't branch per j cleanly, but we keep iterative structure and final store is guarded
            # by the fact we only replace once per iteration.
    # Store results
    for j in range(0, 8):
        tl.store(SelectedIdx_ptr + pid_m * stride_si_m + j * stride_si_n, best_idx[j])


@triton.jit
def _finalize_and_normalize_kernel(
    SelectedScores_ptr,  # [M, 8]
    RoutedScaling,       # float32
    Normalized_ptr,      # [M, 8]
    M, K,
    stride_ssm, stride_ssn,
    stride_nsm, stride_nsn,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    sumv = 0.0
    for j in range(0, K):
        v = tl.load(SelectedScores_ptr + pid_m * stride_ssm + j * stride_ssn)
        sumv += v
    # Compute normalized and scaled
    out = tl.zeros((K,), dtype=tl.float32)
    for j in range(0, K):
        v = tl.load(SelectedScores_ptr + pid_m * stride_ssm + j * stride_ssn)
        out[j] = v / (sumv + eps) * RoutedScaling
    tl.store(Normalized_ptr + pid_m * stride_nsm + 0 * stride_nsn + tl.arange(0, K), out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and contiguity for Triton
        hidden = hidden_states.to(torch.float32).contiguous()      # [M, K]
        weight_t = weight.to(torch.float32).transpose(0, 1).contiguous()  # [K, N] where N=256
        bias = expert_bias.to(torch.float32).contiguous()          # [N]

        M, K = hidden.shape
        N = weight.shape[0]  # number of experts
        assert N == 256, "This implementation assumes 256 experts."
        E = 32  # experts per group
        G = 8   # number of groups

        # 1) Matmul scores [M, N] using Triton
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight_t, scores,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Sigmoid + bias using Triton
        sigmoid_scores = torch.empty_like(scores)
        _sigmoid_bias_kernel[(M, N)](
            scores, bias, sigmoid_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            sigmoid_scores.stride(0), sigmoid_scores.stride(1),
            1,  # bias stride is 1
        )

        # 3) Reshape into groups [M, G, E] in Triton (we'll use torch view since Triton kernel expects raw input)
        #    Note: We need [M, 8, 32] tensor for group kernel. We can create it via view; Triton kernel expects that layout
        #    So we'll create GroupScores [M, 8] and Top2Idx [M, 8, 2] as outputs of a kernel. To avoid complexity, we'll
        #    use a torch view followed by a Triton kernel that treats S as [M, 8, 32] logically. For simplicity, we pass
        #    a pointer to sigmoid_scores and compute per-group max in groups of 32 by reusing E and G.
        group_scores = torch.empty((M, G), dtype=torch.float32, device=sigmoid_scores.device)
        top2_idx = torch.empty((M, G, 2), dtype=torch.int32, device=sigmoid_scores.device)

        # Triton group top-2 kernel
        _group_top2_kernel[(M,)](
            sigmoid_scores.view(M, G, E),
            group_scores, top2_idx,
            M, G, E,
            sigmoid_scores.view(M, G, E).stride(0), sigmoid_scores.view(M, G, E).stride(1), sigmoid_scores.view(M, G, E).stride(2),
            group_scores.stride(0), group_scores.stride(1),
            top2_idx.stride(0), top2_idx.stride(1), top2_idx.stride(2),
        )

        # 4) Select top-4 groups per token using Triton
        top4_group = torch.empty((M, 4), dtype=torch.int32, device=sigmoid_scores.device)
        _select_top4_kernel[(M,)](
            group_scores,
            top4_group,
            M, G,
            group_scores.stride(0), group_scores.stride(1),
            top4_group.stride(0), top4_group.stride(1),
        )

        # 5) Create GroupMask [M, 8] (1.0 for selected groups, 0 otherwise)
        group_mask = torch.zeros((M, G), dtype=torch.float32, device=sigmoid_scores.device)
        # Broadcast group indices to mask
        # For each selected group index, set mask to 1
        # We need to scatter: group_mask[pid_m, top4_group[pid_m]] = 1.0
        # Use torch for this; it's minimal and host-only
        for g in range(4):
            group_mask.scatter_(1, top4_group[:, g].unsqueeze(1), 1.0)
        # group_mask is now [M, 8], 1.0 where selected, 0 otherwise

        # 6) Mask-and-select top-8 from sigmoid_scores using Triton
        selected_idx = torch.empty((M, 8), dtype=torch.int32, device=sigmoid_scores.device)
        _mask_and_select_top8_kernel[(M,)](
            sigmoid_scores, group_mask,
            selected_idx,
            M, N, G, E,
            sigmoid_scores.stride(0), sigmoid_scores.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            selected_idx.stride(0), selected_idx.stride(1),
        )

        # 7) Gather selected expert scores from original sigmoid_scores
        #    selected_idx is [M, 8], we need to gather per row: [M, 8]
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        for j in range(8):
            col = selected_idx[:, j]  # int tensor
            selected_scores[:, j] = sigmoid_scores[:, col]

        # 8) Normalize and apply scaling factor
        normalized = torch.empty((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        eps = 1e-20
        _finalize_and_normalize_kernel[(M,)](
            selected_scores, routed_scaling_factor,
            normalized,
            M, 8,
            selected_scores.stride(0), selected_scores.stride(1),
            normalized.stride(0), normalized.stride(1),
            eps,
        )

        # Return topk_idx (selected expert indices per token) and topk_weight (normalized scores)
        return selected_idx, normalized


def run(*args):
    return ModelNew()(*args)
