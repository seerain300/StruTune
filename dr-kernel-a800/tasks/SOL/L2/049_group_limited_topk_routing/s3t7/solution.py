import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K] = hidden (input to linear)
    B_ptr,  # [K, N] = weight.T (transpose of expert weights)
    C_ptr,  # [M, N] = logits
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: pid_m over rows (tokens), pid_n over column blocks (experts)
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

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _sigmoid_bias_kernel(
    X_ptr,   # [M, N] logits
    Bias_ptr, # [N] expert bias
    Y_ptr,   # [M, N] scores after sigmoid + bias
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_b,
):
    # 2D grid over rows and columns
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * 1 + tl.arange(0, 1)
    offs_n = pid_n * 1 + tl.arange(0, 1)

    # Loop over columns (N dimension)
    for n in range(0, N):
        x = tl.load(X_ptr + offs_m * stride_xm + n * stride_xn)
        b = tl.load(Bias_ptr + n * stride_b)
        y = 1.0 / (1.0 + tl.exp(-x)) + b
        tl.store(Y_ptr + offs_m * stride_ym + n * stride_yn, y)


@triton.jit
def _group_top2_kernel(
    S_ptr,              # [M, 8, 32] scores after sigmoid + bias (contiguous)
    GroupScores_ptr,    # [M, 8] float32
    Top2Idx_ptr,        # [M, 8, 2] int32
    M, G, E,
    stride_sm, stride_sg, stride_se,
    stride_gs_m, stride_gs_g,
    stride_tmi, stride_tmj, stride_tm_k,
):
    # One program per token
    pid_m = tl.program_id(0)
    # Per-group top-2 and sum to produce group_scores
    for g in range(0, G):
        top1_val = -1.0e30
        top2_val = -1.0e30
        top1_idx = 0
        top2_idx = 0
        base = pid_m * stride_sm + g * stride_sg
        for e in range(0, E):
            s = tl.load(S_ptr + base + e * stride_se)
            if s > top1_val:
                top2_val = top1_val
                top2_idx = top1_idx
                top1_val = s
                top1_idx = e
            elif s > top2_val:
                top2_val = s
                top2_idx = e

        # Store group_scores
        tl.store(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_g, top1_val + top2_val)
        # Store top-2 indices for this group
        tl.store(Top2Idx_ptr + pid_m * stride_tmi + g * stride_tmj + 0 * stride_tm_k, top1_idx)
        tl.store(Top2Idx_ptr + pid_m * stride_tmi + g * stride_tmj + 1 * stride_tm_k, top2_idx)


@triton.jit
def _select_top4_groups_kernel(
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
        # Update top4 using insertion
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

    # Store top4 indices
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 0 * stride_gi_k, top4_idx[0])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 1 * stride_gi_k, top4_idx[1])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 2 * stride_gi_k, top4_idx[2])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 3 * stride_gi_k, top4_idx[3])


@triton.jit
def _mask_and_select_final_top8_kernel(
    S_ptr,              # [M, 8, 32] scores after sigmoid + bias
    GroupMask_ptr,      # [M, 8] float32, 1.0 for selected groups, 0 otherwise
    SelectedIdx_ptr,    # [M, 8] int32
    M, G, E,
    stride_sm, stride_sg, stride_se,
    stride_gmm, stride_gmn,
    stride_sim, stride_sin,
):
    # One program per token
    pid_m = tl.program_id(0)

    # Prepare top8 selection from S_masked. Triton supports static loops, so we do iterative top selection.
    best_val = tl.full((8,), -1.0e30, dtype=tl.float32)
    best_idx = tl.zeros((8,), dtype=tl.int32)

    # First find top8
    for g in range(0, G):
        for e in range(0, E):
            s = tl.load(S_ptr + pid_m * stride_sm + g * stride_sg + e * stride_se)
            # group_mask[g] should be 0 or 1
            mask_g = tl.load(GroupMask_ptr + pid_m * stride_gmm + g * stride_gmn)
            if mask_g > 0:
                # Only consider selected groups
                if s > best_val[0]:
                    best_val[3] = best_val[2]
                    best_val[2] = best_val[1]
                    best_val[1] = best_val[0]
                    best_val[0] = s
                    best_idx[3] = best_idx[2]
                    best_idx[2] = best_idx[1]
                    best_idx[1] = best_idx[0]
                    best_idx[0] = g * E + e
                elif s > best_val[1]:
                    best_val[3] = best_val[2]
                    best_val[2] = best_val[1]
                    best_val[1] = s
                    best_idx[3] = best_idx[2]
                    best_idx[2] = best_idx[1]
                    best_idx[1] = g * E + e
                elif s > best_val[2]:
                    best_val[3] = best_val[2]
                    best_val[2] = s
                    best_idx[3] = best_idx[2]
                    best_idx[2] = g * E + e
                elif s > best_val[3]:
                    best_val[3] = s
                    best_idx[3] = g * E + e

    # Store selected indices
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 0 * stride_sin, best_idx[0])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 1 * stride_sin, best_idx[1])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 2 * stride_sin, best_idx[2])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 3 * stride_sin, best_idx[3])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 4 * stride_sin, best_idx[4])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 5 * stride_sin, best_idx[5])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 6 * stride_sin, best_idx[6])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 7 * stride_sin, best_idx[7])


@triton.jit
def _normalize_and_scale_kernel(
    SelectedScores_ptr,   # [M, 8] float32
    OutWeights_ptr,       # [M, 8] float32
    M,
    stride_smi, stride_smj,
    stride_owi, stride_omj,
    eps, scaling_factor,
):
    pid_m = tl.program_id(0)
    s = tl.load(SelectedScores_ptr + pid_m * stride_smi + 0 * stride_smj)
    total = s + tl.load(SelectedScores_ptr + pid_m * stride_smi + 1 * stride_smj) + tl.load(SelectedScores_ptr + pid_m * stride_smi + 2 * stride_smj) + tl.load(SelectedScores_ptr + pid_m * stride_smi + 3 * stride_smj) + tl.load(SelectedScores_ptr + pid_m * stride_smi + 4 * stride_smj) + tl.load(SelectedScores_ptr + pid_m * stride_smi + 5 * stride_smj) + tl.load(SelectedScores_ptr + pid_m * stride_smi + 6 * stride_smj) + tl.load(SelectedScores_ptr + pid_m * stride_smi + 7 * stride_smj)
    total = total + eps
    w0 = s / total * scaling_factor
    tl.store(OutWeights_ptr + pid_m * stride_owi + 0 * stride_omj, w0)
    tl.store(OutWeights_ptr + pid_m * stride_owi + 1 * stride_omj, tl.load(SelectedScores_ptr + pid_m * stride_smi + 1 * stride_smj) / total * scaling_factor)
    tl.store(OutWeights_ptr + pid_m * stride_owi + 2 * stride_omj, tl.load(SelectedScores_ptr + pid_m * stride_smi + 2 * stride_smj) / total * scaling_factor)
    tl.store(OutWeights_ptr + pid_m * stride_owi + 3 * stride_omj, tl.load(SelectedScores_ptr + pid_m * stride_smi + 3 * stride_smj) / total * scaling_factor)
    tl.store(OutWeights_ptr + pid_m * stride_owi + 4 * stride_omj, tl.load(SelectedScores_ptr + pid_m * stride_smi + 4 * stride_smj) / total * scaling_factor)
    tl.store(OutWeights_ptr + pid_m * stride_owi + 5 * stride_omj, tl.load(SelectedScores_ptr + pid_m * stride_smi + 5 * stride_smj) / total * scaling_factor)
    tl.store(OutWeights_ptr + pid_m * stride_owi + 6 * stride_omj, tl.load(SelectedScores_ptr + pid_m * stride_smi + 6 * stride_smj) / total * scaling_factor)
    tl.store(OutWeights_ptr + pid_m * stride_owi + 7 * stride_omj, tl.load(SelectedScores_ptr + pid_m * stride_smi + 7 * stride_smj) / total * scaling_factor)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float,
    ):
        # Ensure dtype and contiguity
        hidden = hidden_states.to(torch.float32).contiguous()
        weight_t = weight.to(torch.float32).t().contiguous()  # [K, N], K=hidden_size, N=256
        bias = expert_bias.to(torch.float32).contiguous()     # [N]

        M, K = hidden.shape
        N = weight_t.shape[1]  # number of experts; expected 256
        assert N == 256, "This implementation assumes 256 experts."
        E = 32  # experts per group
        G = 8   # number of groups

        # 1) Compute logits [M, N] via Triton GEMM
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
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + expert bias via Triton elementwise kernel
        scores = torch.empty_like(logits)
        _sigmoid_bias_kernel[(M, N)](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            bias.stride(0),
            num_warps=4, num_stages=2,
        )

        # 3) Group top-2 and group scores using Triton
        group_scores = torch.empty((M, G), dtype=torch.float32, device=scores.device)
        top2_idx = torch.empty((M, G, 2), dtype=torch.int32, device=scores.device)

        _group_top2_kernel[(M,)](
            scores.view(M, G, E),
            group_scores, top2_idx,
            M, G, E,
            scores.view(M, G, E).stride(0), scores.view(M, G, E).stride(1), scores.view(M, G, E).stride(2),
            group_scores.stride(0), group_scores.stride(1),
            top2_idx.stride(0), top2_idx.stride(1), top2_idx.stride(2),
            num_warps=2, num_stages=2,
        )

        # 4) Select top-4 groups per token using Triton
        top4_group = torch.empty((M, 4), dtype=torch.int32, device=scores.device)
        _select_top4_groups_kernel[(M,)](
            group_scores,
            top4_group,
            M, G,
            group_scores.stride(0), group_scores.stride(1),
            top4_group.stride(0), top4_group.stride(1),
            num_warps=2, num_stages=2,
        )

        # 5) Build group_mask [M, 8]: 1.0 for selected groups, 0 otherwise
        group_mask = torch.empty((M, G), dtype=torch.float32, device=scores.device)
        for g in range(0, G):
            group_mask[:, g] = 0.0
        group_mask.scatter_(1, top4_group, 1.0)  # scatter indices to set 1.0 at selected groups

        # 6) Mask out non-selected groups by setting their scores to -inf in S_masked
        S_masked = scores.clone()  # [M, 256]
        neg_inf = torch.finfo(torch.float32).min
        for g in range(0, G):
            if (group_mask[:, g] == 1.0).any():
                S_masked[:, g * E : (g + 1) * E].fill_(neg_inf)

        # 7) Final top-8 selection using Triton (iterative top selection)
        selected_idx = torch.empty((M, 8), dtype=torch.int32, device=scores.device)
        _mask_and_select_final_top8_kernel[(M,)](
            scores.view(M, G, E),
            group_mask,
            selected_idx,
            M, G, E,
            scores.view(M, G, E).stride(0), scores.view(M, G, E).stride(1), scores.view(M, G, E).stride(2),
            group_mask.stride(0), group_mask.stride(1),
            selected_idx.stride(0), selected_idx.stride(1),
            num_warps=2, num_stages=2,
        )

        # 8) Normalize selected scores and apply scaling factor
        # We need the actual selected scores from S_masked; we can reconstruct by gathering from S_masked using selected_idx.
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        for i in range(0, 8):
            idx = selected_idx[:, i]  # int32
            # Convert to expert index: expert_idx = (idx // E) * E + (idx % E)
            g = (idx // E) * E
            e = idx - g
            selected_scores[:, i] = torch.where(
                group_mask[:, g // E] > 0,
                S_masked[:, idx],  # gather value
                torch.zeros((), dtype=torch.float32, device=scores.device),
            )

        out_weights = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        eps = 1e-20
        _normalize_and_scale_kernel[(M,)](
            selected_scores,
            out_weights,
            M,
            selected_scores.stride(0), selected_scores.stride(1),
            out_weights.stride(0), out_weights.stride(1),
            eps, routed_scaling_factor,
            num_warps=2, num_stages=2,
        )

        return selected_idx, out_weights


def run(*args):
    return ModelNew()(*args)
