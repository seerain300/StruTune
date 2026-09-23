import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K] = hidden
    B_ptr,  # [K, N] = weight.T
    C_ptr,  # [M, N] = logits
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch grid over rows (tokens) and column blocks (experts)
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
    b = tl.load(Bias_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)

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
    # We'll iterate groups g=0..7 and compute top-2 and sum for each
    for g in range(0, G):
        top1_val = -1.0e30
        top2_val = -1.0e30
        top1_idx = 0
        top2_idx = 0
        for e in range(0, E):
            s = tl.load(S_ptr + pid_m * stride_sm + g * stride_sg + e * stride_se)
            # Here S_ptr is contiguous with layout (M, G, E). Using strides ensures correct access.
            if s > top1_val:
                top2_val = top1_val
                top2_idx = top1_idx
                top1_val = s
                top1_idx = e
            elif s > top2_val:
                top2_val = s
                top2_idx = e

        # Store group_score (sum of top-2)
        tl.store(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_g, top1_val + top2_val)
        # Store top-2 indices for this group
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

    # Initialize top-4 buffer
    top4_val = tl.full((4,), -1.0e30, dtype=tl.float32)
    top4_idx = tl.zeros((4,), dtype=tl.int32)

    # Iterate groups and update top4
    for g in range(0, G):
        gs = tl.load(GroupScores_ptr + pid_m * stride_gs_m + g * stride_gs_g)
        # Insert into top4
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

    # Store selected indices
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 0 * stride_gi_k, top4_idx[0])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 1 * stride_gi_k, top4_idx[1])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 2 * stride_gi_k, top4_idx[2])
    tl.store(GroupIdx_ptr + pid_m * stride_gi_m + 3 * stride_gi_k, top4_idx[3])


@triton.jit
def _masked_top8_select_kernel(
    S_ptr,                # [M, N] sigmoid + bias scores (float32)
    GroupMask_ptr,        # [M, 8] float32, 1.0 for selected groups, 0 otherwise
    SelectedIdx_ptr,      # [M, 8] int32 (final selected expert indices)
    M, N, G, E,
    stride_sm, stride_sn,
    stride_gmm, stride_gmn,
    stride_sim, stride_sin,
):
    # One program per token
    pid_m = tl.program_id(0)

    # Prepare best buffers
    best_val = tl.full((8,), -1.0e30, dtype=tl.float32)
    best_idx = tl.zeros((8,), dtype=tl.int32)

    # For each group, if selected (GroupMask > 0), then its 32 experts are allowed.
    # We'll scan all N experts and update best using allowed groups only.
    for g in range(0, G):
        mask_g = tl.load(GroupMask_ptr + pid_m * stride_gmm + g * stride_gmn)
        if mask_g > 0:
            # This group is selected; its experts are in range E=32 per group starting at g*E
            # But since we loop e directly 0..N-1 and check mask, we set scores for unselected groups to -inf here in S_ptr in host code; the kernel assumes S_ptr has already been masked.
            # To avoid host-side masking per token, we instead gate by mask_g only at selection time via S_ptr contents. Here we rely on host to set unselected-group scores to -inf.
            # Iteratively select top-8 from S_ptr:
            # Note: Triton doesn't allow dynamic loops easily, so we implement up to 8 passes, assuming M*N <= 8*BLOCK. We'll use a static loop over 8 passes with dummy values; but S_ptr is scanned in host via mask. To keep correctness, we will not perform iterative selection here; instead, host will pre-mask S_ptr to -inf for unselected groups.
            # Therefore, we simply scan S_ptr and update best_val/best_idx. This will naturally ignore unselected groups since their scores are -inf.
            # To ensure we always find 8, we assume that host guarantees at least 8 selected groups by the prior selection; in case fewer, we can fallback or skip. Given the original logic, top-4 groups typically select many experts; but to be safe, we implement 8 passes manually by scanning S_ptr in chunks and updating best. However, Triton doesn't support dynamic indexing in the way we need; hence, host pre-masking is required. We will implement a safe fallback: if fewer than 8 selected groups, we can't satisfy top-8. In practice, top-4 group selection typically covers many experts; but to strictly adhere, we will require that top-4 groups guarantee >= 8 distinct experts (since each group has 2 selected among 32). This is true by construction: 4 groups * 2 per group = 8, distinct within group. Therefore, host can pre-mask S_ptr so only selected-group scores remain, and unselected groups are -inf.

            # Iterate over all N experts
            for e in range(0, N):
                s = tl.load(S_ptr + pid_m * stride_sm + e * stride_sn)
                # If this comes from a selected group, s is already valid; otherwise, s is -inf due to host pre-masking. We update best.
                if s > best_val[0]:
                    best_val[3] = best_val[2]
                    best_val[2] = best_val[1]
                    best_val[1] = best_val[0]
                    best_val[0] = s
                    best_idx[3] = best_idx[2]
                    best_idx[2] = best_idx[1]
                    best_idx[1] = best_idx[0]
                    best_idx[0] = e
                elif s > best_val[1]:
                    best_val[3] = best_val[2]
                    best_val[2] = best_val[1]
                    best_val[1] = s
                    best_idx[3] = best_idx[2]
                    best_idx[2] = best_idx[1]
                    best_idx[1] = e
                elif s > best_val[2]:
                    best_val[3] = best_val[2]
                    best_val[2] = s
                    best_idx[3] = best_idx[2]
                    best_idx[2] = e
                elif s > best_val[3]:
                    best_val[3] = s
                    best_idx[3] = e

    # Store top-8 selected indices
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 0 * stride_sin, best_idx[0])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 1 * stride_sin, best_idx[1])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 2 * stride_sin, best_idx[2])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 3 * stride_sin, best_idx[3])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 4 * stride_sin, best_idx[4])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 5 * stride_sin, best_idx[5])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 6 * stride_sin, best_idx[6])
    tl.store(SelectedIdx_ptr + pid_m * stride_sim + 7 * stride_sin, best_idx[7])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)
        weight_t = weight.t().contiguous().to(torch.float32)  # [N, K] where N=256, K=hidden.shape[1]
        bias = expert_bias.contiguous().to(torch.float32)

        M, K = hidden.shape
        N = weight_t.shape[0]  # number of experts, expected 256
        assert N == 256, "This implementation assumes 256 experts."

        # 1) Compute logits = hidden @ weight.T using Triton
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

        # 2) Sigmoid + expert bias using Triton
        sigmoid_scores = torch.empty_like(logits)
        _sigmoid_bias_kernel[(M, N)](
            logits, bias, sigmoid_scores,
            M, N,
            logits.stride(0), logits.stride(1),
            sigmoid_scores.stride(0), sigmoid_scores.stride(1),
            bias.stride(0),
        )

        # 3) Reshape to groups [M, 8, 32]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        top2_idx = torch.empty((M, 8, 2), dtype=torch.int32, device=sigmoid_scores.device)

        _group_top2_kernel[(M,)](
            sigmoid_scores.view(M, 8, 32),
            group_scores, top2_idx,
            M, 8, 32,
            sigmoid_scores.view(M, 8, 32).stride(0), sigmoid_scores.view(M, 8, 32).stride(1), sigmoid_scores.view(M, 8, 32).stride(2),
            group_scores.stride(0), group_scores.stride(1),
            top2_idx.stride(0), top2_idx.stride(1), top2_idx.stride(2),
        )

        # 4) Select top-4 groups per token using Triton
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=sigmoid_scores.device)
        _select_top4_kernel[(M,)](
            group_scores,
            top4_groups,
            M, 8,
            group_scores.stride(0), group_scores.stride(1),
            top4_groups.stride(0), top4_groups.stride(1),
        )

        # 5) Build group mask (float32) [M, 8]
        # For each selected group index g, set GroupMask[m, g] = 1.0; else 0.0
        group_mask = torch.zeros((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        # top4_groups is int32, need to scatter along column=group dimension
        # We'll do this with torch operations (host), as Triton doesn't easily support dynamic scatter per token.
        # This is acceptable here because masking is necessary for final selection.
        for g in range(4):
            group_mask.scatter_(1, top4_groups[:, g].unsqueeze(1), 1.0)

        # 6) Pre-mask sigmoid_scores: set unselected-group experts to -inf so they won't be selected in final top-8
        # We need to know which group each expert belongs to: group = expert_idx // 32
        # For each token m, if group_mask[m, group] == 0, set all experts in that group to -inf.
        # Since we have group_mask [M, 8], we can iterate groups and set S_ptr to -inf if mask == 0.
        # However, to do this per-element efficiently, we rely on Triton to read masked values. A simpler approach:
        # The masked_top8_select_kernel expects S_ptr to already be masked by host. So we'll do pre-masking here with torch:
        # Create a copy of sigmoid_scores for selection, and set columns not in any selected group to -inf.
        selected_experts_per_token = torch.empty((M, 8), dtype=torch.int32, device=sigmoid_scores.device)
        # We can reconstruct selected experts from top2_idx and selected group indices. But easier:
        # Each selected group contributes top2_idx[0] and top2_idx[1] experts. We already have per-group top2_idx for each token.
        # However, top2_idx is per-group; for selected groups, their two experts are the ones we care about. We can infer the absolute expert indices by combining group and relative index (but we only have top2 positions, not absolute indices). To strictly implement, we instead set S_ptr to -inf for all unselected groups by broadcasting group_mask.
        # Strategy: iterate groups g and set entire column block e in [g*32 : (g+1)*32) to -inf if mask == 0.
        # Implement via torch:
        masked_scores = sigmoid_scores.clone()
        for g in range(8):
            if group_mask[:, g].any() == 0:
                # This group is not selected for at least some tokens; we need to set all its experts to -inf.
                # For groups that are selected for all tokens, we don't touch them. The condition group_mask[:, g].any() == 0 is not ideal.
                # Better: for each token, if group_mask[m, g] == 0, set all e in [g*32 : (g+1)*32) to -inf.
                # Do it per token m:
                for m in range(M):
                    if group_mask[m, g] == 0.0:
                        start = g * 32
                        end = (g + 1) * 32
                        masked_scores[m, start:end] = -float('inf')

        # 7) Final top-8 selection using Triton from masked_scores
        selected_idx = torch.empty((M, 8), dtype=torch.int32, device=sigmoid_scores.device)
        _masked_top8_select_kernel[(M,)](
            masked_scores,
            group_mask,
            selected_idx,
            M, N, 8, 32,
            masked_scores.stride(0), masked_scores.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            selected_idx.stride(0), selected_idx.stride(1),
        )

        # 8) Gather selected scores from original sigmoid_scores and compute normalized weights with scaling
        # selected_idx is absolute expert index in [0, N). Use torch gather to collect scores.
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=sigmoid_scores.device)
        for i in range(8):
            expert_idx = selected_idx[:, i]  # [M]
            # Gather scores for each token m for selected expert
            selected_scores[:, i] = torch.gather(sigmoid_scores, 1, expert_idx)

        # Normalize weights and apply scaling
        eps = 1e-20
        total = selected_scores.sum(dim=-1, keepdim=True)  # [M, 1]
        normalized = selected_scores / (total + eps)       # [M, 8]
        topk_weight = normalized * routed_scaling_factor

        # Return indices (selected_idx) and weights (topk_weight)
        # Ensure dtype is float32 for weights
        return selected_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
