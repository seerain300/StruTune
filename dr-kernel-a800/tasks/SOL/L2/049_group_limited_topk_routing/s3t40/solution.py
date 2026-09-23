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
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
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
    # 2D tiling
    BLOCK_M = 64
    BLOCK_N = 64
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )
    b = tl.load(Bias_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _group_top2_group_scores_kernel(
    S_ptr,         # [M, 8, 32] = scores_for_routing (after sigmoid + bias)
    G2_ptr,        # [M, 8] = top-2 per group sums
    IDX2_ptr,      # [M, 8, 2] = indices of top-2 per group
    M, N_GROUPS, EXPERTS_PER_GROUP,
    stride_sm, stride_sgrp, stride_sexp,
    stride_gm, stride_gn,
    stride_im, stride_igrp, stride_iexp,
):
    # Each program handles one token (m)
    pid_m = tl.program_id(0)
    # Iterate groups
    for g in range(0, N_GROUPS):
        # Compute top-2 in this group: [experts_per_group] = 32
        scores = tl.load(
            S_ptr + pid_m * stride_sm + g * stride_sgrp + tl.arange(0, EXPERTS_PER_GROUP) * stride_sexp,
            mask=True,
            other=-1e20,  # invalid value
        )
        # Two passes to find top-2: first find max, then find max excluding that
        max_val = tl.full((), -1e20, tl.float32)
        max_idx = 0
        # Find max
        for e in range(0, EXPERTS_PER_GROUP):
            val = scores[e]
            take = val > max_val
            max_val = tl.where(take, val, max_val)
            max_idx = tl.where(take, e, max_idx)
        # Exclude max_idx and find second
        second_val = tl.full((), -1e20, tl.float32)
        second_idx = 0
        for e in range(0, EXPERTS_PER_GROUP):
            val = scores[e]
            is_max = e == max_idx
            cond = (val > second_val) & (~is_max)
            second_val = tl.where(cond, val, second_val)
            second_idx = tl.where(cond, e, second_idx)
        # Store top-2 values and indices
        tl.store(G2_ptr + pid_m * stride_gm + g * stride_gn, max_val + second_val)
        tl.store(IDX2_ptr + pid_m * stride_im + g * stride_igrp + 0 * stride_iexp, max_idx)
        tl.store(IDX2_ptr + pid_m * stride_im + g * stride_igrp + 1 * stride_iexp, second_idx)


@triton.jit
def _select_top4_groups_kernel(
    GROUPSCORES_ptr,  # [M, 8] float32
    GROUPIDX_ptr,     # [M, 4] int32
    M, N_GROUPS,
    stride_gs_m, stride_gs_n,
    stride_gi_m, stride_gi_n,
):
    pid_m = tl.program_id(0)
    # Load group scores
    scores = tl.load(GROUPSCORES_ptr + pid_m * stride_gs_m + tl.arange(0, N_GROUPS) * stride_gs_n,
                     mask=True, other=-1e20)
    # Select top-4 indices
    top1_val = tl.full((), -1e20, tl.float32)
    top1_idx = 0
    for g in range(0, N_GROUPS):
        val = scores[g]
        take = val > top1_val
        top1_val = tl.where(take, val, top1_val)
        top1_idx = tl.where(take, g, top1_idx)
    # Store top1
    tl.store(GROUPIDX_ptr + pid_m * stride_gi_m + 0 * stride_gi_n, top1_idx)
    # top2
    top2_val = tl.full((), -1e20, tl.float32)
    top2_idx = 0
    for g in range(0, N_GROUPS):
        val = scores[g]
        is_g1 = g == top1_idx
        take = (val > top2_val) & (~is_g1)
        top2_val = tl.where(take, val, top2_val)
        top2_idx = tl.where(take, g, top2_idx)
    tl.store(GROUPIDX_ptr + pid_m * stride_gi_m + 1 * stride_gi_n, top2_idx)
    # top3
    top3_val = tl.full((), -1e20, tl.float32)
    top3_idx = 0
    for g in range(0, N_GROUPS):
        val = scores[g]
        is_g1 = g == top1_idx
        is_g2 = g == top2_idx
        take = (val > top3_val) & (~is_g1) & (~is_g2)
        top3_val = tl.where(take, val, top3_val)
        top3_idx = tl.where(take, g, top3_idx)
    tl.store(GROUPIDX_ptr + pid_m * stride_gi_m + 2 * stride_gi_n, top3_idx)
    # top4
    top4_val = tl.full((), -1e20, tl.float32)
    top4_idx = 0
    for g in range(0, N_GROUPS):
        val = scores[g]
        is_g1 = g == top1_idx
        is_g2 = g == top2_idx
        is_g3 = g == top3_idx
        take = (val > top4_val) & (~is_g1) & (~is_g2) & (~is_g3)
        top4_val = tl.where(take, val, top4_val)
        top4_idx = tl.where(take, g, top4_idx)
    tl.store(GROUPIDX_ptr + pid_m * stride_gi_m + 3 * stride_gi_n, top4_idx)


@triton.jit
def _build_group_mask_kernel(
    GROUPIDX_ptr,   # [M, 4] int32
    GROUPMASK_ptr,  # [M, 8] float32
    M, N_GROUPS,
    stride_gi_m, stride_gi_n,
    stride_gm_m, stride_gm_n,
):
    pid_m = tl.program_id(0)
    top_idxs = [0, 1, 2, 3]
    for j in range(0, 4):
        idx = tl.load(GROUPIDX_ptr + pid_m * stride_gi_m + j * stride_gi_n)  # int32
        # cast to int for pointer arithmetic
        idx = idx.to(tl.int32)
        # one-hot at position idx
        for g in range(0, N_GROUPS):
            one = 1.0 if g == idx else 0.0
            tl.store(GROUPMASK_ptr + pid_m * stride_gm_m + g * stride_gm_n, one)


@triton.jit
def _mask_non_selected_kernel(
    SCORES_ptr,         # [M, 256] float32
    GROUPMASK_ptr,      # [M, 8] float32
    MASKED_ptr,         # [M, 256] float32
    M, NUM_EXPERTS,
    stride_s_m, stride_s_n,
    stride_gm_m, stride_gm_n,
    stride_ms_m, stride_ms_n,
):
    pid_m = tl.program_id(0)
    # We assume NUM_EXPERTS == 256 and N_GROUPS == 8 and EXPERTS_PER_GROUP == 32
    for e in range(0, NUM_EXPERTS):
        # determine group for this expert
        g = e // 32
        score = tl.load(SCORES_ptr + pid_m * stride_s_m + e * stride_s_n)
        mask_val = tl.load(GROUPMASK_ptr + pid_m * stride_gm_m + g * stride_gm_n)
        masked = tl.where(mask_val > 0.0, score, -1e20)
        tl.store(MASKED_ptr + pid_m * stride_ms_m + e * stride_ms_n, masked)


@triton.jit
def _select_top8_final_kernel(
    MASKED_ptr,        # [M, 256] float32
    TOPK_idx_ptr,      # [M, 8] int32
    M, NUM_EXPERTS,
    stride_msk_m, stride_msk_n,
    stride_topk_m, stride_topk_n,
):
    pid_m = tl.program_id(0)
    top_vals = [tl.full((), -1e20, tl.float32) for _ in range(8)]
    top_idxs = [tl.full((), -1, tl.int32) for _ in range(8)]
    for e in range(0, NUM_EXPERTS):
        score = tl.load(MASKED_ptr + pid_m * stride_msk_m + e * stride_msk_n)
        # Iterate to find 8 maxima
        for k in range(0, 8):
            if score > top_vals[k]:
                # shift lower positions down
                for kk in range(7, k, -1):
                    top_vals[kk] = top_vals[kk - 1]
                    top_idxs[kk] = top_idxs[kk - 1]
                top_vals[k] = score
                top_idxs[k] = e
                break
    for k in range(0, 8):
        tl.store(TOPK_idx_ptr + pid_m * stride_topk_m + k * stride_topk_n, top_idxs[k])


@triton.jit
def _normalize_and_scale_kernel(
    IDX_ptr,             # [M, 8] int32
    ORIGINAL_ptr,        # [M, 256] float32 (original logits scores before bias)
    OUT_ptr,             # [M, 8] float32
    M, NUM_EXPERTS,
    routed_scaling_factor,
    stride_idx_m, stride_idx_n,
    stride_orig_m, stride_orig_n,
    stride_out_m, stride_out_n,
):
    pid_m = tl.program_id(0)
    # Gather selected scores from original logits (before bias)
    # We need to read 8 indices per token
    for k in range(0, 8):
        idx = tl.load(IDX_ptr + pid_m * stride_idx_m + k * stride_idx_n)
        # idx is int32
        score = tl.load(ORIGINAL_ptr + pid_m * stride_orig_m + idx * stride_orig_n)
        # normalize: divide by sum of selected scores
        # Compute sum of 8 selected scores
        sum_8 = tl.zeros((), dtype=tl.float32)
        for kk in range(0, 8):
            idkk = tl.load(IDX_ptr + pid_m * stride_idx_m + kk * stride_idx_n)
            sk = tl.load(ORIGINAL_ptr + pid_m * stride_orig_m + idkk * stride_orig_n)
            sum_8 += sk
        # epsilon for numerical stability (host sets it to 0.0 in eval; we add 1e-20)
        eps = 1e-20
        norm = score / (sum_8 + eps)
        norm = norm * routed_scaling_factor
        tl.store(OUT_ptr + pid_m * stride_out_m + k * stride_out_n, norm)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts=256, top_k=8, n_group=8, routed_scaling_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.experts_per_group = num_experts // n_group
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure contiguous and float32
        hidden = hidden_states.contiguous().to(torch.float32)  # [M, K]
        weight_t = weight.t().contiguous().to(torch.float32)  # [K, N]
        bias = expert_bias.contiguous().to(torch.float32)     # [N]
        M, K = hidden.shape
        N = self.num_experts

        # 1) GEMM: logits = hidden @ weight.T -> [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _matmul_kernel[grid](
            hidden, weight_t, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        # 2) Sigmoid + bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid2 = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _sigmoid_bias_kernel[grid2](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            bias.stride(0),
            BLOCK_M=64, BLOCK_N=64,
        )

        # Reshape to [M, 8, 32]
        group_scores = scores.view(M, self.n_group, self.experts_per_group)

        # 3) Group top-2 per group and sum to get group_scores [M, 8]
        g2 = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        idx2 = torch.empty((M, self.n_group, 2), dtype=torch.int32, device=hidden.device)
        grid3 = (M,)
        _group_top2_group_scores_kernel[grid3](
            group_scores, g2, idx2,
            M, self.n_group, self.experts_per_group,
            group_scores.stride(0), group_scores.stride(1), group_scores.stride(2),
            g2.stride(0), g2.stride(1),
            idx2.stride(0), idx2.stride(1), idx2.stride(2),
        )

        # 4) Select top-4 groups per token
        group_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        grid4 = (M,)
        _select_top4_groups_kernel[grid4](
            g2, group_idx,
            M, self.n_group,
            g2.stride(0), g2.stride(1),
            group_idx.stride(0), group_idx.stride(1),
        )

        # 5) Build group_mask [M, 8]
        group_mask = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        grid5 = (M,)
        _build_group_mask_kernel[grid5](
            group_idx, group_mask,
            M, self.n_group,
            group_idx.stride(0), group_idx.stride(1),
            group_mask.stride(0), group_mask.stride(1),
        )

        # 6) Mask non-selected groups: create masked_scores [M, N]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid6 = (M,)
        _mask_non_selected_kernel[grid6](
            scores, group_mask, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
        )

        # 7) Select final top-8 experts from masked_scores
        topk_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        grid7 = (M,)
        _select_top8_final_kernel[grid7](
            masked_scores, topk_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
        )

        # 8) Gather selected original logits (before bias) and normalize with scaling
        original_logits = logits  # we already computed logits above
        out = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        grid8 = (M,)
        _normalize_and_scale_kernel[grid8](
            topk_idx, original_logits, out,
            M, N,
            self.top_k,
            topk_idx.stride(0), topk_idx.stride(1),
            original_logits.stride(0), original_logits.stride(1),
            out.stride(0), out.stride(1),
            routed_scaling_factor,
        )

        return topk_idx, out


# The following helper functions mirror the original testing harness interface.
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor: float,
):
    model = ModelNew()
    return model(hidden_states, weight, expert_bias, routed_scaling_factor)


class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
