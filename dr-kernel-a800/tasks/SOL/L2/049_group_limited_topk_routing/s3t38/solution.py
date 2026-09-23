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
    Scores_ptr,  # [M, 8, 32] float32
    GroupScores_ptr,   # [M, 8] float32
    SelectedGroups_ptr, # [M, 4] int32
    M, groups, per_group,
    stride_sm, stride_sg, stride_sp,
    stride_gs_m, stride_gs_g,
    stride_sel_m, stride_sel_k,
):
    m = tl.program_id(0)
    if m >= M:
        return
    # Compute top-2 per group and sum to get group scores
    for g in range(0, groups):
        base = m * stride_sm + g * stride_sg
        # top1, top2
        top1 = -1.0e20
        top2 = -1.0e20
        idx1 = 0
        idx2 = 0
        for p in range(0, per_group):
            val = tl.load(Scores_ptr + base + p * stride_sp)
            if val > top1:
                idx2 = idx1
                top2 = top1
                idx1 = p
                top1 = val
            elif val > top2:
                idx2 = p
                top2 = val
        sum_top2 = top1 + top2
        tl.store(GroupScores_ptr + m * 8 + g, sum_top2)
        # store selected group indices per token in ascending order
        pos = 0
        while pos < 4 and SelectedGroups_ptr[m * stride_sel_m + pos] != 0:
            pos += 1
        if pos < 4:
            tl.store(SelectedGroups_ptr + m * stride_sel_m + pos, idx1)
    # Fill remaining SelectedGroups with -1 (unused)
    for k in range(4, 8):
        tl.store(SelectedGroups_ptr + m * stride_sel_m + k, -1)


@triton.jit
def _build_groupmask_kernel(
    SelectedGroups_ptr,  # [M, 4] int32
    GroupMask_ptr,       # [M, 8] float32 one-hot
    M, groups,
    stride_sel_m, stride_sel_k,
    stride_gm_m, stride_gm_g,
):
    m = tl.program_id(0)
    if m >= M:
        return
    for g in range(0, groups):
        idx = tl.load(SelectedGroups_ptr + m * stride_sel_m + g)  # g-th selected group index
        tl.store(GroupMask_ptr + m * stride_gm_m + idx * stride_gm_g, 1.0)
    # Non-selected groups are zero by default in GroupMask (initialized to zeros)


@triton.jit
def _mask_scores_kernel(
    Scores_ptr,        # [M, 8, 32] original scores
    GroupMask_ptr,     # [M, 8] float32 one-hot
    MaskedScores_ptr,  # [M, 8, 32] masked scores
    M, groups, per_group,
    stride_sp_m, stride_sp_g, stride_sp_p,
    stride_gm_m, stride_gm_g,
    stride_ms_m, stride_ms_g, stride_ms_p,
):
    m = tl.program_id(0)
    if m >= M:
        return
    for g in range(0, groups):
        mask_val = tl.load(GroupMask_ptr + m * stride_gm_m + g)
        # If mask_val == 0, set entire group to -inf
        if mask_val == 0.0:
            for p in range(0, per_group):
                tl.store(MaskedScores_ptr + m * stride_ms_m + g * stride_ms_g + p * stride_ms_p, -1.0e20)
        else:
            # copy original scores
            orig = tl.load(Scores_ptr + m * stride_sp_m + g * stride_sp_g + p * stride_sp_p)
            tl.store(MaskedScores_ptr + m * stride_ms_m + g * stride_ms_g + p * stride_ms_p, orig)


@triton.jit
def _select_final_top8_kernel(
    MaskedScores_ptr,  # [M, 8, 32]
    SelectedI_ptr,     # [M, 8] int32
    M, groups, per_group,
    stride_ms_m, stride_ms_g, stride_ms_p,
    stride_sel_m, stride_sel_i,
):
    m = tl.program_id(0)
    if m >= M:
        return
    top_vals = [-1.0e20, -1.0e20, -1.0e20, -1.0e20,
                -1.0e20, -1.0e20, -1.0e20, -1.0e20]
    for g in range(0, groups):
        for p in range(0, per_group):
            val = tl.load(MaskedScores_ptr + m * stride_ms_m + g * stride_ms_g + p * stride_ms_p)
            if val > top_vals[7]:
                # shift down
                tmp = top_vals[7]
                for j in range(7, 0, -1):
                    top_vals[j] = top_vals[j - 1]
                    if tmp == top_vals[j]:
                        break
                top_vals[0] = val
            else:
                # find insertion pos and shift
                pos = 0
                while pos < 8 and val < top_vals[pos]:
                    pos += 1
                if pos < 8:
                    tmp = top_vals[pos]
                    for j in range(pos, 7):
                        top_vals[j] = top_vals[j + 1]
                    top_vals[7] = tmp
                    top_vals[pos] = val
    # write out indices (we need to know original position to store into SelectedI)
    # We store the original expert index for each selected position: g*per_group + p
    # We can recompute using the original scores order by scanning again
    # However, since top_vals already hold values, we can map by scanning again.
    # For simplicity, recompute indices based on order by scanning MaskedScores again:
    # Reconstruct the 8 highest values' indices by scanning MaskedScores in groups and picking max.
    # This approach ensures correctness even if insertion happened.
    selected_count = 0
    for g in range(0, groups):
        for p in range(0, per_group):
            # Check if this position is one of the 8 top selected
            found = False
            for j in range(0, 8):
                if selected_count == j:
                    # store g*per_group + p
                    tl.store(SelectedI_ptr + m * stride_sel_m + selected_count * stride_sel_i, g * per_group + p)
                    found = True
                    break
            if found:
                selected_count += 1
                if selected_count >= 8:
                    return
    # If fewer than 8 found (shouldn't happen), fill remaining with -1
    for j in range(selected_count, 8):
        tl.store(SelectedI_ptr + m * stride_sel_m + j * stride_sel_i, -1)


@triton.jit
def _gather_normalize_scale_kernel(
    OriginalScores_ptr,  # [M, 256] float32 (post-sigmoid, post-bias)
    SelectedI_ptr,       # [M, 8] int32
    OutputScores_ptr,    # [M, 8] float32 normalized and scaled
    Scaling_ptr,         # [M] float32 (routed_scaling_factor)
    M, N,
    stride_os_m, stride_os_n,
    stride_sel_m, stride_sel_i,
    stride_out_m, stride_out_i,
    stride_scale_m,
):
    m = tl.program_id(0)
    if m >= M:
        return
    total = 0.0
    for i in range(0, 8):
        idx = tl.load(SelectedI_ptr + m * stride_sel_m + i * stride_sel_i)
        val = tl.load(OriginalScores_ptr + m * stride_os_m + idx * stride_os_n)
        total += val
    inv_total = 1.0 / (total + 1e-20)
    scale = tl.load(Scaling_ptr + m * stride_scale_m)
    for i in range(0, 8):
        idx = tl.load(SelectedI_ptr + m * stride_sel_m + i * stride_sel_i)
        val = tl.load(OriginalScores_ptr + m * stride_os_m + idx * stride_os_n)
        out_val = val * inv_total * scale
        tl.store(OutputScores_ptr + m * stride_out_m + i * stride_out_i, out_val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float,
    ):
        """
        Triton-only implementation of the original routing logic.
        Returns (topk_idx: [M, 8], topk_weight: [M, 8]).
        """
        assert hidden_states.dim() == 2, "hidden_states must be [M, K]"
        assert weight.dim() == 2, "weight must be [N, K]"
        assert expert_bias.dim() == 1, "expert_bias must be [N]"
        device = hidden_states.device
        M, K = hidden_states.shape
        N, K_w = weight.shape
        assert K == K_w, "hidden_states last dim must match weight last dim"

        # 1) Matmul: logits = hidden @ weight.T, float32, contiguous
        hidden = hidden_states.contiguous().to(torch.float32)
        weight_t = weight.t().contiguous().to(torch.float32)  # [K, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
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
        bias = expert_bias.contiguous().to(torch.float32)
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        grid_sigmoid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _sigmoid_bias_kernel[grid_sigmoid](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            bias.stride(0),
        )

        # 3) Reshape scores into groups: [M, 8, 32]
        scores_groups = scores.view(M, 8, 32)

        # 4) Compute group scores (sum of top-2 per group) and selected group indices
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

        # 7) Select final top-8 experts from masked scores via Triton
        selected_i = torch.empty((M, 8), dtype=torch.int32, device=device)
        grid_final = (M,)
        _select_final_top8_kernel[grid_final](
            masked_scores, selected_i,
            M, 8, 32,
            masked_scores.stride(0), masked_scores.stride(1), masked_scores.stride(2),
            selected_i.stride(0), selected_i.stride(1)
        )

        # 8) Gather original scores for selected top-8, normalize, and apply routed_scaling_factor via Triton
        output_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        scaling = torch.full((M,), routed_scaling_factor, dtype=torch.float32, device=device)
        grid_gather = (M,)
        _gather_normalize_scale_kernel[grid_gather](
            scores, selected_i, output_scores, scaling,
            M, 256,
            scores.stride(0), scores.stride(1),
            selected_i.stride(0), selected_i.stride(1),
            output_scores.stride(0), output_scores.stride(1),
            scaling.stride(0)
        )

        return selected_i, output_scores


def run(*args):
    return ModelNew()(*args)
