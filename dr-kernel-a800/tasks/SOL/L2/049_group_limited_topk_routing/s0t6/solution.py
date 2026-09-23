import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_proj_kernel(
    hidden_ptr, weight_ptr, logits_ptr,
    M, K, N,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_lm, stride_ln,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Offsets
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    # Accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K, TILE_K):
        offs_k = k + tl.arange(0, TILE_K)
        # A tile: [TILE_M, TILE_K]
        a_ptrs = hidden_ptr + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # B tile: we want B[k, n], but we have weight[n, k] => indexing weight_ptr with n, k
        b_ptrs = weight_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # Accumulate
        acc += tl.dot(a, b)
    # Write back logits
    out_ptrs = logits_ptr + (offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_lm, stride_ln,
    stride_bn,
    stride_om, stride_on,
):
    # Flatten over M*N programs
    pid = tl.program_id(0)
    # Compute row and col
    m = pid // N
    n = pid % N
    if m >= M:
        return
    # Load logits and bias
    logit = tl.load(logits_ptr + m * stride_lm + n * stride_ln)
    bias = tl.load(bias_ptr + n * stride_bn)
    # Sigmoid + bias
    score = 1.0 / (1.0 + tl.exp(-logit)) + bias
    # Store
    tl.store(scores_ptr + m * stride_om + n * stride_on, score)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
):
    # One program per token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # Compute top-2 sum for each group
    for g in range(0, 8):
        start = g * EXP_PER_GROUP
        # Load group values
        vals = tl.zeros((EXP_PER_GROUP,), dtype=tl.float32)
        for i in range(0, EXP_PER_GROUP):
            idx = start + i
            val = tl.load(scores_ptr + pid_m * stride_sm + idx * stride_sn)
            vals[i] = val
        # Compute top-2 (unsorted) and sum
        # Note: torch.topk expects a tensor; here vals is a Triton vector.
        # We can simulate top-2 via reductions:
        # First max
        max1 = -float('inf')
        for i in range(0, EXP_PER_GROUP):
            v = vals[i]
            if v > max1:
                max1 = v
        # Exclude max1 by setting it to -inf
        for i in range(0, EXP_PER_GROUP):
            if vals[i] == max1:
                vals[i] = -float('inf')
        max2 = -float('inf')
        for i in range(0, EXP_PER_GROUP):
            v = vals[i]
            if v > max2:
                max2 = v
        group_score = max1 + max2
        # Store
        tl.store(group_scores_ptr + pid_m * 8 + g, group_score)


@triton.jit
def _select_top4_groups_bubble_kernel(
    group_scores_ptr, top4_groups_ptr,
    M,
    stride_gm, stride_gn,
):
    # One program per token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # Initialize top4 scores/indices
    best_scores = [tl.full((), -float('inf'), dtype=tl.float32) for _ in range(4)]
    selected_idx = [tl.full((), -1, dtype=tl.int32) for _ in range(4)]
    for g in range(0, 8):
        score = tl.load(group_scores_ptr + pid_m * 8 + g * stride_gn)
        for j in range(4):
            if score > best_scores[j]:
                # Shift right and insert
                for k in range(3, j - 1, -1):
                    best_scores[k] = best_scores[k - 1]
                    selected_idx[k] = selected_idx[k - 1]
                best_scores[j] = score
                selected_idx[j] = g
                break
    # Store top-4 indices
    for j in range(4):
        tl.store(top4_groups_ptr + pid_m * 4 + j, selected_idx[j])


@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr, top4_groups_ptr, mask_ptr,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_tm,
):
    # One program per token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    for g in range(0, 8):
        is_selected = False
        # Check if g is in the top4 (top4_groups_ptr contains 4 indices)
        # Compare g against 4 entries
        for j in range(4):
            idx = tl.load(top4_groups_ptr + pid_m * 4 + j)
            if idx == g:
                is_selected = True
                break
        # Mask all scores for non-selected groups to -inf
        if not is_selected:
            start = g * EXP_PER_GROUP
            for i in range(0, EXP_PER_GROUP):
                idx = start + i
                val = tl.load(scores_ptr + pid_m * stride_sm + idx * stride_sn)
                # If not selected, set to -inf
                if val != val:  # NaN check
                    # Do nothing
                    pass
                else:
                    if not is_selected:
                        tl.store(mask_ptr + pid_m * N + idx, -float('inf'))
                    # However, we need to write into scores_ptr: setting masked scores
                    # We'll write back masked scores; but since Triton can't modify input tensor directly,
                    # we instead store -inf into a separate masked_scores tensor.
                    # Here, we'll just write -inf into mask_ptr at positions we want to mask, and in forward we
                    # will use mask_ptr to decide which positions to set in scores. To keep it simple, we
                    # maintain a separate masked_scores tensor in forward, and this kernel writes -inf into
                    # it. For clarity, we'll return now and let forward handle the write-back.
    # Note: This kernel is actually used to populate masked_scores with -inf for non-selected groups; see forward.


@triton.jit
def _select_top8_masked_kernel(
    masked_scores_ptr, top8_indices_ptr,
    M, N,
    stride_mm, stride_mn,
):
    # One program per token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # Initialize best scores and indices
    best_scores = [tl.full((), -float('inf'), dtype=tl.float32) for _ in range(8)]
    selected_idx = [tl.full((), -1, dtype=tl.int32) for _ in range(8)]
    for e in range(0, N):
        score = tl.load(masked_scores_ptr + pid_m * stride_mm + e * stride_mn)
        # Insert into top-8
        for j in range(8):
            if score > best_scores[j]:
                for k in range(7, j - 1, -1):
                    best_scores[k] = best_scores[k - 1]
                    selected_idx[k] = selected_idx[k - 1]
                best_scores[j] = score
                selected_idx[j] = e
                break
    # Store top-8 indices
    for j in range(8):
        tl.store(top8_indices_ptr + pid_m * 8 + j, selected_idx[j])


@triton.jit
def _normalize_and_scale_kernel(
    scores_ptr, top8_indices_ptr, top8_weights_ptr,
    M, N,
    stride_sm, stride_sn,
    stride_im,
):
    # One program per token
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    total_sum = 0.0
    for j in range(8):
        e = tl.load(top8_indices_ptr + pid_m * 8 + j)
        score = tl.load(scores_ptr + pid_m * stride_sm + e * stride_sn)
        total_sum += score
    for j in range(8):
        e = tl.load(top8_indices_ptr + pid_m * 8 + j)
        score = tl.load(scores_ptr + pid_m * stride_sm + e * stride_sn)
        norm = score / (total_sum + 1e-20)
        tl.store(top8_weights_ptr + pid_m * 8 + j, norm)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]
        assert N == 256, "num_experts must be 256"
        assert K == 256, "hidden_dim must be 256"
        assert expert_bias.shape[0] == N, "expert_bias must match num_experts"

        # 1) Triton GEMM: logits = hidden @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        hidden_f32 = hidden_states.contiguous().to(torch.float32)
        weight_f32 = weight.contiguous().to(torch.float32)
        stride_hm, stride_hk = hidden_f32.stride()
        stride_wn, stride_wk = weight_f32.stride()
        stride_lm, stride_ln = logits.stride()
        TILE_M = 64
        TILE_N = 32
        TILE_K = 64
        grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
        _linear_proj_kernel[grid](
            hidden_f32, weight_f32, logits,
            M, K, N,
            stride_hm, stride_hk,
            stride_wn, stride_wk,
            stride_lm, stride_ln,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + expert bias (Triton)
        scores = torch.empty_like(logits)
        stride_sm, stride_sn = logits.stride()
        stride_om, stride_on = scores.stride()
        bias_f32 = expert_bias.contiguous().to(torch.float32)
        stride_bn = bias_f32.stride(0)
        _sigmoid_add_bias_kernel[(M * N,)](
            logits, bias_f32, scores,
            M, N,
            stride_sm, stride_sn,
            stride_bn,
            stride_om, stride_on,
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 sum per token (Triton)
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_gm, stride_gn = group_scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token (Triton)
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        _select_top4_groups_bubble_kernel[(M,)](
            group_scores, top4_groups,
            M,
            stride_gm=stride_gm, stride_gn=1,
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups (Triton). We'll maintain a masked_scores tensor and set -inf for non-selected groups.
        masked_scores = torch.empty_like(scores)  # initialize zeros; we'll set -inf for non-selected
        # Copy scores into masked_scores first
        # Note: Triton kernel will write -inf directly into masked_scores; here we just rely on forward to call the kernel.
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_tm=masked_scores.stride(0),
            num_warps=1, num_stages=1,
        )

        # 6) Select top-8 from masked_scores (Triton)
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_mm=masked_scores.stride(0), stride_mn=masked_scores.stride(1),
            num_warps=1, num_stages=1,
        )

        # 7) Normalize and scale (Triton)
        top8_weights = torch.empty((M, 8), dtype=torch.float32, device=device)
        _normalize_and_scale_kernel[(M,)](
            scores, top8_indices, top8_weights,
            M, N,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_im=top8_indices.stride(0),
            num_warps=1, num_stages=1,
        )

        # Return topk_idx and topk_weight
        # topk_idx: indices of selected experts per token; topk_weight: normalized scores * routed_scaling_factor
        # top8_indices are the selected indices; top8_weights are normalized without scaling.
        # Apply routed_scaling_factor to weights
        topk_idx = top8_indices  # already int32 indices
        topk_weight = top8_weights * routed_scaling_factor

        # Note: For exact matching with the original, we should return (topk_idx, topk_weight).
        # Ensure dtypes: topk_idx is int64 expected by original; cast to int64.
        topk_idx = topk_idx.to(torch.int64)

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
