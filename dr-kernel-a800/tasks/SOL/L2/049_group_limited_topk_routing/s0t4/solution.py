import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_proj_kernel(
    hidden_ptr, weight_ptr, logits_ptr,
    M, N, K,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    # 2D launch: (m_tile, n_tile)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    # Loop over K dimension
    for k in range(0, K, TILE_K):
        k_idx = k + offs_k
        # Load A tile: hidden[offs_m, k_idx]
        a = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hm + k_idx[None, :] * stride_hk,
            mask=(offs_m[:, None] < M) & (k_idx[None, :] < K),
            other=0.0,
        )
        # Load B tile: weight[offs_n, k_idx], then transpose for dot: (k, n)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
        b = tl.load(
            weight_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk,
            mask=(offs_n[:, None] < N) & (k_idx[None, :] < K),
            other=0.0,
        )
        # Dot over K
        acc += tl.sum(a * b, axis=1)
    # Write back
    out = acc
    tl.store(
        logits_ptr + offs_m * stride_om + (pid_n * TILE_N) * stride_on,
        out,
        mask=offs_m < M,
    )


@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_lm, stride_ln,
    stride_bn,
    stride_sm, stride_sn,
):
    # 1D grid over elements
    pid = tl.program_id(0)
    # map pid -> (i, j)
    M_ = tl.load(None)  # not used
    N_ = tl.load(None)  # not used
    i = pid // N
    j = pid % N
    # bounds check
    if (i >= M) or (j >= N):
        return
    val = tl.load(logits_ptr + i * stride_lm + j * stride_ln)
    b = tl.load(bias_ptr + j * stride_bn)
    s = 1.0 / (1.0 + tl.exp(-val)) + b
    tl.store(scores_ptr + i * stride_sm + j * stride_sn, s)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N, n_group, experts_per_group,
    stride_sm, stride_sn,
):
    # one program per token
    t = tl.program_id(0)
    if t >= M:
        return
    # loop over groups
    for g in range(0, n_group):
        start = g * experts_per_group
        # scan for top-2 within this group
        top1 = tl.full((), -1e30, dtype=tl.float32)
        top2 = tl.full((), -1e30, dtype=tl.float32)
        for e in range(0, experts_per_group):
            idx = start + e
            score = tl.load(scores_ptr + t * stride_sm + idx * stride_sn)
            # update top2 if score is greater
            if score > top1:
                top2 = top1
                top1 = score
            elif score > top2:
                top2 = score
        group_sum = top1 + top2
        tl.store(group_scores_ptr + t * n_group + g, group_sum)


@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr,
    M, n_group,
    stride_gm, stride_gn,
):
    # one program per token
    t = tl.program_id(0)
    if t >= M:
        return
    # Load all group scores for this token
    gs = tl.zeros((n_group,), dtype=tl.float32)
    for g in range(0, n_group):
        gs[g] = tl.load(group_scores_ptr + t * n_group + g)
    # Bubble selection of top-4 (descending)
    # Keep selected indices in int32
    selected = tl.zeros((4,), dtype=tl.int32) - 1
    for j in range(4):
        best = -1.0e30
        best_idx = -1
        for g in range(0, n_group):
            if gs[g] > best:
                best = gs[g]
                best_idx = g
        selected[j] = best_idx
        # zero it out
        gs[best_idx] = -1.0e30
    # store
    for j in range(4):
        tl.store(group_idx_ptr + t * 4 + j, selected[j])


@triton.jit
def _final_top8_masked_selection_kernel(
    scores_ptr, group_idx_ptr, selected_idx_ptr, selected_weight_ptr,
    M, N, n_group, experts_per_group, scale_factor, eps,
    stride_sm, stride_sn,
    stride_gim, stride_gin,
    stride_sm_sel, stride_sn_sel,
):
    # one program per token
    t = tl.program_id(0)
    if t >= M:
        return

    # Build group mask: mask = [n_group], 1 for selected groups else 0
    mask = tl.zeros((n_group,), dtype=tl.int32)
    for j in range(4):
        g = tl.load(group_idx_ptr + t * 4 + j).to(tl.int32)
        mask[g] = 1

    # Prepare MaskedScores = scores; set non-selected groups to -inf
    # We will scan over all N experts and set to -inf if mask[group_id] == 0
    # Note: scores_ptr is the original scores after sigmoid + bias
    # Initialize final selection array
    best = tl.zeros((8,), dtype=tl.float32) - 1.0e30
    idx = tl.zeros((8,), dtype=tl.int32) - 1

    for e in range(0, N):
        # Determine group_id for expert e
        g = e // experts_per_group
        # If this expert is in a non-selected group, skip (set to -inf)
        if mask[g] == 0:
            continue
        val = tl.load(scores_ptr + t * stride_sm + e * stride_sn)
        # Iterative top-8 selection: bubble in
        # Note: we can simply overwrite if val > best[j], shifting others down.
        for j in range(7, -1, -1):
            if val > best[j]:
                for k in range(j + 1, 8):
                    best[k - 1] = best[k]
                    idx[k - 1] = idx[k]
                best[j] = val
                idx[j] = e
                break

    # Write selected indices
    for j in range(8):
        tl.store(selected_idx_ptr + t * 8 + j, idx[j])

    # Compute normalized weights: sum of best values, then normalize and scale
    total_sum = 0.0
    for j in range(8):
        total_sum += best[j]
    for j in range(8):
        norm = best[j] / (total_sum + eps)
        scaled = norm * scale_factor
        tl.store(selected_weight_ptr + t * 8 + j, scaled)


# ModelNew: Triton-only forward, no torch ops
class ModelNew(nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0, eps: float = 1e-20):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # We assume:
        # hidden_states: [num_tokens, hidden_dim], dtype float16/float32
        # weight: [num_experts, hidden_dim], dtype float16/float32
        # expert_bias: [num_experts], dtype float32/float16
        # We will compute everything in float32 inside Triton.
        device = hidden_states.device
        num_tokens, hidden_dim = hidden_states.shape
        num_experts = weight.shape[0]
        assert hidden_dim == 256 and num_experts == 256, "This Triton implementation assumes hidden_dim=num_experts=256"
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be CUDA tensors"

        # Ensure contiguous
        hidden = hidden_states.contiguous().to(torch.float32)
        weight_t = weight.contiguous().to(torch.float32)
        bias = expert_bias.contiguous().to(torch.float32)

        # 1) Compute logits with Triton GEMM: [num_tokens, num_experts]
        logits = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=device)
        M, N, K = num_tokens, num_experts, hidden_dim
        stride_hm, stride_hk = hidden.stride()
        stride_wn, stride_wk = weight_t.stride()
        stride_om, stride_on = logits.stride()
        TILE_N = 32
        TILE_K = 64
        grid = (triton.cdiv(M, TILE_N), triton.cdiv(N, TILE_N))
        _linear_proj_kernel[grid](
            hidden, weight_t, logits,
            M, N, K,
            stride_hm, stride_hk,
            stride_wn, stride_wk,
            stride_om, stride_on,
            TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid + bias in Triton, elementwise: scores [num_tokens, num_experts]
        scores = torch.empty_like(logits)
        stride_lm, stride_ln = logits.stride()
        stride_bn = bias.stride(0)
        stride_sm, stride_sn = scores.stride()
        grid_elem = (M * N,)
        _sigmoid_add_bias_kernel[grid_elem](
            logits, bias, scores,
            M, N,
            stride_lm, stride_ln,
            stride_bn,
            stride_sm, stride_sn,
            num_warps=1, num_stages=1,
        )

        # 3) Group aggregation: group_scores [num_tokens, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_sm, stride_sn = scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N, 8, 32,
            stride_sm, stride_sn,
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token: group_idx [num_tokens, 4] int32
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_gm, stride_gn = group_scores.stride()
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, 8,
            stride_gm, stride_gn,
            num_warps=1, num_stages=1,
        )

        # 5) Final top-8 selection with masking and normalization in Triton
        selected_idx = torch.empty((M, 8), dtype=torch.int64, device=device)
        selected_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_sm, stride_sn = scores.stride()
        stride_gim, stride_gin = group_idx.stride()
        stride_sm_sel, stride_sn_sel = selected_idx.stride()
        _final_top8_masked_selection_kernel[(M,)](
            scores, group_idx, selected_idx, selected_weight,
            M, N, 8, 32, self.routed_scaling_factor, self.eps,
            stride_sm, stride_sn,
            stride_gim, stride_gin,
            stride_sm_sel, stride_sn_sel,
            num_warps=1, num_stages=1,
        )

        # Return as requested: topk_idx (int64) and topk_weight (float32)
        return selected_idx, selected_weight


def run(*args):
    return ModelNew()(*args)
