import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_proj_kernel(hidden, weight, logits,
                         M, K, N,
                         stride_hm, stride_hk,
                         stride_wn, stride_wk,
                         stride_lm, stride_ln,
                         TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr):
    # 2D launch: pid0 over rows, pid1 over cols
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    rows = pid0 * TILE_M + tl.arange(0, TILE_M)
    cols = pid1 * TILE_N + tl.arange(0, TILE_N)

    # Accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, TILE_K):
        ks = k0 + tl.arange(0, TILE_K)

        # A_tile: [TILE_M, TILE_K] = hidden[rows, ks]
        A_ptrs = hidden + rows[:, None] * stride_hm + ks[None, :] * stride_hk
        A = tl.load(A_ptrs, mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)

        # B_tile: [TILE_K, TILE_N] = weight[cols, ks] but weight is [N, K], so we access weight.T as [K, N]
        # We need weight.T[ks, cols] = weight[cols, ks]
        B_ptrs = weight + cols[None, :] * stride_wn + ks[:, None] * stride_wk
        B = tl.load(B_ptrs, mask=(cols[None, :] < N) & (ks[:, None] < K), other=0.0)

        acc += tl.dot(A, B)

    # Store results
    out_ptrs = logits + rows[:, None] * stride_lm + cols[None, :] * stride_ln
    tl.store(out_ptrs, acc, mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _sigmoid_add_bias_kernel(logits, bias, scores,
                              M, N,
                              stride_sm, stride_sn,
                              stride_bb,  # bias stride (likely 1)
                              stride_om, stride_on,
                              num_warps: tl.constexpr, num_stages: tl.constexpr):
    # 1D launch over all elements
    idx = tl.program_id(0)
    if idx >= M * N:
        return
    t = idx // N
    e = idx % N
    val = tl.load(logits + t * stride_sm + e * stride_sn)
    val = 1.0 / (1.0 + tl.exp(-val))
    b = tl.load(bias + e * stride_bb)
    val += b
    tl.store(scores + t * stride_om + e * stride_on, val)


@triton.jit
def _group_top2_sum_kernel(scores, group_scores,
                            M, N,
                            EXP_PER_GROUP: tl.constexpr,
                            stride_sm, stride_sn,
                            stride_gm, stride_gn):
    # One program per token
    t = tl.program_id(0)
    if t >= M:
        return
    # Compute top-2 sum for each of the 8 groups
    for g in range(8):
        start = g * EXP_PER_GROUP
        # Loop over 32 experts in this group
        # We need the original scores per token; scores[t, start:start+32]
        for i in range(EXP_PER_GROUP):
            idx = start + i
            score = tl.load(scores + t * stride_sm + idx * stride_sn)
            # Maintain top1 and top2 in registers
            top1 = -float("inf")
            top2 = -float("inf")
            # Simple inner loop to find top-2 among these 32
            for j in range(EXP_PER_GROUP):
                s = tl.load(scores + t * stride_sm + (start + j) * stride_sn)
                if s > top1:
                    top2 = top1
                    top1 = s
                elif s > top2:
                    top2 = s
            # Accumulate into group_scores[t, g]
            tl.store(group_scores + t * stride_gm + g * stride_gn, top1 + top2, mask=True)


@triton.jit
def _select_top4_groups_bubble_kernel(group_scores, selected_groups,
                                      M,
                                      stride_gm, stride_gn):
    # One program per token
    t = tl.program_id(0)
    if t >= M:
        return
    best = tl.full((), -float("inf"), dtype=tl.float32)
    # Initialize selected_groups[4] as -1
    sel = tl.full((4,), -1, dtype=tl.int32)
    # Bubble-like selection of top-4
    for r in range(4):
        # Find current max in group_scores[t, :]
        for g in range(8):
            cur = tl.load(group_scores + t * stride_gm + g * stride_gn)
            if cur > best:
                best = cur
                idx = g
        # Record index
        sel[r] = idx
        # Exclude selected group by setting its score to -inf
        tl.store(group_scores + t * stride_gm + idx * stride_gn, -float("inf"))
        # Prepare next
        best = tl.full((), -float("inf"), dtype=tl.float32)
    # Store selected groups
    for r in range(4):
        tl.store(selected_groups + t * 4 + r, sel[r])


@triton.jit
def _mask_nonselected_groups_kernel(scores, selected_groups, masked_scores,
                                    M, N,
                                    EXP_PER_GROUP: tl.constexpr,
                                    stride_sm, stride_sn,
                                    stride_mm, stride_mn):
    # One program per token
    t = tl.program_id(0)
    if t >= M:
        return
    # Initialize masked_scores with scores
    # Then set non-selected groups to -inf
    for g in range(8):
        start = g * EXP_PER_GROUP
        for i in range(EXP_PER_GROUP):
            idx = start + i
            val = tl.load(scores + t * stride_sm + idx * stride_sn)
            tl.store(masked_scores + t * stride_mm + idx * stride_mn, val)
    # Exclude non-selected groups
    # For each selected group, do nothing; for non-selected, set to -inf
    # selected_groups is int32 [M, 4]
    # We can check which groups are selected by comparing with -1 (we set -1 for non-selected initially).
    # However, we only have top-4 selected; others are not selected. So we set all groups not in sel to -inf.
    for r in range(4):
        g_sel = tl.load(selected_groups + t * 4 + r)  # int32
        start = g_sel * EXP_PER_GROUP
        for i in range(EXP_PER_GROUP):
            idx = start + i
            # Do nothing, keep original
            pass
    # Now set remaining 4 groups to -inf
    for g in range(8):
        if g not in [0, 1, 2, 3]:  # guarded by selected_groups filled
            continue
        # We cannot branch on Python 'if g not in ...' inside Triton. Instead, rely on selected_groups to store -1 for non-selected.
        # We fill selected_groups with actual selected indices, so non-selected groups are not touched above; set them here.
        # But to implement this correctly, we need to explicitly set non-selected groups. We'll detect non-selected by not being in selected_groups.
        # Since Triton lacks Python-side loop control on tensors, we implement exclusion by iterating all groups and checking if they are selected.
        is_selected = False
        for r in range(4):
            if g == tl.load(selected_groups + t * 4 + r):
                is_selected = True
                break
        if not is_selected:
            start = g * EXP_PER_GROUP
            for i in range(EXP_PER_GROUP):
                idx = start + i
                tl.store(masked_scores + t * stride_mm + idx * stride_mn, -float("inf"))


@triton.jit
def _select_top8_masked_kernel(masked_scores, selected_indices,
                                M, N,
                                stride_mm, stride_mn):
    # One program per token
    t = tl.program_id(0)
    if t >= M:
        return
    # We select 8 maxima iteratively
    for r in range(8):
        max_val = -float("inf")
        max_idx = -1
        # Scan all N experts
        for e in range(N):
            val = tl.load(masked_scores + t * stride_mm + e * stride_mn)
            if val > max_val:
                max_val = val
                max_idx = e
        # Record index
        tl.store(selected_indices + t * 8 + r, max_idx)
        # Exclude by setting to -inf
        tl.store(masked_scores + t * stride_mm + max_idx * stride_mn, -float("inf"))


@triton.jit
def _normalize_and_scale_kernel(masked_scores, selected_indices, selected_weights,
                                M, N, routed_scale, eps,
                                stride_mm, stride_mn):
    # One program per token
    t = tl.program_id(0)
    if t >= M:
        return
    total = 0.0
    for r in range(8):
        idx = tl.load(selected_indices + t * 8 + r)
        val = tl.load(masked_scores + t * stride_mm + idx * stride_mn)
        total += val
    denom = total + eps
    for r in range(8):
        idx = tl.load(selected_indices + t * 8 + r)
        val = tl.load(masked_scores + t * stride_mm + idx * stride_mn)
        norm = val / denom
        tl.store(selected_weights + t * 8 + r, norm * routed_scale)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]
        assert weight.shape[1] == K, "weight second dim must equal hidden_dim"
        assert expert_bias.shape[0] == N, "expert_bias must match num_experts"

        # 1) Triton GEMM: logits = hidden @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        hidden_contig = hidden_states.contiguous().to(torch.float32)
        weight_contig = weight.contiguous().to(torch.float32)
        stride_hm, stride_hk = hidden_contig.stride()
        stride_wn, stride_wk = weight_contig.stride()
        stride_lm, stride_ln = logits.stride()
        TILE_M = 64
        TILE_N = 32
        TILE_K = 64
        grid = (_ceil_div(M, TILE_M), _ceil_div(N, TILE_N))
        _linear_proj_kernel[grid](
            hidden_contig, weight_contig, logits,
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
        bias_f32 = expert_bias.contiguous().to(torch.float32)
        stride_bb = bias_f32.stride(0)
        stride_om, stride_on = scores.stride()
        grid_elem = (_ceil_div(M * N, 1),)
        _sigmoid_add_bias_kernel[grid_elem](
            logits, bias_f32, scores,
            M, N,
            stride_sm, stride_sn,
            stride_bb,
            stride_om, stride_on,
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 sum per token
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_gm, stride_gn = group_scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        _select_top4_groups_bubble_kernel[(M,)](
            group_scores, top4_groups,
            M,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups: set masked_scores to -inf for non-selected groups
        masked_scores = torch.empty_like(scores)
        stride_mm, stride_mn = masked_scores.stride()
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=1, num_stages=1,
        )

        # 6) Select top-8 from masked scores
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=1, num_stages=1,
        )

        # 7) Normalize and scale
        top8_weights = torch.empty((M, 8), dtype=torch.float32, device=device)
        _normalize_and_scale_kernel[(M,)](
            masked_scores, top8_indices, top8_weights,
            M, N,
            routed_scale=float(routed_scaling_factor), eps=1e-20,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=1, num_stages=1,
        )

        # Return indices and weights
        # topk_idx: [num_tokens, 8], int
        # topk_weight: [num_tokens, 8], float
        # Note: The original returns indices as int; Triton kernel returns int32; convert to int64 for parity with original.
        topk_idx = top8_indices.to(torch.int64)
        topk_weight = top8_weights

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
