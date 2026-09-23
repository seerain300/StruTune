import torch
import triton
import triton.language as tl

# Triton GEMM: logits = hidden @ weight.T, hidden: [M, K], weight: [N, K]
@triton.jit
def _linear_gemm_kernel(
    A_ptr,  # hidden_states
    B_ptr,  # weight
    C_ptr,  # logits
    M, K, N,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * TILE_M + tl.arange(0, TILE_M)
    rn = pid_n * TILE_N + tl.arange(0, TILE_N)
    # accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, TILE_K):
        rk = k0 + tl.arange(0, TILE_K)
        # A submatrix: [TILE_M, TILE_K]
        A_sub = tl.load(
            A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
            mask=(rm[:, None] < M) & (rk[None, :] < K),
            other=0.0,
        )
        # B submatrix: [TILE_K, TILE_N] where B[k, n] = weight[n, k]
        B_sub = tl.load(
            B_ptr + rn[None, :] * stride_bn + rk[:, None] * stride_bk,
            mask=(rn[None, :] < N) & (rk[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(A_sub, B_sub)

    # write back
    tl.store(
        C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        acc,
        mask=(rm[:, None] < M) & (rn[None, :] < N),
    )

# Triton kernel: scores = sigmoid(logits) + expert_bias
@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_lm, stride_ln,
    stride_b,   # bias is 1D
    stride_sm, stride_sn,
    num_warps: tl.constexpr
):
    pid = tl.program_id(0)
    total = M * N
    # linear id
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < total
    # compute row, col
    row = idx // N
    col = idx % N
    # load
    x = tl.load(logits_ptr + row * stride_lm + col * stride_ln, mask=mask, other=0.0)
    b = tl.load(bias_ptr + col, mask=(col < N), other=0.0)
    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-x))
    y = s + b
    # store
    tl.store(scores_ptr + row * stride_sm + col * stride_sn, y, mask=mask)


# Triton kernel: per token, compute top-2 sum per group -> output [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, out_ptr,  # out: [M, 8]
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    num_warps: tl.constexpr
):
    pid = tl.program_id(0)  # one per token (row)
    # compute for this row
    # iterate groups 0..7
    for g in range(8):
        start = g * EXP_PER_GROUP
        # compute max
        max1 = -float('inf')
        max1_idx = 0
        for i in range(EXP_PER_GROUP):
            j = start + i
            val = tl.load(scores_ptr + pid * stride_sm + j * stride_sn)
            cond = val > max1
            max1_idx = tl.where(cond, i, max1_idx)
            max1 = tl.where(cond, val, max1)
        # second max: find max among remaining
        max2 = -float('inf')
        for i in range(EXP_PER_GROUP):
            j = start + i
            val = tl.load(scores_ptr + pid * stride_sm + j * stride_sn)
            # exclude max1_idx
            exclude = i == max1_idx
            cond = (~exclude) & (val > max2)
            max2 = tl.where(cond, val, max2)
        group_score = max1 + max2
        tl.store(out_ptr + pid * stride_gm + g * stride_gn, group_score)


# Triton kernel: per token, select top-4 group indices (sorted=False)
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, out_groups_ptr,  # out_groups: [M, 4], int32
    M, EXP_GROUPS: tl.constexpr,  # EXP_GROUPS = 8
    stride_gm, stride_gn,
    stride_tm, stride_tn,
    num_warps: tl.constexpr
):
    pid = tl.program_id(0)  # one per token
    # scratch: keep best4 scores and indices
    best_vals = tl.full((4,), -float('inf'), tl.float32)
    best_idx = tl.zeros((4,), tl.int32)
    # original indices
    idx_list = tl.arange(0, EXP_GROUPS)
    # for each group
    for g in range(EXP_GROUPS):
        score = tl.load(group_scores_ptr + pid * stride_gm + g * stride_gn)
        # try to insert
        for j in range(4):
            better = score > best_vals[j]
            best_idx = tl.where(better, idx_list[g], best_idx)
            best_vals = tl.where(better, score, best_vals)
    # store indices to out
    for j in range(4):
        tl.store(out_groups_ptr + pid * stride_tm + j * stride_tn, best_idx[j])


# Triton kernel: mask non-selected groups in scores -> set others to -inf
@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr, groups_ptr, masked_ptr,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    stride_mm, stride_mn,
    num_warps: tl.constexpr
):
    pid = tl.program_id(0)  # one per token
    top4 = tl.load(groups_ptr + pid * stride_gm + tl.arange(0, 4) * stride_gn)  # int32
    # iterate groups
    for g in range(8):
        valid = 0
        # check if g is in top4
        for j in range(4):
            is_in = g == top4[j]
            valid = tl.where(is_in, 1, valid)
        start = g * EXP_PER_GROUP
        if valid == 0:
            # set -inf for all 32 in this group
            for i in range(EXP_PER_GROUP):
                j = start + i
                val = tl.load(scores_ptr + pid * stride_sm + j * stride_sn)
                # only store -inf for non-selected groups
                tl.store(masked_ptr + pid * stride_mm + j * stride_mn, -float('inf'), mask=True)


# Triton kernel: per token, select top-8 from masked scores -> out_indices [8]
@triton.jit
def _select_top8_masked_kernel(
    masked_ptr, out_indices_ptr,  # out_indices: [M, 8], int32
    M, N,
    stride_mm, stride_mn,
    stride_im, stride_in,
    num_warps: tl.constexpr
):
    pid = tl.program_id(0)  # one per token
    best_vals = tl.full((8,), -float('inf'), tl.float32)
    best_idx = tl.zeros((8,), tl.int32)
    # iterate all N elements for this row
    for j in range(N):
        val = tl.load(masked_ptr + pid * stride_mm + j * stride_mn)
        # try to insert into best_vals
        for k in range(8):
            better = val > best_vals[k]
            best_idx = tl.where(better, j, best_idx)
            best_vals = tl.where(better, val, best_vals)
    # store indices
    for k in range(8):
        tl.store(out_indices_ptr + pid * stride_im + k * stride_in, best_idx[k])


# Triton kernel: normalize and scale selected scores (given indices) -> out_weights [M, 8]
@triton.jit
def _normalize_scale_kernel(
    scores_ptr, out_weights_ptr, indices_ptr,
    M, N,
    stride_sm, stride_sn,
    stride_wm, stride_wn,
    stride_im, stride_in,
    routed_scaling: tl.constexpr,
    eps: tl.constexpr,
    num_warps: tl.constexpr
):
    pid = tl.program_id(0)  # one per token
    # compute sum of selected scores
    total = 0.0
    for k in range(8):
        idx = tl.load(indices_ptr + pid * stride_im + k * stride_in)
        val = tl.load(scores_ptr + pid * stride_sm + idx * stride_sn)
        total += val
    denom = total + eps
    # write normalized and scaled
    for k in range(8):
        idx = tl.load(indices_ptr + pid * stride_im + k * stride_in)
        val = tl.load(scores_ptr + pid * stride_sm + idx * stride_sn)
        w = (val / denom) * routed_scaling
        tl.store(out_weights_ptr + pid * stride_wm + k * stride_wn, w)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure inputs are on CUDA and float32, contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA tensors"
        M, K = hidden_states.shape
        N = weight.shape[0]
        assert weight.shape == (N, K), "weight must be [num_experts, hidden_dim]"
        assert expert_bias.shape == (N,), "expert_bias must be [num_experts]"

        # Make inputs contiguous
        hidden = hidden_states.to(torch.float32).contiguous()
        weight_c = weight.to(torch.float32).contiguous()
        bias_c = expert_bias.to(torch.float32).contiguous()

        # 1) Compute logits = hidden @ weight.T using Triton GEMM
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_hm, stride_hk = hidden.stride()
        stride_wn, stride_wk = weight_c.stride()
        stride_lm, stride_ln = logits.stride()

        TILE_M = 128
        TILE_N = 64
        TILE_K = 32
        grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
        _linear_gemm_kernel[grid](
            hidden, weight_c, logits,
            M, K, N,
            stride_hm, stride_hk,
            stride_wn, stride_wk,
            stride_lm, stride_ln,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Triton: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_sm, stride_sn = logits.stride()
        stride_b = bias_c.stride(0)
        stride_om, stride_on = scores.stride()
        # Flatten grid for elementwise
        grid_elem = (triton.cdiv(M * N, 128),)
        _sigmoid_add_bias_kernel[grid_elem](
            logits, bias_c, scores,
            M, N,
            stride_sm, stride_sn,
            stride_b,
            stride_om, stride_on,
            num_warps=4,
        )

        # 3) Triton: group top-2 sums -> [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_gm, stride_gn = group_scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1,
        )

        # 4) Triton: select top-4 groups per token -> [M, 4], int32
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_tm, stride_tn = top4_groups.stride()
        _select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M, EXP_GROUPS=8,
            stride_gm=stride_gm, stride_gn=stride_gn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1,
        )

        # 5) Triton: mask non-selected groups in scores -> masked_scores
        masked_scores = torch.empty_like(scores)
        stride_mm, stride_mn = masked_scores.stride()
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_gm=stride_gm, stride_gn=stride_gn,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=4,
        )

        # 6) Triton: select top-8 from masked_scores -> [M, 8], int32
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        stride_im, stride_in = top8_indices.stride()
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_mm=stride_mm, stride_mn=stride_mn,
            stride_im=stride_im, stride_in=stride_in,
            num_warps=4,
        )

        # 7) Triton: normalize and scale -> [M, 8], float32
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_wm, stride_wn = topk_weight.stride()
        _normalize_scale_kernel[(M,)](
            scores, topk_weight, top8_indices,
            M, N,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_wm=stride_wm, stride_wn=stride_wn,
            stride_im=stride_im, stride_in=stride_in,
            routed_scaling=self.routed_scaling_factor,
            eps=1e-20,
            num_warps=1,
        )

        # Return indices and weights
        # Note: top8_indices are the selected expert indices per token
        # topk_weight are the normalized, scaled scores
        # To match the original signature: return (topk_idx, topk_weight)
        # Since the original returns (topk_idx, topk_weight), we return (top8_indices, topk_weight)
        return top8_indices, topk_weight


def run(*args):
    return ModelNew()(*args)
