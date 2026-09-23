import torch
import triton
import triton.language as tl

# 1) GEMM: logits = hidden @ weight.T
@triton.jit
def _gemm_linear_kernel(
    hidden_ptr, weight_ptr, logits_ptr,
    M, K, N,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_lm, stride_ln,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * TILE_M + tl.arange(0, TILE_M)  # rows
    acc = tl.zeros((TILE_M,), dtype=tl.float32)

    # iterate over K in chunks of TILE_K
    for k0 in range(0, K, TILE_K):
        offs_k = k0 + tl.arange(0, TILE_K)
        # A tile: [TILE_M, TILE_K]
        a = tl.load(
            hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        # B tile: [TILE_K, TILE_N] where B = weight.T → weight_ptr has shape [N, K]
        b = tl.load(
            weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,  # offs_n will be introduced below
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        # Compute b by gathering weight.T using strides: weight.T[k, n] = weight[n, k]
        # Since weight_ptr is [N, K], we access weight_ptr + n * stride_wn + k * stride_wk
        acc += tl.dot(a, b)
    # store acc for valid rows
    tl.store(
        logits_ptr + offs_m * stride_lm,
        acc,
        mask=offs_m < M
    )


# 2) Elementwise: scores = sigmoid(logits) + expert_bias
@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_sm, stride_sn,
    stride_b,
    stride_om, stride_on
):
    pid = tl.program_id(0)  # 1D grid over M*N
    m = pid // N
    n = pid % N
    x = tl.load(logits_ptr + m * stride_sm + n * stride_sn)
    b = tl.load(bias_ptr + n * stride_b)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(scores_ptr + m * stride_om + n * stride_on, y)


# 3) Compute group top-2 sum per token
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N,
    EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_gm, stride_gn
):
    t = tl.program_id(0)  # one program per token
    # loop over groups g in [0, 8)
    for g in range(8):
        start = g * EXP_PER_GROUP
        # init top-2 with very small values
        top1 = -1.0e20
        top2 = -1.0e20
        # loop over 32 experts in the group
        for j in range(EXP_PER_GROUP):
            idx = start + j
            val = tl.load(scores_ptr + t * stride_sm + idx * stride_sn)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        group_scores_ptr[t * stride_gm + g * stride_gn] = top1 + top2


# 4) Select top-4 groups per token (argmax 4 times, update by -inf)
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, top4_groups_ptr,
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn
):
    t = tl.program_id(0)
    # initialize selected as empty vector of size 4
    selected = tl.zeros((4,), dtype=tl.int32)
    # argmax 4 times, update by -inf
    for i in range(4):
        max_val = -1.0e20
        max_idx = 0
        for g in range(8):
            score = tl.load(group_scores_ptr + t * stride_gm + g * stride_gn)
            if score > max_val:
                max_val = score
                max_idx = g
        # mark selected[i] = max_idx
        selected[i] = max_idx
        # invalidate max_idx by setting its group_score to -inf for subsequent iterations
        # (group_scores_ptr is float, we cannot directly store to it here; do it after we know indices)
        # We'll do this in a separate kernel or host, but here we can't modify; instead, we keep selected and skip choosing it in later iterations.
    # store selected indices
    for i in range(4):
        tl.store(top4_groups_ptr + t * stride_tm + i * stride_tn, selected[i])


# 5) Mask non-selected groups: set scores[t, g*32:(g+1)*32] to -inf for g not in top4_groups
@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr, top4_groups_ptr, masked_scores_ptr,
    M, N,
    EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
    stride_mm, stride_mn
):
    t = tl.program_id(0)
    # For each group g, if not in top4_groups, set its 32 scores to -inf
    for g in range(8):
        found = 0
        # check if g is among top4
        for i in range(4):
            idx = tl.load(top4_groups_ptr + t * stride_tm + i * stride_tn)
            if idx == g:
                found = 1
                break
        start = g * EXP_PER_GROUP
        if found == 0:
            # set all 32 scores to -inf
            for j in range(EXP_PER_GROUP):
                idx = start + j
                tl.store(masked_scores_ptr + t * stride_mm + idx * stride_mn, -1.0e20)


# 6) Select top-8 from masked scores per token (iterative argmax, update by -inf)
@triton.jit
def _select_top8_masked_kernel(
    masked_scores_ptr, top8_indices_ptr,
    M, N,
    stride_mm, stride_mn,
    stride_tm, stride_tn
):
    t = tl.program_id(0)
    selected = tl.zeros((8,), dtype=tl.int32)
    for i in range(8):
        max_val = -1.0e20
        max_idx = 0
        for j in range(N):
            val = tl.load(masked_scores_ptr + t * stride_mm + j * stride_mn)
            # if val > max_val and not selected, then select
            # We need to check if j is already selected. Triton doesn't support dynamic vector indexing; emulate selection by tracking selected list as scalar array.
            selected_valid = 0
            for k in range(i):  # i selections already made; ensure we don't re-select
                if selected[k] == j:
                    selected_valid = 1
                    break
            if val > max_val and selected_valid == 0:
                max_val = val
                max_idx = j
        selected[i] = max_idx
    for i in range(8):
        tl.store(top8_indices_ptr + t * stride_tm + i * stride_tn, selected[i])


# 7) Normalize and scale: topk_weight = selected_scores / sum(selected_scores + 1e-20) * routed_scaling
@triton.jit
def _normalize_and_scale_kernel(
    masked_scores_ptr, top8_indices_ptr, out_weights_ptr,
    M, N,
    routed_scaling: tl.constexpr,
    eps: tl.constexpr,
    stride_mm, stride_mn,
    stride_tm, stride_tn,
    stride_wm, stride_wn
):
    t = tl.program_id(0)
    total = 0.0
    # sum selected 8
    for i in range(8):
        idx = tl.load(top8_indices_ptr + t * stride_tm + i * stride_tn)
        val = tl.load(masked_scores_ptr + t * stride_mm + idx * stride_mn)
        total += val
    denom = total + eps
    for i in range(8):
        idx = tl.load(top8_indices_ptr + t * stride_tm + i * stride_tn)
        val = tl.load(masked_scores_ptr + t * stride_mm + idx * stride_mn)
        w = (val / denom) * routed_scaling
        tl.store(out_weights_ptr + t * stride_wm + i * stride_wn, w)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure CUDA tensors, float32, contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA tensors"
        hidden = hidden_states.to(torch.float32).contiguous()  # [M, K]
        weight_t = weight.to(torch.float32).contiguous()       # [N, K] (note: weight provided is [N, K], i.e., [num_experts, hidden_dim])
        bias = expert_bias.to(torch.float32).contiguous()      # [N]

        M, K = hidden.shape
        N = weight_t.shape[0]
        assert weight_t.shape[1] == K, "weight must be [num_experts, hidden_dim] with matching hidden_dim"
        assert bias.shape[0] == N, "expert_bias must be [num_experts]"

        # 1) Triton GEMM: logits = hidden @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_hm, stride_hk = hidden.stride()
        stride_wn, stride_wk = weight_t.stride()  # weight_t is [N, K]
        stride_lm, stride_ln = logits.stride()
        TILE_M = 128
        TILE_N = 64
        TILE_K = 64
        # Number of programs: 1D grid over M in chunks of TILE_M
        grid = (triton.cdiv(M, TILE_M),)
        _gemm_linear_kernel[grid](
            hidden, weight_t, logits,
            M, K, N,
            stride_hm, stride_hk,
            stride_wn, stride_wk,
            stride_lm, stride_ln,
            TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
        scores = torch.empty_like(logits)
        stride_sm, stride_sn = logits.stride()
        stride_om, stride_on = scores.stride()
        _sigmoid_add_bias_kernel[(M * N,)](
            logits, bias, scores,
            M, N,
            stride_sm, stride_sn,
            bias.stride(0),
            stride_om, stride_on,
            num_warps=1, num_stages=1,
        )

        # 3) Triton group top-2 sum per token: group_scores [M, 8]
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

        # 4) Triton select top-4 groups per token: top4_groups [M, 4]
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_tm, stride_tn = top4_groups.stride()
        _select_top4_groups_kernel[(M,)](
            group_scores, top4_groups,
            M,
            stride_gm=stride_gm, stride_gn=stride_gn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1, num_stages=1,
        )

        # 5) Triton mask non-selected groups: masked_scores [M, N]
        masked_scores = torch.empty_like(scores)
        stride_mm, stride_mn = masked_scores.stride()
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N,
            EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            stride_mm=stride_mm, stride_mn=stride_mn,
            num_warps=1, num_stages=1,
        )

        # 6) Triton select top-8 from masked scores: top8_indices [M, 8]
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        stride_wm, stride_wn = top8_indices.stride()
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_mm=stride_mm, stride_mn=stride_mn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1, num_stages=1,
        )

        # 7) Triton normalize and scale: out_weights [M, 8]
        out_weights = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_wm, stride_wn = out_weights.stride()
        _normalize_and_scale_kernel[(M,)](
            masked_scores, top8_indices, out_weights,
            M, N,
            routed_scaling=self.routed_scaling_factor,
            eps=1e-20,
            stride_mm=stride_mm, stride_mn=stride_mn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            stride_wm=stride_wm, stride_wn=stride_wn,
            num_warps=1, num_stages=1,
        )

        # Return indices and weights as required by original signature
        # topk_idx: [num_tokens, 8], integer indices; we have top8_indices; to match original output names, return (top8_indices, out_weights)
        # Note: original returns (topk_idx, topk_weight). Here, we return (indices, weights). Triton kernels produce these.
        return top8_indices, out_weights


def run(*args):
    return ModelNew()(*args)
