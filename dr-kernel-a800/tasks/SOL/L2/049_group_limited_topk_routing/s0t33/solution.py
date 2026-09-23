import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# 1) GEMM kernel: logits = hidden @ weight.T
# hidden: [M, K], weight: [N, K], logits: [M, N]
@triton.jit
def _linear_gemm_kernel(
    hidden_ptr,  # *float32, [M, K]
    weight_ptr,  # *float32, [N, K]
    logits_ptr,  # *float32, [M, N]
    M, K, N,
    stride_hm, stride_hk,  # strides for hidden
    stride_wn, stride_wk,  # strides for weight (N, K)
    stride_lm, stride_ln,  # strides for logits
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # Pointers for current tiles
    a_ptrs = hidden_ptr + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk)  # [TM, TK]
    b_ptrs = weight_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)  # [TK, TN]

    # Accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, TILE_K):
        # Mask for valid m and n
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k + offs_k[:, None] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [TM, TK]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [TK, TN]
        acc += tl.dot(a, b)  # [TM, TN]
        a_ptrs += TILE_K * stride_hk
        b_ptrs += TILE_K * stride_wk

    # Write back
    c_ptrs = logits_ptr + (offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# 2) Elementwise: scores = sigmoid(logits) + expert_bias
# logits: [M, N], bias: [N], scores: [M, N]
@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_sm, stride_sn,   # logits strides
    stride_bn,              # bias stride
    stride_om, stride_on,   # scores strides
):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N
    # Bounds check
    if (row >= M) or (col >= N):
        return
    val = tl.load(logits_ptr + row * stride_sm + col * stride_sn)
    bias = tl.load(bias_ptr + col * stride_bn)
    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-val))
    out = s + bias
    tl.store(scores_ptr + row * stride_om + col * stride_on, out)


# 3) Group top-2 sum per token: group_scores[t, g] = sum(top2 among 32 experts in group g)
# scores: [M, N], group_scores: [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,  # scores strides
    stride_gm, stride_gn,  # group_scores strides
):
    t = tl.program_id(0)
    if t >= M:
        return
    # We will compute per group. Note N must be divisible by EXP_PER_GROUP (here 32).
    # EXP_PER_GROUP is a constexpr to enable compile-time unrolling.
    for g in range(8):
        base = g * EXP_PER_GROUP
        # Load 32 experts in this group, mask for bounds (though N should be divisible by EXP_PER_GROUP)
        offs = base + tl.arange(0, EXP_PER_GROUP)
        vals = tl.load(scores_ptr + t * stride_sm + offs * stride_sn, mask=(offs < N), other=0.0)
        # Compute top-2
        # We do it by selecting max, remove it, then max again.
        m1 = tl.max(vals, axis=0)
        mask_m1 = vals == m1
        vals2 = tl.where(mask_m1, -1e20, vals)
        m2 = tl.max(vals2, axis=0)
        group_scores_ptr[t * stride_gm + g * stride_gn] = m1 + m2


# 4) Select top-4 groups per token: group_idx[t, 0..3] (sorted=False)
# group_scores: [M, 8], group_idx: [M, 4] (int32)
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr,
    M,
    stride_gm, stride_gn,   # group_scores strides
    stride_im, stride_in,   # group_idx strides
):
    t = tl.program_id(0)
    if t >= M:
        return
    # Initialize with -inf
    m1 = tl.full((), -1e20, tl.float32)
    m2 = tl.full((), -1e20, tl.float32)
    m3 = tl.full((), -1e20, tl.float32)
    m4 = tl.full((), -1e20, tl.float32)
    idx1 = tl.full((), -1, tl.int32)
    idx2 = tl.full((), -1, tl.int32)
    idx3 = tl.full((), -1, tl.int32)
    idx4 = tl.full((), -1, tl.int32)

    # Unrolled scan over 8 groups
    for g in range(8):
        score = tl.load(group_scores_ptr + t * stride_gm + g * stride_gn)
        if score > m1:
            m4 = m3
            m3 = m2
            m2 = m1
            m1 = score
            idx4 = idx3
            idx3 = idx2
            idx2 = idx1
            idx1 = g
        elif score > m2:
            m4 = m3
            m3 = m2
            m2 = score
            idx4 = idx3
            idx3 = idx2
            idx2 = g
        elif score > m3:
            m4 = m3
            m3 = score
            idx4 = idx3
            idx3 = g
        elif score > m4:
            m4 = score
            idx4 = g

    # Write out the 4 indices
    tl.store(group_idx_ptr + t * stride_im + 0 * stride_in, idx1)
    tl.store(group_idx_ptr + t * stride_im + 1 * stride_in, idx2)
    tl.store(group_idx_ptr + t * stride_im + 2 * stride_in, idx3)
    tl.store(group_idx_ptr + t * stride_im + 3 * stride_in, idx4)


# 5) Mask non-selected groups: set scores for non-selected groups to -inf
# scores: [M, N], group_idx: [M, 4], masked_scores: [M, N]
@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr, group_idx_ptr, masked_scores_ptr,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_msm, stride_msn,
    stride_im, stride_in,
):
    t = tl.program_id(0)
    if t >= M:
        return
    # For each group, if not in top4, set its 32 entries to -inf
    for g in range(8):
        idx = tl.load(group_idx_ptr + t * stride_im + g * stride_in)
        base = idx * EXP_PER_GROUP
        # Set all entries of that group to -inf
        for j in range(EXP_PER_GROUP):
            col = base + j
            # Load current value, if col < N then store -inf, else keep
            # We know N is 256 and EXP_PER_GROUP*8 == 256, so col < N always true here.
            val = tl.load(scores_ptr + t * stride_sm + col * stride_sn)
            neg_inf = tl.full((), -1e20, tl.float32)
            tl.store(masked_scores_ptr + t * stride_msm + col * stride_msn, neg_inf)


# 6) Select top-8 from masked scores: top8_indices[t, 0..7] (sorted=False)
# masked_scores: [M, N], top8_indices: [M, 8] (int32)
@triton.jit
def _select_top8_masked_kernel(
    masked_scores_ptr, top8_indices_ptr,
    M, N,
    stride_msm, stride_msn,
    stride_tm, stride_tn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    m1 = tl.full((), -1e20, tl.float32)
    m2 = tl.full((), -1e20, tl.float32)
    m3 = tl.full((), -1e20, tl.float32)
    m4 = tl.full((), -1e20, tl.float32)
    m5 = tl.full((), -1e20, tl.float32)
    m6 = tl.full((), -1e20, tl.float32)
    m7 = tl.full((), -1e20, tl.float32)
    m8 = tl.full((), -1e20, tl.float32)
    idx1 = tl.full((), -1, tl.int32)
    idx2 = tl.full((), -1, tl.int32)
    idx3 = tl.full((), -1, tl.int32)
    idx4 = tl.full((), -1, tl.int32)
    idx5 = tl.full((), -1, tl.int32)
    idx6 = tl.full((), -1, tl.int32)
    idx7 = tl.full((), -1, tl.int32)
    idx8 = tl.full((), -1, tl.int32)

    for i in range(N):
        val = tl.load(masked_scores_ptr + t * stride_msm + i * stride_msn)
        if val > m1:
            m8 = m7
            m7 = m6
            m6 = m5
            m5 = m4
            m4 = m3
            m3 = m2
            m2 = m1
            m1 = val
            idx8 = idx7
            idx7 = idx6
            idx6 = idx5
            idx5 = idx4
            idx4 = idx3
            idx3 = idx2
            idx2 = idx1
            idx1 = i
        elif val > m2:
            m8 = m7
            m7 = m6
            m6 = m5
            m5 = m4
            m4 = m3
            m3 = m2
            m2 = val
            idx8 = idx7
            idx7 = idx6
            idx6 = idx5
            idx5 = idx4
            idx4 = idx3
            idx3 = idx2
            idx2 = i
        elif val > m3:
            m8 = m7
            m7 = m6
            m6 = m5
            m5 = m4
            m4 = m3
            m3 = val
            idx8 = idx7
            idx7 = idx6
            idx6 = idx5
            idx5 = idx4
            idx4 = idx3
            idx3 = i
        elif val > m4:
            m8 = m7
            m7 = m6
            m6 = m5
            m5 = m4
            m4 = val
            idx8 = idx7
            idx7 = idx6
            idx6 = idx5
            idx5 = idx4
            idx4 = i
        elif val > m5:
            m8 = m7
            m7 = m6
            m6 = m5
            m5 = val
            idx8 = idx7
            idx7 = idx6
            idx6 = idx5
            idx5 = i
        elif val > m6:
            m8 = m7
            m7 = m6
            m6 = val
            idx8 = idx7
            idx7 = idx6
            idx6 = i
        elif val > m7:
            m8 = m7
            m7 = val
            idx8 = idx7
            idx7 = i
        elif val > m8:
            m8 = val
            idx8 = i

    tl.store(top8_indices_ptr + t * stride_tm + 0 * stride_tn, idx1)
    tl.store(top8_indices_ptr + t * stride_tm + 1 * stride_tn, idx2)
    tl.store(top8_indices_ptr + t * stride_tm + 2 * stride_tn, idx3)
    tl.store(top8_indices_ptr + t * stride_tm + 3 * stride_tn, idx4)
    tl.store(top8_indices_ptr + t * stride_tm + 4 * stride_tn, idx5)
    tl.store(top8_indices_ptr + t * stride_tm + 5 * stride_tn, idx6)
    tl.store(top8_indices_ptr + t * stride_tm + 6 * stride_tn, idx7)
    tl.store(top8_indices_ptr + t * stride_tm + 7 * stride_tn, idx8)


# 7) Normalize and scale: topk_weight = (selected_scores / sum(selected)) * routed_scaling
# scores: [M, N], top8_indices: [M, 8], routed_scaling_factor: float, topk_weight: [M, 8]
@triton.jit
def _normalize_and_scale_kernel(
    scores_ptr, indices_ptr, out_weights_ptr,
    M, N,
    routed_scaling: tl.float32,
    eps: tl.float32,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
    stride_wm, stride_wn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    total = tl.full((), 0.0, tl.float32)
    for k in range(8):
        idx = tl.load(indices_ptr + t * stride_tm + k * stride_tn)
        val = tl.load(scores_ptr + t * stride_sm + idx * stride_sn)
        total += val
    denom = total + eps
    for k in range(8):
        idx = tl.load(indices_ptr + t * stride_tm + k * stride_tn)
        val = tl.load(scores_ptr + t * stride_sm + idx * stride_sn)
        w = (val / denom) * routed_scaling
        tl.store(out_weights_ptr + t * stride_wm + k * stride_wn, w)


class ModelNew(nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0, eps: float = 1e-20):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only implementation. Ensure CUDA and float32, contiguous.
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew.forward requires CUDA device"
        hidden = hidden_states.to(torch.float32).contiguous()
        weight_c = weight.to(torch.float32).contiguous()
        bias_c = expert_bias.to(torch.float32).contiguous()

        M, K = hidden.shape
        N = weight_c.shape[0]
        assert weight_c.shape == (N, K), "weight must be [num_experts, hidden_dim]"
        assert bias_c.shape == (N,), "expert_bias must be [num_experts]"

        # 1) GEMM: logits = hidden @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_hm, stride_hk = hidden.stride()
        stride_wn, stride_wk = weight_c.stride()
        stride_lm, stride_ln = logits.stride()
        TILE_M = 128
        TILE_N = 64
        TILE_K = 64
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

        # 2) Sigmoid + bias: scores = sigmoid(logits) + bias
        scores = torch.empty_like(logits)
        stride_sm, stride_sn = logits.stride()
        stride_om, stride_on = scores.stride()
        grid_elem = (M * N,)
        _sigmoid_add_bias_kernel[grid_elem](
            logits, bias_c, scores,
            M, N,
            stride_sm, stride_sn,
            bias_c.stride(0),
            stride_om, stride_on,
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 sum per token
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_gm, stride_gn = group_scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1, num_stages=1,
        )

        # 4) Select top-4 groups per token
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_im, stride_in = group_idx.stride()
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M,
            stride_gm=stride_gm, stride_gn=stride_gn,
            stride_im=stride_im, stride_in=stride_in,
            num_warps=1, num_stages=1,
        )

        # 5) Mask non-selected groups: set to -inf
        masked_scores = torch.empty_like(scores)
        stride_msm, stride_msn = masked_scores.stride()
        _mask_nonselected_groups_kernel[(M,)](
            scores, group_idx, masked_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_msm=stride_msm, stride_msn=stride_msn,
            stride_im=stride_im, stride_in=stride_in,
            num_warps=1, num_stages=1,
        )

        # 6) Select top-8 from masked scores
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        stride_tm, stride_tn = top8_indices.stride()
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_msm=stride_msm, stride_msn=stride_msn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1, num_stages=1,
        )

        # 7) Normalize and scale to produce topk_weight
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_wm, stride_wn = topk_weight.stride()
        _normalize_and_scale_kernel[(M,)](
            scores, top8_indices, topk_weight,
            M, N,
            self.routed_scaling_factor, self.eps,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            stride_wm=stride_wm, stride_wn=stride_wn,
            num_warps=1, num_stages=1,
        )

        # Return indices and weights as per original signature
        return top8_indices, topk_weight

# Helper functions kept same as original for testing harness:
# get_inputs and get_init_inputs can be used by the evaluation environment.
def get_inputs():
    # Example input generation: hidden [num_tokens, 256], weight [256, 256], bias [256]
    num_tokens = 2048
    hidden = torch.randn(num_tokens, 256, device='cuda', dtype=torch.float32)
    weight = torch.randn(256, 256, device='cuda', dtype=torch.float32)
    expert_bias = torch.randn(256, device='cuda', dtype=torch.float32)
    routed_scaling_factor = 1.0
    return [hidden, weight, expert_bias, routed_scaling_factor]

def get_init_inputs():
    return []


def run(*args):
    return ModelNew()(*args)
