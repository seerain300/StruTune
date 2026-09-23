import torch
import triton
import triton.language as tl


@triton.jit
def _linear_proj_kernel(
    A,  # hidden_states: [M, K], float32
    B,  # weight: [N, K], float32
    C,  # logits: [M, N], float32
    M, K, N,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    k_offsets = tl.arange(0, TILE_K)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    for k in range(0, K, TILE_K):
        a_ptrs = A + m_offsets[:, None] * stride_am + (k + k_offsets[None, :]) * stride_ak  # [BM, BK]
        b_ptrs = B + n_offsets[None, :] * stride_bn + (k + k_offsets[:, None]) * stride_bk  # [BK, BN]
        a_mask = m_offsets[:, None] < M
        b_mask = n_offsets[None, :] < N
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X,  # logits: [M, N], float32
    bias,  # expert_bias: [N], float32
    Y,  # scores: [M, N], float32
    M, N,
    stride_xm, stride_xn,
    stride_bias_n,
    stride_ym, stride_yn,
):
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    if m >= M:
        return
    x = tl.load(X + m * stride_xm + n * stride_xn)
    b = tl.load(bias + n * stride_bias_n)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(Y + m * stride_ym + n * stride_yn, y)


@triton.jit
def _group_top2_sum_kernel(
    scores,  # [M, N], float32
    group_scores,  # [M, 8], float32
    M, N,
    EXP_PER_GROUP: tl.constexpr,  # 32
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(8):
        start = g * EXP_PER_GROUP
        vals = scores[pid, start : start + EXP_PER_GROUP]
        top1 = -float('inf')
        top2 = -float('inf')
        for i in range(EXP_PER_GROUP):
            val_i = vals[i]
            if val_i > top1:
                top2 = top1
                top1 = val_i
            elif val_i > top2:
                top2 = val_i
        group_scores[pid, g] = top1 + top2


@triton.jit
def _select_top4_groups_bubble_kernel(
    group_scores,  # [M, 8], float32
    top4_groups,  # [M, 4], int32
    M,
    stride_gm, stride_gn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    idxs = tl.arange(0, 8)
    scores_vec = tl.load(group_scores + pid * stride_gm + idxs * stride_gn)  # [8]
    # Bubble sort descending
    for j in range(8):
        for k in range(7, j - 1, -1):
            if scores_vec[k] > scores_vec[k - 1]:
                tmp_score = scores_vec[k]
                tmp_idx = idxs[k]
                scores_vec[k] = scores_vec[k - 1]
                idxs[k] = idxs[k - 1]
                scores_vec[k - 1] = tmp_score
                idxs[k - 1] = tmp_idx
    # Write first 4
    for k in range(4):
        tl.store(top4_groups + pid * 4 + k, idxs[k])


@triton.jit
def _mask_nonselected_groups_kernel(
    scores,  # [M, N], float32
    top4_groups,  # [M, 4], int32
    masked_scores,  # [M, N], float32
    M, N, EXP_PER_GROUP: tl.constexpr,  # 32
    stride_sm, stride_sn,
    stride_mm, stride_mn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(8):
        found = False
        for k in range(4):
            selected_group = tl.load(top4_groups + pid * 4 + k)
            if selected_group == g:
                found = True
                break
        if not found:
            start = g * EXP_PER_GROUP
            for e in range(EXP_PER_GROUP):
                ptr = scores + pid * stride_sm + (start + e) * stride_sn
                val = tl.load(ptr)
                tl.store(masked_scores + pid * stride_mm + (start + e) * stride_mn, -float('inf'), mask=(start + e) < N)


@triton.jit
def _select_top8_masked_kernel(
    masked_scores,  # [M, N], float32 (some entries are -inf)
    top8_indices,  # [M, 8], int32
    M, N,
    stride_mm, stride_mn,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best = [tl.full((), -float('inf'), dtype=tl.float32) for _ in range(8)]
    indices = [tl.full((), -1, dtype=tl.int32) for _ in range(8)]
    for e in range(N):
        score = tl.load(masked_scores + pid * stride_mm + e * stride_mn)
        # Treat -inf as invalid; consider only finite scores
        is_valid = score > -float('inf')
        for j in range(8):
            if is_valid and score > best[j]:
                for kk in range(7, j - 1, -1):
                    best[kk] = best[kk - 1]
                    indices[kk] = indices[kk - 1]
                best[j] = score
                indices[j] = e
                break
    for j in range(8):
        tl.store(top8_indices + pid * 8 + j, indices[j])


@triton.jit
def _normalize_and_scale_kernel(
    top8_indices,  # [M, 8], int32
    masked_scores,  # [M, N], float32
    selected_weight,  # [M, 8], float32
    M, N,
    stride_tmi, stride_tnj,  # strides for indices
    stride_mm, stride_mn,    # strides for masked_scores
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = 0.0
    for j in range(8):
        idx = tl.load(top8_indices + pid * 8 + j)
        score = tl.load(masked_scores + pid * stride_mm + idx * stride_mn)
        total += score
    eps = 1e-20
    for j in range(8):
        idx = tl.load(top8_indices + pid * 8 + j)
        score = tl.load(masked_scores + pid * stride_mm + idx * stride_mn)
        norm = score / (total + eps)
        tl.store(selected_weight + pid * 8 + j, norm)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
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
        grid = (triton.cdiv(M, TILE_M), triton.cdiv(N, TILE_N))
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
        stride_bn = bias_f32.stride(0)
        stride_om, stride_on = scores.stride()
        grid_elem = (M * N,)
        _sigmoid_add_bias_kernel[grid_elem](
            logits, bias_f32, scores,
            M, N,
            stride_sm, stride_sn,
            stride_bn,
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

        # 5) Mask non-selected groups: set masked_scores to -inf for


def run(*args):
    return ModelNew()(*args)
