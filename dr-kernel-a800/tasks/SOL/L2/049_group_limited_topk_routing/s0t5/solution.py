import torch
import triton
import triton.language as tl


@triton.jit
def _linear_proj_kernel(
    Hidden_ptr,      # *f32, [M, K]
    Weight_ptr,      # *f32, [N, K] (we will access as W[n, k])
    Out_ptr,         # *f32, [M, N]
    M, K, N,         # int32 sizes
    stride_hm, stride_hk,   # strides for Hidden
    stride_wn, stride_wk,   # strides for Weight
    stride_om, stride_on,   # strides for Out
    TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for k0 in range(0, K, TILE_K):
        k_offsets = k0 + tl.arange(0, TILE_K)
        k_mask = k_offsets < K
        A_ptrs = Hidden_ptr + m_offsets[:, None] * stride_hm + k_offsets[None, :] * stride_hk
        A = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        B_ptrs = Weight_ptr + n_offsets[None, :] * stride_wn + k_offsets[:, None] * stride_wk
        B = tl.load(B_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(A, B)
    Out_ptrs = Out_ptr + m_offsets[:, None] * stride_om + n_offsets[None, :] * stride_on
    tl.store(Out_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _sigmoid_add_bias_kernel(
    In_ptr,          # *f32, [M, N]
    Bias_ptr,        # *f32, [N]
    Out_ptr,         # *f32, [M, N]
    M, N,
    stride_im, stride_in,   # strides for In
    stride_bn,               # stride for Bias
    stride_om, stride_on,   # strides for Out
):
    pid = tl.program_id(0)
    # one program per element
    m = pid // N
    n = pid % N
    if (m >= M) or (n >= N):
        return
    x = tl.load(In_ptr + m * stride_im + n * stride_in)
    b = tl.load(Bias_ptr + n * stride_bn)
    y = 1.0 / (1.0 + tl.exp(-x))
    out = y + b
    tl.store(Out_ptr + m * stride_om + n * stride_on, out)


@triton.jit
def _group_top2_sum_kernel(
    Scores_ptr,      # *f32, [M, N]
    GroupScores_ptr, # *f32, [M, 8]
    M, N,
    stride_sm, stride_sn,   # strides for Scores
    stride_gm, stride_gn,   # strides for GroupScores (gn=8)
):
    t = tl.program_id(0)
    if t >= M:
        return
    n_group = 8
    ep_per_group = N // n_group  # 32
    for g in range(n_group):
        start = g * ep_per_group
        top1 = -float("inf")
        top2 = -float("inf")
        for i in range(ep_per_group):
            val = tl.load(Scores_ptr + t * stride_sm + (start + i) * stride_sn)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        group_sum = top1 + top2
        tl.store(GroupScores_ptr + t * stride_gm + g * stride_gn, group_sum)


@triton.jit
def _select_top4_groups_bubble_kernel(
    GroupScores_ptr,    # *f32, [M, 8]
    GroupIdx_ptr,       # *i32, [M, 4]
    M, n_group,         # n_group=8
    stride_gs_m, stride_gs_n,
    stride_gi_m, stride_gi_n,
):
    t = tl.program_id(0)
    if t >= M:
        return
    selected_idx = [0] * 4
    selected_scores = [0.0] * 4
    for k in range(4):
        best_score = -float("inf")
        best_idx = -1
        for g in range(n_group):
            score = tl.load(GroupScores_ptr + t * stride_gs_m + g * stride_gs_n)
            if score > best_score:
                best_score = score
                best_idx = g
        tl.store(GroupScores_ptr + t * stride_gs_m + best_idx * stride_gs_n, -float("inf"))
        selected_idx[k] = best_idx
        selected_scores[k] = best_score
    for k in range(4):
        tl.store(GroupIdx_ptr + t * stride_gi_m + k * stride_gi_n, selected_idx[k])


@triton.jit
def _mask_nonselected_groups_kernel(
    Scores_ptr,                 # *f32, [M, N]
    GroupIdx_ptr,               # *i32, [M, 4]
    MaskedScores_ptr,           # *f32, [M, N]
    M, N, n_group, ep_per_group, stride_sm, stride_sn,
    stride_gi_m, stride_gi_n,
    stride_mm, stride_mn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    for g in range(n_group):
        found = 0
        for k in range(4):
            gi = tl.load(GroupIdx_ptr + t * stride_gi_m + k * stride_gi_n)  # int32
            if gi == g:
                found = 1
                break
        if found == 0:
            start = g * ep_per_group
            for i in range(ep_per_group):
                ptr = Scores_ptr + t * stride_sm + (start + i) * stride_sn
                val = tl.load(ptr)
                tl.store(MaskedScores_ptr + t * stride_mm + (start + i) * stride_mn, -float("inf"))


@triton.jit
def _select_top8_masked_kernel(
    MaskedScores_ptr,  # *f32, [M, N]
    SelectedIdx_ptr,   # *i32, [M, 8]
    M, N, stride_mm, stride_mn,
):
    t = tl.program_id(0)
    if t >= M:
        return
    best = [0.0] * 8
    idx = [0] * 8
    for e in range(N):
        ptr = MaskedScores_ptr + t * stride_mm + e * stride_mn
        val = tl.load(ptr)
        for j in range(8):
            if val > best[j]:
                for k in range(7, j, -1):
                    best[k] = best[k - 1]
                    idx[k] = idx[k - 1]
                best[j] = val
                idx[j] = e
                break
    for j in range(8):
        tl.store(SelectedIdx_ptr + t * 8 + j, idx[j])


@triton.jit
def _normalize_and_scale_kernel(
    SelectedIdx_ptr,   # *i32, [M, 8]
    SelectedWgt_ptr,   # *f32, [M, 8]
    MaskedScores_ptr,  # *f32, [M, N]
    M, N, routed_scale, stride_mm, stride_mn, stride_mwi, stride_mwj
):
    t = tl.program_id(0)
    if t >= M:
        return
    total = 0.0
    for j in range(8):
        e = tl.load(SelectedIdx_ptr + t * 8 + j)
        ptr = MaskedScores_ptr + t * stride_mm + e * stride_mn
        score = tl.load(ptr)
        total += score
    total += 1e-20
    for j in range(8):
        e = tl.load(SelectedIdx_ptr + t * 8 + j)
        ptr = MaskedScores_ptr + t * stride_mm + e * stride_mn
        score = tl.load(ptr)
        norm = score / total
        tl.store(SelectedWgt_ptr + t * 8 + j, norm * routed_scale)


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
        TILE_N = 64
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
        stride_bn = expert_bias.stride(0)
        stride_om, stride_on = scores.stride()
        _sigmoid_add_bias_kernel[(M * N,)](
            logits, expert_bias, scores,
            M, N,
            stride_sm, stride_sn,
            stride_bn,
            stride_om, stride_on,
            num_warps=1, num_stages=1,
        )

        # 3) Group top-2 sum per token (


def run(*args):
    return ModelNew()(*args)
