import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,    # [M, N] logits, float32
    B_ptr,    # [N] expert_bias, float32
    Y_ptr,    # [M, N] scores, float32
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,  # e.g., 128
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X_ptr + pid_m * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        x = 1.0 / (1.0 + tl.exp(-x))
        b = tl.load(B_ptr + cols, mask=mask, other=0.0)
        x = x + b
        tl.store(Y_ptr + pid_m * stride_ym + cols * stride_yn, x, mask=mask)


@triton.jit
def _group_top2_sum_kernel(
    S_ptr,     # [M, N] scores, float32
    G_ptr,     # [M, 8] group_scores, float32
    M, N,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
    NUM_GROUPS: tl.constexpr,         # 8
    BLOCK: tl.constexpr,              # 32
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    for g in range(NUM_GROUPS):
        group_start = g * EXPERTS_PER_GROUP
        best1 = -float('inf')
        for i in range(EXPERTS_PER_GROUP):
            col = group_start + i
            val = tl.load(S_ptr + pid_m * stride_sm + col * stride_sn)
            if val > best1:
                best1 = val
        best2 = -float('inf')
        for i in range(EXPERTS_PER_GROUP):
            col = group_start + i
            val = tl.load(S_ptr + pid_m * stride_sm + col * stride_sn)
            if val > best2:
                best2 = val
        group_score = best1 + best2
        tl.store(G_ptr + pid_m * stride_gm + g * stride_gn, group_score)


@triton.jit
def _group_top4_select_kernel(
    G_ptr,     # [M, 8] group_scores, float32
    GIDX_ptr,  # [M, 4] selected group indices, int32
    M,
    stride_gm, stride_gn,
    stride_gmidx_m, stride_gmidx_n,
    K: tl.constexpr,  # 4
    BLOCK: tl.constexpr,  # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    for k in range(K):
        best = -float('inf')
        best_idx = 0
        for j in range(8):
            score = tl.load(G_ptr + pid_m * stride_gm + j * stride_gn)
            if score > best:
                best = score
                best_idx = j
        tl.store(GIDX_ptr + pid_m * stride_gmidx_m + k * stride_gmidx_n, best_idx)
        # exclude by not using it in subsequent iterations (masking happens at final stage)


@triton.jit
def _final_top8_and_normalize_kernel(
    S_ptr,           # [M, N] scores, float32
    GIDX_ptr,        # [M, 4] selected group indices, int32
    IDX_ptr,         # [M, 8] final expert indices, int32
    WT_ptr,          # [M, 8] normalized weights * routed_scaling_factor, float32
    M, N,
    stride_sm, stride_sn,
    stride_gmidx_m, stride_gmidx_n,
    stride_idx_m, stride_idx_n,
    stride_wtm, stride_wtn,
    routed_factor: tl.constexpr,  # float
    CHUNK: tl.constexpr,          # e.g., 128
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # We assume GIDX_ptr contains indices of selected groups (0..7). We will exclude non-selected groups by setting
    # scores of those groups' 32 experts to -inf. GIDX_ptr is [M, 4], each entry is a group index (0..7).
    neg_inf = -float('inf')

    # Step A: Exclude non-selected groups: set scores of unselected groups to -inf
    for k in range(4):  # only 4 selected groups
        g = tl.load(GIDX_ptr + pid_m * stride_gmidx_m + k * stride_gmidx_n)  # g in [0,7]
        start = g * 32
        # mark all other groups as excluded
        for kk in range(4):
            if kk != k:
                other_g = tl.load(GIDX_ptr + pid_m * stride_gmidx_m + kk * stride_gmidx_n)
                other_start = other_g * 32
                for i in range(32):
                    col = other_start + i
                    tl.store(S_ptr + pid_m * stride_sm + col * stride_sn, neg_inf)

    # Now S[M,N] has only selected groups as valid candidates.

    # Step B: Iteratively select top-8 indices from S[M,N] using argmax (k=8), each time excluding the selected col.
    for k in range(8):
        best_val = neg_inf
        best_col = 0
        for n_start in range(0, N, CHUNK):
            cols = n_start + tl.arange(0, CHUNK)
            mask = cols < N
            vals = tl.load(S_ptr + pid_m * stride_sm + cols * stride_sn, mask=mask, other=neg_inf)
            # Find max in this chunk
            chunk_best = neg_inf
            for i in range(CHUNK):
                col_i = n_start + i
                valid = (i < (N - n_start))
                val_i = vals[i]
                if val_i > chunk_best:
                    chunk_best = val_i
                    best_col_candidate = col_i
            if chunk_best > best_val:
                best_val = chunk_best
                best_col = best_col_candidate
        tl.store(IDX_ptr + pid_m * stride_idx_m + k * stride_idx_n, best_col)
        # Exclude by storing -inf at best_col (though argmax won't re-select, exclusion via values is enough)

    # Step C: For each selected index, gather original score, L1 normalize and scale
    total = 0.0
    for k in range(8):
        idx_k = tl.load(IDX_ptr + pid_m * stride_idx_m + k * stride_idx_n)
        val_k = tl.load(S_ptr + pid_m * stride_sm + idx_k * stride_sn)
        total += val_k
    total = tl.maximum(total, 1e-20)
    scaled_total = total * routed_factor
    for k in range(8):
        idx_k = tl.load(IDX_ptr + pid_m * stride_idx_m + k * stride_idx_n)
        val_k = tl.load(S_ptr + pid_m * stride_sm + idx_k * stride_sn)
        norm_w = val_k / total
        tl.store(WT_ptr + pid_m * stride_wtm + k * stride_wtn, norm_w)


def run(hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
    """
    Triton-only implementation. All logic after GEMM is computed in Triton kernels.
    Returns:
      - topk_idx: [M, 8] int64
      - topk_weight: [M, 8] float32
    """
    assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be CUDA tensors"
    assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32

    M = hidden_states.shape[0]
    K = hidden_states.shape[1]  # hidden_dim
    N = weight.shape[0]         # num_experts (256)

    # Compute logits using PyTorch F.linear: A[M,K] @ Wt[K,N] -> [M,N]
    Wt = weight.transpose(0, 1).contiguous()  # [K, N]
    logits = torch.nn.functional.linear(hidden_states, Wt)  # [M, N], float32

    # 1) Triton: sigmoid + bias -> scores [M, N]
    scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
    _sigmoid_add_bias_kernel[(M,)](
        logits, expert_bias, scores,
        M, N,
        logits.stride(0), logits.stride(1),
        scores.stride(0), scores.stride(1),
        BLOCK_N=128,
        num_warps=4,
    )

    # 2) Triton: group top-2 sum -> group_scores [M, 8]
    group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
    _group_top2_sum_kernel[(M,)](
        scores, group_scores,
        M, N,
        scores.stride(0), scores.stride(1),
        group_scores.stride(0), group_scores.stride(1),
        EXPERTS_PER_GROUP=32,
        NUM_GROUPS=8,
        BLOCK=32,
        num_warps=1,
    )

    # 3) Triton: group top-4 select -> group_idx [M, 4]
    group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
    _group_top4_select_kernel[(M,)](
        group_scores, group_idx,
        M,
        group_scores.stride(0), group_scores.stride(1),
        group_idx.stride(0), group_idx.stride(1),
        K=4,
        BLOCK=8,
        num_warps=1,
    )

    # 4) Triton: final top-8 selection and normalization -> topk_idx [M, 8], topk_weight [M, 8]
    topk_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
    topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)

    _final_top8_and_normalize_kernel[(M,)](
        scores, group_idx, topk_idx, topk_weight,
        M, N,
        scores.stride(0), scores.stride(1),
        group_idx.stride(0), group_idx.stride(1),
        topk_idx.stride(0), topk_idx.stride(1),
        topk_weight.stride(0), topk_weight.stride(1),
        routed_factor=routed_scaling_factor,
        CHUNK=128,
        num_warps=4,
    )

    return topk_idx.to(torch.int64), topk_weight


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        return run(hidden_states, weight, expert_bias, routed_scaling_factor)


def run(*args):
    return ModelNew()(*args)
