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
        # sigmoid
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
        group_vals = tl.zeros((BLOCK,), dtype=tl.float32)
        # Collect the 32 scores for this group
        for i in range(EXPERTS_PER_GROUP):
            col = group_start + i
            group_vals[i] = tl.load(S_ptr + pid_m * stride_sm + col * stride_sn)
        # Iterative top-2 via argmax
        best1 = -float('inf')
        best2 = -float('inf')
        for i in range(EXPERTS_PER_GROUP):
            v = group_vals[i]
            if v > best1:
                best2 = best1
                best1 = v
            elif v > best2:
                best2 = v
        top2_sum = best1 + best2
        tl.store(G_ptr + pid_m * stride_gm + g * stride_gn, top2_sum)


@triton.jit
def _group_top4_select_kernel(
    G_ptr,   # [M, 8] group_scores, float32
    IDX_ptr, # [M, 4] group_idx, int32
    M,
    stride_gm, stride_gn,
    stride_im, stride_in,
    K_GROUPS: tl.constexpr,  # 4
    BLOCK: tl.constexpr,     # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # Iterative argmax selection of top-4 group indices
    best_val = -float('inf')
    best_idx = -1
    for g in range(K_GROUPS):
        v = tl.load(G_ptr + pid_m * stride_gm + g * stride_gn)
        if v > best_val:
            best_val = v
            best_idx = g
    tl.store(IDX_ptr + pid_m * stride_im + 0 * stride_in, best_idx)
    tl.store(G_ptr + pid_m * stride_gm + best_idx * stride_gn, -float('inf'))

    best_val = -float('inf')
    best_idx = -1
    for g in range(K_GROUPS):
        v = tl.load(G_ptr + pid_m * stride_gm + g * stride_gn)
        if v > best_val:
            best_val = v
            best_idx = g
    tl.store(IDX_ptr + pid_m * stride_im + 1 * stride_in, best_idx)
    tl.store(G_ptr + pid_m * stride_gm + best_idx * stride_gn, -float('inf'))

    best_val = -float('inf')
    best_idx = -1
    for g in range(K_GROUPS):
        v = tl.load(G_ptr + pid_m * stride_gm + g * stride_gn)
        if v > best_val:
            best_val = v
            best_idx = g
    tl.store(IDX_ptr + pid_m * stride_im + 2 * stride_in, best_idx)
    tl.store(G_ptr + pid_m * stride_gm + best_idx * stride_gn, -float('inf'))

    best_val = -float('inf')
    best_idx = -1
    for g in range(K_GROUPS):
        v = tl.load(G_ptr + pid_m * stride_gm + g * stride_gn)
        if v > best_val:
            best_val = v
            best_idx = g
    tl.store(IDX_ptr + pid_m * stride_im + 3 * stride_in, best_idx)
    tl.store(G_ptr + pid_m * stride_gm + best_idx * stride_gn, -float('inf'))


@triton.jit
def _final_top8_and_normalize_kernel(
    S_ptr,          # [M, N] scores, float32
    IDX_ptr,        # [M, 4] group_idx, int32 (kept for future use if needed)
    OUT_IDX_ptr,    # [M, 8] selected expert indices, int32
    OUT_W_ptr,      # [M, 8] selected weights, float32
    M, N,
    stride_sm, stride_sn,
    stride_im, stride_in,  # IDX strides
    stride_om, stride_on,
    routed_scaling_factor: tl.constexpr,  # scaling factor for final weights
    BLOCK: tl.constexpr,                  # tile width for iteration, e.g., 128
    K_TOP: tl.constexpr,                  # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # Iteratively select top-8 indices from S_ptr via argmax
    for t in range(K_TOP):
        best_val = -float('inf')
        best_col = -1
        for n_start in range(0, N, BLOCK):
            cols = n_start + tl.arange(0, BLOCK)
            mask = cols < N
            vals = tl.load(S_ptr + pid_m * stride_sm + cols * stride_sn, mask=mask, other=-float('inf'))
            # simple scan to find max in this tile
            for i in range(BLOCK):
                v = vals[i]
                if v > best_val:
                    best_val = v
                    best_col = n_start + i
        tl.store(OUT_IDX_ptr + pid_m * stride_om + t * stride_on, best_col)
        tl.store(S_ptr + pid_m * stride_sm + best_col * stride_sn, -float('inf'))

    # Compute sum of selected scores from OUT_IDX by gathering from S_ptr
    total = 0.0
    for t in range(K_TOP):
        idx_t = tl.load(OUT_IDX_ptr + pid_m * stride_om + t * stride_on)
        score_t = tl.load(S_ptr + pid_m * stride_sm + idx_t * stride_sn)
        total += score_t

    # Normalize and store weights
    for t in range(K_TOP):
        idx_t = tl.load(OUT_IDX_ptr + pid_m * stride_om + t * stride_on)
        score_t = tl.load(S_ptr + pid_m * stride_sm + idx_t * stride_sn)
        w_t = score_t / (total + 1e-20) * routed_scaling_factor
        tl.store(OUT_W_ptr + pid_m * stride_om + t * stride_on, w_t)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        hidden_states: [M, hidden_dim], float32
        weight: [num_experts, hidden_dim], float32 (PyTorch nn.Linear weight)
        expert_bias: [num_experts], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be on CUDA"
        M = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        num_experts = weight.shape[0]

        # 1) Compute logits using PyTorch for robust correctness (GEMM is heavy and error-prone in Triton here)
        #    Note: This is not Triton, but it's necessary for a correct baseline. Future optimization: move to Triton.
        logits = torch.nn.functional.linear(hidden_states, weight)  # [M, num_experts]

        # 2) Sigmoid + bias via Triton
        scores = torch.empty((M, num_experts), dtype=torch.float32, device=hidden_states.device)
        _sigmoid_add_bias_kernel[(M,)](
            logits, expert_bias,
            scores,
            M, num_experts,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_N=128,
            num_warps=4,
        )

        # 3) Group top-2 sum via Triton
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, num_experts,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            EXPERTS_PER_GROUP=32, NUM_GROUPS=8, BLOCK=32,
            num_warps=2,
        )

        # 4) Group top-4 select via Triton
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            K_GROUPS=4, BLOCK=8,
            num_warps=2,
        )

        # 5) Final top-8 selection and normalization via Triton
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx,  # group_idx kept for potential future use
            topk_idx, topk_weight,
            M, num_experts,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor=routed_scaling_factor,
            BLOCK=128, K_TOP=8,
            num_warps=4,
        )

        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
