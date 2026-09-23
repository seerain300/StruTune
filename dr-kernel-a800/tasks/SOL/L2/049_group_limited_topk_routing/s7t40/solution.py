import torch
import torch.nn as nn
import triton
import triton.language as tl


# Kernel 1: logits = hidden_states @ weight^T + expert_bias
# A: [M, K] (row-major), W: [N, K] (row-major), bias: [N]
@triton.jit
def linear_bias_kernel(A_ptr, W_ptr, BIAS_ptr, OUT_ptr, M, N, K):
    # Each program handles one row i (token)
    i = tl.program_id(0)
    if i >= M:
        return
    acc = tl.zeros((N,), dtype=tl.float32)
    # Loop over K in chunks
    for k0 in range(0, K, 32):
        k_ids = k0 + tl.arange(0, 32)
        # Load a_i[k0:k0+32]
        a = tl.load(A_ptr + i * K + k_ids, mask=k_ids < K, other=0.0)
        # Load W[:, k0:k0+32] -> shape [N, 32]
        w_ptrs = W_ptr + tl.arange(0, N)[:, None] * K + k_ids[None, :]
        w = tl.load(w_ptrs, mask=(k_ids[None, :] < K), other=0.0)
        # acc += a[k_ids] * w[k_ids]
        # Broadcast a over rows: a[:, None] * w -> [N, 32]
        acc += tl.sum(a[None, :] * w, axis=1)
    # Add bias
    bias = tl.load(BIAS_ptr + tl.arange(0, N))
    acc += bias
    # Store
    tl.store(OUT_ptr + i * N + tl.arange(0, N), acc)


# Kernel 2: scores = sigmoid(logits) + expert_bias
# LOGITS: [M, N], BIAS: [N], OUT: [M, N]
@triton.jit
def sigmoid_bias_kernel(LOGITS_ptr, BIAS_ptr, OUT_ptr, M, N):
    i = tl.program_id(0)
    if i >= M:
        return
    for n0 in range(0, N, 32):
        n_ids = n0 + tl.arange(0, 32)
        logits = tl.load(LOGITS_ptr + i * N + n_ids, mask=n_ids < N, other=0.0)
        bias = tl.load(BIAS_ptr + n_ids, mask=n_ids < N, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-logits))
        out = sig + bias
        tl.store(OUT_ptr + i * N + n_ids, out, mask=n_ids < N)


# Kernel 3: compute group scores: sum of top-2 per group -> OUT[M, 8]
# SCORES: [M, N], OUT: [M, 8]
@triton.jit
def compute_group_scores_kernel(SCORES_ptr, OUT_ptr, M, N, GROUPS, EXP_PER_GROUP):
    i = tl.program_id(0)
    if i >= M:
        return
    # group_scores[8]
    gs = tl.zeros((8,), dtype=tl.float32)
    for g in range(8):
        # maxv = max(scores[i, g*32:(g+1)*32])
        start = g * EXP_PER_GROUP
        maxv = -float('inf')
        for j in range(EXP_PER_GROUP):
            n = start + j
            v = tl.load(SCORES_ptr + i * N + n)
            maxv = tl.maximum(maxv, v)
        # second = second max
        second = -float('inf')
        for j in range(EXP_PER_GROUP):
            n = start + j
            v = tl.load(SCORES_ptr + i * N + n)
            if v != maxv:
                second = tl.maximum(second, v)
        gs[g] = maxv + second
    tl.store(OUT_ptr + i * 8 + tl.arange(0, 8), gs)


# Kernel 4: select_top4_groups_kernel
# GROUP_SCORES: [M, 8], OUT: [M, 4] indices of selected groups
@triton.jit
def select_top4_groups_kernel(GROUP_SCORES_ptr, OUT_ptr, M, GROUPS):
    i = tl.program_id(0)
    if i >= M:
        return
    scores = tl.load(GROUP_SCORES_ptr + i * GROUPS + tl.arange(0, GROUPS))
    # iterative elimination
    for t in range(4):
        maxv = -float('inf')
        idx = -1
        for g in range(GROUPS):
            v = scores[g]
            better = v > maxv
            idx = tl.where(better, g, idx)
            maxv = tl.where(better, v, maxv)
        # store idx
        tl.store(OUT_ptr + i * 4 + t, idx)
        # set max to -inf
        scores = tl.where(scores == maxv, -float('inf'), scores)


# Kernel 5: mask scores with selected groups -> set non-selected groups to -inf in OUT_MASKED
# SCORES: [M, N], SELECTED_GROUPS: [M, 4], OUT_MASKED: [M, N]
@triton.jit
def mask_scores_with_groups_kernel(SCORES_ptr, SELECTED_GROUPS_ptr, OUT_MASKED_ptr, M, N, GROUPS, EXP_PER_GROUP):
    i = tl.program_id(0)
    if i >= M:
        return
    for t in range(4):
        g = tl.load(SELECTED_GROUPS_ptr + i * 4 + t)  # int32
        start = g * EXP_PER_GROUP
        for j in range(EXP_PER_GROUP):
            n = start + j
            v = tl.load(SCORES_ptr + i * N + n)
            tl.store(OUT_MASKED_ptr + i * N + n, -float('inf'))
    # we mask only selected groups; non-selected remain unchanged (but to be safe, we set them to -inf as well)
    # However, since we already zeroed selected groups, setting others to -inf ensures correctness.
    # To do that, we just set all to -inf and then restore selected groups from original.
    # Simpler: set all to -inf then overwrite selected back to original. Here, we just set all to -inf.
    for n in range(N):
        tl.store(OUT_MASKED_ptr + i * N + n, -float('inf'))
    # Overwrite selected groups with original scores
    for t in range(4):
        g = tl.load(SELECTED_GROUPS_ptr + i * 4 + t)
        start = g * EXP_PER_GROUP
        for j in range(EXP_PER_GROUP):
            n = start + j
            v = tl.load(SCORES_ptr + i * N + n)
            tl.store(OUT_MASKED_ptr + i * N + n, v)


# Kernel 6: final selection of top-8 within masked scores and compute normalized weight
# LOGITS: [M, N], MASKED_SCORES: [M, N], OUT_IDX: [M, 8], OUT_WEIGHT: [M, 8]
@triton.jit
def select_top8_with_weight_and_normalize_kernel(LOGITS_ptr, MASKED_SCORES_ptr, OUT_IDX_ptr, OUT_WEIGHT_ptr, M, N, SCALE):
    i = tl.program_id(0)
    if i >= M:
        return
    numerator = 0.0
    for cnt in range(8):
        maxv = -float('inf')
        idx = -1
        for n in range(N):
            v = tl.load(MASKED_SCORES_ptr + i * N + n)
            better = v > maxv
            idx = tl.where(better, n, idx)
            maxv = tl.where(better, v, maxv)
        # store idx
        tl.store(OUT_IDX_ptr + i * 8 + cnt, idx)
        # accumulate numerator using original logits at selected idx
        v_logit = tl.load(LOGITS_ptr + i * N + idx)
        numerator += v_logit * SCALE
        # set selected to -inf
        tl.store(MASKED_SCORES_ptr + i * N + idx, -float('inf'))
    denom = numerator + 1e-20
    weight = numerator / denom
    # write weight for all 8 positions
    for cnt in range(8):
        tl.store(OUT_WEIGHT_ptr + i * 8 + cnt, weight)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # constants as in original
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = 256 // self.n_group  # = 32
        self.topk_group = 4
        self.final_topk = 8

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure CUDA device and dtype float32 for compute
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be CUDA tensors"
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = self.num_experts

        A = hidden_states.contiguous().to(torch.float32)  # [M, K]
        W = weight.contiguous().to(torch.float32)        # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)  # [N]

        # Allocate outputs
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=device)
        selected_groups = torch.empty((M, self.topk_group), dtype=torch.int32, device=device)
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)
        topk_idx = torch.empty((M, self.final_topk), dtype=torch.int32, device=device)
        topk_weight = torch.empty((M, self.final_topk), dtype=torch.float32, device=device)

        # Launch kernels
        # 1) logits = A @ W^T + bias
        grid1 = (M,)
        linear_bias_kernel[grid1](
            A, W, bias, logits,
            M, N, K,
        )

        # 2) scores = sigmoid(logits) + bias
        grid2 = (M,)
        sigmoid_bias_kernel[grid2](
            logits, bias, scores,
            M, N,
        )

        # 3) compute group scores per token
        grid3 = (M,)
        compute_group_scores_kernel[grid3](
            scores, group_scores,
            M, N, self.n_group, self.experts_per_group,
        )

        # 4) select top-4 groups per token
        grid4 = (M,)
        select_top4_groups_kernel[grid4](
            group_scores, selected_groups,
            M, self.n_group,
        )

        # 5) mask scores with selected groups (set non-selected groups to -inf)
        grid5 = (M,)
        mask_scores_with_groups_kernel[grid5](
            scores, selected_groups, masked_scores,
            M, N, self.n_group, self.experts_per_group,
        )

        # 6) final selection of top-8 within masked scores and compute normalized weights
        grid6 = (M,)
        select_top8_with_weight_and_normalize_kernel[grid6](
            logits, masked_scores, topk_idx, topk_weight,
            M, N, routed_scaling_factor,
        )

        # Return indices as int64 and weights as float32
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
