import torch
import torch.nn as nn
import triton
import triton.language as tl


# Kernel 1: logits = hidden_states @ weight^T + expert_bias
# A: [M, K] row-major, W: [N, K] row-major, bias: [N], OUT: [M, N]
@triton.jit
def linear_bias_kernel(A_ptr, W_ptr, BIAS_ptr, OUT_ptr, M, N, K):
    i = tl.program_id(0)
    if i >= M:
        return
    acc = tl.zeros((N,), dtype=tl.float32)
    for k0 in range(0, K, 32):
        k_ids = k0 + tl.arange(0, 32)
        a = tl.load(A_ptr + i * K + k_ids, mask=k_ids < K, other=0.0)  # [32]
        w_ptrs = W_ptr + tl.arange(0, N)[:, None] * K + k_ids[None, :]  # [N, 32]
        w = tl.load(w_ptrs, mask=k_ids[None, :] < K, other=0.0)         # [N, 32]
        acc += tl.sum(a[None, :] * w, axis=1)                           # [N]
    bias = tl.load(BIAS_ptr + tl.arange(0, N))
    acc += bias
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


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure on CUDA
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be on CUDA for Triton."
        device = hidden_states.device
        dtype = torch.float32

        # Prepare inputs
        A = hidden_states.to(dtype)           # [M, K]
        W = weight.to(dtype)                  # [N, K]
        bias = expert_bias.to(dtype)          # [N]

        M, K = A.shape
        N = W.shape[0]

        # Allocate logits [M, N]
        logits = torch.empty((M, N), dtype=dtype, device=device)

        # Launch Triton kernel 1: GEMM + bias
        grid1 = (M,)
        linear_bias_kernel[grid1](
            A, W, bias, logits,
            M, N, K,
            num_warps=4
        )

        # Kernel 2: scores = sigmoid(logits) + bias
        scores = torch.empty((M, N), dtype=dtype, device=device)
        grid2 = (M,)
        sigmoid_bias_kernel[grid2](
            logits, bias, scores,
            M, N,
            num_warps=4
        )

        # Compute group scores: sum of top-2 per group
        num_experts = N
        n_group = 8
        experts_per_group = num_experts // n_group  # 32

        group_scores_reshaped = scores.view(M, n_group, experts_per_group)       # [M, 8, 32]
        top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)  # [M, 8, 2]
        group_scores = top2_vals.sum(dim=-1)                                      # [M, 8]

        # Select top-4 groups per token (sorted=False)
        _, group_idx = torch.topk(group_scores, k=4, dim=-1, sorted=False)       # [M, 4]

        # Build group mask [M, 8] -> 1.0 where selected, else 0.0
        group_mask = torch.zeros((M, n_group), dtype=torch.float32, device=device)
        group_mask.scatter_(1, group_idx.to(torch.long), 1.0)                    # [M, 8]

        # Expand to expert level [M, N]
        score_mask = group_mask.unsqueeze(-1).expand(M, n_group, experts_per_group).reshape(M, num_experts)

        # Mask out non-selected groups: set to -inf
        neg_inf = torch.finfo(dtype).min
        masked_scores = scores.masked_fill(score_mask == 0, neg_inf)             # [M, N]

        # Select top-8 experts from masked scores (sorted=False)
        _, topk_idx = torch.topk(masked_scores, k=8, dim=-1, sorted=False)       # [M, 8] (int64 by default)

        # Gather selected expert scores from original logits (for normalization)
        selected_scores = torch.gather(logits, dim=1, index=topk_idx.to(torch.int64))  # [M, 8]

        # Normalize weights and apply scaling factor
        numerator = selected_scores.sum(dim=-1, keepdim=True)                     # [M, 1]
        denominator = numerator + 1e-20
        topk_weight = (numerator / denominator) * routed_scaling_factor           # [M, 8] float32

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
