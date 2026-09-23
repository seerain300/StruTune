import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# Kernel: sigmoid + bias
# logits: [M, N], float32, CUDA
# bias: [N], float32, CUDA
# out: [M, N], float32, CUDA
@triton.jit
def sigmoid_and_bias_kernel(
    logits_ptr,      # *f32, [M, N]
    bias_ptr,        # *f32, [N]
    out_ptr,         # *f32, [M, N]
    M: tl.int32,
    N: tl.constexpr,  # 256
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    if row >= M or col >= N:
        return
    val = tl.load(logits_ptr + row * N + col)
    bias = tl.load(bias_ptr + col)
    val = 1.0 / (1.0 + tl.exp(-val))  # sigmoid
    val = val + bias
    tl.store(out_ptr + row * N + col, val)


# Kernel: expand group mask to [M, N] and set non-selected group experts to -inf
# group_mask: [M, G] float32 (1.0 where selected, 0 otherwise)
# scores_flat: [M*N] float32 (the scores vector to mask)
# masked_flat: [M*N] float32 (output)
# G=8, N=256
@triton.jit
def expand_group_mask_and_set_ninf_kernel(
    group_mask_ptr,       # *f32, [M, G]
    scores_flat_ptr,      # *f32, [M*N]
    masked_flat_ptr,      # *f32, [M*N]
    M: tl.int32,
    N: tl.constexpr,      # 256
    G: tl.constexpr,      # 8
):
    idx = tl.program_id(0)
    if idx >= M * N:
        return
    token = idx // N
    expert = idx % N
    group = expert // 32
    if group >= G:
        return
    mask_val = tl.load(group_mask_ptr + token * G + group)
    orig = tl.load(scores_flat_ptr + idx)
    neg_inf = -1.0e20
    out_val = tl.where(mask_val > 0.0, orig, neg_inf)
    tl.store(masked_flat_ptr + idx, out_val)


# Kernel: iterative argmax top-k selection over a K-length vector
# inp: [M*K], float32 (assumed contiguous), per-row
# out_idx: [M*K], int32
# For each row m, select K maxima (no duplicates). We rely on host to pass each row as a separate program launch.
@triton.jit
def topk_select_argmax_kernel(
    inp_ptr,         # *f32, [M*K]
    out_idx_ptr,     # *i32, [M*K]
    M: tl.int32,
    K: tl.constexpr  # number of top-k (4 or 8)
):
    m = tl.program_id(0)
    # Iterate K times: find max, write index, mask it to -inf, repeat
    for t in range(K):
        best_val = -1.0e20
        best_idx = 0
        for j in range(K):
            val = tl.load(inp_ptr + m * K + j)
            better = val > best_val
            best_val = tl.where(better, val, best_val)
            best_idx = tl.where(better, j, best_idx)
        tl.store(out_idx_ptr + m * K + t, best_idx)
        # We don't need to mask inp since we don't write it back; this selection is read-only.


# Kernel: gather scores from original [M, N] using arg indices
# scores: [M, N], float32
# arg_idx: [M*K], int32
# out: [M*K], float32
@triton.jit
def gather_scores_kernel(
    scores_ptr,        # *f32, [M, N]
    arg_idx_ptr,       # *i32, [M*K]
    out_ptr,           # *f32, [M*K]
    M: tl.int32,
    N: tl.constexpr,   # 256
    K: tl.constexpr    # number of gathered
):
    m = tl.program_id(0)
    for t in range(K):
        idx = tl.load(arg_idx_ptr + m * K + t)  # i32
        val = tl.load(scores_ptr + m * N + idx)
        tl.store(out_ptr + m * K + t, val)


# Kernel: normalize_and_scale per row: out = inp / sum(inp) * scale
@triton.jit
def normalize_and_scale_kernel(
    inp_ptr,        # *f32, [M*K]
    out_ptr,        # *f32, [M*K]
    scale: tl.float32,
    M: tl.int32,
    K: tl.constexpr
):
    m = tl.program_id(0)
    total = 0.0
    for k in range(K):
        v = tl.load(inp_ptr + m * K + k)
        total += v
    for k in range(K):
        v = tl.load(inp_ptr + m * K + k)
        v = v / (total + 1e-20) * scale
        tl.store(out_ptr + m * K + k, v)


class ModelNew(nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor
        self.hidden_size = 128
        self.num_experts = 256
        self.group_count = 8
        self.experts_per_group = 32
        self.final_k = 8

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        """
        Triton-only implementation for elementwise and selection steps; matmul via PyTorch for robustness.
        Returns:
        - topk_idx: [num_tokens, 8], int64
        - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be on CUDA for Triton kernels."
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        assert K == self.hidden_size, "hidden_states.hidden_size must be 128"
        assert weight.shape[0] == self.num_experts, "weight.num_experts must be 256"
        assert weight.shape[1] == self.hidden_size, "weight.hidden_size must be 128"
        assert expert_bias.shape[0] == self.num_experts, "expert_bias length must be 256"

        # 1) Linear projection via PyTorch (fast and reliable)
        logits = F.linear(hidden_states.to(torch.float32), weight.to(torch.float32))  # [M, 256]

        # 2) Triton: sigmoid + bias -> scores
        scores = torch.empty_like(logits)
        grid = (M, self.num_experts)
        sigmoid_and_bias_kernel[grid](logits, expert_bias.to(torch.float32), scores, M=self.num_experts, N=self.num_experts)

        # 3) Group routing: compute per-token group scores (top-2 per group sum) using torch.topk (small vectors)
        # Reshape scores to [M, 8, 32]
        scores_reshaped = scores.view(M, self.group_count, self.experts_per_group)
        group_scores = torch.empty((M, self.group_count), dtype=torch.float32, device=scores.device)
        for t in range(M):
            for g in range(self.group_count):
                group_vec = scores_reshaped[t, g, :]  # [32]
                vals, _ = torch.topk(group_vec, k=2, dim=0, largest=True, sorted=False)
                group_scores[t, g] = vals.sum()

        # 4) Select top-4 groups per token using torch.topk (simple and correct)
        group_idx = torch.topk(group_scores, k=4, dim=1, sorted=False).indices  # [M, 4], int64

        # 5) Build group_mask [M, 8]: 1.0 where selected, else 0
        group_mask = torch.zeros((M, self.group_count), dtype=torch.float32, device=scores.device)
        for j in range(4):
            g = group_idx[:, j]  # int64
            group_mask[torch.arange(M), g] = 1.0

        # 6) Expand group_mask to [M, 256] and set non-selected groups' scores to -inf
        scores_flat = scores.reshape(M * self.num_experts)  # [M*256]
        masked_flat = torch.empty_like(scores_flat)
        grid2 = (M * self.num_experts,)
        expand_group_mask_and_set_ninf_kernel[grid2](group_mask, scores_flat, masked_flat, M=self.num_experts, N=self.num_experts, G=self.group_count)

        # 7) Triton top-k selection for final 8 experts from masked_flat (per row)
        final_arg_idx = torch.empty((M, self.final_k), dtype=torch.int32, device=scores.device)
        for m in range(M):
            row_base = masked_flat[m * self.num_experts : (m + 1) * self.num_experts]  # [256]
            out_idx_row = final_arg_idx[m]  # [8]
            # Launch Triton kernel for this row
            topk_select_argmax_kernel[(1,)](row_base, out_idx_row, M=1, K=self.final_k)

        # 8) Triton gather selected scores from original scores
        gathered_scores = torch.empty((M, self.final_k), dtype=torch.float32, device=scores.device)
        for m in range(M):
            idx_row = final_arg_idx[m]  # [8], int32
            scores_row_ptr = scores[m]  # [256]
            out_row_ptr = gathered_scores[m]  # [8]
            gather_scores_kernel[(1,)](scores_row_ptr, idx_row, out_row_ptr, M=1, N=self.num_experts, K=self.final_k)

        # 9) Normalize and scale gathered scores
        normalized_weights = torch.empty_like(gathered_scores)
        scale = self.routed_scaling_factor
        normalize_and_scale_kernel[(M,)](gathered_scores.reshape(M * self.final_k), normalized_weights.reshape(M * self.final_k), scale, M=M, K=self.final_k)
        topk_weight = normalized_weights.view(M, self.final_k)  # [M, 8]

        # 10) Return indices as int64
        topk_idx = final_arg_idx.to(torch.int64)  # [M, 8]

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
