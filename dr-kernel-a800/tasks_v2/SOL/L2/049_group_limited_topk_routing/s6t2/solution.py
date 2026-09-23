import torch
import torch.nn as nn

import triton
import triton.language as tl

# Kernel 1: logits = hidden_states @ weight.T
# hidden_states: [M, K] where K=hidden_size=128
# weight: [N, K] where N=num_experts=256, row-major
# output logits: [M, N]
@triton.jit
def matmul_logits_kernel(
    a_ptr,        # *f32, [M, K]
    b_ptr,        # *f32, [N, K] (row-major)
    out_ptr,      # *f32, [M, N]
    M: tl.int32,
    N: tl.constexpr,             # 256
    K: tl.constexpr,             # 128
):
    row = tl.program_id(0)  # program id over tokens
    col = tl.program_id(1)  # program id over experts
    acc = 0.0
    for k in range(K):
        a_val = tl.load(a_ptr + row * K + k)
        b_val = tl.load(b_ptr + col * K + k)
        acc += a_val * b_val
    tl.store(out_ptr + row * N + col, acc)


# Kernel 2: scores = sigmoid(logits) + expert_bias
# logits: [M, N], bias: [N], scores: [M, N]
@triton.jit
def sigmoid_bias_kernel(
    logits_ptr,        # *f32, [M, N]
    bias_ptr,          # *f32, [N]
    scores_ptr,        # *f32, [M, N]
    M: tl.int32,
    N: tl.constexpr,             # 256
):
    row = tl.program_id(0)  # one program per row (token)
    cols = tl.arange(0, N)   # N=256
    mask = cols < N
    logits = tl.load(logits_ptr + row * N + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    scores = 1.0 / (1.0 + tl.exp(-logits)) + bias
    tl.store(scores_ptr + row * N + cols, scores, mask=mask)


# Kernel 3: for each token, compute sum of top-2 scores within each group (32 experts) -> [M, 8]
# scores are expected in shape [M, group_count, experts_per_group] with last dim=32 (contiguous).
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,        # *f32, [M, 8, 32]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    group_count: tl.constexpr,      # 8
    experts_per_group: tl.constexpr,  # 32
):
    row = tl.program_id(0)
    for g in range(group_count):
        group_start = g * experts_per_group
        idx = tl.arange(0, 32)
        cols = group_start + idx
        mask = idx < experts_per_group
        vals = tl.load(scores_ptr + row * (group_count * experts_per_group) + g * experts_per_group + idx, mask=mask, other=-float('inf'))
        m1 = tl.max(vals, axis=0)
        vals2 = tl.where(vals == m1, -float('inf'), vals)
        m2 = tl.max(vals2, axis=0)
        tl.store(group_scores_ptr + row * group_count + g, m1 + m2)


# Kernel 4: iterative top-k selection (K is constexpr), returns indices in out_idx [M, K]
# inp_ptr: [M, M_dim] where M_dim can be N (for groups) or number of experts (256).
@triton.jit
def topk_select_kernel(
    inp_ptr,       # *f32, [M, M_dim]
    out_idx_ptr,   # *i32, [M, K]
    M: tl.int32,
    M_dim: tl.int32,
    K: tl.constexpr,             # e.g., 4 or 8
):
    row = tl.program_id(0)
    # Iterative argmax selection
    for k in range(K):
        best_val = -float('inf')
        best_idx = 0
        for j in range(M_dim):
            val = tl.load(inp_ptr + row * M_dim + j)
            better = val > best_val
            best_val = tl.where(better, val, best_val)
            best_idx = tl.where(better, j, best_idx)
        tl.store(out_idx_ptr + row * K + k, best_idx)
        # set selected position to -inf for next iterations
        tl.store(inp_ptr + row * M_dim + best_idx, -float('inf'))


# Kernel 5: build group_mask [M, group_count] with 1 for selected groups
# group_idx: [M, K], where K is 4 in our case. We scatter 1 into group_mask[:, g] for each g in 0..group_count-1.
@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,     # *i32, [M, K]
    group_mask_ptr,    # *i32, [M, group_count], we'll store 0/1
    M: tl.int32,
    K: tl.constexpr,            # 4
    group_count: tl.constexpr,  # 8
):
    row = tl.program_id(0)
    for g in range(group_count):
        # count how many of the K indices equal g
        count = 0
        for k in range(K):
            idx = tl.load(group_idx_ptr + row * K + k)
            count += (idx == g).to(tl.int32)
        # If count == 1, then set group_mask[row, g] = 1
        if count == 1:
            tl.store(group_mask_ptr + row * group_count + g, 1)
        else:
            tl.store(group_mask_ptr + row * group_count + g, 0)


# Kernel 6: expand group_mask to scores and set non-selected groups' scores to -inf
# group_mask: [M, group_count], scores: [M, N], out_scores: [M, N]
@triton.jit
def expand_group_mask_and_set_ninf_kernel(
    group_mask_ptr,    # *i32, [M, group_count]
    scores_ptr,        # *f32, [M, N]
    out_ptr,           # *f32, [M, N]
    M: tl.int32,
    N: tl.constexpr,              # 256
    group_count: tl.constexpr,    # 8
    experts_per_group: tl.constexpr,  # 32
):
    row = tl.program_id(0)
    # Iterate over all experts and set to -inf if their group is not selected
    for col in range(N):
        group_index = col // experts_per_group  # 0..7
        selected = tl.load(group_mask_ptr + row * group_count + group_index)  # int32 0/1
        orig = tl.load(scores_ptr + row * N + col)
        val = tl.where(selected == 1, orig, -float('inf'))
        tl.store(out_ptr + row * N + col, val)


# Kernel 7: gather selected scores from original scores using expert indices (K=8)
@triton.jit
def gather_scores_kernel(
    scores_ptr,       # *f32, [M, N]
    idx_ptr,          # *i32, [M, K]
    out_scores_ptr,   # *f32, [M, K]
    M: tl.int32,
    N: tl.constexpr,              # 256
    K: tl.constexpr               # 8
):
    row = tl.program_id(0)
    for k in range(K):
        idx = tl.load(idx_ptr + row * K + k)
        val = tl.load(scores_ptr + row * N + idx)
        tl.store(out_scores_ptr + row * K + k, val)


# Kernel 8: normalize_and_scale for [M, K] vector, out is scaled normalized values
@triton.jit
def normalize_and_scale_kernel(
    inp_ptr,        # *f32, [M, K]
    out_ptr,        # *f32, [M, K]
    scale: tl.float32,
    M: tl.int32,
    K: tl.constexpr
):
    row = tl.program_id(0)
    total = 0.0
    for k in range(K):
        val = tl.load(inp_ptr + row * K + k)
        total += val
    for k in range(K):
        val = tl.load(inp_ptr + row * K + k)
        val = val / (total + 1e-20) * scale
        tl.store(out_ptr + row * K + k, val)


class ModelNew(nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor
        # fixed sizes per original: hidden_size=128, num_experts=256, group_count=8, group_k=4, final_k=8
        self.hidden_size = 128
        self.num_experts = 256
        self.group_count = 8
        self.experts_per_group = 32
        self.final_k = 8

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        """
        Triton-only implementation.
        Returns:
        - topk_idx: [num_tokens, 8], int64
        - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be on CUDA for Triton kernels."
        M = hidden_states.shape[0]
        assert weight.shape[0] == self.num_experts, "weight.num_experts must be 256"
        assert weight.shape[1] == self.hidden_size, "weight.hidden_size must be 128"
        assert hidden_states.shape[1] == self.hidden_size, "hidden_states.hidden_size must be 128"
        assert expert_bias.shape[0] == self.num_experts, "expert_bias length must be 256"

        # 1) Triton matmul: logits = hidden_states @ weight.T -> [M, 256]
        logits = torch.empty((M, self.num_experts), dtype=torch.float32, device=hidden_states.device)
        grid = (M, self.num_experts)
        matmul_logits_kernel[grid](
            hidden_states.to(torch.float32), weight.to(torch.float32), logits,
            M=M,
            N=self.num_experts, K=self.hidden_size
        )

        # 2) Triton sigmoid + bias: scores = sigmoid(logits) + expert_bias -> [M, 256]
        scores = torch.empty_like(logits)
        grid2 = (M,)
        sigmoid_bias_kernel[grid2](logits, expert_bias.to(torch.float32), scores, M=M, N=self.num_experts)

        # 3) Reshape scores into [M, 8, 32] and compute group_scores (sum of top-2 per group) -> [M, 8]
        # We need to form the reshaped view and then run the Triton kernel.
        group_scores = torch.empty((M, self.group_count), dtype=torch.float32, device=scores.device)
        # Create a temporary tensor with the required layout [M, 8, 32]
        scores_reshaped = scores.view(M, self.group_count, self.experts_per_group)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](scores_reshaped, group_scores, M=M, group_count=self.group_count, experts_per_group=self.experts_per_group)

        # 4) Triton top-4 group selection: group_idx [M, 4]
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=scores.device)
        # M_dim is 8 (group_scores has 8 groups)
        M_dim = self.group_count
        grid4 = (M,)
        topk_select_kernel[grid4](group_scores, group_idx, M=M, M_dim=M_dim, K=4)

        # 5) Build group_mask [M, 8], 1 where group selected, else 0
        group_mask = torch.empty((M, self.group_count), dtype=torch.int32, device=scores.device)
        grid5 = (M,)
        build_group_mask_kernel[grid5](group_idx, group_mask, M=M, K=4, group_count=self.group_count)

        # 6) Expand group_mask to scores and set non-selected groups' scores to -inf -> masked_scores [M, 256]
        masked_scores = torch.empty_like(scores)
        grid6 = (M,)
        expand_group_mask_and_set_ninf_kernel[grid6](group_mask, scores, masked_scores, M=M, N=self.num_experts, group_count=self.group_count, experts_per_group=self.experts_per_group)

        # 7) Triton top-8 expert selection: topk_idx [M, 8]
        topk_idx = torch.empty((M, self.final_k), dtype=torch.int32, device=scores.device)
        grid7 = (M,)
        topk_select_kernel[grid7](masked_scores, topk_idx, M=M, M_dim=self.num_experts, K=self.final_k)

        # 8) Gather selected scores from original scores (use original scores, not masked_scores)
        selected_scores = torch.empty((M, self.final_k), dtype=torch.float32, device=scores.device)
        grid8 = (M,)
        gather_scores_kernel[grid8](scores, topk_idx, selected_scores, M=M, N=self.num_experts, K=self.final_k)

        # 9) Normalize and scale topk_weight
        topk_weight = torch.empty_like(selected_scores)
        grid9 = (M,)
        normalize_and_scale_kernel[grid9](selected_scores, topk_weight, self.routed_scaling_factor, M=M, K=self.final_k)

        # Return as required: int64 indices and float32 weights
        return topk_idx.to(torch.int64), topk_weight


def run(*args):
    return ModelNew()(*args)
