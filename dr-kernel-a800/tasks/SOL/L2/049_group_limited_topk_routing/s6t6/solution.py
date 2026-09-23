import torch
import torch.nn as nn

import triton
import triton.language as tl

# 1) Triton matmul: logits = hidden_states @ weight.T
# hidden_states: [M, K], row-major
# weight: [N, K], row-major (PyTorch stores weight as [num_experts, hidden_size])
# output: logits [M, N]
@triton.jit
def matmul_logits_kernel(
    a_ptr,         # *f32, [M, K]
    w_ptr,         # *f32, [N, K] (weight)
    out_ptr,       # *f32, [M, N]
    M: tl.int32,                       # number of tokens
    N: tl.constexpr,                   # num_experts = 256
    K: tl.constexpr                     # hidden_size = 128
):
    row = tl.program_id(0)  # token id
    col = tl.program_id(1)  # expert id
    acc = 0.0
    # loop over hidden_size dimension
    for k in range(K):
        a_val = tl.load(a_ptr + row * K + k)
        w_val = tl.load(w_ptr + col * K + k)
        acc += a_val * w_val
    tl.store(out_ptr + row * N + col, acc)


# 2) Triton: scores = sigmoid(logits) + expert_bias
# logits: [M, N], bias: [N]
@triton.jit
def sigmoid_add_bias_2d_kernel(
    logits_ptr,    # *f32, [M, N]
    bias_ptr,      # *f32, [N]
    out_ptr,       # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    val = tl.load(logits_ptr + row * N + col)
    sig = 1.0 / (1.0 + tl.exp(-val))
    b = tl.load(bias_ptr + col)
    out = sig + b
    tl.store(out_ptr + row * N + col, out)


# 3) Triton: compute group_scores [M, 8] from scores [M, 8, 32] flat buffer
# scores_flat: [M*GROUP_COUNT*EXPERTS_PER_GROUP] = [M*8*32]
@triton.jit
def group_top2_sum_kernel(
    scores_flat_ptr,   # *f32, [M*GROUP_COUNT*EXPERTS_PER_GROUP]
    out_group_ptr,     # *f32, [M, GROUP_COUNT]
    M: tl.int32,
    GROUP_COUNT: tl.constexpr,            # 8
    EXPERTS_PER_GROUP: tl.constexpr,      # 32
):
    pid = tl.program_id(0)  # token id
    for g in range(GROUP_COUNT):
        start = pid * GROUP_COUNT * EXPERTS_PER_GROUP + g * EXPERTS_PER_GROUP
        top1 = -float('inf')
        top2 = -float('inf')
        for i in range(EXPERTS_PER_GROUP):
            idx = start + i
            v = tl.load(scores_flat_ptr + idx)
            if v > top1:
                top2 = top1
                top1 = v
            elif v > top2:
                top2 = v
        out = top1 + top2
        tl.store(out_group_ptr + pid * GROUP_COUNT + g, out)


# 4) Triton: top-4 group selection indices per token
# group_scores: [M, 8], out_idx: [M, 4] (int32)
@triton.jit
def topk_group_kernel(
    group_scores_ptr,  # *f32, [M, 8]
    out_idx_ptr,       # *i32, [M, 4]
    M: tl.int32,
    GROUP_COUNT: tl.constexpr,    # 8
    K_GROUP: tl.constexpr,        # 4
):
    pid = tl.program_id(0)
    selected = tl.zeros((GROUP_COUNT,), dtype=tl.int32)
    idx_buf = tl.zeros((GROUP_COUNT,), dtype=tl.int32)
    for i in range(GROUP_COUNT):
        idx_buf[i] = i

    for t in range(K_GROUP):
        max_val = -float('inf')
        chosen = 0
        for i in range(GROUP_COUNT):
            val = tl.load(group_scores_ptr + pid * GROUP_COUNT + i)
            if selected[i] == 0:
                if val > max_val:
                    max_val = val
                    chosen = i
        selected[chosen] = 1
        tl.store(out_idx_ptr + pid * K_GROUP + t, chosen)


# 5) Triton: build group_mask [M, 8] from group_idx [M, 4]
@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,     # *i32, [M, 4]
    out_mask_ptr,      # *i32, [M, 8]
    M: tl.int32,
    GROUP_COUNT: tl.constexpr,    # 8
    K_GROUP: tl.constexpr,        # 4
):
    pid = tl.program_id(0)
    for g in range(GROUP_COUNT):
        found = 0
        for t in range(K_GROUP):
            idx = tl.load(group_idx_ptr + pid * K_GROUP + t)
            if g == idx:
                found = 1
                break
        tl.store(out_mask_ptr + pid * GROUP_COUNT + g, found)


# 6) Triton: expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores
# masked_scores: [M, 256]
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,     # *i32, [M, 8]
    scores_ptr,         # *f32, [M, 256] (original scores)
    masked_scores_ptr,  # *f32, [M, 256]
    M: tl.int32,
    NUM_EXPERTS: tl.constexpr,     # 256
    GROUP_COUNT: tl.constexpr,     # 8
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    for e in range(NUM_EXPERTS):
        group_id = e // EXPERTS_PER_GROUP
        group_selected = tl.load(group_mask_ptr + pid * GROUP_COUNT + group_id)  # 0 or 1
        if group_selected == 0:
            val = tl.load(scores_ptr + pid * NUM_EXPERTS + e)
            tl.store(masked_scores_ptr + pid * NUM_EXPERTS + e, -float('inf'))
        else:
            val = tl.load(scores_ptr + pid * NUM_EXPERTS + e)
            tl.store(masked_scores_ptr + pid * NUM_EXPERTS + e, val)


# 7) Triton: final top-8 selection indices from masked_scores -> [M, 8]
@triton.jit
def final_topk_indices_kernel(
    masked_scores_ptr,    # *f32, [M, 256]
    out_idx_ptr,          # *i32, [M, 8]
    M: tl.int32,
    NUM_EXPERTS: tl.constexpr,  # 256
    FINAL_K: tl.constexpr,      # 8
):
    pid = tl.program_id(0)
    selected = tl.zeros((NUM_EXPERTS,), dtype=tl.int32)
    idx_buf = tl.zeros((NUM_EXPERTS,), dtype=tl.int32)
    for i in range(NUM_EXPERTS):
        idx_buf[i] = i
    for t in range(FINAL_K):
        max_val = -float('inf')
        chosen = 0
        for i in range(NUM_EXPERTS):
            val = tl.load(masked_scores_ptr + pid * NUM_EXPERTS + i)
            if selected[i] == 0:
                if val > max_val:
                    max_val = val
                    chosen = i
        selected[chosen] = 1
        tl.store(out_idx_ptr + pid * FINAL_K + t, chosen)


# 8) Triton: gather original scores for selected experts into [M, 8]
@triton.jit
def gather_scores_kernel(
    scores_ptr,              # *f32, [M, 256]
    selected_idx_ptr,        # *i32, [M, 8]
    out_scores_ptr,          # *f32, [M, 8]
    M: tl.int32,
    NUM_EXPERTS: tl.constexpr,  # 256
    FINAL_K: tl.constexpr,      # 8
):
    pid = tl.program_id(0)
    for t in range(FINAL_K):
        idx = tl.load(selected_idx_ptr + pid * FINAL_K + t)
        val = tl.load(scores_ptr + pid * NUM_EXPERTS + idx)
        tl.store(out_scores_ptr + pid * FINAL_K + t, val)


# 9) Triton: normalize_and_scale for [M, 8] -> [M, 8]
@triton.jit
def normalize_and_scale_kernel(
    inp_ptr,         # *f32, [M, 8]
    out_ptr,         # *f32, [M, 8]
    scale: tl.float32,
    M: tl.int32,
    K: tl.constexpr  # 8
):
    pid = tl.program_id(0)
    total = 0.0
    for t in range(K):
        val = tl.load(inp_ptr + pid * K + t)
        total += val
    for t in range(K):
        val = tl.load(inp_ptr + pid * K + t)
        out_val = val / (total + 1e-20) * scale
        tl.store(out_ptr + pid * K + t, out_val)


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
        Triton-only implementation. All numerical work done inside Triton kernels.
        Returns:
        - topk_idx: [num_tokens, 8], int64
        - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be on CUDA for Triton kernels."
        M = hidden_states.shape[0]
        assert hidden_states.shape[1] == self.hidden_size, "hidden_states.hidden_size must be 128"
        assert weight.shape[0] == self.num_experts, "weight.num_experts must be 256"
        assert weight.shape[1] == self.hidden_size, "weight.hidden_size must be 128"
        assert expert_bias.shape[0] == self.num_experts, "expert_bias length must be 256"

        # 1)


def run(*args):
    return ModelNew()(*args)
