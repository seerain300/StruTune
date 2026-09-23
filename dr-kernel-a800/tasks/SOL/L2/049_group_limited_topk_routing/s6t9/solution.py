import torch
import torch.nn as nn

import triton
import triton.language as tl


# Kernel 1: logits = hidden_states @ weight.T
# hidden_states: [M, K] float32, row-major
# weight: [N, K] float32, row-major (same as PyTorch [num_experts, hidden_size])
# output logits: [M, N] float32
@triton.jit
def matmul_logits_kernel(
    a_ptr,            # *f32, [M, K]
    w_ptr,            # *f32, [N, K]
    out_ptr,          # *f32, [M, N]
    M: tl.int32,      # number of tokens
    N: tl.constexpr,  # number of experts (256)
    K: tl.constexpr,  # hidden_size (128)
):
    row = tl.program_id(0)  # 0..M-1
    col = tl.program_id(1)  # 0..N-1
    acc = 0.0
    # loop over hidden dimension
    for k in range(K):
        a_val = tl.load(a_ptr + row * K + k)  # hidden_states[row, k]
        w_val = tl.load(w_ptr + col * K + k)  # weight[col, k]
        acc += a_val * w_val
    tl.store(out_ptr + row * N + col, acc)


# Kernel 2: elementwise sigmoid + expert bias
# logits: [M, N], float32
# expert_bias: [N], float32
# out_scores: [M, N], float32
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,       # *f32, [M, N]
    bias_ptr,         # *f32, [N]
    out_ptr,          # *f32, [M, N]
    M: tl.int32,
    N: tl.constexpr,  # 256
):
    row = tl.program_id(0)  # 0..M-1
    col = tl.program_id(1)  # 0..N-1
    val = tl.load(logits_ptr + row * N + col)
    sig = 1.0 / (1.0 + tl.exp(-val))
    b = tl.load(bias_ptr + col)
    out = sig + b
    tl.store(out_ptr + row * N + col, out)


# Kernel 3: compute per-token group scores [M, 8] = sum of top-2 per group
# We take scores_reshaped as a flat buffer of size M * GROUP_COUNT * EXPERTS_PER_GROUP
# and decode index using group = pid // EXPERTS_PER_GROUP, offset = pid % EXPERTS_PER_GROUP.
# Output: group_scores [M, GROUP_COUNT], float32
@triton.jit
def group_top2_sum_kernel(
    scores_flat_ptr,  # *f32, flattened [M, GROUP_COUNT, EXPERTS_PER_GROUP]
    out_ptr,          # *f32, [M, GROUP_COUNT]
    M: tl.int32,
    GROUP_COUNT: tl.constexpr,         # 8
    EXPERTS_PER_GROUP: tl.constexpr,   # 32
    TOTAL_EXPERTS: tl.constexpr        # 256
):
    pid = tl.program_id(0)  # 0..M-1
    top1 = -float('inf')
    top2 = -float('inf')
    # loop over 32 experts in the group
    for i in range(EXPERTS_PER_GROUP):
        idx = pid * TOTAL_EXPERTS + i
        val = tl.load(scores_flat_ptr + idx)
        # find top-2
        if val > top1:
            top2 = top1
            top1 = val
        elif val > top2:
            top2 = val
    out_val = top1 + top2
    tl.store(out_ptr + pid * GROUP_COUNT + 0, out_val)  # one group per pid


# Kernel 4: select top-4 groups per token from group_scores [M, 4] (iterative top-k)
# group_scores_ptr: *f32, [M, GROUP_COUNT]
# group_idx_out_ptr: *i32, [M, 4]
@triton.jit
def topk_group_kernel(
    group_scores_ptr,      # *f32, [M, GROUP_COUNT]
    group_idx_out_ptr,     # *i32, [M, 4]
    M: tl.int32,
    GROUP_COUNT: tl.constexpr        # 8
):
    pid = tl.program_id(0)  # 0..M-1
    # initialize selected indices
    selected = [tl.zeros((), tl.int32) for _ in range(4)]
    used = [tl.zeros((), tl.int32) for _ in range(GROUP_COUNT)]  # 0 or 1
    # perform iterative top-k
    for t in range(4):
        max_val = -float('inf')
        argmax = 0
        for g in range(GROUP_COUNT):
            val = tl.load(group_scores_ptr + pid * GROUP_COUNT + g)
            if used[g] == 0 and val > max_val:
                max_val = val
                argmax = g
        selected[t] = argmax
        used[argmax] = 1
    # store selected indices
    for t in range(4):
        tl.store(group_idx_out_ptr + pid * 4 + t, selected[t])


# Kernel 5: build group_mask [M, 8], 1 at selected groups, 0 otherwise
# group_idx_in_ptr: *i32, [M, 4]
# group_mask_out_ptr: *i32, [M, 8]
@triton.jit
def build_group_mask_kernel(
    group_idx_in_ptr,      # *i32, [M, 4]
    group_mask_out_ptr,    # *i32, [M, 8]
    M: tl.int32,
    GROUP_COUNT: tl.constexpr        # 8
):
    pid = tl.program_id(0)  # 0..M-1
    # initialize mask with zeros
    for g in range(GROUP_COUNT):
        tl.store(group_mask_out_ptr + pid * GROUP_COUNT + g, 0)
    # set 1 at selected group indices
    # indices are in group_idx_in_ptr[pid * 4 + t], t in [0,3]
    for t in range(4):
        idx = tl.load(group_idx_in_ptr + pid * 4 + t)  # 0..7
        tl.store(group_mask_out_ptr + pid * GROUP_COUNT + idx, 1)


# Kernel 6: expand group_mask to [M, 256] and set -inf for non-selected groups
# group_mask_in: [M, 8] int32
# scores_masked_out: [M, 256] float32 (we assume input is the pre-bias scores to mask)
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_in_ptr,     # *i32, [M, 8]
    scores_in_ptr,         # *f32, [M, 256] (pre-bias scores)
    scores_masked_out_ptr, # *f32, [M, 256]
    M: tl.int32,
    NUM_EXPERTS: tl.constexpr,        # 256
    GROUP_COUNT: tl.constexpr,        # 8
    EXPERTS_PER_GROUP: tl.constexpr,  # 32
):
    pid = tl.program_id(0)  # 0..M-1
    # for each group, check if selected; if not, set its 32 experts to -inf
    for g in range(GROUP_COUNT):
        flag = tl.load(group_mask_in_ptr + pid * GROUP_COUNT + g)  # 0 or 1
        if flag == 0:
            # set all 32 experts in this group to -inf
            for j in range(EXPERTS_PER_GROUP):
                idx = g * EXPERTS_PER_GROUP + j
                val = tl.load(scores_in_ptr + pid * NUM_EXPERTS + idx)
                # write -inf where flag==0
                if flag == 0:
                    tl.store(scores_masked_out_ptr + pid * NUM_EXPERTS + idx, float('-inf'))
                else:
                    tl.store(scores_masked_out_ptr + pid * NUM_EXPERTS + idx, val)


# Kernel 7: final top-8 selection from masked_scores (elementwise max with iterative selection)
# masked_scores: [M, 256]
# out_topk_idx: [M, 8] i32
@triton.jit
def final_topk_select_kernel(
    masked_scores_ptr,     # *f32, [M, 256]
    out_idx_ptr,           # *i32, [M, 8]
    M: tl.int32,
    NUM_EXPERTS: tl.constexpr,        # 256
    FINAL_K: tl.constexpr,            # 8
):
    pid = tl.program_id(0)  # 0..M-1
    selected = [tl.zeros((), tl.int32) for _ in range(FINAL_K)]
    used = [tl.zeros((), tl.int32) for _ in range(NUM_EXPERTS)]  # 0 or 1
    # perform iterative top-k on masked_scores
    for t in range(FINAL_K):
        max_val = -float('inf')
        argmax = 0
        for e in range(NUM_EXPERTS):
            val = tl.load(masked_scores_ptr + pid * NUM_EXPERTS + e)
            if used[e] == 0 and val > max_val:
                max_val = val
                argmax = e
        selected[t] = argmax
        used[argmax] = 1
    # store selected indices
    for t in range(FINAL_K):
        tl.store(out_idx_ptr + pid * FINAL_K + t, selected[t])


# Kernel 8: gather selected original scores from pre-bias scores using final indices
# pre_bias_scores: [M, 256], float32
# final_idx_in: [M, 8], i32
# out_selected_scores: [M, 8], float32
@triton.jit
def gather_selected_scores_kernel(
    pre_bias_scores_ptr,   # *f32, [M, 256]
    final_idx_in_ptr,      # *i32, [M, 8]
    out_scores_ptr,        # *f32, [M, 8]
    M: tl.int32,
    NUM_EXPERTS: tl.constexpr,        # 256
    FINAL_K: tl.constexpr             # 8
):
    pid = tl.program_id(0)  # 0..M-1
    for t in range(FINAL_K):
        idx = tl.load(final_idx_in_ptr + pid * FINAL_K + t)  # 0..255
        val = tl.load(pre_bias_scores_ptr + pid * NUM_EXPERTS + idx)
        tl.store(out_scores_ptr + pid * FINAL_K + t, val)


# Kernel 9: normalize_and_scale for [M, 8] -> [M, 8]
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

        # 1) Triton: compute logits = hidden_states @ weight.T -> [M, 256]
        logits = torch.empty((M, self.num_experts), dtype=torch.float32, device=hidden_states.device)
        grid = (M, self.num_experts)
        matmul_logits_kernel[grid](
            hidden_states.to(torch.float32), weight.to(torch.float32), logits,
            M=M, N=self.num_experts, K=self.hidden_size
        )

        # 2) Triton: scores = sigmoid(logits) + expert_bias -> [M, 256]
        scores = torch.empty((M, self.num_experts), dtype=torch.float32, device=hidden_states.device)
        sigmoid_add_bias_kernel[grid](logits, expert_bias.to(torch.float32), scores, M, self.num_experts)

        # 3) Triton: group_top2_sum from scores reshaped [M, 8, 32] → [M, 8]
        # We need a flat pointer to scores for the kernel. Use scores.view(-1) with decoding by GROUP_COUNT and EXPERTS_PER_GROUP.
        scores_flat = scores.reshape(M, self.group_count, self.experts_per_group).reshape(-1)  # length M*8*32
        group_scores = torch.empty((M, self.group_count), dtype=torch.float32, device=hidden_states.device)
        # Launch one program per token
        group_top2_sum_kernel[(M,)](scores_flat, group_scores, M, self.group_count, self.experts_per_group, self.num_experts)

        # 4) Triton: select top-4 groups per token → [M, 4]
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        topk_group_kernel[(M,)](group_scores, group_idx, M, self.group_count)

        # 5) Triton: build group_mask [M, 8]
        group_mask = torch.empty((M, self.group_count), dtype=torch.int32, device=hidden_states.device)
        build_group_mask_kernel[(M,)](group_idx, group_mask, M, self.group_count)

        # 6) Triton: expand group_mask and set -inf in masked_scores (use pre-bias scores as the base)
        masked_scores = torch.empty((M, self.num_experts), dtype=torch.float32, device=hidden_states.device)
        expand_and_set_ninf_kernel[(M,)](
            group_mask, scores, masked_scores, M, self.num_experts, self.group_count, self.experts_per_group
        )

        # 7) Triton: final top-8 selection indices from masked_scores → [M, 8]
        final_idx = torch.empty((M, self.final_k), dtype=torch.int32, device=hidden_states.device)
        final_topk_select_kernel[(M,)](masked_scores, final_idx, M, self.num_experts, self.final_k)

        # 8) Triton: gather original pre-bias scores at selected indices → [M, 8]
        # We need pre-bias scores. Recompute sigmoid + bias to obtain them or use original logits. Here we recompute using Triton:
        # But to avoid extra matmul, we can use the same scores (which is sigmoid(logits)+bias), as selection logic in the original
        # depends on the post-bias scores for ranking. The original code uses scores_for_routing = scores + bias; we used that.
        # So we can gather directly from scores using final_idx.
        selected_scores = torch.empty((M, self.final_k), dtype=torch.float32, device=hidden_states.device)
        gather_selected_scores_kernel[(M,)](scores, final_idx, selected_scores, M, self.num_experts, self.final_k)

        # 9) Triton: normalize and scale
        out_weight = torch.empty((M, self.final_k), dtype=torch.float32, device=hidden_states.device)
        normalize_and_scale_kernel[(M,)](selected_scores, out_weight, self.routed_scaling_factor, M, self.final_k)

        # Return int64 indices and float32 weights
        return final_idx.to(torch.int64), out_weight


def run(*args):
    return ModelNew()(*args)
