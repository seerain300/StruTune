import torch
import triton
import triton.language as tl


# 1) Triton matmul: logits = hidden_states @ weight.T
@triton.jit
def matmul_logits_kernel(
    a_ptr,               # *f32, [M, K]
    b_ptr,               # *f32, [N, K]
    out_ptr,             # *f32, [M, N]
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bn: tl.int32, stride_bk: tl.int32,
    stride_om: tl.int32, stride_on: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        a_val = tl.load(a_ptr + pid_m * stride_am + k * stride_ak)
        b_val = tl.load(b_ptr + pid_n * stride_bn + k * stride_bk)
        acc += a_val * b_val
    tl.store(out_ptr + pid_m * stride_om + pid_n * stride_on, acc)


# 2) Triton: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,          # *f32, [M, N]
    bias_ptr,            # *f32, [N]
    scores_ptr,          # *f32, [M, N]
    M: tl.int32, N: tl.int32,
    stride_lm: tl.int32, stride_ln: tl.int32,
    stride_sm: tl.int32, stride_sn: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    val = tl.load(logits_ptr + pid_m * stride_lm + pid_n * stride_ln)
    bias_val = tl.load(bias_ptr + pid_n)
    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-val))
    tl.store(scores_ptr + pid_m * stride_sm + pid_n * stride_sn, s + bias_val)


# 3) Triton: group_top2_sum_kernel — compute group_scores [M, 8]
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,          # *f32, [M, N]
    group_scores_ptr,    # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,   # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        total = 0.0
        base = g * experts_per_group
        # find top-2 in this group
        top1 = -float('inf')
        top2 = -float('inf')
        for e in range(experts_per_group):
            idx = base + e
            val = tl.load(scores_ptr + pid * N + idx)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


# 4) Triton: topk_group_kernel (K=4) — return indices (int32) of selected groups per token
@triton.jit
def topk_group_kernel(
    group_scores_ptr,    # *f32, [M, 8]
    selected_idx_ptr,    # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,     # 4
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for g in range(8):
        val = tl.load(group_scores_ptr + pid * 8 + g)
        for j in range(K):
            if val > best_vals[j]:
                # shift down
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


# 5) Triton: build_group_mask_kernel — given selected group_idx [M, 4], set group_mask [M, 8] to 1 at selected positions
@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,       # *int32, [M, 4]
    group_mask_ptr,      # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # initialize zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


# 6) Triton: expand_and_set_ninf_kernel — expand group_mask [M, 8] to [M, 256], set non-selected groups' 32 entries to -inf
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256], will be mutated in-place
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,  # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Expand mask to [M, 256]
    for e in range(N):
        # compute which group this expert belongs to
        group_id = (e // experts_per_group)
        if tl.load(group_mask_ptr + pid * group_count + group_id) == 0.0:
            # set to -inf
            tl.store(masked_scores_ptr + pid * N + e, -float('inf'))
        else:
            # keep original value at this position; since masked_scores_ptr is not provided initially, we need to
            # rely on the caller to pass the original scores for this step. Here we just set -inf for non-selected
            # group entries; in practice, masked_scores should be initialized to the original scores. In this
            # implementation, the caller initializes masked_scores to original scores and we only set -inf for
            # non-selected groups.
            pass


# 7) Triton: final_topk_kernel (K=8) — select top-8 experts from masked_scores [M, 256]
@triton.jit
def final_topk_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    final_idx_ptr,       # *int32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(256):
        val = tl.load(masked_scores_ptr + pid * 256 + n)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(final_idx_ptr + pid * K + t, best_idxs[t])


# 8) Triton: gather_original_scores_kernel — gather original scores for selected indices
@triton.jit
def gather_original_scores_kernel(
    scores_ptr,          # *f32, [M, N]
    indices_ptr,         # *int32, [M, 8]
    gathered_ptr,        # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(indices_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * N + idx)
        tl.store(gathered_ptr + pid * K + t, val)


# 9) Triton: normalize_and_scale_kernel — normalize by sum of 8 scores and scale
@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,        # *f32, [M, 8]
    out_ptr,             # *f32, [M, 8]
    scale: tl.float32,
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = 0.0
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        total += val
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        out_val = val / (total + 1e-20) * scale
        tl.store(out_ptr + pid * K + t, out_val)


class ModelNew(torch.nn.Module):
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
        Triton-only implementation. All numerical work is done in Triton kernels.
        Returns:
        - topk_idx: [num_tokens, 8], int64
        - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be on CUDA for Triton kernels."
        M = hidden_states.shape[0]
        assert hidden_states.shape[1] == self.hidden_size, "hidden_states must have shape [num_tokens, 128]"
        assert weight.shape[0] == self.num_experts, "weight.num_experts must be 256"
        assert weight.shape[1] == self.hidden_size, "weight.hidden_size must be 128"
        assert expert_bias.shape[0] == self.num_experts, "expert_bias length must be 256"

        # 1) Compute logits = hidden_states @ weight.T in Triton
        logits = torch.empty((M, self.num_experts), device=hidden_states.device, dtype=torch.float32)
        grid = (M, self.num_experts)
        matmul_logits_kernel[grid](
            hidden_states, weight, logits,
            M, self.hidden_size, self.num_experts,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
        )

        # 2) Compute scores = sigmoid(logits) + expert_bias in Triton
        scores = torch.empty((M, self.num_experts), device=hidden_states.device, dtype=torch.float32)
        grid2 = (M, self.num_experts)
        sigmoid_add_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, self.num_experts,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # 3) Compute group_scores [M, 8] (sum of top-2 per group)
        group_scores = torch.empty((M, self.group_count), device=hidden_states.device, dtype=torch.float32)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](
            scores, group_scores,
            M, self.num_experts,
            self.group_count, self.experts_per_group,
        )

        # 4) Select top-4 groups per token
        group_idx = torch.empty((M, 4), device=hidden_states.device, dtype=torch.int32)
        grid4 = (M,)
        topk_group_kernel[grid4](
            group_scores, group_idx,
            M, 4,
        )

        # 5) Build group_mask [M, 8]
        group_mask = torch.empty((M, self.group_count), device=hidden_states.device, dtype=torch.float32)
        grid5 = (M,)
        build_group_mask_kernel[grid5](
            group_idx, group_mask,
            M, 4, self.group_count,
        )

        # 6) Expand group_mask to [M, 256] and set non-selected groups' 32 entries to -inf
        # Initialize masked_scores to original scores
        masked_scores = scores.clone()
        grid6 = (M,)
        # Note: expand_and_set_ninf_kernel sets -inf only for non-selected groups. For selected groups, it keeps original scores.
        expand_and_set_ninf_kernel[grid6](
            group_mask, masked_scores,
            M, self.num_experts,
            self.group_count, self.experts_per_group,
        )

        # 7) Select final top-8 experts from masked_scores
        final_idx = torch.empty((M, self.final_k), device=hidden_states.device, dtype=torch.int32)
        grid7 = (M,)
        final_topk_kernel[grid7](
            masked_scores, final_idx,
            M, 8,
        )

        # 8) Gather original scores for those 8
        gathered = torch.empty((M, self.final_k), device=hidden_states.device, dtype=torch.float32)
        grid8 = (M,)
        gather_original_scores_kernel[grid8](
            scores, final_idx, gathered,
            M, 8,
        )

        # 9) Normalize and scale
        topk_weight = torch.empty((M, self.final_k), device=hidden_states.device, dtype=torch.float32)
        grid9 = (M,)
        normalize_and_scale_kernel[grid9](
            gathered, topk_weight,
            self.routed_scaling_factor, M, 8,
        )

        # Return indices (int64) and weights (float32)
        topk_idx = final_idx.to(torch.int64)
        return topk_idx, topk_weight


# If you want to quickly test locally:
# model = ModelNew(routed_scaling_factor=1.0).cuda()
# hidden_states = torch.randn(2048, 128, device='cuda', dtype=torch.float32)
# weight = torch.randn(256, 128, device='cuda', dtype=torch.float32)
# expert_bias = torch.randn(256, device='cuda', dtype=torch.float32)
# idx, weight = model(hidden_states, weight, expert_bias)
# print(idx.shape, weight.shape)


def run(*args):
    return ModelNew()(*args)
