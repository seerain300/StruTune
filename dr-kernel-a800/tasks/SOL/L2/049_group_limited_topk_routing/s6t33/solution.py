import torch
import triton
import triton.language as tl


# Kernel 1: matmul_logits_kernel
# Computes logits = hidden_states @ weight.T
# hidden_states: [M, K], weight: [N, K], logits: [M, N]
@triton.jit
def matmul_logits_kernel(
    hidden_ptr,      # *f32, [M, K]
    weight_ptr,      # *f32, [N, K]
    logits_ptr,      # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        a = tl.load(hidden_ptr + pid_m * K + k)
        b = tl.load(weight_ptr + pid_n * K + k)
        acc += a * b
    tl.store(logits_ptr + pid_m * N + pid_n, acc)


# Kernel 2: sigmoid_add_bias_kernel
# scores = sigmoid(logits) + expert_bias
# logits: [M, N], expert_bias: [N], scores: [M, N]
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,      # *f32, [M, N]
    bias_ptr,        # *f32, [N]
    scores_ptr,      # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    val = tl.load(logits_ptr + pid_m * N + pid_n)
    val = 1.0 / (1.0 + tl.exp(-val))  # sigmoid
    bias_val = tl.load(bias_ptr + pid_n)
    val += bias_val
    tl.store(scores_ptr + pid_m * N + pid_n, val)


# Kernel 3: group_top2_sum_kernel
# Computes per-token group_scores [M, 8] as sum of top-2 per group from scores [M, N]
# We decode group = n // 32, offset = n % 32. Passing scores as [M, N] pointer.
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,            # *f32, [M, N]
    group_scores_ptr,      # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,           # 256
    group_count: tl.constexpr,       # 8
    experts_per_group: tl.constexpr, # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    for g in range(group_count):
        base = g * experts_per_group
        top1 = -float('inf')
        top2 = -float('inf')
        for o in range(experts_per_group):
            idx = base + o
            val = tl.load(scores_ptr + pid * N + idx)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


# Kernel 4: topk_group_kernel (select top-4 groups per token)
@triton.jit
def topk_group_kernel(
    group_scores_ptr,      # *f32, [M, 8]
    selected_idx_ptr,      # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,       # 4
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
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


# Kernel 5: build_group_mask_kernel
# Set group_mask [M, 8]: 1 at selected groups (given selected_idx), 0 elsewhere
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
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


# Kernel 6: expand_and_set_ninf_kernel
# Expand group_mask [M, 8] to [M, 256] and set non-selected groups to -inf in masked_scores
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256], to be mutated
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,       # 8
    experts_per_group: tl.constexpr, # 32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    g = pid_n // experts_per_group
    keep = tl.load(group_mask_ptr + pid_m * group_count + g)
    if keep != 1.0:
        val = tl.load(masked_scores_ptr + pid_m * N + pid_n)
        tl.store(masked_scores_ptr + pid_m * N + pid_n, -float('inf'))


# Kernel 7: masked_topk_experts_kernel (select top-8 from masked_scores [M, 256])
@triton.jit
def masked_topk_experts_kernel(
    masked_scores_ptr,    # *f32, [M, 256]
    selected_experts_ptr, # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,          # 256
    K3: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K3,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K3,), dtype=tl.int32)
    for n in range(0, N):
        val = tl.load(masked_scores_ptr + pid * N + n)
        for j in range(K3):
            if val > best_vals[j]:
                for jj in range(K3 - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K3):
        tl.store(selected_experts_ptr + pid * K3 + t, best_idxs[t])


# Kernel 8: gather_original_scores_kernel
# Gather original scores at selected positions from original_scores [M, N]
@triton.jit
def gather_original_scores_kernel(
    original_scores_ptr,  # *f32, [M, N]
    selected_experts_ptr, # *int32, [M, 8]
    gathered_ptr,         # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,
    K3: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K3):
        idx = tl.load(selected_experts_ptr + pid * K3 + t)
        val = tl.load(original_scores_ptr + pid * N + idx)
        tl.store(gathered_ptr + pid * K3 + t, val)


# Kernel 9: normalize_and_scale_kernel
# Normalize gathered scores per token: divide by sum + eps, then scale by routed_scaling_factor
@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,        # *f32, [M, 8]
    normalized_ptr,      # *f32, [M, 8]
    M: tl.int32,
    K3: tl.constexpr,    # 8
    scale: tl.float32,
    eps: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = 0.0
    for t in range(K3):
        total += tl.load(gathered_ptr + pid * K3 + t)
    total = total + eps
    inv = 1.0 / total
    for t in range(K3):
        val = tl.load(gathered_ptr + pid * K3 + t) * inv * scale
        tl.store(normalized_ptr + pid * K3 + t, val)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0, eps: float = 1e-20):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure tensors are on CUDA and dtype float32
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors."
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = 128
        N = 256
        group_count = 8
        experts_per_group = 32
        K2 = 4  # top-4 groups
        K3 = 8  # final top-8 experts

        # 1) Compute logits = hidden_states @ weight.T in Triton
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_matmul = (M, N)
        matmul_logits_kernel[grid_matmul](hidden_states, weight, logits, M, N, K)

        # 2) scores = sigmoid(logits) + expert_bias in Triton
        scores = torch.empty_like(logits)
        bias = expert_bias.to(torch.float32).contiguous()
        grid_sigmoid = (M, N)
        sigmoid_add_bias_kernel[grid_sigmoid](logits, bias, scores, M, N)

        # 3) Compute per-token group_scores [M, 8] (sum of top-2 per group)
        group_scores = torch.empty((M, group_count), device=device, dtype=torch.float32)
        grid_g = (M,)
        group_top2_sum_kernel[grid_g](scores, group_scores, M, N, group_count, experts_per_group)

        # 4) Select top-4 groups per token → [M, 4] int32
        selected_group_idx = torch.empty((M, K2), device=device, dtype=torch.int32)
        grid_t = (M,)
        topk_group_kernel[grid_t](group_scores, selected_group_idx, M, K2)

        # 5) Build group_mask [M, 8]
        group_mask = torch.empty((M, group_count), device=device, dtype=torch.float32)
        grid_b = (M,)
        build_group_mask_kernel[grid_b](selected_group_idx, group_mask, M, K2, group_count)

        # 6) Expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        # Initialize masked_scores = scores (will set -inf where group_mask == 0)
        masked_scores.copy_(scores)
        grid_e = (M, N)
        expand_and_set_ninf_kernel[grid_e](group_mask, masked_scores, M, N, group_count, experts_per_group)

        # 7) Select final top-8 experts from masked_scores → [M, 8] int32
        selected_expert_idx = torch.empty((M, K3), device=device, dtype=torch.int32)
        grid_s = (M,)
        masked_topk_experts_kernel[grid_s](masked_scores, selected_expert_idx, M, N, K3)

        # 8) Gather original scores for selected experts (pre-bias) → [M, 8] float32
        original_selected_scores = torch.empty((M, K3), device=device, dtype=torch.float32)
        grid_go = (M,)
        gather_original_scores_kernel[grid_go](scores, selected_expert_idx, original_selected_scores, M, N, K3)

        # 9) Normalize per token and apply routed_scaling_factor → [M, 8] float32
        topk_weight = torch.empty((M, K3), device=device, dtype=torch.float32)
        grid_n = (M,)
        normalize_and_scale_kernel[grid_n](original_selected_scores, topk_weight, M, K3, self.routed_scaling_factor, self.eps)

        # Return indices and weights
        topk_idx = selected_expert_idx
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
