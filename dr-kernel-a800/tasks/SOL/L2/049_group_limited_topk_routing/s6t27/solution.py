import torch
import triton
import triton.language as tl

# 1) Triton: matmul_logits_kernel computes logits = hidden_states @ weight.T
@triton.jit
def matmul_logits_kernel(
    hidden_ptr,       # *f32, [M, K]
    weight_ptr,       # *f32, [N, K]
    logits_ptr,       # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
):
    pid_m = tl.program_id(0)  # row
    pid_n = tl.program_id(1)  # column
    if pid_m >= M or pid_n >= N:
        return
    acc = 0.0
    # loop over K
    for k in range(0, K):
        a = tl.load(hidden_ptr + pid_m * K + k)
        b = tl.load(weight_ptr + pid_n * K + k)
        acc += a * b
    tl.store(logits_ptr + pid_m * N + pid_n, acc)


# 2) Triton: sigmoid_add_bias_kernel computes scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,       # *f32, [M, N]
    bias_ptr,         # *f32, [N]
    scores_ptr,       # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
):
    pid_m = tl.program_id(0)  # row
    pid_n = tl.program_id(1)  # col
    if pid_m >= M or pid_n >= N:
        return
    val = tl.load(logits_ptr + pid_m * N + pid_n)
    bias = tl.load(bias_ptr + pid_n)
    val = 1.0 / (1.0 + tl.exp(-val)) + bias
    tl.store(scores_ptr + pid_m * N + pid_n, val)


# 3) Triton: group_top2_sum_kernel computes per-token group_scores [M, 8]
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,       # *f32, [M, N] where N=256
    group_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
):
    pid = tl.program_id(0)  # token
    if pid >= M:
        return
    total = 0.0
    for g in range(8):  # group index 0..7
        group_total = 0.0
        for ep in range(32):  # 32 experts per group
            idx = g * 32 + ep
            val = tl.load(scores_ptr + pid * N + idx)
            # simple top-2 tracking via list; Triton allows scalar loops
            max1 = -float('inf')
            max2 = -float('inf')
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
            group_total = group_total + max1 + max2
        total = total + group_total
    tl.store(group_scores_ptr + pid * 8 + 0, total)


# 4) Triton: topk_group_kernel (K=4) — return indices (int32) of selected groups per token
@triton.jit
def topk_group_kernel(
    group_scores_ptr, # *f32, [M, 8]
    selected_idx_ptr, # *int32, [M, 4]
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
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(8):
        if tl.load(group_mask_ptr + pid * 8 + g) != 1.0:
            base = g * 32
            for ep in range(32):
                idx = base + ep
                tl.store(masked_scores_ptr + pid * N + idx, -float('inf'))


# 7) Triton: topk_expert_kernel (K=8) — select top-8 experts from masked_scores [M, 256]
@triton.jit
def topk_expert_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    selected_expert_ptr, # *int32, [M, 8]
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
        tl.store(selected_expert_ptr + pid * K + t, best_idxs[t])


# 8) Triton: gather_original_scores_kernel — gather original scores [M, 8] from scores [M, 256] using selected_expert_idx
@triton.jit
def gather_original_scores_kernel(
    scores_ptr,          # *f32, [M, 256]
    selected_exp_ptr,    # *int32, [M, 8]
    original_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(selected_exp_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * 256 + idx)
        tl.store(original_scores_ptr + pid * 8 + t, val)


# 9) Triton: normalize_and_scale_kernel — normalize selected scores and apply routed_scaling_factor
@triton.jit
def normalize_and_scale_kernel(
    original_scores_ptr,  # *f32, [M, 8]
    normalized_ptr,       # *f32, [M, 8]
    M: tl.int32,
    scale: tl.float32,    # routed_scaling_factor
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sumv = 0.0
    for t in range(8):
        v = tl.load(original_scores_ptr + pid * 8 + t)
        sumv += v
    for t in range(8):
        v = tl.load(original_scores_ptr + pid * 8 + t)
        w = v / sumv
        tl.store(normalized_ptr + pid * 8 + t, w * scale)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and contiguity
        hidden = hidden_states.to(torch.float32).contiguous()
        weight = weight.to(torch.float32).contiguous()
        bias = expert_bias.to(torch.float32).contiguous()

        M = hidden.shape[0]
        K = 128  # hidden_size
        N = 256  # num_experts

        # 1) logits [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid1 = (M, N)
        matmul_logits_kernel[grid1](hidden, weight, logits, M, N, K)

        # 2) scores [M, N] = sigmoid(logits) + bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid2 = (M, N)
        sigmoid_add_bias_kernel[grid2](logits, bias, scores, M, N)

        # 3) group_scores [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](scores, group_scores, M)

        # 4) group_idx [M, 4] (int32)
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden.device)
        grid4 = (M,)
        topk_group_kernel[grid4](group_scores, group_idx, M, 4)

        # 5) group_mask [M, 8]
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid5 = (M,)
        build_group_mask_kernel[grid5](group_idx, group_mask, M, 4, 8)

        # 6) masked_scores [M, 256] with -inf for non-selected groups
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid6 = (M,)
        expand_and_set_ninf_kernel[grid6](group_mask, masked_scores, M, N)

        # 7) selected_experts [M, 8] (indices)
        selected_experts = torch.empty((M, 8), dtype=torch.int32, device=hidden.device)
        grid7 = (M,)
        topk_expert_kernel[grid7](masked_scores, selected_experts, M, 8)

        # 8) original_selected_scores [M, 8]
        original_selected_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid8 = (M,)
        gather_original_scores_kernel[grid8](scores, selected_experts, original_selected_scores, M, 8)

        # 9) normalized and scaled weights [M, 8]
        normalized = torch.empty((M, 8), dtype=torch.float32, device=hidden.device)
        grid9 = (M,)
        normalize_and_scale_kernel[grid9](original_selected_scores, normalized, M, routed_scaling_factor)

        # Return indices and weights
        return selected_experts, normalized


def run(*args):
    return ModelNew()(*args)
