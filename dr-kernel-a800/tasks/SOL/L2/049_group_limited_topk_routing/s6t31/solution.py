import torch
import triton
import triton.language as tl


# Kernel 1: logits = hidden_states @ weight.T
@triton.jit
def matmul_logits_kernel(
    A_ptr,  # *f32, [M, K]
    B_ptr,  # *f32, [N, K]
    C_ptr,  # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        a = tl.load(A_ptr + pid_m * K + k)
        b = tl.load(B_ptr + k * N)  # B_ptr indexed by [K, N], so k*N + n
        # b = tl.load(B_ptr + k * N + pid_n) is incorrect; we must have a 2D pointer type
        # Triton requires passing B as [N, K] contiguous and indexing [pid_n, k]
        # Implement a 2D tile or load per n; here we use simple loop over N and use stride per n
        # To keep it simple and correct, we'll use Triton's tl.dot-like pattern via loop:
        # Better: use a 2D launch and tl.dot. Since we only use one program per m, loop over N and accumulate
        # But Triton doesn't support tl.dot; we implement outer product style accumulation:
        # However, Triton requires B to be [N, K]. We'll adjust below to use correct indexing.
    # Note: The above block is a placeholder. Triton matmul typically expects B in [N, K], so we redefine below:
    # We'll change the signature to pass B as [N, K] and do:
    pass  # placeholder, replaced below


# Better matmul kernel with 2D grid: C[m, n] = sum_k A[m, k] * B[n, k]
@triton.jit
def matmul_logits_kernel_2d(
    A_ptr,  # *f32, [M, K]
    B_ptr,  # *f32, [N, K]
    C_ptr,  # *f32, [M, N]
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
        a = tl.load(A_ptr + pid_m * K + k)
        b = tl.load(B_ptr + pid_n * K + k)
        acc += a * b
    tl.store(C_ptr + pid_m * N + pid_n, acc)


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,  # *f32, [M, N]
    bias_ptr,    # *f32, [N]
    out_ptr,     # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    x = tl.load(logits_ptr + pid_m * N + pid_n)
    b = tl.load(bias_ptr + pid_n)
    y = 1.0 / (1.0 + tl.exp(-x))
    y = y + b
    tl.store(out_ptr + pid_m * N + pid_n, y)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,    # *f32, [M, N]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,        # 256
    group_count: tl.constexpr,        # 8
    experts_per_group: tl.constexpr,  # 32
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
            # update top-2
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


@triton.jit
def topk_group_kernel(
    group_scores_ptr,  # *f32, [M, 8]
    selected_idx_ptr,  # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,   # 4
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
    # set zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


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
    neg_inf = -float('inf')
    for g in range(group_count):
        if tl.load(group_mask_ptr + pid * group_count + g) == 0.0:
            base = g * experts_per_group
            for o in range(experts_per_group):
                idx = base + o
                tl.store(masked_scores_ptr + pid * N + idx, neg_inf)


@triton.jit
def masked_topk_experts_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    selected_experts_ptr,  # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(N):
        val = tl.load(masked_scores_ptr + pid * N + n)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(selected_experts_ptr + pid * K + t, best_idxs[t])


@triton.jit
def gather_original_scores_kernel(
    scores_ptr,          # *f32, [M, N]
    selected_experts_ptr,  # *int32, [M, 8]
    gathered_ptr,        # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(selected_experts_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * N + idx)
        tl.store(gathered_ptr + pid * K + t, val)


@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,        # *f32, [M, 8]
    out_ptr,             # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
    scale: tl.float32,   # routed_scaling_factor
    eps: tl.float32,     # small epsilon
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sum_val = 0.0
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        sum_val += val
    inv = 1.0 / (sum_val + eps)
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t) * inv * scale
        tl.store(out_ptr + pid * K + t, val)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0, eps: float = 1e-20):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        """
        hidden_states: [M, 128], float32, CUDA
        weight: [256, 128], float32, CUDA
        expert_bias: [256], float32, CUDA
        Returns:
        - topk_idx: [M, 8], int32 (indices of selected experts per token)
        - topk_weight: [M, 8], float32 (normalized and scaled weights)
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "Use float32"
        M = hidden_states.shape[0]
        K = 128
        N = 256
        group_count = 8
        experts_per_group = 32
        K2 = 4  # top-k groups
        K3 = 8  # final top-k experts

        # 1) Compute logits = hidden_states @ weight.T using Triton
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid = (M, N)
        matmul_logits_kernel_2d[grid](hidden_states, weight, logits, M, N, K)

        # 2) Compute scores = sigmoid(logits) + expert_bias using Triton
        scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid2 = (M, N)
        sigmoid_add_bias_kernel[grid2](logits, expert_bias, scores, M, N)

        # 3) Compute group_scores [M, 8]: sum of top-2 per group
        group_scores = torch.empty((M, group_count), device=hidden_states.device, dtype=torch.float32)
        grid3 = (M,)
        group_top2_sum_kernel[grid3](scores, group_scores, M, N, group_count, experts_per_group)

        # 4) Select top-4 groups per token
        selected_group_idx = torch.empty((M, K2), device=hidden_states.device, dtype=torch.int32)
        grid4 = (M,)
        topk_group_kernel[grid4](group_scores, selected_group_idx, M, K2)

        # 5) Build group_mask [M, 8]
        group_mask = torch.empty((M, group_count), device=hidden_states.device, dtype=torch.float32)
        grid5 = (M,)
        build_group_mask_kernel[grid5](selected_group_idx, group_mask, M, K2, group_count)

        # 6) Expand group_mask to [M, 256] and set non-selected group entries to -inf
        masked_scores = scores.clone()  # will be mutated in-place by Triton
        grid6 = (M,)
        expand_and_set_ninf_kernel[grid6](group_mask, masked_scores, M, N, group_count, experts_per_group)

        # 7) Select final top-8 experts from masked_scores
        selected_expert_idx = torch.empty((M, K3), device=hidden_states.device, dtype=torch.int32)
        grid7 = (M,)
        masked_topk_experts_kernel[grid7](masked_scores, selected_expert_idx, M, N, K3)

        # 8) Gather original scores for selected experts (pre-bias)
        original_selected_scores = torch.empty((M, K3), device=hidden_states.device, dtype=torch.float32)
        grid8 = (M,)
        gather_original_scores_kernel[grid8](scores, selected_expert_idx, original_selected_scores, M, N, K3)

        # 9) Normalize and apply routed_scaling_factor
        topk_weight = torch.empty((M, K3), device=hidden_states.device, dtype=torch.float32)
        grid9 = (M,)
        normalize_and_scale_kernel[grid9](original_selected_scores, topk_weight, M, K3, self.routed_scaling_factor, self.eps)

        # Return indices (int32) and weights (float32)
        topk_idx = selected_expert_idx  # [M, 8]
        return topk_idx, topk_weight    # [M, 8]


def run(*args):
    return ModelNew()(*args)
