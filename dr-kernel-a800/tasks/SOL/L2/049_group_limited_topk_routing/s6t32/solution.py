import torch
import triton
import triton.language as tl


# 1) Triton matmul: logits = hidden_states @ weight.T
# hidden_states: [M, K], weight: [N, K], logits: [M, N]
@triton.jit
def matmul_logits_kernel(
    A_ptr,       # *f32, [M, K]
    B_ptr,       # *f32, [N, K] (weight, row-major)
    C_ptr,       # *f32, [M, N] (output logits)
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int32,  # stride for A along M
    stride_ak: tl.int32,  # stride for A along K
    stride_bn: tl.int32,  # stride for B along N (rows)
    stride_bk: tl.int32,  # stride for B along K (cols)
    stride_cm: tl.int32,  # stride for C along M
    stride_cn: tl.int32,  # stride for C along N
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        a = tl.load(A_ptr + pid_m * stride_am + k * stride_ak)
        b = tl.load(B_ptr + pid_n * stride_bn + k * stride_bk)
        acc += a * b
    tl.store(C_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


# 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,   # *f32, [M, N]
    bias_ptr,     # *f32, [N]
    scores_ptr,   # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_lm: tl.int32,  # stride logits along M
    stride_ln: tl.int32,  # stride logits along N
    stride_sm: tl.int32,  # stride scores along M
    stride_sn: tl.int32,  # stride scores along N
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    l = tl.load(logits_ptr + pid_m * stride_lm + pid_n * stride_ln)
    b = tl.load(bias_ptr + pid_n)
    s = 1.0 / (1.0 + tl.exp(-l)) + b
    tl.store(scores_ptr + pid_m * stride_sm + pid_n * stride_sn, s)


# 3) Triton: per-token group top-2 sum → group_scores [M, 8]
# scores_ptr is [M, N] flattened, group_count=8, experts_per_group=32
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,       # *f32, [M*N]
    group_scores_ptr, # *f32, [M*8]
    M: tl.int32,
    N: tl.int32,           # 256
    group_count: tl.constexpr,    # 8
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
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


# 4) Triton: top-k (K=4) on group_scores, return indices [M, 4] (int32)
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


# 5) Triton: build group_mask [M, 8] from selected group indices
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


# 6) Triton: expand group_mask to [M, 256], set non-selected groups to -inf in masked_scores
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256], will be mutated
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,    # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        if tl.load(group_mask_ptr + pid * group_count + g) != 1.0:
            base = g * experts_per_group
            for o in range(experts_per_group):
                idx = base + o
                val = tl.load(masked_scores_ptr + pid * N + idx)
                tl.store(masked_scores_ptr + pid * N + idx, -float('inf'))


# 7) Triton: masked top-8 selection per token on masked_scores → selected_expert_idx [M, 8] (int32)
@triton.jit
def masked_topk_experts_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    selected_idx_ptr,    # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    K3: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K3,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K3,), dtype=tl.int32)
    for n in range(N):
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
        tl.store(selected_idx_ptr + pid * K3 + t, best_idxs[t])


# 8) Triton: gather original scores for selected experts from original scores tensor
@triton.jit
def gather_original_scores_kernel(
    original_scores_ptr, # *f32, [M, N]
    selected_idx_ptr,    # *int32, [M, 8]
    gathered_ptr,        # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,
    K3: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K3):
        n_idx = tl.load(selected_idx_ptr + pid * K3 + t)
        s = tl.load(original_scores_ptr + pid * N + n_idx)
        tl.store(gathered_ptr + pid * K3 + t, s)


# 9) Triton: normalize gathered scores per token and scale by routed_scaling_factor
@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,        # *f32, [M, 8]
    out_ptr,             # *f32, [M, 8]
    M: tl.int32,
    K3: tl.constexpr,    # 8
    scaling: tl.float32,
    eps: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    denom = 0.0
    for t in range(K3):
        s = tl.load(gathered_ptr + pid * K3 + t)
        denom += s
    # denom > 0 by construction (we select real top-k), but guard with eps
    inv = 1.0 / (denom + eps)
    for t in range(K3):
        s = tl.load(gathered_ptr + pid * K3 + t) * inv
        s = s * scaling
        tl.store(out_ptr + pid * K3 + t, s)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float, eps: float = 1e-20):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure device and dtype
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be CUDA tensors"
        M = hidden_states.shape[0]
        K = 128  # hidden size
        N = 256  # number of experts
        group_count = 8
        experts_per_group = 32
        K2 = 4
        K3 = 8

        # 1) Triton matmul: logits = hidden_states @ weight.T
        # hidden_states: [M, K], weight: [N, K]
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid = (M, N)
        matmul_logits_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
        )

        # 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
        scores = torch.empty_like(logits)  # [M, N], float32
        bias = expert_bias.to(torch.float32).contiguous()  # [N]
        grid = (M, N)
        sigmoid_add_bias_kernel[grid](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # 3) Triton group top-2 sum → group_scores [M, 8]
        group_scores = torch.empty((M, group_count), device=scores.device, dtype=torch.float32)
        group_top2_sum_kernel[(M,)](
            scores.flatten(), group_scores,
            M, N,
            group_count=group_count,
            experts_per_group=experts_per_group,
        )

        # 4) Triton: top-4 groups per token → selected_idx [M, 4] (int32)
        selected_idx = torch.empty((M, K2), device=scores.device, dtype=torch.int32)
        topk_group_kernel[(M,)](
            group_scores, selected_idx,
            M, K=K2,
        )

        # 5) Triton: build group_mask [M, 8]
        group_mask = torch.empty((M, group_count), device=scores.device, dtype=torch.float32)
        build_group_mask_kernel[(M,)](
            selected_idx, group_mask,
            M, K=K2, group_count=group_count,
        )

        # 6) Triton: expand to [M, 256] and set non-selected groups to -inf
        masked_scores = scores.clone()
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N,
            group_count=group_count, experts_per_group=experts_per_group,
        )

        # 7) Triton: masked top-8 selection → selected_expert_idx [M, 8] (int32)
        selected_expert_idx = torch.empty((M, K3), device=scores.device, dtype=torch.int32)
        masked_topk_experts_kernel[(M,)](
            masked_scores, selected_expert_idx,
            M, N, K3=K3,
        )

        # 8) Triton: gather original scores for selected experts (pre-bias) → [M, 8] (float32)
        original_selected_scores = torch.empty((M, K3), device=scores.device, dtype=torch.float32)
        gather_original_scores_kernel[(M,)](
            scores, selected_expert_idx, original_selected_scores,
            M, N, K3=K3,
        )

        # 9) Triton: normalize and scale → final topk_weight [M, 8]
        topk_weight = torch.empty((M, K3), device=scores.device, dtype=torch.float32)
        normalize_and_scale_kernel[(M,)](
            original_selected_scores, topk_weight,
            M, K3=K3, scaling=self.routed_scaling_factor, eps=self.eps,
        )

        # Return indices (int32) and weights (float32)
        topk_idx = selected_expert_idx  # [M, 8]
        return topk_idx, topk_weight    # [M, 8]


def run(*args):
    return ModelNew()(*args)
