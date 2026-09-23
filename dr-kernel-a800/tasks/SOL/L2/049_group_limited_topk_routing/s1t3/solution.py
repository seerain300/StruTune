import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Row-wise matmul logits = hidden @ weight^T
@triton.jit
def _row_matmul_kernel(
    hidden_ptr,         # *f32, [num_tokens, hidden_dim]
    weight_ptr,         # *f32, [num_experts, hidden_dim]
    out_ptr,            # *f32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_experts: tl.constexpr,
):
    token_id = tl.program_id(0)
    base_hidden = token_id * hidden_dim
    base_out = token_id * num_experts
    acc = tl.zeros([num_experts], dtype=tl.float32)
    for j in range(0, hidden_dim):
        h = tl.load(hidden_ptr + base_hidden + j)
        for e in range(0, num_experts):
            w = tl.load(weight_ptr + e * hidden_dim + j)
            acc[e] += h * w
    for e in range(0, num_experts):
        tl.store(out_ptr + base_out + e, acc[e])


# Kernel 2: Sigmoid elementwise on logits
@triton.jit
def _sigmoid_kernel(
    logits_ptr,         # *f32, [num_tokens, num_experts]
    out_ptr,            # *f32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
):
    token_id = tl.program_id(0)
    base = token_id * num_experts
    for e in range(0, num_experts):
        x = tl.load(logits_ptr + base + e)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + base + e, y)


# Kernel 3: Add expert bias
@triton.jit
def _add_bias_kernel(
    scores_ptr,         # *f32, [num_tokens, num_experts]
    bias_ptr,           # *f32, [num_experts]
    out_ptr,            # *f32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
):
    token_id = tl.program_id(0)
    base = token_id * num_experts
    for e in range(0, num_experts):
        s = tl.load(scores_ptr + base + e)
        b = tl.load(bias_ptr + e)
        tl.store(out_ptr + base + e, s + b)


# Kernel 4: Group top-2 reduction per token → group_scores [num_tokens, n_group]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,         # *f32, [num_tokens, num_experts]
    out_group_scores_ptr,  # *f32, [num_tokens, n_group]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    n_group: tl.constexpr,
    experts_per_group: tl.constexpr,
):
    token_id = tl.program_id(0)
    base = token_id * num_experts
    for g in range(0, n_group):
        group_start = g * experts_per_group
        max1 = -float('inf')
        max2 = -float('inf')
        for i in range(0, experts_per_group):
            e = group_start + i
            v = tl.load(scores_ptr + base + e)
            if v > max1:
                max2 = max1
                max1 = v
            elif v > max2:
                max2 = v
        tl.store(out_group_scores_ptr + token_id * n_group + g, max1 + max2)


# Kernel 5: Top-4 group selection per token (iterative argmax scan)
@triton.jit
def _top4_groups_kernel(
    group_scores_ptr,   # *f32, [num_tokens, n_group]
    out_group_idx_ptr,  # *i32, [num_tokens, 4]
    num_tokens: tl.constexpr,
    n_group: tl.constexpr,
    K_GROUP: tl.constexpr,  # 4
):
    token_id = tl.program_id(0)
    base_out = token_id * K_GROUP
    best_idx = tl.zeros([K_GROUP], dtype=tl.int32)
    best_val = tl.zeros([K_GROUP], dtype=tl.float32)
    for k in range(0, K_GROUP):
        best_idx[k] = 0
        best_val[k] = -float('inf')
    for g in range(0, n_group):
        score = tl.load(group_scores_ptr + token_id * n_group + g)
        for j in range(0, K_GROUP):
            if score > best_val[j]:
                # shift down
                for r in range(K_GROUP - 1, j, -1):
                    best_val[r] = best_val[r - 1]
                    best_idx[r] = best_idx[r - 1]
                best_val[j] = score
                best_idx[j] = g
                break
    for k in range(0, K_GROUP):
        tl.store(out_group_idx_ptr + base_out + k, best_idx[k])


# Kernel 6: Expand group mask to expert-level score_mask [num_tokens, num_experts]
@triton.jit
def _mask_expand_kernel(
    group_idx_ptr,      # *i32, [num_tokens, 4]
    score_mask_ptr,     # *i32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    n_group: tl.constexpr,
    experts_per_group: tl.constexpr,
):
    token_id = tl.program_id(0)
    base_out = token_id * num_experts
    # Initialize score_mask to zeros
    for e in range(0, num_experts):
        tl.store(score_mask_ptr + base_out + e, 0)
    # Set ones for selected groups
    for k in range(0, 4):
        g = tl.load(group_idx_ptr + token_id * 4 + k)  # i32
        start = g * experts_per_group
        for i in range(0, experts_per_group):
            e = start + i
            tl.store(score_mask_ptr + base_out + e, 1)


# Kernel 7: Masked fill: set non-selected group experts to NEG_INF in masked_scores
@triton.jit
def _masked_fill_kernel(
    scores_ptr,         # *f32, [num_tokens, num_experts]
    score_mask_ptr,     # *i32, [num_tokens, num_experts] (0/1)
    output_ptr,         # *f32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    NEG_INF: tl.constexpr,  # e.g., -1.0e20
):
    token_id = tl.program_id(0)
    base = token_id * num_experts
    for e in range(0, num_experts):
        val = tl.load(scores_ptr + base + e)
        mask = tl.load(score_mask_ptr + base + e)  # int32 0 or 1
        out = val if mask != 0 else NEG_INF
        tl.store(output_ptr + base + e, out)


# Kernel 8: Final top-8 selection on masked_scores via iterative argmax
@triton.jit
def _top8_indices_vals_kernel(
    masked_scores_ptr,  # *f32, [num_tokens, num_experts]
    out_top8_idx_ptr,   # *i32, [num_tokens, 8]
    out_top8_vals_ptr,  # *f32, [num_tokens, 8]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    K: tl.constexpr,  # 8
):
    token_id = tl.program_id(0)
    base_scores = token_id * num_experts
    base_out_idx = token_id * K
    base_out_vals = token_id * K
    best_idx = tl.zeros([K], dtype=tl.int32)
    best_val = tl.zeros([K], dtype=tl.float32)
    for k in range(0, K):
        best_idx[k] = 0
        best_val[k] = -float('inf')
    for e in range(0, num_experts):
        x = tl.load(masked_scores_ptr + base_scores + e)
        for j in range(0, K):
            if x > best_val[j]:
                for r in range(K - 1, j, -1):
                    best_val[r] = best_val[r - 1]
                    best_idx[r] = best_idx[r - 1]
                best_val[j] = x
                best_idx[j] = e
                break
    for k in range(0, K):
        tl.store(out_top8_idx_ptr + base_out_idx + k, best_idx[k])
        tl.store(out_top8_vals_ptr + base_out_vals + k, best_val[k])


# Kernel 9: Normalize top-8 values and apply scaling factor
@triton.jit
def _normalize_scale_top8_kernel(
    vals_ptr,           # *f32, [num_tokens, 8]
    out_weight_ptr,     # *f32, [num_tokens, 8]
    routed_scaling_factor: tl.constexpr,
    num_tokens: tl.constexpr,
    top_k: tl.constexpr,  # 8
):
    token_id = tl.program_id(0)
    base_vals = token_id * top_k
    total = 0.0
    for k in range(0, top_k):
        v = tl.load(vals_ptr + base_vals + k)
        total += v
    denom = total + 1e-20
    for k in range(0, top_k):
        v = tl.load(vals_ptr + base_vals + k)
        w = v / denom
        w = w * routed_scaling_factor
        tl.store(out_weight_ptr + base_vals + k, w)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int, routed_scaling_factor: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.routed_scaling_factor = routed_scaling_factor
        # Original constants
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure Triton availability and CUDA tensors
        assert TRITON_AVAILABLE, "Triton is not available"
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be CUDA tensors"

        # Cast to float32 for Triton
        hidden = hidden_states.contiguous().to(torch.float32)      # [num_tokens, hidden_dim]
        weight = weight.contiguous().to(torch.float32)             # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)          # [num_experts]

        num_tokens = hidden.shape[0]
        hidden_dim = hidden.shape[1]
        assert hidden_dim == self.hidden_dim, f"hidden_dim mismatch: expected {self.hidden_dim}, got {hidden_dim}"

        # Intermediate and output tensors
        logits = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((num_tokens, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((num_tokens, self.topk_group), dtype=torch.int32, device=hidden.device)
        score_mask = torch.empty((num_tokens, self.num_experts), dtype=torch.int32, device=hidden.device)
        masked_scores = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((num_tokens, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((num_tokens, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((num_tokens, self.top_k), dtype=torch.float32, device=hidden.device)

        # Launch kernels
        _row_matmul_kernel[(num_tokens,)](
            hidden, weight, logits,
            num_tokens, hidden_dim, self.num_experts,
        )

        _sigmoid_kernel[(num_tokens,)](
            logits, scores,
            num_tokens, self.num_experts,
        )

        _add_bias_kernel[(num_tokens,)](
            scores, bias, scores_for_routing,
            num_tokens, self.num_experts,
        )

        _group_top2_sum_kernel[(num_tokens,)](
            scores_for_routing, group_scores,
            num_tokens, self.num_experts, self.n_group, self.experts_per_group,
        )

        _


def run(*args):
    return ModelNew()(*args)
