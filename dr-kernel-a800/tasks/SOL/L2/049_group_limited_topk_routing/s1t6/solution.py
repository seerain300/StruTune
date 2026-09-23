import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Row-wise matmul to compute logits = hidden @ weight^T
# hidden: [num_tokens, hidden_dim], weight: [num_experts, hidden_dim], logits: [num_tokens, num_experts]
@triton.jit
def _row_matmul_kernel(
    hidden_ptr,        # *f32, [num_tokens, hidden_dim]
    weight_ptr,        # *f32, [num_experts, hidden_dim]
    logits_ptr,        # *f32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    hidden_dim: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0)
    base_out = token_id * num_experts
    acc = tl.zeros((num_experts,), dtype=tl.float32)
    # Tile over hidden_dim
    for h in range(0, hidden_dim, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < hidden_dim
        # Load hidden row chunk
        hidden_row = tl.load(hidden_ptr + token_id * hidden_dim + h_offsets, mask=mask_h, other=0.0)
        # Accumulate dot-products over weight rows
        for e in range(0, num_experts):
            w = tl.load(weight_ptr + e * hidden_dim + h_offsets, mask=mask_h, other=0.0)
            acc[e] += tl.sum(hidden_row * w, axis=0)
    # Write logits
    for e in range(0, num_experts):
        tl.store(logits_ptr + base_out + e, acc[e])


# Kernel 2: Elementwise sigmoid on logits -> scores
@triton.jit
def _sigmoid_kernel(
    logits_ptr,        # *f32, [num_tokens, num_experts]
    scores_ptr,        # *f32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
):
    token_id = tl.program_id(0)
    base = token_id * num_experts
    for e in range(0, num_experts):
        x = tl.load(logits_ptr + base + e)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(scores_ptr + base + e, y)


# Kernel 3: Add expert bias to scores
@triton.jit
def _add_bias_kernel(
    scores_ptr,        # *f32, [num_tokens, num_experts]
    bias_ptr,          # *f32, [num_experts]
    scores_out_ptr,    # *f32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
):
    token_id = tl.program_id(0)
    base = token_id * num_experts
    for e in range(0, num_experts):
        s = tl.load(scores_ptr + base + e)
        b = tl.load(bias_ptr + e)
        tl.store(scores_out_ptr + base + e, s + b)


# Kernel 4: Compute per-group top-2 and sum -> group_scores [num_tokens, n_group]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,        # *f32, [num_tokens, num_experts]
    group_scores_ptr,  # *f32, [num_tokens, n_group]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    n_group: tl.constexpr,
    experts_per_group: tl.constexpr,
):
    token_id = tl.program_id(0)
    base_out = token_id * n_group
    for g in range(0, n_group):
        group_total = 0.0
        group_base = g * experts_per_group
        # Compute top-2 within this group
        for i in range(0, experts_per_group):
            e = group_base + i
            x = tl.load(scores_ptr + token_id * num_experts + e)
            group_total = tl.maximum(group_total, x)
            # Second max after removing the first
            x2 = -1.0e20
            for ii in range(0, experts_per_group):
                ex = tl.load(scores_ptr + token_id * num_experts + group_base + ii)
                if ex != group_total:
                    x2 = tl.maximum(x2, ex)
            group_total += x2
        tl.store(group_scores_ptr + base_out + g, group_total)


# Kernel 5: Select top-4 groups per token (iterative argmax)
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # *f32, [num_tokens, n_group]
    group_idx_ptr,     # *i32, [num_tokens, 4]
    num_tokens: tl.constexpr,
    n_group: tl.constexpr,
):
    token_id = tl.program_id(0)
    base_out = token_id * 4
    for k in range(0, 4):
        max_val = -1.0e20
        max_idx = 0
        for g in range(0, n_group):
            val = tl.load(group_scores_ptr + token_id * n_group + g)
            if val > max_val:
                max_val = val
                max_idx = g
        tl.store(group_idx_ptr + base_out + k, max_idx)
        # Zero out the selected to avoid re-selection in subsequent iterations
        tl.store(group_scores_ptr + token_id * n_group + max_idx, -1.0e20)


# Kernel 6: Build group mask: score_mask[i, e] = 1 if e belongs to one of selected groups, else 0
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr,     # *i32, [num_tokens, 4]
    score_mask_ptr,    # *i32, [num_tokens, num_experts]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    n_group: tl.constexpr,
    experts_per_group: tl.constexpr,
):
    token_id = tl.program_id(0)
    base_out = token_id * num_experts
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
    scores_ptr,         # *f32, [num_tokens, num_experts] (scores_for_routing)
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
        m = tl.load(score_mask_ptr + base + e)  # int32 0 or 1
        res = val if m != 0 else NEG_INF
        tl.store(output_ptr + base + e, res)


# Kernel 8: Final top-8 selection from masked_scores (iterative argmax), write indices and values
@triton.jit
def _top8_indices_vals_kernel(
    masked_scores_ptr, # *f32, [num_tokens, num_experts]
    out_idx_ptr,       # *i32, [num_tokens, 8]
    out_vals_ptr,      # *f32, [num_tokens, 8]
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    top_k: tl.constexpr,  # 8
):
    token_id = tl.program_id(0)
    base = token_id * num_experts
    for k in range(0, top_k):
        max_val = -1.0e20
        max_idx = 0
        for e in range(0, num_experts):
            x = tl.load(masked_scores_ptr + base + e)
            if x > max_val:
                max_val = x
                max_idx = e
        tl.store(out_idx_ptr + token_id * top_k + k, max_idx)
        tl.store(out_vals_ptr + token_id * top_k + k, max_val)
        # To avoid revisiting, set the chosen element to -inf
        tl.store(masked_scores_ptr + base + max_idx, -1.0e20)


# Kernel 9: Normalize and apply scaling factor to top-8 values
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
        # Constants from the original code
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure Triton is available and tensors are on CUDA
        if not TRITON_AVAILABLE or not (hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda):
            raise RuntimeError("Triton is not available or inputs are not on CUDA. Please run on CUDA with Triton installed.")

        # Make inputs contiguous and FP32
        hidden = hidden_states.contiguous().to(torch.float32)       # [num_tokens, hidden_dim]
        weight = weight.contiguous().to(torch.float32)              # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)           # [num_experts]

        num_tokens = hidden.shape[0]
        hidden_dim = hidden.shape[1]
        assert hidden_dim == self.hidden_dim, f"hidden_dim mismatch: expected {self.hidden_dim}, got {hidden_dim}"

        # Allocate outputs and intermediates
        logits = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((num_tokens, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((num_tokens, self.topk_group), dtype=torch.int32, device=hidden.device)
        score_mask = torch.empty((num_tokens, self.num_experts), dtype=torch.int32, device=hidden.device)  # expert-level mask
        masked_scores = torch.empty((num_tokens, self.num_experts), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((num_tokens, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((num_tokens, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((num_tokens, self.top_k), dtype=torch.float32, device=hidden.device)

        # Triton kernel launches (all computation in Triton, no torch ops in host)
        _row_matmul_kernel[(num_tokens,)](
            hidden, weight, logits,
            num_tokens=num_tokens, hidden_dim=hidden_dim, num_experts=self.num_experts, BLOCK_H=64,
        )

        _sigmoid_kernel[(num_tokens,)](
            logits, scores,
            num_tokens=num_tokens, num_experts=self.num_experts,
        )

        _add_bias_kernel[(num_tokens,)](
            scores, bias, scores_for_routing,
            num_tokens=num_tokens, num_experts=self.num_experts,
        )

        _group_top2_sum_kernel[(num_tokens,)](
            scores_for_routing, group_scores,
            num_tokens=num_tokens, num_experts=self.num_experts, n_group=self.n_group, experts_per_group=self.experts_per_group,
        )

        _select_top4_groups_kernel[(num_tokens,)](
            group_scores, group_idx,
            num_tokens=num_tokens, n_group=self.n_group,
        )

        _build_group_mask_kernel[(num_tokens,)](
            group_idx, score_mask,
            num_tokens=num_tokens, num_experts=self.num_experts, n_group=self.n_group, experts_per_group=self.experts_per_group,
        )

        _masked_fill_kernel[(num_tokens,)](
            scores_for_routing, score_mask, masked_scores,
            num_tokens=num_tokens, num_experts=self.num_experts, NEG_INF=-1.0e20,
        )

        _top8_indices_vals_kernel[(num_tokens,)](
            masked_scores, top8_idx, top8_vals,
            num_tokens=num_tokens, num_experts=self.num_experts, top_k=self.top_k,
        )

        _normalize_scale_top8_kernel[(num_tokens,)](
            top8_vals, topk_weight,
            routed_scaling_factor=self.routed_scaling_factor, num_tokens=num_tokens, top_k=self.top_k,
        )

        # Return indices and normalized weights as torch tensors
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
