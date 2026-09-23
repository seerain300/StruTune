import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Matmul for logits = hidden @ weight^T
# hidden: [M, K], weight: [N, K], out: [M, N]
@triton.jit
def _matmul_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, K, N,           # int32 sizes
    stride_hm, stride_hk,   # strides for hidden
    stride_wm, stride_wk,   # strides for weight
):
    row = tl.program_id(0)
    # Accumulator for this output row
    acc = tl.zeros((N,), dtype=tl.float32)
    # Loop over hidden_dim K in tiles
    for k0 in range(0, K, 32):
        # For each K-chunk, accumulate dot products with all N experts
        for n0 in range(0, N, 32):
            # Compute partial accumulation for the N tile
            for n_idx in range(32):
                col = n0 + n_idx
                # Load weight[col, k:k+32]
                w_vals = tl.load(
                    weight_ptr + col * stride_wm + (k0 + tl.arange(0, 32)) * stride_wk,
                    mask=(col < N) & ((k0 + tl.arange(0, 32)) < K),
                    other=0.0,
                )
                # Load hidden[row, k:k+32]
                h_vals = tl.load(
                    hidden_ptr + row * stride_hm + (k0 + tl.arange(0, 32)) * stride_hk,
                    mask=(k0 + tl.arange(0, 32)) < K,
                    other=0.0,
                )
                acc[col] += tl.sum(w_vals * h_vals, axis=0)
    # Store the row
    for n_idx in range(0, N):
        tl.store(out_ptr + row * N + n_idx, acc[n_idx])


# Kernel 2: Elementwise sigmoid over flat tensor
@triton.jit
def _sigmoid_flat_kernel(in_ptr, out_ptr, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < L
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


# Kernel 3: Add expert bias (elementwise over [M, N])
@triton.jit
def _add_bias_kernel(scores_ptr, bias_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # one program per row
    for row in range(0, M):
        for n in range(0, N, BLOCK):
            offs = n + tl.arange(0, BLOCK)
            mask = offs < N
            s = tl.load(scores_ptr + row * N + offs, mask=mask, other=0.0)
            b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + row * N + offs, s + b, mask=mask)


# Kernel 4: Group top-2 sum for each token (8 groups, 32 experts per group)
@triton.jit
def _group_top2_sum_kernel(scores_ptr, group_scores_ptr, M, N, n_group, experts_per_group, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # token id
    for g in range(0, n_group):
        base = g * experts_per_group
        vals = tl.load(scores_ptr + pid * N + base + tl.arange(0, BLOCK), mask=True, other=-float('inf'))  # BLOCK=32
        # Pass 1: max
        max_val = -float('inf')
        max_idx = 0
        for i in range(0, BLOCK):
            vi = vals[i]
            if vi > max_val:
                max_val = vi
                max_idx = i
        # Pass 2: second max
        second_val = -float('inf')
        for i in range(0, BLOCK):
            vi = vals[i]
            if vi > second_val and i != max_idx:
                second_val = vi
        tl.store(group_scores_ptr + pid * n_group + g, max_val + second_val)


# Kernel 5: Select top-4 groups per token (iterative argmax on group_scores)
@triton.jit
def _select_top4_groups_kernel(group_scores_ptr, group_idx_ptr, M, n_group, k: tl.constexpr):
    pid = tl.program_id(0)  # token id
    for t in range(0, k):
        max_val = -float('inf')
        best_group = 0
        for g in range(0, n_group):
            score = tl.load(group_scores_ptr + pid * n_group + g)
            if score > max_val:
                max_val = score
                best_group = g
        tl.store(group_idx_ptr + pid * k + t, best_group)


# Kernel 6: Build expert-level mask from selected group indices (1 for selected, 0 otherwise)
@triton.jit
def _build_group_mask_kernel(group_idx_ptr, score_mask_ptr, M, n_group, experts_per_group, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # token id
    for t in range(0, 4):  # topk_group = 4
        g = tl.load(group_idx_ptr + pid * 4 + t)
        base = g * experts_per_group
        for i in range(0, BLOCK):
            exp = base + i
            # linear index into expert dimension
            tl.store(score_mask_ptr + pid * 256 + exp, 1)


# Kernel 7: Masked fill: set masked_scores to -inf where score_mask == 0
@triton.jit
def _masked_fill_kernel(scores_ptr, score_mask_ptr, masked_ptr, M, N, NEG_INF: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # token id
    for n in range(0, N, BLOCK):
        offs = n + tl.arange(0, BLOCK)
        mask_exp = offs < N
        s = tl.load(scores_ptr + pid * N + offs, mask=mask_exp, other=0.0)
        m = tl.load(score_mask_ptr + pid * N + offs, mask=mask_exp, other=0).to(tl.float32)
        out = tl.where(m == 1, s, NEG_INF)
        tl.store(masked_ptr + pid * N + offs, out, mask=mask_exp)


# Kernel 8: Final top-8 selection from masked_scores (iterative argmax)
@triton.jit
def _top8_select_kernel(masked_ptr, top8_idx_ptr, top8_vals_ptr, M, N, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # token id
    for t in range(0, TOPK):
        max_val = -float('inf')
        best = 0
        for n in range(0, N, BLOCK):
            offs = n + tl.arange(0, BLOCK)
            mask = offs < N
            vals = tl.load(masked_ptr + pid * N + offs, mask=mask, other=-float('inf'))
            # reduce to get max in block (simple pass)
            for i in range(0, BLOCK):
                vi = vals[i]
                if vi > max_val:
                    max_val = vi
                    best = offs[i]
        tl.store(top8_idx_ptr + pid * TOPK + t, best)
        tl.store(top8_vals_ptr + pid * TOPK + t, max_val)


# Kernel 9: Normalize selected values and apply scaling factor
@triton.jit
def _normalize_scale_kernel(top8_vals_ptr, topk_weight_ptr, M, TOPK: tl.constexpr, eps: tl.constexpr, scaling: tl.constexpr):
    pid = tl.program_id(0)  # token id
    s = 0.0
    for t in range(0, TOPK):
        val = tl.load(top8_vals_ptr + pid * TOPK + t)
        s += val
    for t in range(0, TOPK):
        val = tl.load(top8_vals_ptr + pid * TOPK + t)
        normalized = val / (s + eps)
        scaled = normalized * scaling
        tl.store(topk_weight_ptr + pid * TOPK + t, scaled)


class ModelNew(nn.Module):
    def __init__(self, num_experts: int = 256, hidden_dim: int = None, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.n_group = 8
        self.experts_per_group = num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward: enforce CUDA and FP32
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        hidden = hidden_states.contiguous().to(torch.float32)       # [M, hidden_dim]
        weight = weight.contiguous().to(torch.float32)              # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)           # [num_experts]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim] and hidden [M, hidden_dim]."

        # 1) Matmul: logits = hidden @ weight^T using Triton
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid = (M,)
        _matmul_kernel[grid](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(0), weight.stride(1),
        )

        # 2) Sigmoid in Triton (elementwise over logits)
        logits_flat = logits.view(-1)  # [M*N]
        sig_out = torch.empty_like(logits_flat, dtype=torch.float32, device=logits.device)
        BLOCK_SIGMOID = 1024
        grid_sig = (triton.cdiv(logits_flat.numel(), BLOCK_SIGMOID),)
        _sigmoid_flat_kernel[grid_sig](logits_flat, sig_out, logits_flat.numel(), BLOCK_SIGMOID)
        scores = sig_out.view(M, N)

        # 3) Add expert bias
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=scores.device)
        BLOCK_BIAS = 128
        _add_bias_kernel[(M,)](
            scores, bias, scores_for_routing,
            M, N, BLOCK_BIAS
        )

        # 4) Group top-2 sum for each token
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=scores.device)
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group, self.experts_per_group, 32
        )

        # 5) Select top-4 groups per token
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=scores.device)
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group, 4
        )

        # 6) Build expert-level group mask
        score_mask = torch.empty((M, N), dtype=torch.int32, device=scores.device)  # 0/1
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, self.n_group, self.experts_per_group, 32
        )

        # 7) Masked fill: set non-selected to -inf
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=scores.device)
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M,)](
            scores_for_routing, score_mask, masked_scores,
            M, N, NEG_INF, 32
        )

        # 8) Final top-8 selection from masked scores
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=scores.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=scores.device)
        _top8_select_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N, 8, 32
        )

        # 9) Normalize and scale
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=scores.device)
        eps = 1e-20
        _normalize_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, 8, eps, self.routed_scaling_factor
        )

        # Return indices and weights (indices in [0..N-1], weights normalized and scaled)
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
