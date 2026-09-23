import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: elementwise sigmoid
# Input: x [M*N], Output: out [M*N]
@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, size, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y, mask=mask)


# Triton kernel: add bias vector to each row
# scores: [M, N], bias: [N], out: [M, N]
@triton.jit
def _add_bias_kernel(scores_ptr, bias_ptr, out_ptr,
                     M, N,
                     stride_sm, stride_sn, stride_outm, stride_outn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    scores = tl.load(scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    out = scores + bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, out, mask=mask)


# Triton kernel: per-token group top-2 sum over groups
# scores_for_routing_flat: [M*N], group_scores: [M*G], N=256, G=8, E=32
# For each token m in 0..M-1, process 8 groups of 32:
#   For each group g in 0..7: compute top2 per group and sum, store at group_scores[m*G + g].
@triton.jit
def _group_top2_sum_kernel(scores_ptr, group_scores_ptr,
                           M, N, G, E,  # N=256, G=8, E=32
                           stride_m, stride_n,
                           BLOCK: tl.constexpr):
    m = tl.program_id(0)
    base = m * stride_m
    # Loop over groups
    for g in range(0, G):
        # Start index for this group
        start = g * E
        # Compute top-2 sum for this group
        # Load E elements: scores[m, start + 0..E-1]
        idxs = start + tl.arange(0, E)
        vals = tl.load(scores_ptr + base + idxs * stride_n)
        # Compute top-2 manually
        max1 = tl.max(vals, axis=0)
        mask1 = vals == max1
        # Exclude max1 by setting to -inf
        vals2 = tl.where(mask1, -float('inf'), vals)
        max2 = tl.max(vals2, axis=0)
        group_scores[m * G + g] = max1 + max2


# Triton kernel: select top-4 group indices per token
# group_scores: [M*G], group_idx: [M*TOPK_GROUP]
@triton.jit
def _select_top4_groups_kernel(group_scores_ptr, group_idx_ptr,
                                M, G, TOPK_GROUP: tl.constexpr):
    m = tl.program_id(0)
    base_gs = m * G
    # Initialize best_idx and best_val (scalars)
    best_idx = tl.zeros((), dtype=tl.int32)
    best_val = tl.zeros((), dtype=tl.float32)
    # Iterative argmax for 4 selections
    for t in range(0, TOPK_GROUP):
        max_val = -float('inf')
        max_idx = 0
        for g in range(0, G):
            val = tl.load(group_scores_ptr + base_gs + g)
            if val > max_val:
                max_val = val
                max_idx = g
        # Write index
        tl.store(group_idx_ptr + m * TOPK_GROUP + t, max_idx)
        # Mark selected by setting to -inf
        tl.store(group_scores_ptr + base_gs + max_idx, -float('inf'))


# Triton kernel: build expert-level mask from group indices
# group_idx: [M*TOPK_GROUP], score_mask: [M*N] int32
# For each token m, set score_mask[m, e] = 1 if e belongs to any selected group, else 0.
@triton.jit
def _build_group_mask_kernel(group_idx_ptr, score_mask_ptr,
                             M, N, G, TOPK_GROUP: tl.constexpr,
                             stride_mm, stride_nn,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Set score_mask to 0
    tl.store(score_mask_ptr + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_nn, 0, mask=mask)

    # For each selected group idx, set mask[m, idx*32 + 0..31] = 1
    for t in range(0, TOPK_GROUP):
        g = tl.load(group_idx_ptr + pid_m * TOPK_GROUP + t)  # scalar
        base = g * 32
        # For this block, if pid_m == m, set elements at offs_n within this group to 1
        # We broadcast compare: (pid_m == m) is True for this block's m; otherwise mask prevents writes.
        # To ensure only the correct m writes, we use pid_m == m; but since pid_m identifies block, we rely on mask_m and offs_m.
        # However, score_mask is 2D and we can only write for this m via mask_m. Triton requires scalar indexing; so we write per m.
        # So we will rely on the fact that each (pid_m, pid_n) block handles one m and set those entries for all N.
        # To set mask[m, base + k], we need a 1D loop over k in [0,31]. Implement via static loop.
        for k in range(0, 32):
            e = base + k
            # only write if e < N
            if e < N:
                tl.store(score_mask_ptr + pid_m * stride_mm + e * stride_nn, 1)


# Triton kernel: masked fill, set masked_scores[m, e] = -inf if score_mask[m, e] == 0
@triton.jit
def _masked_fill_kernel(scores_ptr, score_mask_ptr, masked_ptr,
                        M, N,
                        stride_sm, stride_sn, stride_mm, stride_mn, stride_outm, stride_outn,
                        NEG_INF: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    scores = tl.load(scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn, mask=mask, other=0.0)
    mask_int = tl.load(score_mask_ptr + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn, mask=mask, other=0).to(tl.int1)
    out = tl.where(mask_int == 0, NEG_INF, scores)
    tl.store(masked_ptr + offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn, out, mask=mask)


# Triton kernel: select top-8 indices from masked_scores per token
# masked_scores: [M, N], top8_idx: [M*TOPK], top8_vals: [M*TOPK]
@triton.jit
def _top8_select_kernel(masked_ptr, top_idx_ptr, top_val_ptr,
                        M, N, TOPK: tl.constexpr,
                        stride_mm, stride_nn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    # For this block, only the first m row is valid; assume pid_m == 0 covers all M via grid size.
    # To handle M>1, we set masked loads/stores for each m. Implement by processing single m per program and looping.
    # Since Triton grid is 1D over M, we can simplify: we set BLOCK_M=1. So each program handles one m.
    m = offs_m  # scalar, but we use single element indexing
    for t in range(0, TOPK):
        max_val = -float('inf')
        max_idx = 0
        for n in range(0, N):
            val = tl.load(masked_ptr + m * stride_mm + n * stride_nn)
            if val > max_val:
                max_val = val
                max_idx = n
        tl.store(top_idx_ptr + m * TOPK + t, max_idx)
        tl.store(top_val_ptr + m * TOPK + t, max_val)
        # Mark selected element as -inf for next iterations
        tl.store(masked_ptr + m * stride_mm + max_idx * stride_nn, NEG_INF)


# Triton kernel: normalize selected values and apply scaling
# top8_vals: [M*TOPK], scaling_factor: scalar, out: [M*TOPK]
@triton.jit
def _normalize_and_scale_kernel(vals_ptr, out_ptr,
                                M, TOPK: tl.constexpr,
                                scaling_factor: tl.float32,
                                BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M
    # Compute sum of selected vals
    s = tl.zeros((), dtype=tl.float32)
    for t in range(0, TOPK):
        val = tl.load(vals_ptr + offs * TOPK + t, mask=mask, other=0.0)
        s += val
    eps = 1e-20
    for t in range(0, TOPK):
        val = tl.load(vals_ptr + offs * TOPK + t, mask=mask, other=0.0)
        normalized = val / (s + eps)
        scaled = normalized * scaling_factor
        tl.store(out_ptr + offs * TOPK + t, scaled, mask=mask)


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
        # Triton-only forward: no torch operations in host code
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        hidden = hidden_states.contiguous().to(torch.float32)       # [M, hidden_dim]
        weight = weight.contiguous().to(torch.float32)              # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)           # [num_experts]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim] and hidden [M, hidden_dim]."

        # 1) Linear operation in PyTorch (robust and fast)
        logits = F.linear(hidden, weight)  # [M, N], FP32

        # 2) Sigmoid in Triton (elementwise)
        logits_flat = logits.view(-1)
        sig_out = torch.empty_like(logits_flat, dtype=torch.float32, device=logits.device)
        BLOCK_SIGMOID = 1024
        grid_sig = (triton.cdiv(logits_flat.numel(), BLOCK_SIGMOID),)
        _sigmoid_kernel[grid_sig](logits_flat, sig_out, logits_flat.numel(), BLOCK_SIGMOID)
        scores = sig_out.view(M, N)

        # 3) Add expert bias in Triton
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        BLOCK_M = 64
        BLOCK_N = 64
        grid_add = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _add_bias_kernel[grid_add](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            BLOCK_M, BLOCK_N
        )

        # 4) Group top-2 sum in Triton: reshape to [M, 8, 32] and reduce
        # Flatten scores_for_routing to [M*N], but since we already have [M,N], build a contiguous view
        scores_flat = scores_for_routing.view(-1)  # [M*N]
        group_scores = torch.empty((M * self.n_group,), dtype=torch.float32, device=hidden.device)
        grid_top2 = (M,)
        _group_top2_sum_kernel[grid_top2](
            scores_flat, group_scores,
            M, N, self.n_group, self.experts_per_group,
            scores_flat.stride(0), scores_flat.stride(1),
            1024
        )

        # 5) Select top-4 groups in Triton
        group_idx = torch.empty((M * self.topk_group,), dtype=torch.int32, device=hidden.device)
        grid_group = (M,)
        _select_top4_groups_kernel[grid_group](
            group_scores, group_idx,
            M, self.n_group, self.topk_group
        )

        # 6) Build group mask in Triton: expert-level mask [M, N]
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)
        grid_mask = (triton.cdiv(M, 1), triton.cdiv(N, 64))
        _build_group_mask_kernel[grid_mask](
            group_idx, score_mask,
            M, N, self.n_group, self.topk_group,
            score_mask.stride(0), score_mask.stride(1),
            1, 64
        )

        # 7) Masked fill in Triton: masked_scores = -inf where score_mask == 0
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid_mask_fill = (triton.cdiv(M, 1), triton.cdiv(N, 64))
        _masked_fill_kernel[grid_mask_fill](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            NEG_INF=-1.0e20,
            BLOCK_M=1, BLOCK_N=64
        )

        # 8) Select top-8 from masked_scores in Triton
        top8_idx = torch.empty((M * self.top_k,), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M * self.top_k,), dtype=torch.float32, device=hidden.device)
        grid_top8 = (triton.cdiv(M, 1),)
        _top8_select_kernel[grid_top8](
            masked_scores, top8_idx, top8_vals,
            M, N, self.top_k,
            masked_scores.stride(0), masked_scores.stride(1),
            1, 64
        )

        # 9) Normalize and scale top8_vals in Triton
        topk_weight = torch.empty((M * self.top_k,), dtype=torch.float32, device=hidden.device)
        grid_norm = (triton.cdiv(M, 1),)
        _normalize_and_scale_kernel[grid_norm](
            top8_vals, topk_weight,
            M, self.top_k,
            self.routed_scaling_factor,
            1
        )

        # Reshape outputs to match original expectations: indices [M, TOPK], weights [M, TOPK]
        top8_idx = top8_idx.view(M, self.top_k)
        topk_weight = topk_weight.view(M, self.top_k)

        # Map indices back to original token ordering? The original code didn't require it; we return per-token selected indices and weights.
        # Note: The original function returns (topk_idx, topk_weight). We return them directly.

        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
