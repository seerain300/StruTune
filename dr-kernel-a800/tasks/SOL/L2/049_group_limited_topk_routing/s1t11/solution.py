import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Matmul logits = hidden @ weight^T
# hidden: [M, K], weight: [N, K], out: [M, N]
@triton.jit
def _matmul_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, K, N,           # int32 sizes
    stride_hm, stride_hk,   # strides for hidden
    stride_wk, stride_wn,   # strides for weight (wk = K dim, wn = N dim)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Pointers to tile
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        # Compute hidden tile pointers [BLOCK_M, BLOCK_K]
        hidden_ptrs = hidden_ptr + m_offsets[:, None] * stride_hm + (k + k_offsets[None, :]) * stride_hk
        # Compute weight tile pointers [BLOCK_K, BLOCK_N] using weight[e, j]
        weight_ptrs = weight_ptr + (k + k_offsets[:, None]) * stride_wk + n_offsets[None, :] * stride_wn

        # Masks for out-of-bounds
        h_mask = (m_offsets[:, None] < M) & (k + k_offsets[None, :] < K)
        w_mask = (k + k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles
        a = tl.load(hidden_ptrs, mask=h_mask, other=0.0)   # [BLOCK_M, BLOCK_K]
        b = tl.load(weight_ptrs, mask=w_mask, other=0.0)   # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)

    # Write output
    out_ptrs = out_ptr + m_offsets[:, None] * N + n_offsets[None, :]
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# Kernel 2: Sigmoid on logits -> scores
@triton.jit
def _sigmoid_kernel(
    in_ptr, out_ptr, M, N, NUM_TOKENS, NUM_EXPERTS,
):
    # 2D launch over (NUM_TOKENS, NUM_EXPERTS)
    token = tl.program_id(0)
    expert = tl.program_id(1)
    # Bounds check
    if token >= NUM_TOKENS or expert >= NUM_EXPERTS:
        return
    x = tl.load(in_ptr + token * NUM_EXPERTS + expert)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + token * NUM_EXPERTS + expert, y)


# Kernel 3: Add bias to scores -> scores_for_routing
@triton.jit
def _add_bias_kernel(
    scores_ptr, bias_ptr, out_ptr, M, N,
):
    token = tl.program_id(0)
    expert = tl.program_id(1)
    if token >= M or expert >= N:
        return
    score = tl.load(scores_ptr + token * N + expert)
    bias = tl.load(bias_ptr + expert)
    tl.store(out_ptr + token * N + expert, score + bias)


# Kernel 4: Compute per-group top-2 sum -> group_scores [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N, n_group, experts_per_group,
):
    token = tl.program_id(0)
    if token >= M:
        return
    # Loop over groups
    for g in range(0, n_group):
        group_sum = 0.0
        group_base = g * experts_per_group
        # Compute top-2 within this group of size 32
        max1 = -1.0e20
        max2 = -1.0e20
        for e in range(0, experts_per_group):
            exp_idx = group_base + e
            if exp_idx < N:
                val = tl.load(scores_ptr + token * N + exp_idx)
                if val > max1:
                    max2 = max1
                    max1 = val
                elif val > max2:
                    max2 = val
        group_sum = max1 + max2
        tl.store(group_scores_ptr + token * n_group + g, group_sum)


# Kernel 5: Select top-4 groups per token -> group_idx [M, 4]
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr, M, n_group,
):
    token = tl.program_id(0)
    if token >= M:
        return
    current = -1.0e20
    for i in range(0, 4):
        maxv = -1.0e20
        maxj = -1
        for g in range(0, n_group):
            v = tl.load(group_scores_ptr + token * n_group + g)
            if v > maxv:
                maxv = v
                maxj = g
        # Write selected group
        tl.store(group_idx_ptr + token * 4 + i, maxj)
        # Invalidate it by setting to -inf
        tl.store(group_scores_ptr + token * n_group + maxj, -1.0e20)


# Kernel 6: Build expert-level group mask from group_idx -> score_mask [M, 256] int32
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr, score_mask_ptr, M, n_group, experts_per_group,
):
    token = tl.program_id(0)
    if token >= M:
        return
    for i in range(0, 4):
        g = tl.load(group_idx_ptr + token * 4 + i)  # int32
        base = g * experts_per_group
        for e in range(0, experts_per_group):
            exp_idx = base + e
            tl.store(score_mask_ptr + token * 256 + exp_idx, 1)
    # Initialize all to 0 first by setting to 0 then override selected groups
    # We assume score_mask is zero-initialized by host.
    # Nothing to do here since host zeros it.


# Kernel 7: Masked fill: masked_scores[i, e] = -inf if score_mask[i, e] == 0 else scores_for_routing[i, e]
@triton.jit
def _masked_fill_kernel(
    scores_ptr, mask_ptr, masked_ptr, M, N, NEG_INF,
):
    token = tl.program_id(0)
    expert = tl.program_id(1)
    if token >= M or expert >= N:
        return
    score = tl.load(scores_ptr + token * N + expert)
    m = tl.load(mask_ptr + token * N + expert)
    out = score
    if m == 0:
        out = NEG_INF
    tl.store(masked_ptr + token * N + expert, out)


# Kernel 8: Top-8 selection from masked_scores -> top8_idx [M, 8], top8_vals [M, 8]
@triton.jit
def _top8_selection_kernel(
    masked_ptr, idx_ptr, vals_ptr, M, N, K_TOP,
):
    token = tl.program_id(0)
    if token >= M:
        return
    for k in range(0, K_TOP):
        maxv = -1.0e20
        maxj = -1
        for e in range(0, N):
            val = tl.load(masked_ptr + token * N + e)
            if val > maxv:
                maxv = val
                maxj = e
        tl.store(idx_ptr + token * K_TOP + k, maxj)
        tl.store(vals_ptr + token * K_TOP + k, maxv)


# Kernel 9: Normalize top-8 vals and apply scaling factor -> topk_weight [M, 8]
@triton.jit
def _normalize_and_scale_kernel(
    vals_ptr, scale, out_ptr, M, K_TOP,
):
    token = tl.program_id(0)
    if token >= M:
        return
    sumv = 0.0
    for k in range(0, K_TOP):
        v = tl.load(vals_ptr + token * K_TOP + k)
        sumv += v
    for k in range(0, K_TOP):
        v = tl.load(vals_ptr + token * K_TOP + k)
        norm = v / (sumv + 1e-20) * scale
        tl.store(out_ptr + token * K_TOP + k, norm)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the routing computation.
    Entry point as requested: ModelNew.
    """
    def __init__(self, hidden_dim: int = 768, num_experts: int = 256, n_group: int = 8, top_k: int = 8):
        super().__init__()
        # Keep constants for kernel invocations
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.n_group = n_group
        self.experts_per_group = num_experts // n_group
        self.top_k = top_k
        self.topk_group = 4

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)           # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim] and hidden [M, hidden_dim]."

        # Allocate outputs and intermediates
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)  # 0/1 mask, int32
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)

        # Triton kernel launches
        # 1) Matmul for logits
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid_matmul](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),
            num_warps=4,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M, N)](
            logits, scores,
            M, N, M, N,
            num_warps=1,
        )

        # 3) Add bias
        _add_bias_kernel[(M, N)](
            scores, bias, scores_for_routing,
            M, N,
            num_warps=1,
        )

        # 4) Group top-2 sum
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group, self.experts_per_group,
            num_warps=1,
        )

        # 5) Select top-4 groups
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            num_warps=1,
        )

        # 6) Build group mask (assume score_mask zero-initialized by host)
        # Host initializes score_mask to zeros
        score_mask.zero_()

        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, self.n_group, self.experts_per_group,
            num_warps=1,
        )

        # 7) Masked fill
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M, N)](
            scores_for_routing, score_mask, masked_scores,
            M, N, NEG_INF,
            num_warps=1,
        )

        # 8) Top-8 selection from masked scores
        _top8_selection_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N, self.top_k,
            num_warps=1,
        )

        # 9) Normalize and apply scaling factor
        scale = 1.0  # routed_scaling_factor from original code; default 1.0, keep as 1.0 for equivalence
        _normalize_and_scale_kernel[(M,)](
            top8_vals, scale, topk_weight,
            M, self.top_k,
            num_warps=1,
        )

        # Return indices and normalized weights
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
