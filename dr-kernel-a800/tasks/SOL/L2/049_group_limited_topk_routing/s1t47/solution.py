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
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ids for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the current tile's A and B blocks
    A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Masks to avoid out-of-bounds
        a_mask = (offs_m[:, None] < M) & (k + offs_k[None, :] < K)
        b_mask = (k + offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        # Accumulate
        acc += tl.dot(a, b)
        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store results
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


# Kernel 2: Sigmoid elementwise
@triton.jit
def _sigmoid_kernel(
    inp_ptr, out_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n = pid_n
    if m >= M or n >= N:
        return
    val = tl.load(inp_ptr + m * stride_im + n * stride_in)
    val = 1.0 / (1.0 + tl.exp(-val))
    tl.store(out_ptr + m * stride_om + n * stride_on, val)


# Kernel 3: Add expert bias (bias is length N)
@triton.jit
def _add_bias_kernel(
    inp_ptr, bias_ptr, out_ptr,
    M, N,
    stride_im, stride_in,
    stride_on,
    stride_b,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n = pid_n
    if m >= M or n >= N:
        return
    val = tl.load(inp_ptr + m * stride_im + n * stride_in)
    b = tl.load(bias_ptr + n * stride_b)
    tl.store(out_ptr + m * stride_on + n * stride_on, val + b)


# Kernel 4: Per-group top-2 sum over groups of 32 experts
# Input: scores_for_routing [M, N], Output: group_scores [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    inp_ptr, out_ptr,
    M, N, n_group, experts_per_group,
    stride_im, stride_in,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    # Iterate groups
    for g in range(0, n_group):
        base = g * experts_per_group
        max1 = -1e20
        max2 = -1e20
        for e in range(0, experts_per_group):
            idx = base + e
            if idx >= N:
                break
            val = tl.load(inp_ptr + m * stride_im + idx * stride_in)
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
        tl.store(out_ptr + m * stride_om + g * stride_on, max1 + max2)


# Kernel 5: Select top-4 groups (iterative argmax) from group_scores [M, 8]
@triton.jit
def _select_top4_groups_kernel(
    inp_ptr, out_ptr,
    M, n_group,
    stride_im, stride_in,
    stride_om, stride_on,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    gs = inp_ptr + m * stride_im  # pointer to start of group_scores row
    out_row = out_ptr + m * stride_om
    top_vals = tl.zeros((n_group,), dtype=tl.float32) - 1e20
    top_idx = tl.zeros((n_group,), dtype=tl.int32)
    for g in range(0, n_group):
        val = tl.load(gs + g * stride_in)
        idx = g
        # Bubble to the top positions
        for j in range(0, n_group):
            if j >= 4:
                break
            if val > top_vals[j]:
                # Shift bigger values to the right
                for k in range(3, j - 1, -1):
                    top_vals[k] = top_vals[k - 1]
                    top_idx[k] = top_idx[k - 1]
                top_vals[j] = val
                top_idx[j] = idx
                break
    # Write indices
    for j in range(0, 4):
        tl.store(out_row + j * stride_on, top_idx[j])


# Kernel 6: Build expert-level mask: 1 for selected groups, 0 otherwise
# Input: group_idx [M, 4], Output: score_mask [M, N] (int32)
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr, score_mask_ptr,
    M, N, n_group, experts_per_group,
    stride_mg, stride_ge, stride_mn, stride_ne,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    # Iterate selected groups
    for g in range(0, 4):
        g_idx = tl.load(group_idx_ptr + m * stride_mg + g * stride_ge)
        base = g_idx * experts_per_group
        for e in range(0, experts_per_group):
            idx = base + e
            if idx < N:
                ptr = score_mask_ptr + m * stride_mn + idx * stride_ne
                one = tl.full((1,), 1, dtype=tl.int32)
                tl.store(ptr, one)


# Kernel 7: Masked fill: set non-selected experts to -inf
@triton.jit
def _masked_fill_kernel(
    inp_ptr, mask_ptr, out_ptr,
    M, N,
    stride_im, stride_in,
    stride_mn, stride_ne,
    stride_on,
    NEG_INF: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    for n in range(0, N):
        sel = tl.load(mask_ptr + m * stride_mn + n * stride_ne)
        val = tl.load(inp_ptr + m * stride_im + n * stride_in)
        if sel == 0:
            val = NEG_INF
        tl.store(out_ptr + m * stride_on + n * stride_on, val)


# Kernel 8: Final top-8 selection from masked_scores [M, N]
@triton.jit
def _final_top8_kernel(
    inp_ptr, idx_ptr, vals_ptr,
    M, N,
    stride_im, stride_in,
    stride_im_idx, stride_in_idx,
    stride_im_vals, stride_in_vals,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    row_ptr = inp_ptr + m * stride_im
    row_idx_ptr = idx_ptr + m * stride_im_idx
    row_vals_ptr = vals_ptr + m * stride_im_vals
    # Iterative argmax 8 times
    for j in range(0, 8):
        max_val = -1e20
        max_idx = 0
        for n in range(0, N):
            val = tl.load(row_ptr + n * stride_in)
            if val > max_val:
                max_val = val
                max_idx = n
        tl.store(row_vals_ptr + j * stride_in_vals, max_val)
        tl.store(row_idx_ptr + j * stride_in_idx, max_idx)
        # Mark selected by setting to -inf
        if max_idx < N:
            tl.store(row_ptr + max_idx * stride_in, -1e20)


# Kernel 9: Normalize and scale selected routing weights
@triton.jit
def _normalize_scale_kernel(
    vals_ptr, weight_ptr, out_ptr,
    M, K,
    stride_vm, stride_vk,
    stride_om, stride_on,
    scale: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    row_vals_ptr = vals_ptr + m * stride_vm
    row_weight_ptr = weight_ptr + m * stride_om
    row_out_ptr = out_ptr + m * stride_om
    # Sum selected vals (top-8)
    total = tl.zeros((), dtype=tl.float32)
    for j in range(0, 8):
        val = tl.load(row_vals_ptr + j * stride_vk)
        total += val
    # Normalize and scale
    total = total + 1e-20  # epsilon
    for j in range(0, 8):
        val = tl.load(row_vals_ptr + j * stride_vk)
        out = val / total * scale
        tl.store(row_out_ptr + j * stride_on, out)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int = 768, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward. No torch ops in host code.
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        hidden = hidden_states.contiguous().to(torch.float32)       # [M, hidden_dim]
        weight = weight.contiguous().to(torch.float32)              # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)           # [num_experts]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim], hidden [M, hidden_dim]."

        # Allocate intermediates and outputs
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)  # expert-level mask
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)

        # 1) Triton matmul for logits = hidden @ weight^T
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M, N)](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # 3) Add bias
        _add_bias_kernel[(M, N)](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores.stride(1),
            scores_for_routing.stride(0),
            bias.stride(0),
        )

        # 4) Group top-2 sum: [M, 8]
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group, self.experts_per_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
        )

        # 5) Select top-4 groups: [M, 4]
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
        )

        # 6) Build expert-level mask: [M, N]
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N, self.n_group, self.experts_per_group,
            group_idx.stride(0), group_idx.stride(1),  # strides for group_idx (M, 4) here we use strides(0,1) as int? We pass ints
            score_mask.stride(0), score_mask.stride(1),
        )

        # Note: The above kernel uses strides from group_idx and score_mask. For group_idx (M,4), stride(0)=4, stride(1)=1.
        # We pass them explicitly:
        stride_mg = group_idx.stride(0)
        stride_ge = group_idx.stride(1)
        stride_mn = score_mask.stride(0)
        stride_ne = score_mask.stride(1)

        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N, self.n_group, self.experts_per_group,
            stride_mg, stride_ge, stride_mn, stride_ne,
        )

        # 7) Masked fill: set non-selected to -inf
        _masked_fill_kernel[(M, N)](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            masked_scores.stride(0),
            NEG_INF=-1e20,
        )

        # 8) Final top-8 selection from masked_scores
        _final_top8_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            top8_vals.stride(0), top8_vals.stride(1),
        )

        # 9) Normalize and scale selected routing weights
        _normalize_scale_kernel[(M,)](
            top8_vals, topk_weight, topk_weight,
            M, self.top_k,
            top8_vals.stride(0), top8_vals.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            scale=self.routed_scaling_factor,
        )

        # Return selected indices and normalized weights
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
