import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Matmul for logits = hidden @ weight^T
# hidden: [M, K] = [num_tokens, hidden_dim], weight: [N, K] = [num_experts, hidden_dim], out: [M, N] = [num_tokens, num_experts]
@triton.jit
def _matmul_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, N, K,           # int32 sizes
    stride_hm, stride_hk,   # strides for hidden
    stride_wk, stride_wn,   # strides for weight
    stride_om, stride_on,   # strides for out
):
    # 2D grid over tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * 64
    n0 = pid_n * 64

    offs_m = m0 + tl.arange(0, 64)
    offs_n = n0 + tl.arange(0, 64)
    offs_k = tl.arange(0, 64)

    # Initialize accumulator
    acc = tl.zeros((64, 64), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, 64):
        k = k0 + offs_k
        # Compute pointers for hidden and weight tiles
        hidden_tile_ptr = hidden_ptr + (offs_m[:, None] * stride_hm) + (k[None, :] * stride_hk)  # [64, 64]
        weight_tile_ptr = weight_ptr + (k[:, None] * stride_wk) + (offs_n[None, :] * stride_wn)  # [64, 64]

        # Masks for boundary
        mask_h = (offs_m[:, None] < M) & (k[None, :] < K)
        mask_w = (k[:, None] < K) & (offs_n[None, :] < N)

        h = tl.load(hidden_tile_ptr, mask=mask_h, other=0.0)  # [64, 64]
        w = tl.load(weight_tile_ptr, mask=mask_w, other=0.0)  # [64, 64]
        acc += tl.dot(h, w)

    # Write back to output
    out_tile_ptr = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_tile_ptr, acc, mask=mask_out)


# Kernel 2: Sigmoid elementwise
@triton.jit
def _sigmoid_kernel(
    in_ptr, out_ptr, M, N,
    stride_im, stride_in, stride_om, stride_on,
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < (M * N)
    in_tile_ptr = in_ptr + (offs * stride_im)
    out_tile_ptr = out_ptr + (offs * stride_om)
    x = tl.load(in_tile_ptr, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_tile_ptr, y, mask=mask)


# Kernel 3: Add expert bias elementwise over N (num_experts)
@triton.jit
def _add_bias_kernel(
    scores_ptr, bias_ptr, out_ptr, M, N,
    stride_sm, stride_sn, stride_on,
    stride_bn,
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < (M * N)
    scores_tile_ptr = scores_ptr + (offs * stride_sm)
    out_tile_ptr = out_ptr + (offs * stride_on)
    s = tl.load(scores_tile_ptr, mask=mask, other=0.0)
    b = tl.load(bias_ptr + (offs % N) * stride_bn, mask=mask, other=0.0)  # broadcast bias over M
    y = s + b
    tl.store(out_tile_ptr, y, mask=mask)


# Kernel 4: Compute per-group top-2 sum -> group_scores [M, 8]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N, n_group, experts_per_group,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
):
    m = tl.program_id(0)  # one program per token
    # Iterate over groups: for each g in [0..7], find top-2 among 32 experts and sum
    for g in range(0, 8):
        start = g * 32
        group_total = 0.0
        # Loop over 32 experts in this group
        for i in range(0, 32):
            idx = start + i
            val = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
            # For simplicity, we maintain the top-2 using pairwise comparisons; we'll just store the max and second max via scanning.
            # Initialize candidates
            best1 = val
            best2 = -1.0e20
            # Scan remaining 31 positions (i+1..31)
            for j in range(i + 1, 32):
                cand = tl.load(scores_ptr + m * stride_sm + (start + j) * stride_sn)
                # Update best1 and best2
                if cand > best1:
                    best2 = best1
                    best1 = cand
                elif cand > best2:
                    best2 = cand
            group_total += best1 + best2
        # Store group total for this token
        tl.store(group_scores_ptr + m * stride_gm + g * stride_gn, group_total)


# Kernel 5: Select top-4 group indices per token
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr,
    M, n_group, topk_group,
    stride_gs_m, stride_gs_n,
):
    m = tl.program_id(0)  # one program per token
    # Iteratively argmax over 8 groups, write indices
    for k in range(0, 4):
        max_val = -1.0e20
        max_idx = 0
        for g in range(0, 8):
            val = tl.load(group_scores_ptr + m * stride_gs_m + g * stride_gs_n)
            if val > max_val:
                max_val = val
                max_idx = g
        # Write selected index
        tl.store(group_idx_ptr + m * 4 + k, max_idx)
        # Set selected group score to -inf for next selection
        tl.store(group_scores_ptr + m * stride_gs_m + max_idx * stride_gs_n, -1.0e20)


# Kernel 6: Build expert-level group mask from group_idx
# group_idx: [M, 4], score_mask: [M, 256], set 1 for selected groups' 32 experts, else 0
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr, score_mask_ptr,
    M, n_group, topk_group, experts_per_group,
    stride_gm, stride_gn,  # group_idx strides
    stride_smm, stride_smn,  # score_mask strides
):
    m = tl.program_id(0)
    for k in range(0, 4):
        g = tl.load(group_idx_ptr + m * topk_group + k)  # int32
        start = g * 32
        for i in range(0, 32):
            e = start + i
            mask_ptr = score_mask_ptr + m * stride_smm + e * stride_smn
            tl.store(mask_ptr, 1)
    # Initialize remaining entries to 0
    # score_mask is int32: set zeros for non-selected groups' 32 positions (already set above)


# Kernel 7: Masked fill: set masked_scores[i, e] = -inf if score_mask[i, e] == 0 else scores_for_routing[i, e]
@triton.jit
def _masked_fill_kernel(
    scores_ptr, score_mask_ptr, masked_ptr,
    M, N,
    stride_sm_m, stride_sm_n, stride_mm_m, stride_mm_n,
    stride_smm, stride_smn,
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < (M * N)
    scores_tile_ptr = scores_ptr + offs * stride_sm_m
    mask_tile_ptr = score_mask_ptr + offs * stride_smm
    masked_tile_ptr = masked_ptr + offs * stride_mm_m
    s = tl.load(scores_tile_ptr, mask=mask, other=0.0)
    m = tl.load(mask_tile_ptr, mask=mask, other=0)  # int32
    neg_inf = -1.0e20
    y = tl.where(m != 0, s, neg_inf)
    tl.store(masked_tile_ptr, y, mask=mask)


# Kernel 8: Final top-8 selection from masked_scores per token
@triton.jit
def _top8_select_kernel(
    masked_ptr, top8_idx_ptr, top8_vals_ptr,
    M, N,
    stride_mm_m, stride_mm_n,
):
    m = tl.program_id(0)
    best_vals = tl.full((8,), -1.0e20, dtype=tl.float32)
    best_idxs = tl.full((8,), 0, dtype=tl.int32)
    for e in range(0, N):
        val = tl.load(masked_ptr + m * stride_mm_m + e * stride_mm_n)
        for j in range(0, 8):
            if val > best_vals[j]:
                # Shift down
                for l in range(7, j, -1):
                    best_vals[l] = best_vals[l - 1]
                    best_idxs[l] = best_idxs[l - 1]
                best_vals[j] = val
                best_idxs[j] = e
                break
    # Store indices and values
    for j in range(0, 8):
        tl.store(top8_idx_ptr + m * 8 + j, best_idxs[j])
        tl.store(top8_vals_ptr + m * 8 + j, best_vals[j])


# Kernel 9: Normalize selected values and apply scaling factor
@triton.jit
def _normalize_and_scale_kernel(
    top8_vals_ptr, topk_weight_ptr,
    M, top_k, scaling_factor,
    stride_v_m, stride_v_n,
):
    m = tl.program_id(0)
    total = 0.0
    for j in range(0, top_k):
        val = tl.load(top8_vals_ptr + m * stride_v_m + j * stride_v_n)
        total += val
    inv = 1.0 / (total + 1e-20)
    for j in range(0, top_k):
        val = tl.load(top8_vals_ptr + m * stride_v_m + j * stride_v_n)
        w = val * inv * scaling_factor
        tl.store(topk_weight_ptr + m * stride_v_m + j * stride_v_n, w)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int = 768, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = num_experts = 256 // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward: ensure tensors are on CUDA
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Make inputs contiguous and FP32
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
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)  # expert-level mask
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernels
        # 1) Matmul for logits
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _matmul_kernel[grid](
            hidden, weight, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            num_warps=4,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M * N,)](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=4,
        )

        # 3) Add bias
        _add_bias_kernel[(M * N,)](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            bias.stride(0),
            num_warps=4,
        )

        # 4) Group top-2 sum
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group, self.experts_per_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            num_warps=2,
        )

        # 5) Select top-4 groups
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group, self.topk_group,
            group_scores.stride(0), group_scores.stride(1),
            num_warps=2,
        )

        # 6) Build group mask
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, self.n_group, self.topk_group, self.experts_per_group,
            group_idx.stride(0), group_idx.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            num_warps=2,
        )

        # 7) Masked fill to -inf for non-selected experts
        _masked_fill_kernel[(M * N,)](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            num_warps=4,
        )

        # 8) Final top-8 selection from masked_scores
        _top8_select_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=4,
        )

        # 9) Normalize and scale
        _normalize_and_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, self.top_k, self.routed_scaling_factor,
            top8_vals.stride(0), top8_vals.stride(1),
            num_warps=2,
        )

        # Return indices (int32) and weights (float32)
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
