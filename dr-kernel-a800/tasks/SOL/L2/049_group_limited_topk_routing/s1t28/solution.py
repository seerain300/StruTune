import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 2: Elementwise sigmoid on scores (FP32), out[i, :] = 1 / (1 + exp(-scores[i, :]))
@triton.jit
def _sigmoid_kernel(
    in_ptr, out_ptr,
    M, N,
    in_stride0, in_stride1, out_stride0, out_stride1,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    # bounds check
    if row >= M:
        return
    # pointers to the row
    in_row_ptr = in_ptr + row * in_stride0
    out_row_ptr = out_ptr + row * out_stride0

    # Iterate over columns
    for j in range(0, N):
        val = tl.load(in_row_ptr + j * in_stride1)
        val = 1.0 / (1.0 + tl.exp(-val))
        tl.store(out_row_ptr + j * out_stride1, val)


# Kernel 3: Add bias (vector of length N) to each row: out[i, :] = in[i, :] + bias
@triton.jit
def _add_bias_kernel(
    in_ptr, out_ptr, bias_ptr,
    M, N,
    in_stride0, in_stride1, out_stride0, out_stride1, bias_stride0,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    in_row_ptr = in_ptr + row * in_stride0
    out_row_ptr = out_ptr + row * out_stride0
    for j in range(0, N):
        val = tl.load(in_row_ptr + j * in_stride1)
        b = tl.load(bias_ptr + j * bias_stride0)
        tl.store(out_row_ptr + j * out_stride1, val + b)


# Kernel 4: Compute group top-2 sum per token, reshape [M, n_group, experts_per_group]
# scores_ptr: [M, N], group_scores_ptr: [M, n_group]
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N, n_group, experts_per_group,
    scores_stride0, scores_stride1, group_scores_stride0, group_scores_stride1,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    base = row * scores_stride0
    # For each group g in [0..n_group-1]
    for g in range(0, n_group):
        start = g * experts_per_group
        # Initialize top2
        top1 = tl.full((), -1.0e20, tl.float32)
        top2 = tl.full((), -1.0e20, tl.float32)
        # Loop over experts in this group
        for k in range(0, experts_per_group):
            idx = start + k
            val = tl.load(scores_ptr + base + idx * scores_stride1)
            # Update top2
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        group_sum = top1 + top2
        tl.store(group_scores_ptr + row * group_scores_stride0 + g * group_scores_stride1, group_sum)


# Kernel 5: Select top4 groups per token from group_scores [M, n_group] -> group_idx [M, 4]
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr,
    M, n_group,
    group_scores_stride0, group_scores_stride1,
    group_idx_stride0, group_idx_stride1,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    # Initialize selected_idx and max_val
    selected_idx = tl.zeros((4,), dtype=tl.int32)
    max_val = tl.full((4,), -1.0e20, tl.float32)
    # First pass: find top-4 indices
    for g in range(0, n_group):
        val = tl.load(group_scores_ptr + row * group_scores_stride0 + g * group_scores_stride1)
        # For k in 0..3: if val > max[k] and not selected, replace
        # Note: we can maintain top-4 in registers; since n_group=8, this is fine.
        # We use a simple iterative approach comparing to current maxima.
        # We'll implement a small helper to update top-4.
        # Here, we just iterate and update positions 0..3 if better.
        # This is a straightforward argmax over 4 slots.
        # We'll perform 4 comparisons sequentially.
        # Compare to position 0
        if val > max_val[0]:
            # Shift down maxima
            max_val[1] = max_val[0]
            max_val[2] = max_val[1]
            max_val[3] = max_val[2]
            max_val[0] = val
            selected_idx[1] = selected_idx[0]
            selected_idx[2] = selected_idx[1]
            selected_idx[3] = selected_idx[2]
            selected_idx[0] = g
        else:
            # Compare to position 1
            if val > max_val[1]:
                max_val[2] = max_val[1]
                max_val[1] = val
                selected_idx[2] = selected_idx[1]
                selected_idx[1] = g
            else:
                # Compare to position 2
                if val > max_val[2]:
                    max_val[3] = max_val[2]
                    max_val[2] = val
                    selected_idx[3] = selected_idx[2]
                    selected_idx[2] = g
                else:
                    # Compare to position 3
                    if val > max_val[3]:
                        max_val[3] = val
                        selected_idx[3] = g
    # Store result
    for k in range(0, 4):
        tl.store(group_idx_ptr + row * group_idx_stride0 + k * group_idx_stride1, selected_idx[k])


# Kernel 6: Build expert-level mask from selected group indices:
# score_mask[i, e] = 1 if e in selected groups else 0
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr, score_mask_ptr,
    M, N, n_group,
    group_idx_stride0, group_idx_stride1,
    score_mask_stride0, score_mask_stride1,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    # Initialize mask row to zeros
    base_mask = row * score_mask_stride0
    for j in range(0, N):
        tl.store(score_mask_ptr + base_mask + j * score_mask_stride1, 0)
    # Set 1 for selected groups
    # Each group has 32 experts
    ep = N // n_group
    for k in range(0, n_group):
        g = tl.load(group_idx_ptr + row * group_idx_stride0 + k * group_idx_stride1)
        start = g * ep
        for kk in range(0, ep):
            tl.store(score_mask_ptr + base_mask + (start + kk) * score_mask_stride1, 1)


# Kernel 7: Masked fill: masked_scores[i, e] = -inf if score_mask[i, e] == 0 else scores_for_routing[i, e]
@triton.jit
def _masked_fill_kernel(
    scores_ptr, mask_ptr, out_ptr,
    M, N,
    scores_stride0, scores_stride1, mask_stride0, mask_stride1, out_stride0, out_stride1,
    NEG_INF: tl.constexpr,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    scores_row_ptr = scores_ptr + row * scores_stride0
    mask_row_ptr = mask_ptr + row * mask_stride0
    out_row_ptr = out_ptr + row * out_stride0
    for j in range(0, N):
        s = tl.load(scores_row_ptr + j * scores_stride1)
        m = tl.load(mask_row_ptr + j * mask_stride1)
        # if mask == 0, set to NEG_INF, else keep s
        val = tl.where(m != 0, s, NEG_INF)
        tl.store(out_row_ptr + j * out_stride1, val)


# Kernel 8: Final top-8 selection from masked_scores (ignore non-selected by -inf), write indices and values
@triton.jit
def _final_top8_kernel(
    masked_ptr, out_idx_ptr, out_vals_ptr,
    M, N,
    masked_stride0, masked_stride1,
    out_idx_stride0, out_idx_stride1,
    out_vals_stride0, out_vals_stride1,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    base_mask = row * masked_stride0
    # Initialize top-8 arrays
    top_val = tl.full((8,), -1.0e20, tl.float32)
    top_idx = tl.zeros((8,), dtype=tl.int32)
    for j in range(0, N):
        val = tl.load(masked_ptr + base_mask + j * masked_stride1)
        # Update top-8
        for k in range(0, 8):
            if val > top_val[k]:
                # Shift down
                for l in range(7, k, -1):
                    top_val[l] = top_val[l - 1]
                    top_idx[l] = top_idx[l - 1]
                top_val[k] = val
                top_idx[k] = j
                break  # once inserted, move on
    # Store results
    for k in range(0, 8):
        tl.store(out_idx_ptr + row * out_idx_stride0 + k * out_idx_stride1, top_idx[k])
        tl.store(out_vals_ptr + row * out_vals_stride0 + k * out_vals_stride1, top_val[k])


# Kernel 9: Normalize top-8 selected values and apply scaling factor: out = vals / sum(vals+eps) * scale
@triton.jit
def _normalize_scale_kernel(
    vals_ptr, out_ptr,
    M, K,
    vals_stride0, vals_stride1, out_stride0, out_stride1,
    EPS: tl.constexpr, SCALING: tl.constexpr,
    num_warps: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    base = row * vals_stride0
    total = tl.full((), 0.0, tl.float32)
    for k in range(0, K):
        v = tl.load(vals_ptr + base + k * vals_stride1)
        total += v
    # total > 0 by construction (we select 8 valid), add EPS for stability
    total = total + EPS
    for k in range(0, K):
        v = tl.load(vals_ptr + base + k * vals_stride1)
        norm = v / total
        tl.store(out_ptr + base + k * out_stride1, norm * SCALING)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int = 768, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure Triton availability and CUDA
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")
        if hidden_states.shape[1] != self.hidden_dim:
            raise ValueError(f"hidden_states must have hidden_dim={self.hidden_dim}, got {hidden_states.shape[1]}")
        if weight.shape[0] != self.num_experts or weight.shape[1] != self.hidden_dim:
            raise ValueError(f"weight must be [num_experts={self.num_experts}, hidden_dim={self.hidden_dim}], got {tuple(weight.shape)}")

        # Make tensors contiguous and FP32 for computation
        hidden = hidden_states.contiguous().to(torch.float32)   # [M, hidden_dim]
        weight = weight.contiguous().to(torch.float32)          # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)       # [num_experts]

        M = hidden.shape[0]
        N = self.num_experts

        # 1) Compute logits using torch F.linear (for correctness and robustness)
        logits = F.linear(hidden, weight)  # [M, N], FP32

        # 2) Allocate intermediate outputs
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)   # 0/1
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)

        # 3) Triton sigmoid
        _sigmoid_kernel[(M,)](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1), scores.stride(0), scores.stride(1),
            num_warps=4,
        )

        # 4) Triton add bias
        _add_bias_kernel[(M,)](
            scores, scores_for_routing, bias,
            M, N,
            scores.stride(0), scores.stride(1), scores_for_routing.stride(0), scores_for_routing.stride(1), bias.stride(0),
            num_warps=4,
        )

        # 5) Triton group top-2 sum
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group, self.experts_per_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1), group_scores.stride(0), group_scores.stride(1),
            num_warps=1,
        )

        # 6) Triton select top-4 groups
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=1,
        )

        # 7) Triton build group mask
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N, self.n_group,
            group_idx.stride(0), group_idx.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            num_warps=1,
        )

        # 8) Triton masked fill
        _masked_fill_kernel[(M,)](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            NEG_INF=-1.0e20,
            num_warps=4,
        )

        # 9) Triton final top-8 selection (indices and values)
        _final_top8_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            top8_vals.stride(0), top8_vals.stride(1),
            num_warps=1,
        )

        # 10) Triton normalize + scale to produce topk_weight
        EPS = 1e-20
        _normalize_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, self.top_k,
            top8_vals.stride(0), top8_vals.stride(1), topk_weight.stride(0), topk_weight.stride(1),
            EPS=EPS, SCALING=self.routed_scaling_factor,
            num_warps=1,
        )

        # Return indices (int32) and normalized weights (float32)
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
