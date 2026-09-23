import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: elementwise sigmoid on a 2D tensor
@triton.jit
def _sigmoid_kernel(scores_ptr, out_ptr, M, N, stride_sm, stride_sn, stride_om, stride_on):
    m = tl.program_id(0)
    for e in range(0, N):
        x = tl.load(scores_ptr + m * stride_sm + e * stride_sn)
        # sigmoid: 1 / (1 + exp(-x))
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + m * stride_om + e * stride_on, y)


# Triton kernel: add expert bias (vector of size N) to scores (MxN), in-place to out
@triton.jit
def _add_bias_kernel(scores_ptr, bias_ptr, out_ptr, M, N, stride_sm, stride_sn, stride_bm, stride_bn, stride_om, stride_on):
    # We assume bias is 1D vector [N]; Triton can index by scalar. Here we use row-wise loop.
    m = tl.program_id(0)
    for e in range(0, N):
        x = tl.load(scores_ptr + m * stride_sm + e * stride_sn)
        b = tl.load(bias_ptr + e * stride_bn)  # bias[e]
        y = x + b
        tl.store(out_ptr + m * stride_om + e * stride_on, y)


# Triton kernel: compute group_scores = sum of top-2 per group for each token
# scores: [M, N], group_scores: [M, n_group]
@triton.jit
def _group_top2_sum_kernel(scores_ptr, group_scores_ptr, M, N, n_group, stride_sm, stride_sn, stride_gsm, stride_gsn, experts_per_group: tl.constexpr):
    m = tl.program_id(0)
    for g in range(0, n_group):
        start = g * experts_per_group
        best1 = -1.0e20
        best2 = -1.0e20
        for e in range(0, experts_per_group):
            idx = start + e
            val = tl.load(scores_ptr + m * stride_sm + idx * stride_sn)
            # update top-2
            if val > best1:
                best2 = best1
                best1 = val
            elif val > best2:
                best2 = val
        sum_top2 = best1 + best2
        tl.store(group_scores_ptr + m * stride_gsm + g * stride_gsn, sum_top2)


# Triton kernel: select top-4 group indices based on group_scores, out: [M, 4]
@triton.jit
def _select_top4_groups_kernel(group_scores_ptr, group_idx_ptr, M, n_group, stride_gsm, stride_gsn, stride_gim, stride_gin):
    m = tl.program_id(0)
    candidates = tl.zeros((4, 2), dtype=tl.float32)  # (idx, score)
    for i in range(0, 4):
        candidates[i, 1] = -1.0e20
    for g in range(0, n_group):
        score = tl.load(group_scores_ptr + m * stride_gsm + g * stride_gsn)
        pos = 0
        while pos < 4 and candidates[pos, 1] > score:
            pos += 1
        if pos < 4:
            for j in range(3, pos, -1):
                candidates[j, 0] = candidates[j-1, 0]
                candidates[j, 1] = candidates[j-1, 1]
            candidates[pos, 0] = g
            candidates[pos, 1] = score
    for i in range(0, 4):
        idx = int(candidates[i, 0])
        tl.store(group_idx_ptr + m * stride_gim + i * stride_gin, tl.full((1,), idx, tl.int32))


# Triton kernel: build expert-level mask from group_idx: score_mask[m, e] = 1 if e belongs to selected group else 0
@triton.jit
def _build_group_mask_kernel(group_idx_ptr, score_mask_ptr, M, N, stride_gim, stride_gin, stride_sms, stride_ssn, n_group: tl.constexpr, experts_per_group: tl.constexpr):
    m = tl.program_id(0)
    for e in range(0, N):
        g = e // experts_per_group
        found = tl.full((1,), 0, tl.int32)
        for k in range(0, 4):
            idx = tl.load(group_idx_ptr + m * stride_gim + k * stride_gin)
            if g == idx:
                found = tl.full((1,), 1, tl.int32)
                break
        tl.store(score_mask_ptr + m * stride_sms + e * stride_ssn, found)


# Triton kernel: masked fill: if score_mask == 0, set scores to -inf
@triton.jit
def _masked_fill_kernel(scores_ptr, score_mask_ptr, masked_scores_ptr, M, N, stride_sm, stride_sn, stride_sms, stride_ssn, stride_msm, stride_msn, NEG_INF: tl.constexpr):
    m = tl.program_id(0)
    for e in range(0, N):
        val = tl.load(scores_ptr + m * stride_sm + e * stride_sn)
        keep = tl.load(score_mask_ptr + m * stride_sms + e * stride_ssn)
        if keep == 0:
            val = NEG_INF
        tl.store(masked_scores_ptr + m * stride_msm + e * stride_msn, val)


# Triton kernel: final top-8 selection from masked_scores per token
@triton.jit
def _final_top8_kernel(masked_scores_ptr, top_idx_ptr, top_vals_ptr, M, N, stride_msm, stride_msn, stride_tim, stride_tin, stride_vim, stride_vin):
    m = tl.program_id(0)
    top8 = tl.zeros((8, 2), dtype=tl.float32)  # (index in [0..N-1], value)
    top8[:, 1] = -1.0e20
    for e in range(0, N):
        val = tl.load(masked_scores_ptr + m * stride_msm + e * stride_msn)
        pos = 0
        while pos < 8 and top8[pos, 1] > val:
            pos += 1
        if pos < 8:
            for j in range(7, pos, -1):
                top8[j, 0] = top8[j-1, 0]
                top8[j, 1] = top8[j-1, 1]
            top8[pos, 0] = tl.full((1,), e, tl.int32)
            top8[pos, 1] = val
    for i in range(0, 8):
        idx = int(top8[i, 0])
        tl.store(top_idx_ptr + m * stride_tim + i * stride_tin, tl.full((1,), idx, tl.int32))
        tl.store(top_vals_ptr + m * stride_vim + i * stride_vin, top8[i, 1])


# Triton kernel: normalize selected values and apply scaling factor
@triton.jit
def _normalize_scale_kernel(top_vals_ptr, topk_weight_ptr, M, stride_tv, stride_tvn, stride_tw, stride_twn, scale_factor: tl.constexpr):
    m = tl.program_id(0)
    total = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, 8):
        val = tl.load(top_vals_ptr + m * stride_tv + i * stride_tvn)
        total += val
    total = tl.maximum(total, 1e-20)
    inv_total = 1.0 / total
    for i in range(0, 8):
        val = tl.load(top_vals_ptr + m * stride_tv + i * stride_tvn)
        val = val * inv_total * scale_factor
        tl.store(topk_weight_ptr + m * stride_tw + i * stride_twn, val)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.num_experts = 256
        self.hidden_dim = hidden_dim
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # If Triton/CUDA unavailable, fallback to original PyTorch (optional, but environment requires Triton)
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            # Fallback path: compute exactly like original, but ensure correctness
            logits = F.linear(hidden_states.to(torch.float32), weight.to(torch.float32))  # [M, 256]
            scores = torch.sigmoid(logits)                                       # [M, 256]
            scores_for_routing = scores + expert_bias.to(torch.float32)          # [M, 256]
            # Group top-2 sum
            group_scores_reshaped = scores_for_routing.view(-1, self.n_group, self.experts_per_group)
            top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
            group_scores = top2_vals.sum(dim=-1)                                # [M, 8]
            # Select top-4 groups
            _, group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)  # [M, 4]
            # Build expert-level mask
            score_mask = torch.zeros((group_scores.shape[0], self.num_experts), dtype=torch.int32, device=hidden_states.device)
            for g in range(self.n_group):
                start = g * self.experts_per_group
                for k in range(self.topk_group):
                    idx = int(group_idx[:, k])  # already int64 from torch.topk
                    if idx == g:
                        score_mask[:, start:start + self.experts_per_group] = 1
                        break
            # Masked fill
            masked_scores = scores_for_routing.masked_fill(score_mask == 0, -1.0e20)
            # Final top-8 selection
            _, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)  # [M, 8]
            # Gather selected logits
            selected_scores = F.linear(hidden_states, weight)  # gather logits, not masked
            selected_scores = selected_scores[:, topk_idx]                             # [M, 8]
            # Normalize and scale
            denom = selected_scores.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = (selected_scores / denom) * self.routed_scaling_factor
            return topk_idx, topk_weight

        # Triton path: compute with Triton kernels
        # Compute logits using PyTorch F.linear to ensure correctness across hidden_dim variations
        logits = F.linear(hidden_states.to(torch.float32), weight.to(torch.float32))  # [M, 256]

        num_tokens, num_experts = logits.shape
        assert num_experts == self.num_experts, "Expected num_experts == 256"
        assert self.num_experts == self.n_group * self.experts_per_group, "Grouping must divide num_experts"

        # Prepare outputs and intermediates
        scores = torch.empty_like(logits)
        # Sigmoid in Triton
        _sigmoid_kernel[(num_tokens,)](
            logits, scores,
            num_tokens, num_experts,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # Add expert bias in Triton
        scores_for_routing = torch.empty_like(scores)
        _add_bias_kernel[(num_tokens,)](
            scores, expert_bias.to(torch.float32), scores_for_routing,
            num_tokens, num_experts,
            scores.stride(0), scores.stride(1),
            expert_bias.stride(0), expert_bias.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
        )

        # Group top-2 sum
        group_scores = torch.empty((num_tokens, self.n_group), dtype=torch.float32, device=scores.device)
        _group_top2_sum_kernel[(num_tokens,)](
            scores_for_routing, group_scores,
            num_tokens, num_experts, self.n_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            self.experts_per_group,
        )

        # Select top-4 groups
        group_idx = torch.empty((num_tokens, self.topk_group), dtype=torch.int32, device=scores.device)
        _select_top4_groups_kernel[(num_tokens,)](
            group_scores, group_idx,
            num_tokens, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
        )

        # Build expert-level mask (int32, 0/1)
        score_mask = torch.empty((num_tokens, num_experts), dtype=torch.int32, device=scores.device)
        _build_group_mask_kernel[(num_tokens,)](
            group_idx, score_mask,
            num_tokens, num_experts,
            group_idx.stride(0), group_idx.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            self.n_group,
            self.experts_per_group,
        )

        # Masked fill: set non-selected to -inf
        masked_scores = torch.empty_like(scores_for_routing)
        NEG_INF = -1.0e20
        _masked_fill_kernel[(num_tokens,)](
            scores_for_routing, score_mask, masked_scores,
            num_tokens, num_experts,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            NEG_INF,
        )

        # Final top-8 selection from masked_scores
        top8_idx = torch.empty((num_tokens, self.top_k), dtype=torch.int32, device=scores.device)
        top8_vals = torch.empty((num_tokens, self.top_k), dtype=torch.float32, device=scores.device)
        _final_top8_kernel


def run(*args):
    return ModelNew()(*args)
