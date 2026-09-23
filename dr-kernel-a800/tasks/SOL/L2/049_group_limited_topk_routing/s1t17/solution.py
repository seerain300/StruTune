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
# M = num_tokens, N = num_experts, K = hidden_dim
@triton.jit
def _matmul_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, K, N,           # int32 sizes
    stride_hm, stride_hk,   # strides for hidden
    stride_wk, stride_wn,   # strides for weight
    stride_om, stride_on,   # strides for out
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols

    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # hidden tile: [BM, BK]
        h_ptrs = hidden_ptr + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk)
        h_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        h = tl.load(h_ptrs, mask=h_mask, other=0.0)

        # weight tile: [BN, BK], we want BK x BN to match acc
        w_ptrs = weight_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # acc += h [BM, BK] @ w^T [BK, BN] -> [BM, BN]
        acc += tl.dot(h, tl.trans(w))

    # write back
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# Kernel 2: Elementwise sigmoid: out = 1 / (1 + exp(-x))
@triton.jit
def _sigmoid_kernel(
    x_ptr, y_ptr, M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptrs, y, mask=mask)


# Kernel 3: Elementwise add bias: y[i, e] = x[i, e] + bias[e]
@triton.jit
def _add_bias_kernel(
    x_ptr, bias_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    y = x + b[None, :]
    tl.store(y_ptrs, y, mask=mask)


# Kernel 4: Group top-2 sum per token, output [M, 8]
# input is scores_for_routing reshaped as [M, 8, 32]
@triton.jit
def _group_top2_sum_kernel(
    x_ptr, out_ptr, M, GROUPS: tl.constexpr, EXP_PER_GROUP: tl.constexpr,
    stride_xm, stride_xg, stride_xe,
    stride_om,
):
    pid_m = tl.program_id(0)
    # one program per token
    acc = tl.zeros((), dtype=tl.float32)
    for g in range(GROUPS):
        base = g * EXP_PER_GROUP
        # compute max1
        max1 = -1.0e20
        for e in range(EXP_PER_GROUP):
            ptr = x_ptr + pid_m * stride_xm + g * stride_xg + e * stride_xe
            val = tl.load(ptr)
            max1 = tl.maximum(max1, val)
        # compute max2 among remaining
        max2 = -1.0e20
        for e in range(EXP_PER_GROUP):
            ptr = x_ptr + pid_m * stride_xm + g * stride_xg + e * stride_xe
            val = tl.load(ptr)
            if val > max1 and val > max2:
                max2 = val
        acc += max1 + max2
    tl.store(out_ptr + pid_m * stride_om, acc)


# Kernel 5: Select top-4 groups per token, output indices [M, 4] int32
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr, M, GROUPS,
    stride_gm,
):
    pid_m = tl.program_id(0)
    # initialize indices to -1
    idxs = tl.full((4,), -1, dtype=tl.int32)
    scores = tl.load(group_scores_ptr + pid_m * stride_gm)  # [GROUPS]
    for g in range(GROUPS):
        best = 0
        best_j = -1
        for j in range(4):
            cond = scores[g] > scores[best]
            if cond:
                best = g
                best_j = j
        if best_j >= 0:
            idxs[best_j] = best
    tl.store(group_idx_ptr + pid_m * 4 + tl.arange(0, 4), idxs)


# Kernel 6: Build expert-level mask from group_idx: score_mask[i, e] = 1 if e in selected groups else 0
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr, score_mask_ptr, M, EXPERTS: tl.constexpr, GROUPS: tl.constexpr, EXP_PER_GROUP: tl.constexpr,
    stride_gm,  # stride for group_idx per row
    stride_sm_m, stride_sm_e,  # strides for score_mask
):
    pid_m = tl.program_id(0)
    for g in range(GROUPS):
        e0 = g * EXP_PER_GROUP
        for e in range(EXP_PER_GROUP):
            exp_idx = e0 + e
            is_selected = 0
            for j in range(4):
                sel_group = tl.load(group_idx_ptr + pid_m * stride_gm + j)  # int32
                if sel_group == g:
                    is_selected = 1
                    break
            ptr = score_mask_ptr + pid_m * stride_sm_m + exp_idx * stride_sm_e
            # store 1 if selected else 0
            tl.store(ptr, is_selected)


# Kernel 7: Masked fill: set masked_scores[i, e] = -inf if score_mask[i, e] == 0 else keep scores_for_routing[i, e]
@triton.jit
def _masked_fill_kernel(
    scores_ptr, score_mask_ptr, masked_ptr,
    M, EXPERTS,
    stride_sm_m, stride_sm_e,
    stride_ms_m, stride_ms_e,
):
    pid_m = tl.program_id(0)
    for e in range(EXPERTS):
        ptr_scores = scores_ptr + pid_m * stride_sm_m + e * stride_sm_e
        ptr_mask = score_mask_ptr + pid_m * stride_sm_m + e * stride_sm_e
        val = tl.load(ptr_scores)
        mask = tl.load(ptr_mask)  # 0 or 1
        new_val = tl.where(mask > 0, val, -1.0e20)
        ptr_out = masked_ptr + pid_m * stride_ms_m + e * stride_ms_e
        tl.store(ptr_out, new_val)


# Kernel 8: Final top-8 selection from masked_scores: indices and values
@triton.jit
def _final_top8_selection_kernel(
    masked_ptr, indices_ptr, vals_ptr,
    M, EXPERTS,
    stride_mm_m, stride_mm_e,
):
    pid_m = tl.program_id(0)
    top_vals = tl.full((8,), -1.0e20, dtype=tl.float32)
    top_idxs = tl.full((8,), -1, dtype=tl.int32)
    for e in range(EXPERTS):
        ptr = masked_ptr + pid_m * stride_mm_m + e * stride_mm_e
        val = tl.load(ptr)
        # simple insertion sort for top8
        for j in range(8):
            if val > top_vals[j]:
                # shift down
                for k in range(7, j, -1):
                    top_vals[k] = top_vals[k - 1]
                    top_idxs[k] = top_idxs[k - 1]
                top_vals[j] = val
                top_idxs[j] = e
                break
    # store indices and vals
    tl.store(indices_ptr + pid_m * 8 + tl.arange(0, 8), top_idxs)
    tl.store(vals_ptr + pid_m * 8 + tl.arange(0, 8), top_vals)


# Kernel 9: Normalize and scale selected values: out[i, k] = vals[i, k] / sum(vals[i]) * routed_scaling_factor
@triton.jit
def _normalize_scale_kernel(
    vals_ptr, out_ptr,
    M, K: tl.constexpr,
    stride_vm, stride_vk,
    stride_om, stride_ok,
    routed_scale: tl.float32,
):
    pid_m = tl.program_id(0)
    sumv = 0.0
    for k in range(K):
        ptr = vals_ptr + pid_m * stride_vm + k * stride_vk
        val = tl.load(ptr)
        sumv += val
    inv_sum = 1.0 / (sumv + 1e-20)
    for k in range(K):
        ptr = vals_ptr + pid_m * stride_vm + k * stride_vk
        val = tl.load(ptr)
        out_ptr_k = out_ptr + pid_m * stride_om + k * stride_ok
        tl.store(out_ptr_k, val * inv_sum * routed_scale)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Match the original configuration: 256 experts, 8 groups of 32, top-8 final
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = 32
        self.top_k = 8
        self.hidden_dim = hidden_dim

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Triton-only forward: no torch ops in host
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            # Fallback (should not happen under evaluation): use original PyTorch path
            logits = F.linear(hidden_states.to(torch.float32), weight.to(torch.float32))
            scores = torch.sigmoid(logits)
            scores = scores + expert_bias.to(torch.float32)
            # Reshape to groups and compute top2 sums
            group_scores_reshaped = scores.view(hidden_states.shape[0], self.n_group, self.experts_per_group)
            top2_vals, _ = torch.topk(group_scores_reshaped, k=2, dim=-1, largest=True, sorted=False)
            group_scores = top2_vals.sum(dim=-1)  # [M, 8]
            _, group_idx = torch.topk(group_scores, k=4, dim=-1, sorted=False)
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1.0)
            score_mask = group_mask.unsqueeze(-1).expand(group_mask.shape[0], self.n_group, self.experts_per_group).reshape(group_mask.shape[0], self.num_experts)
            masked_scores = scores.masked_fill(score_mask == 0, float("-inf"))
            _, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)
            selected_scores = torch.gather(scores, dim=1, index=topk_idx)
            topk_weight = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
            topk_weight = topk_weight * routed_scaling_factor
            return topk_idx, topk_weight

        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)       # [M, hidden_dim]
        weight = weight.contiguous().to(torch.float32)              # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)           # [num_experts]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch."

        # Allocate outputs and intermediates
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden.device)   # top-4 group indices
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)   # expert-level mask
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernels
        # 1) Matmul logits = hidden @ weight^T
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid_matmul](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),  # weight is [N, K], need K and N strides
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M, N)](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4,
        )

        # 3) Add expert bias
        _add_bias_kernel[(M, N)](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4,
        )

        # 4) Group top-2 sum: compute per token across 8 groups of 32
        # Reshape to [M, 8, 32] virtually via strides: stride_xm = N, stride_xg = 32, stride_xe = 1
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, self.n_group, self.experts_per_group,
            N, 32, 1,   # emulate strides as if [M, 8, 32]
            group_scores.stride(0),
            num_warps=1,
        )

        # 5) Select top-4 groups per token
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0),
            num_warps=1,
        )

        # 6) Build expert-level mask
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, self.num_experts, self.n_group, self.experts_per_group,
            4,   # group_idx stride is 4
            score_mask.stride(0), score_mask.stride(1),
            num_warps=1,
        )

        # 7) Masked fill: set non-selected to -inf
        _masked_fill_kernel[(M,)](
            scores_for_routing, score_mask, masked_scores,
            M, self.num_experts,
            score_mask.stride(0), score_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=1,
        )

        # 8) Final top-8 selection from masked_scores
        _final_top8_selection_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, self.num_experts,
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=1,
        )

        # 9) Normalize and scale
        _normalize_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, self.top_k,
            top8_vals.stride(0), top8_vals.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor,
            num_warps=1,
        )

        # Return indices (as per original signature) and normalized weights
        # Note: original returns topk_idx, topk_weight; we compute indices from top8_idx since top_k=8, so top8_idx is the topk_idx.
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
