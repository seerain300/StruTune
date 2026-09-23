import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Matmul for logits = hidden @ weight^T
# hidden: [M, K] = [num_tokens, hidden_dim]
# weight: [N, K] = [num_experts, hidden_dim]
# out: [M, N] = [num_tokens, num_experts]
@triton.jit
def _matmul_kernel(
    hidden_ptr, weight_ptr, out_ptr,
    M, N, K,
    stride_hm, stride_hk, stride_wk, stride_wn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        # hidden tile: [BLOCK_M, BLOCK_K]
        h_ptrs = hidden_ptr + rm[:, None] * stride_hm + rk[None, :] * stride_hk
        # weight tile (transposed use): we want weight[k, n] loaded as [BLOCK_K, BLOCK_N]
        w_ptrs = weight_ptr + rk[:, None] * stride_wk + rn[None, :] * stride_wn
        h = tl.load(h_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        # accumulate
        acc += tl.dot(h, w)
    # store
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


# Kernel 2: Sigmoid
# in: [M, N], out: [M, N]
@triton.jit
def _sigmoid_kernel(in_ptr, out_ptr, M, N, stride_im, stride_in, stride_om, stride_on):
    for m in range(0, M):
        for n in range(0, N):
            x = tl.load(in_ptr + m * stride_im + n * stride_in)
            y = 1.0 / (1.0 + tl.exp(-x))
            tl.store(out_ptr + m * stride_om + n * stride_on, y)


# Kernel 3: Add bias (vector) to each column
# in: [M, N], bias: [N], out: [M, N]
@triton.jit
def _add_bias_kernel(in_ptr, bias_ptr, out_ptr, M, N, stride_im, stride_in, stride_om, stride_on):
    for m in range(0, M):
        for n in range(0, N):
            x = tl.load(in_ptr + m * stride_im + n * stride_in)
            b = tl.load(bias_ptr + n)
            y = x + b
            tl.store(out_ptr + m * stride_om + n * stride_on, y)


# Kernel 4: Group top-2 sum for 8 groups of 32
# inp: [M, N], out: [M, G] where G=8
@triton.jit
def _group_top2_sum_kernel(inp_ptr, out_ptr, M, N, G, EPG, stride_im, stride_in, stride_om, stride_on):
    # EPG = 32
    for m in range(0, M):
        group_scores = tl.zeros((G,), dtype=tl.float32)
        for g in range(0, G):
            base = g * EPG
            max1 = -float('inf')
            max2 = -float('inf')
            for e in range(0, EPG):
                idx = base + e
                val = tl.load(inp_ptr + m * stride_im + idx * stride_in)
                if val > max1:
                    max2 = max1
                    max1 = val
                elif val > max2:
                    max2 = val
            group_scores[g] = max1 + max2
        for g in range(0, G):
            tl.store(out_ptr + m * stride_om + g * stride_on, group_scores[g])


# Kernel 5: Select top-4 groups (argmax indices) from group_scores
# group_scores: [M, G], out: [M, 4]
@triton.jit
def _select_top4_groups_kernel(group_scores_ptr, group_idx_ptr, M, G, stride_gm, stride_gn, stride_im, stride_in):
    # Iterative argmax for 4 positions
    for m in range(0, M):
        # top4 values and indices
        top1_val = -float('inf')
        top1_idx = 0
        top2_val = -float('inf')
        top2_idx = 0
        top3_val = -float('inf')
        top3_idx = 0
        top4_val = -float('inf')
        top4_idx = 0
        # Scan groups
        for g in range(0, G):
            val = tl.load(group_scores_ptr + m * stride_gm + g * stride_gn)
            # update top1
            if val > top1_val:
                top4_val = top3_val
                top4_idx = top3_idx
                top3_val = top2_val
                top3_idx = top2_idx
                top2_val = top1_val
                top2_idx = top1_idx
                top1_val = val
                top1_idx = g
            elif val > top2_val:
                top4_val = top3_val
                top4_idx = top3_idx
                top3_val = top2_val
                top3_idx = top2_idx
                top2_val = val
                top2_idx = g
            elif val > top3_val:
                top4_val = top3_val
                top4_idx = top3_idx
                top3_val = val
                top3_idx = g
            elif val > top4_val:
                top4_val = val
                top4_idx = g
        # store indices
        tl.store(group_idx_ptr + m * stride_im + 0 * stride_in, top1_idx)
        tl.store(group_idx_ptr + m * stride_im + 1 * stride_in, top2_idx)
        tl.store(group_idx_ptr + m * stride_im + 2 * stride_in, top3_idx)
        tl.store(group_idx_ptr + m * stride_im + 3 * stride_in, top4_idx)


# Kernel 6: Build expert-level mask from group_idx
# group_idx: [M, 4], out score_mask: [M, N], set 1 for experts in selected groups, else 0
@triton.jit
def _build_group_mask_kernel(group_idx_ptr, score_mask_ptr, M, N, G, EPG, stride_gm, stride_gn, stride_sm, stride_sn):
    # We process each token row m
    for m in range(0, M):
        # for each selected group index
        for k in range(0, 4):  # top-4 groups
            g = tl.load(group_idx_ptr + m * stride_gm + k * stride_gn)
            base = g * EPG
            for e in range(0, EPG):
                idx = base + e
                mask_val = 1
                tl.store(score_mask_ptr + m * stride_sm + idx * stride_sn, mask_val)


# Kernel 7: Masked fill: set masked_scores[i, e] = -inf if score_mask[i, e] == 0, else scores_for_routing[i, e]
@triton.jit
def _masked_fill_kernel(scores_ptr, score_mask_ptr, masked_ptr, M, N, stride_sm, stride_sn, stride_mm, stride_mn, NEG_INF):
    for m in range(0, M):
        for n in range(0, N):
            s = tl.load(scores_ptr + m * stride_sm + n * stride_sn)
            mask = tl.load(score_mask_ptr + m * stride_sm + n * stride_sn)
            val = tl.where(mask != 0, s, NEG_INF)
            tl.store(masked_ptr + m * stride_mm + n * stride_mn, val)


# Kernel 8: Final top-8 selection from masked_scores (argmax indices and values)
@triton.jit
def _final_top8_kernel(masked_ptr, idx_ptr, vals_ptr, M, N, stride_mm, stride_mn, stride_im, stride_in, stride_vm, stride_vn):
    for m in range(0, M):
        top1_val = -float('inf')
        top1_idx = 0
        # iterate to get top-8
        for i in range(0, 8):
            best_val = -float('inf')
            best_idx = 0
            for n in range(0, N):
                val = tl.load(masked_ptr + m * stride_mm + n * stride_mn)
                if val > best_val:
                    best_val = val
                    best_idx = n
            # store and remove it from future consideration
            tl.store(idx_ptr + m * stride_im + i * stride_in, best_idx)
            tl.store(vals_ptr + m * stride_vm + i * stride_vn, best_val)
            # mark used (set to -inf) to avoid re-selecting
            tl.store(masked_ptr + m * stride_mm + best_idx * stride_mn, -float('inf'))


# Kernel 9: Normalize selected values and apply scaling factor
@triton.jit
def _normalize_and_scale_kernel(vals_ptr, out_ptr, M, K, SCALE, stride_vm, stride_vn, stride_om, stride_on):
    for m in range(0, M):
        total = 0.0
        for k in range(0, K):
            v = tl.load(vals_ptr + m * stride_vm + k * stride_vn)
            total += v
        inv = 1.0 / (total + 1e-20)
        for k in range(0, K):
            v = tl.load(vals_ptr + m * stride_vm + k * stride_vn) * inv
            v = v * SCALE
            tl.store(out_ptr + m * stride_om + k * stride_on, v)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 256, routed_scaling_factor: float = 1.0):
        super().__init__()
        # assert constants from the original code
        assert num_experts == 256, "num_experts must be 256"
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.n_group = 8
        self.experts_per_group = num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward; ensure tensors are on CUDA
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
        # 1) Matmul for logits: [M, N] = [num_tokens, num_experts]
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1), weight.stride(1), weight.stride(0), logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M, N)](
            logits, scores,
            M, N, logits.stride(0), logits.stride(1), scores.stride(0), scores.stride(1),
        )

        # 3) Add bias
        _add_bias_kernel[(M, N)](
            scores, bias, scores_for_routing,
            M, N, scores.stride(0), scores.stride(1), scores_for_routing.stride(0), scores_for_routing.stride(1),
        )

        # 4) Group top-2 sum (G=8, EPG=32)
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group, self.experts_per_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1), group_scores.stride(0), group_scores.stride(1),
        )

        # 5) Select top-4 groups
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1), group_idx.stride(0), group_idx.stride(1),
        )

        # 6) Build expert-level mask
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N, self.n_group, self.experts_per_group,
            group_idx.stride(0), group_idx.stride(1), score_mask.stride(0), score_mask.stride(1),
        )

        # 7) Masked fill: set non-selected to -inf
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M, N)](
            scores_for_routing, score_mask, masked_scores,
            M, N, scores_for_routing.stride(0), scores_for_routing.stride(1), masked_scores.stride(0), masked_scores.stride(1), NEG_INF,
        )

        # 8) Final top-8 selection
        _final_top8_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1), top8_idx.stride(0), top8_idx.stride(1), top8_vals.stride(0), top8_vals.stride(1),
        )

        # 9) Normalize and scale
        _normalize_and_scale_kernel[(M, self.top_k)](
            top8_vals, topk_weight,
            M, self.top_k, self.routed_scaling_factor,
            top8_vals.stride(0), top8_vals.stride(1), topk_weight.stride(0), topk_weight.stride(1),
        )

        # Return topk indices and normalized weights (scaled)
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
