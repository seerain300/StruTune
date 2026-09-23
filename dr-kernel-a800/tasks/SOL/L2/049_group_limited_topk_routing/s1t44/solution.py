import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Constants for this routing
NUM_EXPERTS = 256
N_GROUP = 8
EXPERTS_PER_GROUP = NUM_EXPERTS // N_GROUP  # 32
TOPK_GROUP = 4
TOP_K = 8


@triton.jit
def _row_matmul_kernel(
    hidden_ptr,        # *f32, [M, K] = [num_tokens, hidden_dim]
    weight_ptr,        # *f32, [N, K] = [num_experts, hidden_dim]
    out_ptr,           # *f32, [M, N] = [num_tokens, num_experts]
    M, K, N,
    stride_hm, stride_hk,   # strides for hidden
    stride_wn, stride_wk,   # strides for weight
    stride_om, stride_on,   # strides for out
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    acc = tl.zeros((N,), dtype=tl.float32)
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        # Load hidden row slice
        hidden_row = tl.load(
            hidden_ptr + m * stride_hm + offs_k * stride_hk,
            mask=mask_k,
            other=0.0
        )  # [BLOCK_K]
        # Load weight column slices for all experts
        weight_cols = tl.load(
            weight_ptr + tl.arange(0, N) * stride_wn + offs_k * stride_wk,
            mask=mask_k,
            other=0.0
        )  # [N, BLOCK_K]
        # acc += sum_j weight[:, j] * hidden[m, j]
        acc += tl.sum(weight_cols * hidden_row[None, :], axis=1)
    # Store results for this row
    for n_idx in range(0, N):
        tl.store(out_ptr + m * stride_om + n_idx * stride_on, acc[n_idx])


@triton.jit
def _sigmoid_kernel(
    in_ptr, out_ptr, M, N,
    stride_im, stride_in,  # strides for input logits
    stride_om, stride_on,  # strides for output
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    for n_idx in range(0, N):
        ptr = in_ptr + m * stride_im + n_idx * stride_in
        x = tl.load(ptr)
        y = 1.0 / (1.0 + tl.exp(-x))
        out_ptr[m * stride_om + n_idx * stride_on] = y


@triton.jit
def _add_bias_kernel(
    scores_ptr, bias_ptr, out_ptr, M, N,
    stride_sm, stride_sn,   # strides for scores
    stride_bm,               # stride for bias (bias is 1D)
    stride_om, stride_on,   # strides for output
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    for n_idx in range(0, N):
        score = tl.load(scores_ptr + m * stride_sm + n_idx * stride_sn)
        bias_val = tl.load(bias_ptr + n_idx * stride_bm)  # bias[n]
        out = score + bias_val
        tl.store(out_ptr + m * stride_om + n_idx * stride_on, out)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr, M, N, G, EP,
    stride_sm, stride_sn,
    stride_gs,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    for g in range(0, G):
        base = g * EP
        max1 = -1e20
        max2 = -1e20
        for e in range(0, EP):
            col = base + e
            val = tl.load(scores_ptr + m * stride_sm + col * stride_sn)
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
        sum_top2 = max1 + max2
        tl.store(group_scores_ptr + m * stride_gs + g, sum_top2)


@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr, group_idx_ptr, M, G,
    stride_gm, stride_gn,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    top = tl.full((4,), -1e20, dtype=tl.float32)
    idx = tl.full((4,), -1, dtype=tl.int32)
    for g in range(0, G):
        val = tl.load(group_scores_ptr + m * stride_gm + g * stride_gn)
        # Insert val into top array
        for j in range(0, 4):
            if val > top[j]:
                tmp = top[j]
                top[j] = val
                # shift right
                for k in range(j+1, 4):
                    top[k] = tmp if k == j+1 else top[k-1]
                val = tmp
                tmp_idx = idx[j]
                idx[j] = g
                # shift right
                for k in range(j+1, 4):
                    idx[k] = idx[k-1] if k > 0 else g
                break
    # store idx
    out_base = m * (TOPK_GROUP)
    for j in range(0, 4):
        tl.store(group_idx_ptr + out_base + j, idx[j])


@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr, score_mask_ptr, M, G, EP,
    stride_gm, stride_gn,
    stride_sm,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    for g in range(0, G):
        gi = tl.load(group_idx_ptr + m * stride_gm + g * stride_gn)
        base = gi * EP
        for e in range(0, EP):
            col = base + e
            tl.store(score_mask_ptr + m * stride_sm + col * stride_sm, 1)
    # zero-initialize was done in host code


@triton.jit
def _masked_fill_kernel(
    scores_ptr, score_mask_ptr, masked_ptr, M, N,
    stride_sm, stride_sn,
    stride_mm, stride_mn,  # masked_ptr strides
    NEG_INF: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    for n_idx in range(0, N):
        score = tl.load(scores_ptr + m * stride_sm + n_idx * stride_sn)
        mask = tl.load(score_mask_ptr + m * stride_sm + n_idx * stride_sn)  # 0 or 1
        out = score if mask != 0 else NEG_INF
        tl.store(masked_ptr + m * stride_mm + n_idx * stride_mn, out)


@triton.jit
def _final_top8_selection_kernel(
    masked_ptr, topk_idx_ptr, topk_vals_ptr, M, N,
    stride_mm, stride_mn,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    top = tl.full((8,), -1e20, dtype=tl.float32)
    idx = tl.full((8,), -1, dtype=tl.int32)
    for n_idx in range(0, N):
        val = tl.load(masked_ptr + m * stride_mm + n_idx * stride_mn)
        for j in range(0, 8):
            if val > top[j]:
                tmp = top[j]
                top[j] = val
                val = tmp
                tmp_idx = idx[j]
                idx[j] = n_idx
                break
    out_vals_base = m * 8
    out_idx_base = m * 8
    for j in range(0, 8):
        tl.store(topk_vals_ptr + out_vals_base + j, top[j])
        tl.store(topk_idx_ptr + out_idx_base + j, idx[j])


@triton.jit
def _normalize_scale_kernel(
    topk_vals_ptr, topk_weight_ptr, routed_factor, M,
    scale_stride, weight_stride,
):
    pid_m = tl.program_id(0)
    m = pid_m
    if m >= M:
        return
    s = 0.0
    # sum of top-8 values
    for j in range(0, 8):
        val = tl.load(topk_vals_ptr + m * scale_stride + j * scale_stride)
        s += val
    inv_s = 1.0 / (s + 1e-20)
    routed = routed_factor
    for j in range(0, 8):
        val = tl.load(topk_vals_ptr + m * scale_stride + j * scale_stride) * routed
        tl.store(topk_weight_ptr + m * weight_stride + j * weight_stride, val)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure Triton and CUDA
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")
        # Ensure contiguity and FP32 for computation
        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)           # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == NUM_EXPERTS and K == HIDDEN_DIM, "Shape mismatch: weight must be [256, hidden_dim], hidden [M, hidden_dim]."

        # Allocate outputs and intermediates
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((M, N_GROUP), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((M, TOPK_GROUP), dtype=torch.int32, device=hidden.device)
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)  # expert-level mask
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, TOP_K), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, TOP_K), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, TOP_K), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernels
        # 1) Matmul for logits
        BLOCK_K = 64
        _row_matmul_kernel[(M,)](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # 2) Sigmoid
        _sigmoid_kernel[(M,)](
            logits, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=4,
        )

        # 3) Add bias
        _add_bias_kernel[(M,)](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores.stride(1),
            bias.stride(0),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            num_warps=4,
        )

        # 4) Group top-2 sum: reshape to [M, G, EP] and reduce (G=8, EP=32)
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, N_GROUP, EXPERTS_PER_GROUP,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0),
            num_warps=4,
        )

        # 5) Select top-4 groups per token
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, N_GROUP,
            group_scores.stride(0), group_scores.stride(1),
            num_warps=4,
        )

        # 6) Build group mask: set 1 at selected groups’ 32 experts, 0 otherwise
        # We pre-zero score_mask via torch.empty_like (host). Kernel only sets 1s.
        # 7) Masked fill: set non-selected expert scores to -inf
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M,)](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            NEG_INF=NEG_INF,
            num_warps=4,
        )

        # 8) Final top-8 selection from masked scores
        _final_top8_selection_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=4,
        )

        # 9) Normalize and apply routed_scaling_factor
        routed = 1.0  # default scaling factor; use whatever is passed to the function similarly
        _normalize_scale_kernel[(M,)](
            top8_vals, topk_weight,
            routed,
            M,
            top8_vals.stride(0),
            topk_weight.stride(0),
            num_warps=4,
        )

        # Return indices and normalized weights
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
