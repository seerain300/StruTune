import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr, bias_ptr, scores_ptr,
    M, N,
    stride_sm, stride_sn,
    stride_b,  # bias_ptr stride (bias is 1D)
    stride_om, stride_on,
    num_warps: tl.constexpr
):
    # Elementwise: scores = sigmoid(logits) + bias
    for row in range(0, M * N, 1):
        m = row // N
        n = row % N
        val = tl.load(logits_ptr + m * stride_sm + n * stride_sn)
        bias_val = tl.load(bias_ptr + n * stride_b)
        val = 1.0 / (1.0 + tl.exp(-val)) + bias_val
        tl.store(scores_ptr + m * stride_om + n * stride_on, val)


@triton.jit
def _group_top2_sum_kernel(
    scores_ptr, group_scores_ptr,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_gm, stride_gn,
    num_warps: tl.constexpr
):
    # Compute group_scores[t, g] = sum of top-2 among scores[t, g*32:(g+1)*32]
    for t in range(0, M):
        base = t * stride_sm
        group_acc = 0.0
        for g in range(0, 8):
            start = g * EXP_PER_GROUP
            top1 = -float('inf')
            top2 = -float('inf')
            # Reduce across this group
            for i in range(0, EXP_PER_GROUP):
                idx = start + i
                val = tl.load(scores_ptr + base + idx * stride_sn)
                if val > top1:
                    top2 = top1
                    top1 = val
                elif val > top2:
                    top2 = val
            group_acc += (top1 + top2)
        tl.store(group_scores_ptr + t * stride_gm + g * stride_gn, group_acc)


@triton.jit
def _select_top4_groups_bubble_kernel(
    group_scores_ptr, top4_groups_ptr,
    M,
    stride_gm, stride_gn,
    stride_tm, stride_tn,
    num_warps: tl.constexpr
):
    # For each token, select top-4 group indices (sorted=False)
    for t in range(0, M):
        vec = tl.load(group_scores_ptr + t * stride_gm + 0 * stride_gn + tl.arange(0, 8))
        idxs = tl.zeros((8,), dtype=tl.int32)
        for k in range(0, 4):
            max_val = -float('inf')
            max_idx = 0
            for i in range(0, 8):
                val = vec[i]
                if val > max_val:
                    max_val = val
                    max_idx = i
            # Mark selected
            idxs[k] = max_idx
            # Zero it for next selection
            vec = tl.where(tl.arange(0, 8) == max_idx, -float('inf'), vec)
        tl.store(top4_groups_ptr + t * stride_tm + 0 * stride_tn + tl.arange(0, 4), idxs[:4])


@triton.jit
def _mask_nonselected_groups_kernel(
    scores_ptr, top4_groups_ptr, masked_scores_ptr,
    M, N, EXP_PER_GROUP: tl.constexpr,
    stride_sm, stride_sn,
    stride_tm, stride_tn,
    stride_msm, stride_msn,
    num_warps: tl.constexpr
):
    # For each token, set scores of non-selected groups to -inf
    for t in range(0, M):
        selected = tl.load(top4_groups_ptr + t * stride_tm + tl.arange(0, 4), mask=tl.arange(0, 4) < 4, other=0).to(tl.int32)
        for g in range(0, 8):
            if not any(selected == g):
                start = g * EXP_PER_GROUP
                for i in range(0, EXP_PER_GROUP):
                    idx = start + i
                    val = tl.load(scores_ptr + t * stride_sm + idx * stride_sn)
                    new_val = tl.where(g == g, val, -float('inf'))
                    tl.store(masked_scores_ptr + t * stride_msm + idx * stride_msn, new_val)


@triton.jit
def _select_top8_masked_kernel(
    masked_scores_ptr, top8_indices_ptr,
    M, N,
    stride_msm, stride_msn,
    stride_tm, stride_tn,
    num_warps: tl.constexpr
):
    # For each token, iteratively select max 8 times (sorted=False)
    for t in range(0, M):
        vec = tl.load(masked_scores_ptr + t * stride_msm + tl.arange(0, N) * stride_msn)
        idxs = tl.zeros((8,), dtype=tl.int32)
        for k in range(0, 8):
            max_val = -float('inf')
            max_idx = 0
            for i in range(0, N):
                val = vec[i]
                if val > max_val:
                    max_val = val
                    max_idx = i
            idxs[k] = max_idx
            vec = tl.where(tl.arange(0, N) == max_idx, -float('inf'), vec)
        tl.store(top8_indices_ptr + t * stride_tm + tl.arange(0, 8) * stride_tn, idxs)


@triton.jit
def _normalize_and_scale_kernel(
    masked_scores_ptr, top8_indices_ptr, out_weights_ptr,
    M, N,
    routed_scaling, eps: tl.constexpr,
    stride_msm, stride_msn,
    stride_tm, stride_tn,
    stride_wm, stride_wn,
    num_warps: tl.constexpr
):
    # For each token, normalize selected 8 scores by sum + eps and scale
    for t in range(0, M):
        indices = tl.load(top8_indices_ptr + t * stride_tm + tl.arange(0, 8) * stride_tn)
        total = 0.0
        for k in range(0, 8):
            idx = indices[k]
            val = tl.load(masked_scores_ptr + t * stride_msm + idx * stride_msn)
            total += val
        denom = total + eps
        for k in range(0, 8):
            idx = indices[k]
            val = tl.load(masked_scores_ptr + t * stride_msm + idx * stride_msn)
            w = (val / denom) * routed_scaling
            tl.store(out_weights_ptr + t * stride_wm + k * stride_wn, w)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure CUDA tensors, float32, contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA tensors"
        hidden = hidden_states.to(torch.float32).contiguous()
        weight_c = weight.to(torch.float32).contiguous()
        bias_c = expert_bias.to(torch.float32).contiguous()

        M, K = hidden.shape
        N = weight_c.shape[0]
        assert weight_c.shape == (N, K), "weight must be [num_experts, hidden_dim]"
        assert bias_c.shape == (N,), "expert_bias must be [num_experts]"

        # 1) PyTorch GEMM for logits (robust and exact)
        logits = F.linear(hidden, weight_c, None).contiguous()  # [M, N]

        # 2) Triton: sigmoid + expert bias → scores
        scores = torch.empty_like(logits)
        stride_lm, stride_ln = logits.stride()
        stride_sm, stride_sn = scores.stride()
        stride_b = bias_c.stride(0)
        _sigmoid_add_bias_kernel[(M * N,)](
            logits, bias_c, scores,
            M, N,
            stride_lm, stride_ln,
            stride_b,
            stride_sm, stride_sn,
            num_warps=1
        )

        # 3) Triton: group top-2 sum → group_scores [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_gm, stride_gn = group_scores.stride()
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_gm=stride_gm, stride_gn=stride_gn,
            num_warps=1
        )

        # 4) Triton: select top-4 groups per token → top4_groups [M, 4]
        top4_groups = torch.empty((M, 4), dtype=torch.int32, device=device)
        stride_tm, stride_tn = top4_groups.stride()
        _select_top4_groups_bubble_kernel[(M,)](
            group_scores, top4_groups,
            M,
            stride_gm=stride_gm, stride_gn=stride_gn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1
        )

        # 5) Triton: mask non-selected groups in scores → masked_scores [M, N]
        masked_scores = torch.empty_like(scores)
        stride_msm, stride_msn = masked_scores.stride()
        _mask_nonselected_groups_kernel[(M,)](
            scores, top4_groups, masked_scores,
            M, N, EXP_PER_GROUP=32,
            stride_sm=stride_sm, stride_sn=stride_sn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            stride_msm=stride_msm, stride_msn=stride_msn,
            num_warps=1
        )

        # 6) Triton: select top-8 from masked scores → top8_indices [M, 8]
        top8_indices = torch.empty((M, 8), dtype=torch.int32, device=device)
        stride_pi_tm, stride_pi_tn = top8_indices.stride()
        _select_top8_masked_kernel[(M,)](
            masked_scores, top8_indices,
            M, N,
            stride_msm=stride_msm, stride_msn=stride_msn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            num_warps=1
        )

        # 7) Triton: normalize and scale → topk_weight [M, 8]
        out_weights = torch.empty((M, 8), dtype=torch.float32, device=device)
        stride_wm, stride_wn = out_weights.stride()
        _normalize_and_scale_kernel[(M,)](
            masked_scores, top8_indices, out_weights,
            M, N,
            self.routed_scaling_factor, eps=1e-20,
            stride_msm=stride_msm, stride_msn=stride_msn,
            stride_tm=stride_tm, stride_tn=stride_tn,
            stride_wm=stride_wm, stride_wn=stride_wn,
            num_warps=1
        )

        # Return indices and weights; original returns (topk_idx, topk_weight)
        # top8_indices are the selected expert indices per token
        return top8_indices, out_weights


def run(*args):
    return ModelNew()(*args)
