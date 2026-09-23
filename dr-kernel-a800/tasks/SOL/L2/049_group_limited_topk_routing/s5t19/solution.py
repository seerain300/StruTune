import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn,
                         BLOCK: tl.constexpr):
    # 2D launch over tokens and columns
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_offsets = n_block * BLOCK + tl.arange(0, BLOCK)
    mask = n_offsets < N
    x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
    b = tl.load(Bias_ptr + n_offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    y = y + b
    tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, y, mask=mask)


@triton.jit
def group_top2_sum_kernel(S_ptr, Out_ptr,
                          M, G, E,
                          stride_Sm, stride_Sg, stride_Se,
                          stride_Outm, stride_Outg,
                          BLOCK_E: tl.constexpr):
    # One program per (token, group) pair
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    # Compute top-2 across E=32 in this group
    vals = tl.zeros([2], dtype=tl.float32)
    # Simple scan: find max, then second max excluding it
    for e in range(0, E):
        s = tl.load(S_ptr + pid_m * stride_Sm + pid_g * stride_Sg + e * stride_Se)
        if e == 0:
            vals[0] = s
            continue
        if s > vals[0]:
            vals[1] = vals[0]
            vals[0] = s
        elif s > vals[1]:
            vals[1] = s
    sum2 = vals[0] + vals[1]
    tl.store(Out_ptr + pid_m * stride_Outm + pid_g * stride_Outg, sum2)


@triton.jit
def arg_topk_arg_kernel(X_ptr, K_ptr,
                        M, N, K_num,
                        stride_Xm, stride_Xn,
                        stride_Km, stride_Kk,
                        BLOCK_N: tl.constexpr):
    # One program per token; compute top-K indices in descending order
    m = tl.program_id(0)
    # Initialize top-k buffers
    topv = tl.full([BLOCK_N], -float('inf'), dtype=tl.float32)
    topix = tl.zeros([BLOCK_N], dtype=tl.int32)
    # Iterate over N columns
    for n in range(0, N):
        x = tl.load(X_ptr + m * stride_Xm + n * stride_Xn)
        for k in range(0, K_num):
            if x > topv[k]:
                # Insert x at position k, push others down
                topix[k+1:K_num] = topix[k:K_num-1]
                topv[k+1:K_num] = topv[k:K_num-1]
                topix[k] = n
                topv[k] = x
                break
    # Store top-k indices in descending order
    for k in range(0, K_num):
        tl.store(K_ptr + m * stride_Km + k * stride_Kk, topix[k])


@triton.jit
def set_group_mask_kernel(Idx_ptr, Mask_ptr,
                           M, G,
                           stride_Idxm, stride_Idxk,
                           stride_Maskm, stride_Maskg,
                           BLOCK_G: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    for j in range(0, 4):
        g = tl.load(Idx_ptr + m * stride_Idxm + j * stride_Idxk)  # int32
        tl.store(Mask_ptr + m * stride_Maskm + g * stride_Maskg, 1.0)


@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, ExpMask_ptr,
                              M, G, E,
                              stride_Gm, stride_Gg,
                              stride_Em, stride_En,
                              BLOCK_E: tl.constexpr):
    # One program per (token, group)
    m = tl.program_id(0)
    g = tl.program_id(1)
    # Load group mask value
    vm = tl.load(GroupMask_ptr + m * stride_Gm + g * stride_Gg)
    start = g * E
    for e in range(0, E):
        n = start + e
        # Store vm at ExpMask[m, n]
        # We need m for stride_Em and n for stride_En
        tl.store(ExpMask_ptr + m * stride_Em + n * stride_En, vm)


@triton.jit
def mask_scores_groups_kernel(S_ptr, ExpMask_ptr, SMasked_ptr,
                               M, N,
                               stride_Sm, stride_Sn,
                               stride_Em, stride_En,
                               stride_SMaskm, stride_SMaskn,
                               BLOCK_N: tl.constexpr):
    # One program per token row
    m = tl.program_id(0)
    for n in range(0, N):
        valid = tl.load(ExpMask_ptr + m * stride_Em + n * stride_En)  # 0.0 or 1.0
        s = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        new = tl.where(valid > 0.0, s, -float('inf'))
        tl.store(SMasked_ptr + m * stride_SMaskm + n * stride_SMaskn, new)


@triton.jit
def normalize_scale_kernel(Idx_ptr, Scores_ptr, Scale, Out_ptr,
                            M, K,
                            stride_Idxm, stride_Idxk,
                            stride_Sm, stride_Sk,
                            stride_Om, stride_Ok,
                            BLOCK_K: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    denom = 0.0
    for k in range(0, K):
        idx = tl.load(Idx_ptr + m * stride_Idxm + k * stride_Idxk)
        s = tl.load(Scores_ptr + m * stride_Sm + idx * stride_Sk)
        denom += s
    inv = 1.0 / (denom + 1e-20)
    for k in range(0, K):
        idx = tl.load(Idx_ptr + m * stride_Idxm + k * stride_Idxk)
        s = tl.load(Scores_ptr + m * stride_Sm + idx * stride_Sk)
        ns = s * inv * Scale
        tl.store(Out_ptr + m * stride_Om + k * stride_Ok, ns)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Compute logits via PyTorch GEMV (robust and fast)
        logits = torch.nn.functional.linear(hidden_states.to(torch.float32), weight.to(torch.float32))

        # Triton: sigmoid + bias
        M, N = logits.shape
        scores = torch.empty_like(logits)
        BLOCK = 128
        grid_sig = (M, triton.cdiv(N, BLOCK))
        sigmoid_bias_kernel[grid_sig](
            logits, expert_bias, scores,
            M, N,
            scores.stride(0), scores.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK
        )

        # Triton: per-group top-2 aggregation (groups of 32)
        G = 8
        E = N // G  # 32
        group_scores = torch.empty((M, G), device=scores.device, dtype=torch.float32)
        grid_top2 = (M, G)
        group_top2_sum_kernel[grid_top2](
            scores, group_scores,
            M, G, E,
            scores.stride(0), scores.stride(1), scores.stride(1),  # stride_Se is 1
            group_scores.stride(0), group_scores.stride(1),
            E
        )

        # Triton: per-token top-4 groups (arg-topk)
        group_idx = torch.empty((M, 4), device=scores.device, dtype=torch.int32)
        grid_topk_groups = (M,)
        arg_topk_arg_kernel[grid_topk_groups](
            group_scores, group_idx,
            M, G, 4,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            G
        )

        # Triton: set per-token group mask to 1.0 at selected groups
        group_mask = torch.empty((M, G), device=scores.device, dtype=torch.float32)
        grid_set = (M,)
        set_group_mask_kernel[grid_set](
            group_idx, group_mask,
            M, G,
            group_idx.stride(0), group_idx.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            4
        )

        # Triton: expand group mask to [M, N]
        exp_mask = torch.empty((M, N), device=scores.device, dtype=torch.float32)
        grid_expand = (M, G)
        expand_group_mask_kernel[grid_expand](
            group_mask, exp_mask,
            M, G, E,
            group_mask.stride(0), group_mask.stride(1),
            exp_mask.stride(0), exp_mask.stride(1),
            E
        )

        # Triton: mask scores: non-selected group entries set to -inf
        masked_scores = torch.empty_like(scores)
        grid_mask_scores = (M,)
        mask_scores_groups_kernel[grid_mask_scores](
            scores, exp_mask, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            exp_mask.stride(0), exp_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            N
        )

        # Triton: per-token top-8 experts from masked scores
        topk_idx = torch.empty((M, 8), device=scores.device, dtype=torch.int32)
        grid_topk_exp = (M,)
        arg_topk_arg_kernel[grid_topk_exp](
            masked_scores, topk_idx,
            M, N, 8,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            N
        )

        # Triton: normalize and scale
        selected_scores = torch.empty((M, 8), device=scores.device, dtype=torch.float32)
        # We need to gather selected_scores from original 'scores' at indices topk_idx. We'll do this in Triton via a small gather loop.
        # To avoid extra PyTorch ops, we compute gathered scores in Triton:
        out_weight = torch.empty((M, 8), device=scores.device, dtype=torch.float32)
        grid_norm = (M,)
        normalize_scale_kernel[grid_norm](
            topk_idx, scores, routed_scaling_factor, out_weight,
            M, 8,
            topk_idx.stride(0), topk_idx.stride(1),
            scores.stride(0), scores.stride(1),
            out_weight.stride(0), out_weight.stride(1),
            8
        )

        return topk_idx, out_weight


def run(*args):
    return ModelNew()(*args)
