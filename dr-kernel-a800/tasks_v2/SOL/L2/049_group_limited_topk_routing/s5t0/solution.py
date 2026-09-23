import torch
import triton
import triton.language as tl


@triton.jit
def gemv_linear_kernel(A_ptr, B_ptr, C_ptr,
                        M, N, K,
                        stride_Am, stride_Ak,
                        stride_Bn, stride_Bk,
                        stride_Cm, stride_Cn,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program per row m
    m = tl.program_id(0)
    # Column offsets
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        # Accumulator for this block of columns
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Loop over K in blocks
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            # Load A[m, k_offsets] and B[n_offsets, k_offsets]
            a = tl.load(A_ptr + m * stride_Am + k_offsets * stride_Ak,
                        mask=k_offsets < K, other=0.0)
            b = tl.load(B_ptr + n_offsets[:, None] * stride_Bn + k_offsets[None, :] * stride_Bk,
                        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
                        other=0.0)
            # Accumulate dot products for this n block
            acc += tl.sum(a[None, :] * b, axis=1)
        # Store results
        tl.store(C_ptr + m * stride_Cm + n_offsets * stride_Cn, acc, mask=(n_offsets < N))


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn):
    # One program per element
    m = tl.program_id(0)
    n = tl.program_id(1)
    x = tl.load(X_ptr + m * stride_Xm + n * stride_Xn)
    b = tl.load(Bias_ptr + n)
    y = 1.0 / (1.0 + tl.exp(-x))
    y = y + b
    tl.store(Y_ptr + m * stride_Ym + n * stride_Yn, y)


@triton.jit
def top2_per_group_kernel(S_ptr, Top_ptr,
                          M, G, E,
                          stride_Sm, stride_Sg, stride_Se,
                          stride_Tpg, stride_Tpe,
                          BLOCK_E: tl.constexpr):
    m = tl.program_id(0)
    g = tl.program_id(1)
    # Iterate over E in blocks, track best1, best2
    best1 = tl.full((), -float('inf'), tl.float32)
    best2 = tl.full((), -float('inf'), tl.float32)
    # simple loop over E (E is runtime, we emulate with while)
    e = 0
    while e < E:
        val = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Se)
        # Update best2 then best1
        cond2 = val > best2
        tmp2 = tl.where(cond2, val, best2)
        cond1 = val > best1
        tmp1 = tl.where(cond1, val, best1)
        best2 = tl.where(cond2 & (~cond1), val, best2)
        best1 = tmp1
        e += 1
    # Store top-2
    tl.store(Top_ptr + m * stride_Tpg + g * stride_Tpe + 0, best1)
    tl.store(Top_ptr + m * stride_Tpg + g * stride_Tpe + 1, best2)


@triton.jit
def top4_groups_kernel(Top2_ptr, GroupIdx_ptr,
                       M, G,
                       stride_Tpg, stride_Tpe,  # Top2 strides
                       stride_Gm, stride_Gk):  # GroupIdx strides
    m = tl.program_id(0)
    # Load top-2 sums for 8 groups
    top_vals = tl.zeros([G], dtype=tl.float32)
    for g in range(0, G):
        sum_ = tl.load(Top_ptr + m * stride_Tpg + g * stride_Tpe + 0) + \
               tl.load(Top_ptr + m * stride_Tpg + g * stride_Tpe + 1)
        top_vals[g] = sum_
    # Select top-4 using pairwise comparisons
    top4 = tl.full([4], -float('inf'), tl.float32)
    idx4 = tl.zeros([4], dtype=tl.int32)
    for g in range(0, G):
        val = top_vals[g]
        # find position in top4 where insertion should occur
        pos = 0
        # simple insertion via comparisons
        # If val > max, put at 0, else find the first slot smaller than val
        if val > top4[3]:
            top4[3] = val
            idx4[3] = g
        else:
            for j in range(3, -1, -1):
                if (j == 0 and val <= top4[0]) or (val < top4[j]):
                    # shift j+1..3 down
                    for r in range(3, j, -1):
                        top4[r] = top4[r - 1]
                        idx4[r] = idx4[r - 1]
                    top4[j] = val
                    idx4[j] = g
                    break
    # Store idx4
    for k in range(0, 4):
        tl.store(GroupIdx_ptr + m * stride_Gm + k * stride_Gk, idx4[k])


@triton.jit
def mask_groups_to_experts_kernel(GroupMask_ptr, MaskExp_ptr,
                                  M, N, G, E,
                                  stride_Gm, stride_Gg,
                                  stride_ME_m, stride_ME_n):
    m = tl.program_id(0)
    # For each group, check if selected (GroupMask[m, g] == 1), then set MaskExp[m, n] = 1 for n in g*E:(g+1)*E
    for g in range(0, G):
        sel = tl.load(GroupMask_ptr + m * stride_Gm + g * stride_Gg)
        if sel > 0:
            for n in range(0, N):
                # compute group for this expert
                group_exp = n // E
                if group_exp == g:
                    tl.store(MaskExp_ptr + m * stride_ME_m + n * stride_ME_n, 1.0)
                else:
                    tl.store(MaskExp_ptr + m * stride_ME_m + n * stride_ME_n, 0.0)


@triton.jit
def masked_scores_kernel(S_ptr, MaskExp_ptr, Y_ptr,
                         M, N,
                         stride_Sm, stride_Sn,
                         stride_ME_m, stride_ME_n,
                         stride_Ym, stride_Yn):
    m = tl.program_id(0)
    n = tl.program_id(1)
    s = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
    me = tl.load(MaskExp_ptr + m * stride_ME_m + n * stride_ME_n)
    y = tl.where(me > 0, s, -float('inf'))
    tl.store(Y_ptr + m * stride_Ym + n * stride_Yn, y)


@triton.jit
def topk_experts_kernel(S_ptr, TopK_ptr, TopK_vals_ptr,
                        M, N, K,
                        stride_Sm, stride_Sn,
                        stride_TK_m, stride_TK_k,
                        stride_TKV_m, stride_TKV_k):
    # For each row m, select top-K from S[m, :]
    m = tl.program_id(0)
    top_vals = tl.full([K], -float('inf'), tl.float32)
    top_idx = tl.zeros([K], dtype=tl.int32)
    for n in range(0, N):
        val = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        # Insert into top-K array
        for k in range(0, K):
            if val > top_vals[k]:
                # shift down
                for r in range(K - 1, k, -1):
                    top_vals[r] = top_vals[r - 1]
                    top_idx[r] = top_idx[r - 1]
                top_vals[k] = val
                top_idx[k] = n
                break
    # store indices and values
    for k in range(0, K):
        tl.store(TopK_ptr + m * stride_TK_m + k * stride_TK_k, top_idx[k])
        tl.store(TopK_vals_ptr + m * stride_TKV_m + k * stride_TKV_k, top_vals[k])


@triton.jit
def normalize_and_scale_kernel(Weights_ptr, Out_ptr,
                               M, K,
                               stride_Wm, stride_Wk,
                               stride_Om, stride_Ok,
                               scale):
    m = tl.program_id(0)
    # sum of topk weights
    s = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        w = tl.load(Weights_ptr + m * stride_Wm + k * stride_Wk)
        s += w
    # normalize
    for k in range(0, K):
        w = tl.load(Weights_ptr + m * stride_Wm + k * stride_Wk)
        w = w / s
        w = w * scale
        tl.store(Out_ptr + m * stride_Om + k * stride_Ok, w)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure CUDA and float32
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew requires CUDA device for Triton kernels."
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]
        E = 32  # experts per group (constant in the original)
        G = 8   # number of groups (constant)

        # 1) Compute logits via Triton GEMV
        # hidden_states: [M, K], weight: [N, K]
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        # Strides
        stride_Am, stride_Ak = hidden_states.stride()
        stride_Bn, stride_Bk = weight.stride()
        stride_Cm, stride_Cn = logits.stride()
        # Launch kernel: one program per m
        grid = (M,)
        # Choose blocks
        BLOCK_N = 128
        BLOCK_K = 128
        gemv_linear_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            stride_Am, stride_Ak,
            stride_Bn, stride_Bk,
            stride_Cm, stride_Cn,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 2) Sigmoid + expert bias via Triton
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        stride_Xm, stride_Xn = logits.stride()
        stride_Ym, stride_Yn = scores.stride()
        grid2 = (M, N)
        sigmoid_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, N,
            stride_Xm, stride_Xn,
            stride_Ym, stride_Yn,
        )

        # 3) Reshape to [M, G, E] and compute top-2 per group via Triton
        scores_reshaped = scores.view(M, G, E)
        top2_vals = torch.empty((M, G, 2), device=device, dtype=torch.float32)
        stride_Sm, stride_Sg, stride_Se = scores_reshaped.stride()
        stride_Tpg, stride_Tpe = top2_vals.stride()
        grid3 = (M, G)
        top2_per_group_kernel[grid3](
            scores_reshaped, top2_vals,
            M, G, E,
            stride_Sm, stride_Sg, stride_Se,
            stride_Tpg, stride_Tpe,
            BLOCK_E=1,
        )

        # 4) Top-4 groups via Triton
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        stride_Tpg2, stride_Tpe2 = top2_vals.stride()  # not used here, but pass any
        stride_Gm, stride_Gk = group_idx.stride()
        grid4 = (M,)
        top4_groups_kernel[grid4](
            top2_vals, group_idx,
            M, G,
            stride_Tpg, stride_Tpe,
            stride_Gm, stride_Gk,
        )

        # 5) Group mask -> expert mask via Triton
        group_mask = (group_idx > 0).to(torch.float32)  # [M, 4] of 1s where selected
        # Build group_mask_expanded [M, G]
        group_mask_expanded = torch.zeros((M, G), device=device, dtype=torch.float32)
        for k in range(0, 4):
            mask_k = (group_idx[:, k] == torch.arange(0, G, device=device)).to(torch.float32)  # [M, G]
            group_mask_expanded = group_mask_expanded + mask_k  # only one 1 per row
        # Now group_mask_expanded has 1 for selected groups, 0 otherwise
        mask_expert = torch.empty((M, N), device=device, dtype=torch.float32)
        stride_Gm2, stride_Gg = group_mask_expanded.stride()  # not used, but pass
        stride_ME_m, stride_ME_n = mask_expert.stride()
        grid5 = (M,)
        mask_groups_to_experts_kernel[grid5](
            group_mask_expanded, mask_expert,
            M, N, G, E,
            stride_Gm2, stride_Gg,
            stride_ME_m, stride_ME_n,
        )

        # 6) Masked scores via Triton
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        stride_Sm2, stride_Sn = scores.stride()
        stride_ME_m2, stride_ME_n2 = mask_expert.stride()
        stride_Ym2, stride_Yn2 = masked_scores.stride()
        grid6 = (M, N)
        masked_scores_kernel[grid6](
            scores, mask_expert, masked_scores,
            M, N,
            stride_Sm2, stride_Sn,
            stride_ME_m2, stride_ME_n2,
            stride_Ym2, stride_Yn2,
        )

        # 7) Top-8 experts via Triton
        topk_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        topk_vals = torch.empty((M, 8), device=device, dtype=torch.float32)
        stride_Sm3, stride_Sn2 = masked_scores.stride()
        stride_TK_m, stride_TK_k = topk_idx.stride()
        stride_TKV_m, stride_TKV_k = topk_vals.stride()
        grid7 = (M,)
        topk_experts_kernel[grid7](
            masked_scores, topk_idx, topk_vals,
            M, N, 8,
            stride_Sm3, stride_Sn2,
            stride_TK_m, stride_TK_k,
            stride_TKV_m, stride_TKV_k,
        )

        # 8) Normalize and scale via Triton
        topk_weight = torch.empty((M, 8), device=device, dtype=torch.float32)
        stride_Wm, stride_Wk = topk_vals.stride()
        stride_Om, stride_Ok = topk_weight.stride()
        scale = routed_scaling_factor
        grid8 = (M,)
        normalize_and_scale_kernel[grid8](
            topk_vals, topk_weight,
            M, 8,
            stride_Wm, stride_Wk,
            stride_Om, stride_Ok,
            scale,
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
