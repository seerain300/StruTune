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
    # 2D grid: program_id(0)=token, program_id(1)=expert block
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Reduction over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load hidden for this token and chunk: A[m, k]
        a = tl.load(A_ptr + m * stride_Am + k_offsets * stride_Ak, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight chunk: B[n, k] -> shape [BLOCK_N, BLOCK_K]
        b = tl.load(B_ptr + n_offsets[:, None] * stride_Bn + k_offsets[None, :] * stride_Bk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        # Accumulate: acc[n] += sum_k (A[m,k] * B[n,k])
        acc += tl.sum(b * a[None, :], axis=1)

    # Store results to C[m, n_offsets]
    tl.store(C_ptr + m * stride_Cm + n_offsets * stride_Cn, acc, mask=mask_n)


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Bn,
                         stride_Ym, stride_Yn,
                         BLOCK_N: tl.constexpr):
    # One program per token row
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = n_offsets < N
        x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
        b = tl.load(Bias_ptr + n_offsets * stride_Bn, mask=mask, other=0.0)
        y = 1.0 / (1.0 + tl.exp(-x)) + b
        tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, y, mask=mask)


@triton.jit
def top2_per_group_kernel(S_ptr, GroupScores_ptr,
                          M, G, E,
                          stride_Sm, stride_Sg, stride_Sexp,
                          BLOCK_E: tl.constexpr):
    # One program per (token, group)
    pid = tl.program_id(0)
    m = pid // G
    g = pid % G
    if m >= M:
        return
    exp_offsets = g * E + tl.arange(0, BLOCK_E)
    mask = exp_offsets < (G * E)
    scores = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + exp_offsets * stride_Sexp, mask=mask, other=0.0)
    # Sort/selection within BLOCK_E (E=32 is small). Compute top-2 via argmax twice.
    # First max
    max_val = -float('inf')
    max_idx = 0
    for i in range(BLOCK_E):
        val = scores[i]
        if val > max_val:
            max_val = val
            max_idx = i
    scores = tl.where(tl.arange(0, BLOCK_E) == max_idx, -float('inf'), scores)
    sec_val = -float('inf')
    for i in range(BLOCK_E):
        val = scores[i]
        if val > sec_val:
            sec_val = val
    group_score = max_val + sec_val
    tl.store(GroupScores_ptr + m * G + g, group_score)


@triton.jit
def argtopk_groups_kernel(GroupScores_ptr, GroupIdx_ptr,
                           M, K,
                           stride_Sm, stride_Sk,
                           stride_Ikm, stride_Ikn,
                           BLOCK_K: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    best_vals = tl.full([K], -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros([K], dtype=tl.int32)
    for k in range(0, 8):
        val = tl.load(GroupScores_ptr + m * stride_Sm + k * stride_Sk)
        for j in range(K):
            if val > best_vals[j]:
                tmp = best_vals[j]
                best_vals[j] = val
                val = tmp
                tmp_idx = best_idxs[j]
                best_idxs[j] = k
                idx = tmp_idx
    for j in range(K):
        tl.store(GroupIdx_ptr + m * stride_Ikm + j * stride_Ikn, best_idxs[j])


@triton.jit
def scatter_group_mask_kernel(GroupIdx_ptr, GroupMask_ptr,
                              M, G,
                              stride_Ikm, stride_Ikn,
                              stride_Gm, stride_Gn,
                              BLOCK_G: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    for j in range(0, G):
        idx_j = tl.load(GroupIdx_ptr + m * stride_Ikm + j * stride_Ikn)
        tl.store(GroupMask_ptr + m * stride_Gm + idx_j * stride_Gn, 1.0)


@triton.jit
def expand_mask_to_N_kernel(GroupMask_ptr, Expanded_ptr,
                             M, G, N,
                             stride_Gm, stride_Gn,
                             stride_Em, stride_En,
                             BLOCK_N: tl.constexpr):
    # One program per token row
    m = tl.program_id(0)
    # GroupMask is [M, G], Expanded is [M, N]
    for g in range(0, G):
        mask_val = tl.load(GroupMask_ptr + m * stride_Gm + g * stride_Gn)
        # repeat across N: group contributes 32 experts; map g to expert block
        exp_block = g * 32
        for n_start in range(0, N, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N
            # compute expert index within block: exp_idx = n_offsets - exp_block
            exp_idx = n_offsets - exp_block
            valid = (exp_idx >= 0) & (exp_idx < 32) & mask_n
            # For valid, set to mask_val, else 0
            val = tl.where(valid, mask_val, 0.0)
            tl.store(Expanded_ptr + m * stride_Em + n_offsets * stride_En, val, mask=mask_n)


@triton.jit
def mask_scores_kernel(Scores_ptr, Expanded_ptr, Masked_ptr,
                        M, N,
                        stride_Sm, stride_Sn,
                        stride_Em, stride_En,
                        stride_Cm, stride_Cn,
                        BLOCK_N: tl.constexpr):
    # One program per token row
    m = tl.program_id(0)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        scores = tl.load(Scores_ptr + m * stride_Sm + n_offsets * stride_Sn, mask=mask_n, other=0.0)
        mask_vals = tl.load(Expanded_ptr + m * stride_Em + n_offsets * stride_En, mask=mask_n, other=0.0)
        scores_masked = tl.where(mask_vals > 0, scores, -float('inf'))
        tl.store(Masked_ptr + m * stride_Cm + n_offsets * stride_Cn, scores_masked, mask=mask_n)


@triton.jit
def argtopk_experts_kernel(X_ptr, Indices_ptr,
                            M, N,
                            stride_Xm, stride_Xn,
                            stride_Ikm, stride_Ikn,
                            BLOCK_N: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    best_vals = tl.full([8], -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros([8], dtype=tl.int32)
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        vals = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask_n, other=-float('inf'))
        for j in range(8):
            # Select top-8
            if j < N:
                val_j = vals[j]
            else:
                val_j = -float('inf')
            for k in range(8):
                if val_j > best_vals[k]:
                    tmp = best_vals[k]
                    best_vals[k] = val_j
                    val_j = tmp
                    tmp_idx = best_idxs[k]
                    best_idxs[k] = j
                    idx = tmp_idx
    for j in range(8):
        tl.store(Indices_ptr + m * stride_Ikm + j * stride_Ikn, best_idxs[j])


@triton.jit
def gather_normalize_scale_kernel(Indices_ptr, X_ptr, Weight_ptr,
                                   M, N, routed_scaling_factor,
                                   stride_Ikm, stride_Ikn,
                                   stride_Xm, stride_Xn,
                                   stride_Wm, stride_Wn,
                                   BLOCK_N: tl.constexpr):
    # One program per token
    m = tl.program_id(0)
    sum_scores = 0.0
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N
        # Gather selected scores: read Indices[m, j], then X[m, Indices[m, j]]
        # We loop j=0..7 and gather each, accumulate sum_scores
        # For simplicity, assume we can compute selected scores by reading Indices and X_ptr via loads.
        # Triton doesn't support arbitrary vectorized gather, so we perform scalar loads for each j.
        # But here we can recompute by scanning X_ptr, selecting top-8; however, Indices already contain top-8.
        # To implement gather, we need to load each index and then load X[m, index]. We'll do that via scalar loop.
        # Build a small array of selected scores by iterating Indices columns.
        sel_vals = tl.zeros([8], dtype=tl.float32)
        for j in range(8):
            idx_j = tl.load(Indices_ptr + m * stride_Ikm + j * stride_Ikn)
            # idx_j is scalar int32; cast to int64 for pointer arithmetic
            val = tl.load(X_ptr + m * stride_Xm + idx_j.to(tl.int64) * stride_Xn)
            sel_vals[j] = val
        # Now normalize and scale
        denom = 0.0
        for j in range(8):
            denom += sel_vals[j]
        for j in range(8):
            w = tl.load(Weight_ptr + m * stride_Wm + j * stride_Wn)
            out = (sel_vals[j] / denom) * routed_scaling_factor
            tl.store(Weight_ptr + m * stride_Wm + j * stride_Wn, out)

# Entry point
class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float):
        super().__init__()
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        hidden_states: [M, K], float32, CUDA
        weight: [N, K] = [256, K], float32, CUDA
        expert_bias: [N], float32, CUDA
        returns:
        - topk_idx: [M, 8], int32
        - topk_weight: [M, 8], float32
        """
        # 1) Triton GEMV: logits = hidden_states @ weight.T -> [M, N], float32
        M = hidden_states.shape[0]
        N = 256
        K = hidden_states.shape[1]
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)

        # Strides for A [M, K], B [N, K], C [M, N]
        stride_Am = hidden_states.stride(0)
        stride_Ak = hidden_states.stride(1)
        stride_Bn = weight.stride(0)  # stride along N dimension
        stride_Bk = weight.stride(1)  # stride along K dimension
        stride_Cm = logits.stride(0)
        stride_Cn = logits.stride(1)

        # Launch GEMV
        BLOCK_N = 256
        BLOCK_K = 64
        grid = (M, (N + BLOCK_N - 1) // BLOCK_N)
        gemv_linear_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            stride_Am, stride_Ak,
            stride_Bn, stride_Bk,
            stride_Cm, stride_Cn,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # 2) Triton sigmoid + add expert bias -> scores_for_routing
        scores = torch.empty_like(logits)
        stride_Sm = scores.stride(0)
        stride_Sn = scores.stride(1)
        stride_Bn_bias = expert_bias.stride(0)
        sigmoid_bias_kernel[(M,)](
            logits, expert_bias, scores,
            M, N,
            stride_Sm, stride_Sn,
            stride_Bn_bias,
            stride_Sm, stride_Sn,
            BLOCK_N=256,
            num_warps=4,
        )

        # 3) Triton: per-group top-2 aggregation -> group_scores [M, 8]
        group_scores = torch.empty((M, 8), device=scores.device, dtype=torch.float32)
        # Reshape scores into [M, 8, 32] and call kernel
        stride_Sm_gs = scores.stride(0)
        stride_Sg_gs = 32  # group dimension stride is elements (32)
        stride_Sexp_gs = 1  # contiguous within group
        top2_per_group_kernel[(M * 8,)](
            scores, group_scores,
            M, 8, 32,
            stride_Sm_gs, stride_Sg_gs, stride_Sexp_gs,
            BLOCK_E=32,
            num_warps=4,
        )

        # 4) Triton arg-topk groups (K=4) -> group_idx [M, 4]
        group_idx = torch.empty((M, 4), device=group_scores.device, dtype=torch.int32)
        argtopk_groups_kernel[(M,)](
            group_scores, group_idx,
            M, 4,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            BLOCK_K=4,
            num_warps=4,
        )

        # 5) Triton scatter group_idx to group_mask [M, 8]
        group_mask = torch.empty((M, 8), device=group_scores.device, dtype=torch.float32)
        scatter_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M, 8,
            group_idx.stride(0), group_idx.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            BLOCK_G=8,
            num_warps=4,
        )

        # 6) Triton expand group_mask [M, 8] to expanded_mask [M, N]
        expanded_mask = torch.empty((M, N), device=group_scores.device, dtype=torch.float32)
        expand_mask_to_N_kernel[(M,)](
            group_mask, expanded_mask,
            M, 8, N,
            group_mask.stride(0), group_mask.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1),
            BLOCK_N=256,
            num_warps=4,
        )

        # 7) Triton mask non-selected group scores to -inf in masked_scores
        masked_scores = torch.empty_like(scores)
        mask_scores_kernel[(M,)](
            scores, expanded_mask, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            BLOCK_N=256,
            num_warps=4,
        )

        # 8) Triton arg-topk experts (K=8) on masked_scores -> topk_idx [M, 8]
        topk_idx = torch.empty((M, 8), device=group_scores.device, dtype=torch.int32)
        argtopk_experts_kernel[(M,)](
            masked_scores, topk_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            BLOCK_N=256,
            num_warps=4,
        )

        # 9) Triton gather and normalize, then scale to produce topk_weight [M, 8]
        # We need to read selected scores from masked_scores at topk_idx. Triton kernel gather:
        topk_weight = torch.empty((M, 8), device=group_scores.device, dtype=torch.float32)
        # To implement gather in Triton: read indices and load corresponding values from masked_scores
        # We can reinitialize topk_weight to zeros and then fill by loading gathered values.
        # However, Triton kernels typically write outputs; to fill weight, we'll use a gather-normalize-scale kernel that writes final weight.
        # The provided environment expects returning topk_idx and topk_weight; but gathering from masked_scores requires reading per-index which Triton does via loads using indices vector. We'll emulate by computing selected scores directly via argtopk_experts_kernel output and then normalize. For simplicity, we read masked_scores and use topk_idx to compute gathered values in Triton:
        # Build a kernel that fills topk_weight with gathered normalized scaled values.
        # We'll use gather_normalize_scale_kernel but we need selected_scores; since Triton doesn't provide a way to return gathered values, we instead compute gathered values here by reading masked_scores at indices. For correctness, we'll read masked_scores at indices to get selected_scores. Triton can do this by loading vector indexed by Indices_ptr.

        # Initialize topk_weight to zeros
        topk_weight.zero_()
        # Launch gather-normalize-scale kernel that writes final weights
        gather_normalize_scale_kernel[(M,)](
            topk_idx, masked_scores, topk_weight,
            M, N, self.routed_scaling_factor,
            topk_idx.stride(0), topk_idx.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            BLOCK_N=256,
            num_warps=4,
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
