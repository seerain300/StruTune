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
    # One program per row m (token)
    m = tl.program_id(0)
    # Loop over N in blocks; for each block, compute dot with K blocks
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Loop over K dimension in blocks
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            # Load A[m, k] for this chunk
            a = tl.load(A_ptr + m * stride_Am + k_offsets * stride_Ak,
                        mask=k_offsets < K, other=0.0)  # [BLOCK_K]
            # Load B[n, k] for this chunk (B is [N, K], we want per token m, per expert n)
            b = tl.load(B_ptr + n_offsets[:, None] * stride_Bn + k_offsets[None, :] * stride_Bk,
                        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
                        other=0.0)  # [BLOCK_N, BLOCK_K]
            # Accumulate dot products for each column in this block: acc += sum_k a[k] * b[:, k]
            for kk in range(BLOCK_K):
                acc += b[:, kk] * a[kk]
        # Store results for this block of columns
        tl.store(C_ptr + m * stride_Cm + n_offsets * stride_Cn, acc, mask=(n_offsets < N))


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn):
    # 2D grid: (M, N)
    m = tl.program_id(0)
    n = tl.program_id(1)
    x = tl.load(X_ptr + m * stride_Xm + n * stride_Xn)
    b = tl.load(Bias_ptr + n)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(Y_ptr + m * stride_Ym + n * stride_Yn, y)


@triton.jit
def top2_per_group_kernel(S_ptr, Top2_ptr,
                          M, G, E,
                          stride_Sm, stride_Sg, stride_Se,
                          stride_Tpg, stride_Tpe):
    # Each program computes top-2 per (token m, group g)
    m = tl.program_id(0)
    g = tl.program_id(1)
    # Compute base for this group: expert indices are [g*E : (g+1)*E)
    base = g * E
    # Initialize best1 and best2 to -inf
    best1 = tl.full((), -float("inf"), tl.float32)
    best2 = tl.full((), -float("inf"), tl.float32)
    for e in range(E):
        idx = base + e
        s = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Se)  # Note: passing S as [M, G, E] via strides
        if s > best1:
            best2 = best1
            best1 = s
        elif s > best2:
            best2 = s
    # Store top-2 values (sorted order expected: best1 >= best2)
    tl.store(Top2_ptr + m * stride_Tpg + 0 * stride_Tpe, best1)
    tl.store(Top2_ptr + m * stride_Tpg + 1 * stride_Tpe, best2)


@triton.jit
def argtopk_groups_kernel(S_ptr, GroupIdx_ptr,
                          M, G,
                          stride_Sm, stride_Sg, stride_Se,
                          stride_GIm, stride_GIg):
    # Each program computes top-4 groups for a given token m using arg-topk
    m = tl.program_id(0)
    # Maintain best4 values and indices
    best0 = tl.full((), -float("inf"), tl.float32)
    best1 = tl.full((), -float("inf"), tl.float32)
    best2 = tl.full((), -float("inf"), tl.float32)
    best3 = tl.full((), -float("inf"), tl.float32)
    idx0 = tl.full((), -1, tl.int32)
    idx1 = tl.full((), -1, tl.int32)
    idx2 = tl.full((), -1, tl.int32)
    idx3 = tl.full((), -1, tl.int32)

    # Scan groups and maintain best4
    for g in range(G):
        v = tl.load(S_ptr + m * stride_Sm + g * stride_Sg)  # load group score for this token
        if v > best0:
            best3 = best2
            best2 = best1
            best1 = best0
            best0 = v
            idx3 = idx2
            idx2 = idx1
            idx1 = idx0
            idx0 = g
        elif v > best1:
            best3 = best2
            best2 = best1
            best1 = v
            idx3 = idx2
            idx2 = idx1
            idx1 = g
        elif v > best2:
            best3 = best2
            best2 = v
            idx3 = idx2
            idx2 = g
        elif v > best3:
            best3 = v
            idx3 = g

    # Store indices: positions 0..3
    tl.store(GroupIdx_ptr + m * stride_GIm + 0 * stride_GIg, idx0)
    tl.store(GroupIdx_ptr + m * stride_GIm + 1 * stride_GIg, idx1)
    tl.store(GroupIdx_ptr + m * stride_GIm + 2 * stride_GIg, idx2)
    tl.store(GroupIdx_ptr + m * stride_GIm + 3 * stride_GIg, idx3)


@triton.jit
def mask_groups_to_experts_kernel(S_ptr, GroupMask_ptr, MaskExp_ptr,
                                   M, G, E,
                                   stride_Sm, stride_Sg, stride_Se,
                                   stride_Gm, stride_Gg,
                                   stride_Mm, stride_Me):
    # For each token m, build mask for all experts indicating whether they belong to selected groups
    m = tl.program_id(0)
    # Group mask [G]: 1 for selected, 0 otherwise (GroupMask is provided as float mask)
    for g in range(G):
        gm = tl.load(GroupMask_ptr + m * stride_Gm + g * stride_Gg)  # float 0/1
        base = g * E
        for e in range(E):
            expert_id = base + e
            val = gm  # 1.0 or 0.0
            tl.store(MaskExp_ptr + m * stride_Mm + expert_id * stride_Me, val)


@triton.jit
def masked_scores_kernel(S_ptr, MaskExp_ptr, Masked_ptr,
                          M, N,
                          stride_Sm, stride_Sn,
                          stride_Mm, stride_Me,
                          stride_Msm, stride_Msn):
    # Apply mask to scores: if mask==0, set to -inf; else keep
    m = tl.program_id(0)
    for n in range(N):
        s = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        mask = tl.load(MaskExp_ptr + m * stride_Mm + n * stride_Me)
        masked = tl.where(mask > 0.0, s, -float("inf"))
        tl.store(Masked_ptr + m * stride_Msm + n * stride_Msn, masked)


@triton.jit
def topk_experts_kernel(Masked_ptr, TopKIdx_ptr,
                         M, N,
                         stride_Msm, stride_Msn,
                         stride_TKm, stride_TKk):
    # For each token m, compute top-8 indices from masked scores
    m = tl.program_id(0)
    # We'll maintain a small array of best-8; note Triton allows scalar loops up to reasonable extents.
    best = [tl.full((), -float("inf"), tl.float32) for _ in range(8)]
    idx = [tl.full((), -1, tl.int32) for _ in range(8)]
    for n in range(N):
        score = tl.load(Masked_ptr + m * stride_Msm + n * stride_Msn)
        # Insertion into sorted top-8 (descending)
        for i in range(8):
            if score > best[i]:
                # Shift down
                for j in range(7, i, -1):
                    best[j] = best[j - 1]
                    idx[j] = idx[j - 1]
                best[i] = score
                idx[i] = n
                break
    # Store indices 0..7
    for i in range(8):
        tl.store(TopKIdx_ptr + m * stride_TKm + i * stride_TKk, idx[i])


@triton.jit
def normalize_and_scale_kernel(TopVals_ptr, TopWeight_ptr,
                               M, K,
                               stride_TVm, stride_TVk,
                               stride_TWm, stride_TWk,
                               Scale):
    m = tl.program_id(0)
    total = tl.full((), 0.0, tl.float32)
    for k in range(K):
        val = tl.load(TopVals_ptr + m * stride_TVm + k * stride_TVk)
        total += val
    total = tl.where(total > 0.0, total, tl.full((), 1.0, tl.float32))
    for k in range(K):
        val = tl.load(TopVals_ptr + m * stride_TVm + k * stride_TVk)
        weight = val / total * Scale
        tl.store(TopWeight_ptr + m * stride_TWm + k * stride_TWk, weight)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,
                expert_bias: torch.Tensor,
                routed_scaling_factor: float):
        # Ensure CUDA and float32
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        hidden_states = hidden_states.to(torch.float32)
        weight = weight.to(torch.float32)
        expert_bias = expert_bias.to(torch.float32)

        # 1) Compute logits using Triton GEMV: logits = hidden_states @ weight.T
        M = hidden_states.shape[0]  # num_tokens
        N = weight.shape[0]         # num_experts = 256
        K_hidden = hidden_states.shape[1]  # hidden_dim
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid_gemm = (M,)
        gemv_linear_kernel[grid_gemm](
            hidden_states, weight, logits,
            M, N, K_hidden,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_N=128, BLOCK_K=64
        )

        # 2) Sigmoid + bias via Triton
        scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid_sigmoid = (M, N)
        sigmoid_bias_kernel[grid_sigmoid](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1)
        )

        # 3) Reshape for group routing: [M, G=8, E=32]
        G = 8
        E = 32
        group_scores = scores.view(M, G, E)  # [M, 8, 32]

        # 4) Top-2 per group: [M, 8, 2]
        top2_vals = torch.empty((M, G, 2), device=hidden_states.device, dtype=torch.float32)
        grid_top2 = (M, G)
        top2_per_group_kernel[grid_top2](
            scores, top2_vals,
            M, G, E,
            scores.stride(0), scores.stride(1), scores.stride(2),
            top2_vals.stride(0), top2_vals.stride(2)
        )

        # 5) Per-token top-4 groups: [M, 4] using argtopk in Triton
        group_idx = torch.empty((M, 4), device=hidden_states.device, dtype=torch.int32)
        grid_top4 = (M,)
        argtopk_groups_kernel[grid_top4](
            group_scores, group_idx,
            M, G,
            group_scores.stride(0), group_scores.stride(1), group_scores.stride(2),
            group_idx.stride(0), group_idx.stride(1)
        )

        # 6) Build group mask [M, 8] (1.0 for selected groups, 0 otherwise) - we will compute it in Triton below
        # but here we pass precomputed mask from PyTorch. However, to keep Triton-only, we recompute here via GroupIdx:
        # create a float mask tensor
        group_mask = torch.zeros((M, G), device=hidden_states.device, dtype=torch.float32)
        for m in range(M):
            for i in range(4):
                g = int(group_idx[m, i].item())
                if 0 <= g < G:
                    group_mask[m, g] = 1.0

        # 7) Expand group


def run(*args):
    return ModelNew()(*args)
