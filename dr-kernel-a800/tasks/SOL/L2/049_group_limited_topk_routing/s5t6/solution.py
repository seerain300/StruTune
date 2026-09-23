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
    # Loop over N (experts) in blocks
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        # Loop over hidden dimension in blocks
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            # Load hidden for this token and chunk: A[m, k] -> [BLOCK_K]
            a = tl.load(A_ptr + m * stride_Am + k_offsets * stride_Ak,
                        mask=k_offsets < K, other=0.0)
            # Load weight for this chunk: B[n, k] -> [BLOCK_N, BLOCK_K]
            b = tl.load(B_ptr + n_offsets[:, None] * stride_Bn + k_offsets[None, :] * stride_Bk,
                        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
                        other=0.0)
            # Accumulate dot products per expert column
            for kk in range(BLOCK_K):
                acc += b[:, kk] * a[kk]
        # Store the block of logits
        tl.store(C_ptr + m * stride_Cm + n_offsets * stride_Cn, acc, mask=(n_offsets < N))


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn):
    # One program per element (m, n)
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
    # Compute top-2 per group for each token m
    m = tl.program_id(0)
    g = tl.program_id(1)
    # Initialize best values
    best1 = tl.full((), -1.0e20, tl.float32)
    best2 = tl.full((), -1.0e20, tl.float32)
    idx1 = tl.full((), -1, tl.int32)
    idx2 = tl.full((), -1, tl.int32)
    # Loop over experts in group g (size E)
    for e in range(0, E):
        score = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Se)
        # Insert into best1, best2
        if score > best1:
            best2 = best1
            idx2 = idx1
            best1 = score
            idx1 = tl.full((), e, tl.int32)
        elif score > best2:
            best2 = score
            idx2 = tl.full((), e, tl.int32)
    # Store results at positions (g,0) and (g,1) in Top2[m, g, :]
    tl.store(Top2_ptr + m * stride_Tpg + 0 * stride_Tpe, best1)
    tl.store(Top2_ptr + m * stride_Tpg + 1 * stride_Tpe, best2)


@triton.jit
def argtopk_groups_kernel(S_ptr, ArgTopK_ptr,
                          M, G,
                          stride_Sm, stride_Sg, stride_Se,
                          stride_ATm, stride_ATk):
    # Compute per-token top-4 group indices
    m = tl.program_id(0)
    # Load group scores for this token into a vector and compute argtopk manually
    group_scores = tl.zeros([G], dtype=tl.float32)
    for g in range(0, G):
        group_scores[g] = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + 0 * stride_Se)
    # Maintain k=4 best values and indices
    best = tl.zeros([4], dtype=tl.float32)
    idx = tl.zeros([4], dtype=tl.int32)
    for g in range(0, G):
        score = group_scores[g]
        # Insert into best positions
        if score > best[0]:
            best[1:] = best[:-1]
            best[0] = score
            idx[1:] = idx[:-1]
            idx[0] = tl.full((), g, tl.int32)
        elif score > best[1]:
            best[2:] = best[1:3]
            best[1] = score
            idx[2:] = idx[1:3]
            idx[1] = tl.full((), g, tl.int32)
        elif score > best[2]:
            best[3] = score
            idx[3] = tl.full((), g, tl.int32)
        elif score > best[3]:
            best[3] = score
            idx[3] = tl.full((), g, tl.int32)
    # Store indices
    for j in range(4):
        tl.store(ArgTopK_ptr + m * stride_ATm + j * stride_ATk, idx[j])


@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, Expanded_ptr,
                             M, G, E,
                             stride_GMm, stride_GMg,
                             stride_Epm, stride_Epe):
    # Expand group_mask [M, G] to [M, N] where N=G*E
    m = tl.program_id(0)
    # Load group_mask for this token
    for g in range(0, G):
        mask_val = tl.load(GroupMask_ptr + m * stride_GMm + g * stride_GMg)
        # Write to all experts in this group
        for e in range(0, E):
            expert_id = g * E + e
            tl.store(Expanded_ptr + m * stride_Epm + expert_id * stride_Epe, mask_val)


@triton.jit
def mask_scores_kernel(S_ptr, Mask_ptr, Masked_ptr,
                        M, N,
                        stride_Sm, stride_Sn,
                        stride_Mm, stride_Mn,
                        stride_Msm, stride_Msn):
    # Apply mask: if mask == 0, set score to -inf
    m = tl.program_id(0)
    for n in range(0, N):
        score = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        mask = tl.load(Mask_ptr + m * stride_Mm + n * stride_Mn)
        new_score = tl.where(mask == 0.0, -1.0e20, score)
        tl.store(Masked_ptr + m * stride_Msm + n * stride_Msn, new_score)


@triton.jit
def topk_experts_kernel(S_ptr, TopK_idx_ptr,
                         M, N, K_TOP,
                         stride_Sm, stride_Sn,
                         stride_TKm, stride_TKn):
    # Compute per-token top-K_TOP indices (descending)
    m = tl.program_id(0)
    vals = tl.zeros([N], dtype=tl.float32)
    idxs = tl.zeros([N], dtype=tl.int32)
    for n in range(0, N):
        vals[n] = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        idxs[n] = tl.full((), n, tl.int32)
    # Selection loop: pick top K_TOP
    for k in range(0, K_TOP):
        best = -1.0e20
        pos = -1
        for n in range(0, N):
            if vals[n] > best:
                best = vals[n]
                pos = n
        # Record index
        tl.store(TopK_idx_ptr + m * stride_TKm + k * stride_TKn, idxs[pos])
        # Mark as used by setting to -inf
        vals[pos] = -1.0e20


@triton.jit
def normalize_and_scale_kernel(TopK_idx_ptr, Selected_ptr, Scale,
                               M, N, K_TOP,
                               stride_TKm, stride_TKn,
                               stride_SMm, stride_SMn):
    # Gather selected scores and normalize per token
    m = tl.program_id(0)
    total = tl.full((), 0.0, tl.float32)
    for k in range(0, K_TOP):
        idx = tl.load(TopK_idx_ptr + m * stride_TKm + k * stride_TKn)
        # Gather score (we assume Selected_ptr already contains gathered scores)
        score = tl.load(Selected_ptr + m * stride_SMm + k * stride_SMn)
        total += score
    total = tl.where(total > 0.0, total, tl.full((), 1.0, tl.float32))
    for k in range(0, K_TOP):
        idx = tl.load(TopK_idx_ptr + m * stride_TKm + k * stride_TKn)
        score = tl.load(Selected_ptr + m * stride_SMm + k * stride_SMn)
        weight = score / total * Scale
        tl.store(Selected_ptr + m * stride_SMm + k * stride_SMn, weight)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,
                expert_bias: torch.Tensor,
                routed_scaling_factor: float):
        # Ensure inputs are CUDA tensors and float32
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All inputs must be CUDA tensors"
        hidden_states = hidden_states.to(torch.float32)
        weight = weight.to(torch.float32)
        expert_bias = expert_bias.to(torch.float32)

        # 1) Compute logits via Triton GEMV: [M, N] where N=256
        M = hidden_states.shape[0]
        N = weight.shape[0]
        K = hidden_states.shape[1]
        logits = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid_gmv = (M,)
        gemv_linear_kernel[grid_gmv](
            hidden_states, weight, logits,
            M, N, K,
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

        # 5) Per-token top-4 groups: [M, 4] via Triton
        group_idx = torch.empty((M, 4), device=hidden_states.device, dtype=torch.int32)
        grid_top4 = (M,)
        argtopk_groups_kernel[grid_top4](
            top2_vals[:, :, 0],  # use the first top2 component (sum of top-2 per group is what we select)
            group_idx,
            M, G,
            top2_vals.stride(0), top2_vals.stride(1), top2_vals.stride(2),
            group_idx.stride(0), group_idx.stride(1)
        )

        # 6) Build group mask [M, G] (1.0 for selected groups, 0 otherwise) via PyTorch using group_idx
        group_mask = torch.zeros((M, G), device=hidden_states.device, dtype=torch.float32)
        for m in range(M):
            for j in range(4):
                g = int(group_idx[m, j].item())
                if 0 <= g < G:
                    group_mask[m, g] = 1.0

        # 7) Expand group mask to expert level: [M, N] with N=G*E
        expanded_mask = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid_expanded = (M,)
        expand_group_mask_kernel[grid_expanded](
            group_mask, expanded_mask,
            M, G, E,
            group_mask.stride(0), group_mask.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1)
        )

        # 8) Mask scores: set non-selected to -inf
        masked_scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid_mask = (M, N)
        mask_scores_kernel[grid_mask](
            scores, expanded_mask, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            expanded_mask.stride(0), expanded_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1)
        )

        # 9) Select per-token top-8 experts from masked_scores
        topk_idx = torch.empty((M, 8), device=hidden_states.device, dtype=torch.int32)
        grid_topk = (M,)
        topk_experts_kernel[grid_topk](
            masked_scores, topk_idx,
            M, N, 8,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1)
        )

        # 10) Gather selected scores from original scores and normalize + scale
        selected_scores = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        # We need to gather scores at indices topk_idx: selected_scores[m, k] = scores[m, topk_idx[m,k]]
        for m in range(M):
            for k in range(8):
                idx = int(topk_idx[m, k].item())
                selected_scores[m, k] = scores[m, idx]
        # Normalize and scale via Triton
        topk_weight = torch.empty((M, 8), device=hidden_states.device, dtype=torch.float32)
        grid_norm = (M,)
        normalize_and_scale_kernel[grid_norm](
            topk_idx, selected_scores, routed_scaling_factor,
            M, N, 8,
            topk_idx.stride(0), topk_idx.stride(1),
            selected_scores.stride(0), selected_scores.stride(1),
            topk_weight.stride(0), topk_weight.stride(1)
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
