import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn):
    # Compute y = sigmoid(x) + bias[n] for each element (m, n)
    m = tl.program_id(0)
    n = tl.program_id(1)
    x = tl.load(X_ptr + m * stride_Xm + n * stride_Xn)
    b = tl.load(Bias_ptr + n)
    y = 1.0 / (1.0 + tl.exp(-x)) + b
    tl.store(Y_ptr + m * stride_Ym + n * stride_Yn, y)


@triton.jit
def top2_per_group_kernel(S_ptr, Top_ptr,
                          M, G, E,
                          stride_Sm, stride_Sg, stride_Se,
                          stride_Tpg, stride_Tpe):
    # Compute per-group top-2 scores and sum them: S: [M, G, E], Top: [M, G, 2]
    m = tl.program_id(0)
    g = tl.program_id(1)
    max1 = tl.full((), -float('inf'), tl.float32)
    for e in range(E):
        val = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Se)
        if val > max1:
            max1 = val
    max2 = tl.full((), -float('inf'), tl.float32)
    for e in range(E):
        val = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Se)
        if val > max2 and val != max1:
            max2 = val
    tl.store(Top_ptr + m * stride_Tpg + g * stride_Tpe, max1 + max2)


@triton.jit
def topk_groups_arg_kernel(S_ptr, TopIdx_ptr,
                            M, G, K,
                            stride_Sm, stride_Sg,
                            stride_Tim, stride_Tik):
    # Per-token top-K groups: S: [M, G], TopIdx: [M, K] int32
    m = tl.program_id(0)
    best_vals = tl.zeros([K], dtype=tl.float32)  # initialize to -inf
    best_idx = tl.zeros([K], dtype=tl.int32)
    for i in range(K):
        best_vals[i] = -float('inf')
        best_idx[i] = -1
    for g in range(G):
        val = tl.load(S_ptr + m * stride_Sm + g * stride_Sg)
        # Insertion into sorted array (descending)
        for i in range(K):
            if val > best_vals[i]:
                # Shift down
                for j in range(K - 1, i, -1):
                    best_vals[j] = best_vals[j - 1]
                    best_idx[j] = best_idx[j - 1]
                best_vals[i] = val
                best_idx[i] = g
                break
    for i in range(K):
        tl.store(TopIdx_ptr + m * stride_Tim + i * stride_Tik, best_idx[i])


@triton.jit
def expand_group_mask_kernel(GroupMask_ptr, MaskExp_ptr,
                              M, G, E,
                              stride_Gm, stride_Gg,
                              stride_ME_m, stride_ME_e):
    # GroupMask: [M, G] float32 mask per group; MaskExp: [M, E*G] float32 mask expanded
    # Each selected group's 32 experts are set to 1.0
    m = tl.program_id(0)
    for g in range(G):
        flag = tl.load(GroupMask_ptr + m * stride_Gm + g * stride_Gg)  # 0.0 or 1.0
        if flag == 1.0:
            # For each expert in this group: e in [0..E-1]
            for e in range(E):
                # Linear index for MaskExp: offset by group
                # We write to all experts in this group starting at m*E*G + g*E + e
                base = m * (G * E) + g * E
                tl.store(MaskExp_ptr + base + e, 1.0)
        # If flag == 0.0, leave MaskExp entries for this group at 0.0 (allocated zeros)


@triton.jit
def masked_scores_kernel(S_ptr, MaskExp_ptr, SMasked_ptr,
                          M, N,
                          stride_Sm, stride_Sn,
                          stride_MEm, stride_MEe,
                          stride_SMm, stride_SMn):
    # S: [M, N], MaskExp: [M, N], SMasked: [M, N]
    m = tl.program_id(0)
    for n in range(N):
        mask = tl.load(MaskExp_ptr + m * stride_MEm + n * stride_MEe)
        val = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        if mask == 0.0:
            val = -float('inf')
        tl.store(SMasked_ptr + m * stride_SMm + n * stride_SMn, val)


@triton.jit
def topk_experts_arg_kernel(S_ptr, TopIdx_ptr,
                             M, N, K,
                             stride_Sm, stride_Sn,
                             stride_Tim, stride_Tik):
    # Per-token top-K experts: S: [M, N], TopIdx: [M, K] int32
    m = tl.program_id(0)
    best_vals = tl.zeros([K], dtype=tl.float32)  # initialize to -inf
    best_idx = tl.zeros([K], dtype=tl.int32)
    for i in range(K):
        best_vals[i] = -float('inf')
        best_idx[i] = -1
    for n in range(N):
        val = tl.load(S_ptr + m * stride_Sm + n * stride_Sn)
        for i in range(K):
            if val > best_vals[i]:
                # Shift down
                for j in range(K - 1, i, -1):
                    best_vals[j] = best_vals[j - 1]
                    best_idx[j] = best_idx[j - 1]
                best_vals[i] = val
                best_idx[i] = n
                break
    for i in range(K):
        tl.store(TopIdx_ptr + m * stride_Tim + i * stride_Tik, best_idx[i])


@triton.jit
def normalize_and_scale_kernel(TopIdx_ptr, Selected_ptr, Scale, TopWeight_ptr,
                               M, K,
                               stride_Tim, stride_Tik,
                               stride_Sm, stride_Sn,
                               stride_TWm, stride_TWk):
    # Selected: [M, K], TopIdx: [M, K] int32, TopWeight: [M, K]
    m = tl.program_id(0)
    total = tl.full((), 0.0, tl.float32)
    for k in range(K):
        idx = tl.load(TopIdx_ptr + m * stride_Tim + k * stride_Tik)  # int32
        val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sn)  # gather selected score
        total += val
    total = tl.where(total > 0.0, total, 1.0)
    for k in range(K):
        idx = tl.load(TopIdx_ptr + m * stride_Tim + k * stride_Tik)
        val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sn)
        w = (val / total) * Scale
        tl.store(TopWeight_ptr + m * stride_TWm + k * stride_TWk, w)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                weight: torch.Tensor,
                expert_bias: torch.Tensor,
                routed_scaling_factor: float):
        # Ensure inputs are CUDA tensors and float32
        device = hidden_states.device
        M = hidden_states.shape[0]  # num_tokens
        K_hidden = hidden_states.shape[1]  # hidden_dim
        N = weight.shape[0]  # num_experts = 256
        G = 8
        E = 32

        hidden_states = hidden_states.to(torch.float32).contiguous()
        weight = weight.to(torch.float32).contiguous()
        expert_bias = expert_bias.to(torch.float32).contiguous()

        # 1) Compute logits using PyTorch F.linear: logits [M, N]
        logits = torch.nn.functional.linear(hidden_states, weight)  # [M, N]

        # 2) Apply sigmoid and add expert bias via Triton
        scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_sigmoid = (M, N)
        sigmoid_bias_kernel[grid_sigmoid](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1)
        )

        # 3) Reshape for group routing: [M, G=8, E=32]
        group_scores = scores.view(M, G, E)  # [M, 8, 32]

        # 4) Top-2 per group: [M, 8, 2]
        top2_vals = torch.empty((M, G, 2), device=device, dtype=torch.float32)
        grid_top2 = (M, G)
        top2_per_group_kernel[grid_top2](
            scores, top2_vals,
            M, G, E,
            scores.stride(0), scores.stride(1), scores.stride(2),
            top2_vals.stride(0), top2_vals.stride(2)
        )

        # 5) Per-token top-4 groups: [M, 4] via Triton arg-topk
        # We need per-token top-4 among groups based on summed top2_vals
        group_score_per_token = torch.empty((M, G), device=device, dtype=torch.float32)
        for m in range(M):
            for g in range(G):
                group_score_per_token[m, g] = top2_vals[m, g, 0] + top2_vals[m, g, 1]
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        grid_top4 = (M,)
        topk_groups_arg_kernel[grid_top4](
            group_score_per_token, group_idx,
            M, G, 4,
            group_score_per_token.stride(0), group_score_per_token.stride(1),
            group_idx.stride(0), group_idx.stride(1)
        )

        # 6) Build group mask [M, 8] in Triton (or PyTorch): 1.0 for selected groups
        group_mask = torch.zeros((M, G), device=device, dtype=torch.float32)
        # Set selected groups to 1.0 based on group_idx
        for m in range(M):
            for j in range(4):
                g = int(group_idx[m, j].item())
                if 0 <= g < G:
                    group_mask[m, g] = 1.0

        # 7) Expand group mask to expert level [M, N] in Triton
        mask_exp = torch.empty((M, G * E), device=device, dtype=torch.float32)  # we'll store in [M, E*G] style
        grid_expand = (M,)
        expand_group_mask_kernel[grid_expand](
            group_mask, mask_exp,
            M, G, E,
            group_mask.stride(0), group_mask.stride(1),
            mask_exp.stride(0), 1  # stride_ME_e per expert; since we write linearly, use 1 for simplicity
        )
        # Note: In Triton, we wrote linearly into mask_exp with base m*(G*E) + g*E + e, so this layout is correct.

        # 8) Masked scores: set non-selected experts to -inf in Triton
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        grid_masked = (M,)
        masked_scores_kernel[grid_masked](
            scores, mask_exp, masked_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            mask_exp.stride(0), 1,  # stride over expert dimension is 1 here
            masked_scores.stride(0), masked_scores.stride(1)
        )

        # 9) Per-token top-8 experts from masked scores via Triton
        topk_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        grid_topk_exp = (M,)
        topk_experts_arg_kernel[grid_topk_exp](
            masked_scores, topk_idx,
            M, N, 8,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1)
        )

        # 10) Normalize and scale selected weights via Triton
        selected_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        # Gather selected scores from original scores at indices topk_idx
        # Since masked_scores has -inf for non-selected, topk_idx selects from actual values.
        for m in range(M):
            for k in range(8):
                idx = int(topk_idx[m, k].item())
                selected_scores[m, k] = scores[m, idx]

        topk_weight = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid_norm = (M,)
        normalize_and_scale_kernel[grid_norm](
            topk_idx, selected_scores, routed_scaling_factor,
            M, 8,
            topk_idx.stride(0), topk_idx.stride(1),
            selected_scores.stride(0), selected_scores.stride(1),
            topk_weight.stride(0), topk_weight.stride(1)
        )

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
