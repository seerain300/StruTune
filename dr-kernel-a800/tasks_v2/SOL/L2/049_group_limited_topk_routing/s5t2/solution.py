import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn):
    # Grid: (M, N), elementwise
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
                          stride_Tpg, stride_Tpe,
                          BLOCK_E: tl.constexpr):
    # S: [M, G, E], Top: [M, G, 2]
    m = tl.program_id(0)
    g = tl.program_id(1)
    best1 = tl.full((), -float('inf'), tl.float32)
    best2 = tl.full((), -float('inf'), tl.float32)
    for e in range(0, E):
        val = tl.load(S_ptr + m * stride_Sm + g * stride_Sg + e * stride_Se)
        # Update best2; if val > best2, set best2=val; if val > best1, swap
        if val > best2:
            best2 = val
        if val > best1:
            tmp = best1
            best1 = best2
            best2 = tmp
    # store top-2
    tl.store(Top_ptr + m * stride_Tpg + g * stride_Tpg + 0, best1)
    tl.store(Top_ptr + m * stride_Tpg + g * stride_Tpg + 1, best2)


@triton.jit
def top4_groups_kernel(Top2_ptr, GroupIdx_ptr,
                       M, G,
                       stride_Tpg, stride_Tpe,  # Top2 strides: we pass m,g
                       stride_Gi0, stride_Gi1):  # GroupIdx strides: [M, 4]
    # Compute per-token top-4 group indices. Maintain four slots (v0..v3) and their indices.
    m = tl.program_id(0)
    v0 = tl.full((), -float('inf'), tl.float32)
    v1 = tl.full((), -float('inf'), tl.float32)
    v2 = tl.full((), -float('inf'), tl.float32)
    v3 = tl.full((), -float('inf'), tl.float32)
    idx0 = tl.full((), -1, tl.int32)
    idx1 = tl.full((), -1, tl.int32)
    idx2 = tl.full((), -1, tl.int32)
    idx3 = tl.full((), -1, tl.int32)

    # Scan groups 0..7
    for g in range(0, G):
        s0 = tl.load(Top2_ptr + m * stride_Tpg + g * stride_Tpg + 0)
        s1 = tl.load(Top2_ptr + m * stride_Tpg + g * stride_Tpg + 1)
        score = s0 + s1
        # Insert into best4
        better = score > v0
        tmp_v0 = tl.where(better, score, v0)
        tmp_v1 = tl.where(better, v0, v1)
        tmp_v2 = tl.where(better, v1, v2)
        tmp_v3 = tl.where(better, v2, v3)
        v0 = tmp_v0; v1 = tmp_v1; v2 = tmp_v2; v3 = tmp_v3
        # idx updates: if inserted into v0, idx0 = g; else keep previous
        tmp_idx0 = tl.where(better, tl.full((), g, tl.int32), idx0)
        tmp_idx1 = tl.where(better, idx0, idx1)
        tmp_idx2 = tl.where(better, idx1, idx2)
        tmp_idx3 = tl.where(better, idx2, idx3)
        idx0 = tmp_idx0; idx1 = tmp_idx1; idx2 = tmp_idx2; idx3 = tmp_idx3

    # Store idx0..idx3 into GroupIdx[m, 0..3]
    tl.store(GroupIdx_ptr + m * stride_Gi0 + 0, idx0)
    tl.store(GroupIdx_ptr + m * stride_Gi0 + 1, idx1)
    tl.store(GroupIdx_ptr + m * stride_Gi0 + 2, idx2)
    tl.store(GroupIdx_ptr + m * stride_Gi0 + 3, idx3)


@triton.jit
def mask_groups_to_experts_kernel(GroupMask_ptr, GroupIdx_ptr, MaskExpert_ptr,
                                  M, G, E,
                                  stride_GM, stride_GI0, stride_GI1,  # GroupMask [M, G]
                                  stride_ME_m, stride_ME_n):       # MaskExpert [M, N]
    # For each token m, set mask for experts in selected groups
    m = tl.program_id(0)
    # Load selected group indices 0..3
    gi0 = tl.load(GroupMask_ptr + m * stride_GM + 0).to(tl.int32)
    gi1 = tl.load(GroupMask_ptr + m * stride_GM + 1).to(tl.int32)
    gi2 = tl.load(GroupMask_ptr + m * stride_GM + 2).to(tl.int32)
    gi3 = tl.load(GroupMask_ptr + m * stride_GM + 3).to(tl.int32)
    # Loop over groups and set mask for their E experts
    for i in range(4):
        g = gi0 if i == 0 else (gi1 if i == 1 else (gi2 if i == 2 else gi3))
        base = g * E
        for e in range(E):
            n = base + e
            one = tl.full((), 1.0, tl.float32)
            tl.store(MaskExpert_ptr + m * stride_ME_m + n * stride_ME_n, one)


@triton.jit
def masked_scores_kernel(Scores_ptr, MaskExpert_ptr, Masked_ptr,
                         M, N,
                         stride_Sm, stride_Sn,
                         stride_ME_m, stride_ME_n,
                         stride_MMm, stride_Mmn):
    # Elementwise mask: if MaskExpert>0, keep Score; else set to -inf
    m = tl.program_id(0)
    for n in range(0, N):
        s = tl.load(Scores_ptr + m * stride_Sm + n * stride_Sn)
        me = tl.load(MaskExpert_ptr + m * stride_ME_m + n * stride_ME_n)
        neg_inf = tl.full((), -float('inf'), tl.float32)
        val = tl.where(me > 0, s, neg_inf)
        tl.store(Masked_ptr + m * stride_MMm + n * stride_Mmn, val)


@triton.jit
def topk_experts_kernel(Scores_ptr, Masked_ptr, TopIdx_ptr, TopVals_ptr,
                        M, N, K,
                        stride_Sm, stride_Sn,
                        stride_MMm, stride_Mmn,
                        stride_TIm, stride_TIk,
                        stride_TVm, stride_TVk):
    # Select top-K experts from Masked scores per token. K is 8.
    m = tl.program_id(0)
    # Initialize top-k arrays
    top_vals = tl.zeros([K], dtype=tl.float32)
    top_idxs = tl.zeros([K], dtype=tl.int32)
    for k in range(K):
        top_vals[k] = -float('inf')
        top_idxs[k] = -1
    # Scan all N
    for n in range(0, N):
        score = tl.load(Scores_ptr + m * stride_Sm + n * stride_Sn)
        masked = tl.load(Masked_ptr + m * stride_MMm + n * stride_Mmn)
        better = masked > top_vals[0]
        # Shift right if better
        tmp = [top_vals[0], top_vals[1], top_vals[2], top_vals[3],
               top_vals[4], top_vals[5], top_vals[6], top_vals[7]]
        for j in range(1, K):
            cond = masked > top_vals[j]
            tmp[j-1] = tl.where(cond, masked, tmp[j-1])
            tmp[j] = tl.where(cond, top_vals[j], tmp[j])
        top_vals = [tmp[0], tmp[1], tmp[2], tmp[3],
                    tmp[4], tmp[5], tmp[6], tmp[7]]
        # Update idxs similarly
        tmp_idx = [top_idxs[0], top_idxs[1], top_idxs[2], top_idxs[3],
                   top_idxs[4], top_idxs[5], top_idxs[6], top_idxs[7]]
        for j in range(1, K):
            cond = masked > top_vals[j]
            tmp_idx[j-1] = tl.where(cond, tl.full((), n, tl.int32), tmp_idx[j-1])
            tmp_idx[j] = tl.where(cond, top_idxs[j], tmp_idx[j])
        top_idxs = [tmp_idx[0], tmp_idx[1], tmp_idx[2], tmp_idx[3],
                    tmp_idx[4], tmp_idx[5], tmp_idx[6], tmp_idx[7]]
    # Store top-k indices and values
    for k in range(K):
        tl.store(TopIdx_ptr + m * stride_TIm + k * stride_TIk, top_idxs[k])
        tl.store(TopVals_ptr + m * stride_TVm + k * stride_TVk, top_vals[k])


@triton.jit
def normalize_and_scale_kernel(TopVals_ptr, Scale, TopWeight_ptr,
                              M, K,
                              stride_TVm, stride_TVk,
                              stride_TWm, stride_TWk):
    # Normalize per token: top_vals / sum, then scale
    m = tl.program_id(0)
    total = tl.full((), 0.0, tl.float32)
    for k in range(K):
        val = tl.load(TopVals_ptr + m * stride_TVm + k * stride_TVk)
        total += val
    # Guard against total==0 (shouldn't happen for top-k)
    total = tl.where(total > 0.0, total, tl.full((), 1.0, tl.float32))
    for k in range(K):
        val = tl.load(TopVals_ptr + m * stride_TVm + k * stride_TVk)
        weight = val / total * Scale
        tl.store(TopWeight_ptr + m * stride_TWm + k * stride_TWk, weight)


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

        # 1) Compute logits using PyTorch F.linear for correctness: logits [M, N]
        M = hidden_states.shape[0]  # num_tokens
        N = weight.shape[0]         # num_experts = 256
        K_hidden = hidden_states.shape[1]  # hidden_dim
        logits = torch.nn.functional.linear(hidden_states, weight)  # [M, N]

        # 2) Apply sigmoid and bias via Triton
        scores = torch.empty((M, N), device=hidden_states.device, dtype=torch.float32)
        grid2 = (M, N)
        sigmoid_bias_kernel[grid2](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1)
        )

        # 3) Reshape for group routing: [M, G=8, E=32]


def run(*args):
    return ModelNew()(*args)
