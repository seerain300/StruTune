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
    # One Triton program per token m
    m = tl.program_id(0)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    # Accumulate over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + m * stride_Am + k_offsets * stride_Ak,
                    mask=k_offsets < K, other=0.0)  # [BLOCK_K]
        # For each chunk of N, compute dot with 'a'
        for n_start in range(0, N, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            b = tl.load(B_ptr + n_offsets * stride_Bn + k_offsets * stride_Bk,
                        mask=(n_offsets < N)[:, None] & (k_offsets < K)[None, :],
                        other=0.0)  # [BLOCK_N, BLOCK_K]
            acc[n_start:n_start+BLOCK_N] += tl.sum(b * a[None, :], axis=1)
    # Store results
    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        tl.store(C_ptr + m * stride_Cm + n_offsets * stride_Cn,
                 acc[n_start:n_start+BLOCK_N], mask=n_offsets < N)


@triton.jit
def sigmoid_bias_kernel(X_ptr, Bias_ptr, Y_ptr,
                         M, N,
                         stride_Xm, stride_Xn,
                         stride_Ym, stride_Yn,
                         BLOCK: tl.constexpr):
    # 2D grid over tokens and expert blocks
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_offsets = n_block * BLOCK + tl.arange(0, BLOCK)
    mask = n_offsets < N
    x = tl.load(X_ptr + m * stride_Xm + n_offsets * stride_Xn, mask=mask, other=0.0)
    bias = tl.load(Bias_ptr + n_offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x)) + bias
    tl.store(Y_ptr + m * stride_Ym + n_offsets * stride_Yn, y, mask=mask)


@triton.jit
def group_top2_kernel(S_ptr, GroupS_ptr,
                      M, N, G, E,
                      stride_Sm, stride_Sn,
                      stride_GSm, stride_GSn,
                      BLOCK: tl.constexpr):
    # For each token m, compute per-group top-2 scores and sum
    for m in range(0, M):
        base = m * stride_Sm
        for g in range(0, G):
            start = g * E
            vals = []
            for e in range(0, E):
                val = tl.load(S_ptr + base + (start + e) * stride_Sn)
                vals.append(val)
            top1 = -float('inf')
            top2 = -float('inf')
            for i in range(E):
                v = vals[i]
                if v > top1:
                    top2 = top1
                    top1 = v
                elif v > top2:
                    top2 = v
            tl.store(GroupS_ptr + m * stride_GSm + g * stride_GSn, top1 + top2)


@triton.jit
def topk_arg_kernel(V_ptr, Indices_ptr,
                    M, N, K,
                    stride_Vm, stride_Vn,
                    stride_Ism, stride_Isn,
                    BLOCK: tl.constexpr):
    # Generic arg-topk per row
    for m in range(0, M):
        base = m * stride_Vm
        vals = tl.load(V_ptr + base + tl.arange(0, N) * stride_Vn)
        topk_vals = tl.zeros([K], dtype=tl.float32)
        topk_idx = tl.zeros([K], dtype=tl.int32)
        for i in range(N):
            v = vals[i]
            for j in range(K):
                if v > topk_vals[j]:
                    for l in range(K - 1, 0, -1):
                        topk_vals[l] = topk_vals[l - 1]
                        topk_idx[l] = topk_idx[l - 1]
                    topk_vals[0] = v
                    topk_idx[0] = i
                    break
        tl.store(Indices_ptr + m * stride_Ism + tl.arange(0, K) * stride_Isn, topk_idx)


@triton.jit
def mask_groups_kernel(Scores_ptr, GroupMask_ptr, Masked_ptr,
                        M, N, G, E,
                        stride_Sm, stride_Sn,
                        stride_Gm, stride_Gn,
                        stride_Mm, stride_Mn,
                        BLOCK: tl.constexpr):
    # For each token m, set scores to -inf for non-selected groups
    for m in range(0, M):
        base = m * stride_Sm
        for g in range(0, G):
            flag = tl.load(GroupMask_ptr + m * stride_Gm + g * stride_Gn)
            start = g * E
            for e in range(0, E):
                idx = start + e
                val = tl.load(Scores_ptr + base + idx * stride_Sn)
                new_val = tl.where(flag == 1, val, -float('inf'))
                tl.store(Masked_ptr + base + idx * stride_Mn, new_val)


@triton.jit
def normalize_scale_kernel(Indices_ptr, Selected_ptr, Scaling, Out_ptr,
                           M, N, K,
                           stride_Ism, stride_Isn,
                           stride_Sm, stride_Sn,
                           stride_Om, stride_On):
    # For each token m, normalize selected scores and scale
    for m in range(0, M):
        base_idx = m * stride_Ism
        # Compute sum of selected scores
        ssum = 0.0
        for k in range(K):
            idx = tl.load(Indices_ptr + base_idx + k * stride_Isn)
            val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sn)
            ssum += val
        inv = 1.0 / (ssum + 1e-20)
        for k in range(K):
            idx = tl.load(Indices_ptr + base_idx + k * stride_Isn)
            val = tl.load(Selected_ptr + m * stride_Sm + idx * stride_Sn) * inv * Scaling
            tl.store(Out_ptr + m * stride_Om + k * stride_On, val)


class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Dimensions
        M = hidden_states.shape[0]  # num_tokens
        K = hidden_states.shape[1]  # hidden_dim
        N = 256  # num_experts (constant as in original)

        device = hidden_states.device
        dtype = torch.float32

        # Ensure inputs are contiguous
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()
        expert_bias = expert_bias.contiguous()

        # 1) Compute logits via Triton GEMV: C[M, N] = hidden_states[M, K] @ weight[N, K]^T
        logits = torch.empty((M, N), device=device, dtype=dtype)
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (M,)
        gemv_linear_kernel[grid](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 2) Sigmoid + add expert bias via Triton
        scores = torch.empty((M, N), device=device, dtype=dtype)
        BLOCK = 128
        grid_sig = (M, (N + BLOCK - 1) // BLOCK)
        sigmoid_bias_kernel[grid_sig](
            logits, expert_bias.to(dtype), scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK=BLOCK
        )

        # 3) Compute per-token group top-2 scores and aggregated group_scores [M, 8]
        group_scores = torch.empty((M, 8), device=device, dtype=dtype)
        G = 8
        E = N // G  # 32
        group_top2_kernel[(M,)](
            scores, group_scores,
            M, N, G, E,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1)
        )

        # 4) Per-token top-4 groups via Triton arg-topk
        group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        grid_tpk = (M,)
        topk_arg_kernel[grid_tpk](
            group_scores, group_idx,
            M, 8, 4,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1)
        )

        # 5) Build group mask and expand to [M, N], set -inf for non-selected groups using Triton
        group_mask = torch.empty((M, 8), device=device, dtype=dtype)
        # Manually set selected groups to 1.0
        for m in range(M):
            for g_sel in range(4):
                g = int(group_idx[m, g_sel].item())
                group_mask[m, g] = 1.0
        masked_scores = torch.empty((M, N), device=device, dtype=dtype)
        grid_mask = (M, (N + BLOCK - 1) // BLOCK)
        mask_groups_kernel[grid_mask](
            scores, group_mask, masked_scores,
            M, N, G, E,
            scores.stride(0), scores.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
            BLOCK=BLOCK
        )

        # 6) Per-token top-8 experts from masked scores via Triton
        topk_idx = torch.empty((M, 8), device=device, dtype=torch.int32)
        grid_topk = (M,)
        topk_arg_kernel[grid_topk](
            masked_scores, topk_idx,
            M, N, 8,
            masked_scores.stride(0), masked_scores.stride(1),
            topk_idx.stride(0), topk_idx.stride(1)
        )

        # 7) Normalize selected scores and scale via Triton
        selected_scores = torch.empty((M, 8), device=device, dtype=dtype)
        # Gather selected scores from original 'scores' at indices 'topk_idx'
        for m in range(M):
            for k in range(8):
                idx = int(topk_idx[m, k].item())
                selected_scores[m, k] = scores[m, idx]
        out_weight = torch.empty((M, 8), device=device, dtype=dtype)
        grid_norm = (M,)
        normalize_scale_kernel[grid_norm](
            topk_idx, selected_scores, self.routed_scaling_factor, out_weight,
            M, N, 8,
            topk_idx.stride(0), topk_idx.stride(1),
            selected_scores.stride(0), selected_scores.stride(1),
            out_weight.stride(0), out_weight.stride(1)
        )

        return topk_idx, out_weight


def run(*args):
    return ModelNew()(*args)
