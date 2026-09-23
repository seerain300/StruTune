import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_AxB_kernel(
    A_ptr,  # *const float, shape [M, K]
    B_ptr,  # *const float, shape [K, N] (B is weight.T)
    C_ptr,  # *float, shape [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    X_ptr,  # *const float, shape [M, N]
    B_ptr,  # *const float, shape [N]
    Y_ptr,  # *float, shape [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Load tile of X and bias B
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    x_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=x_mask, other=0.0)
    b = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    # Compute sigmoid(x) + b (broadcast over rows)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = s + b[None, :]
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, y, mask=x_mask)


@triton.jit
def _group_top2_sum_kernel(
    S_ptr,  # *const float, shape [M, N]
    GROUP_SCORES_ptr,  # *float, shape [M, 8]
    M, N,
    stride_sm, stride_sn,
    stride_gs_m, stride_gs_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    # Each program handles one token m
    offs_n = tl.arange(0, BLOCK_N)
    # We assume N is a multiple of 32; if not, mask handles boundary
    for g in range(8):
        base = g * 32
        col_idx = base + offs_n  # [32]
        mask = col_idx < N
        s_ptrs = S_ptr + (pid_m * stride_sm + col_idx * stride_sn)
        vals = tl.load(s_ptrs, mask=mask, other=-1e20)
        # Iterative top-2 selection within 32 elements
        best1 = -1e20
        best2 = -1e20
        # Scan cols in order to find top-2 (since BLOCK_N=32, this is fine)
        for j in range(32):
            v = vals[j]
            if v > best1:
                best2 = best1
                best1 = v
            elif v > best2:
                best2 = v
        group_scores_ptr = GROUP_SCORES_ptr + (pid_m * stride_gs_m + g * stride_gs_n)
        tl.store(group_scores_ptr, best1 + best2)


@triton.jit
def _group_top4_select_kernel(
    GROUP_SCORES_ptr,  # *const float, shape [M, 8]
    GROUP_IDX_ptr,     # *int32, shape [M, 4]
    M, N,
    stride_gs_m, stride_gs_n,
    stride_gi_m, stride_gi_n,
    BLOCK_M: tl.constexpr,
):
    # Simple per-row selection using while loops; N is not used here.
    for m in range(M):
        gs = tl.zeros((8,), dtype=tl.float32)
        for g in range(8):
            ptr = GROUP_SCORES_ptr + (m * stride_gs_m + g * stride_gs_n)
            gs[g] = tl.load(ptr)
        # Iterative argmax: select top-4
        selected = tl.zeros((4,), dtype=tl.int32)  # store indices 0..7
        for t in range(4):
            maxv = -1e20
            idx = -1
            for g in range(8):
                if gs[g] > maxv:
                    maxv = gs[g]
                    idx = g
            selected[t] = idx
            gs[idx] = -1e20  # exclude
        for t in range(4):
            ptr = GROUP_IDX_ptr + (m * stride_gi_m + t * stride_gi_n)
            tl.store(ptr, selected[t])


@triton.jit
def _final_top8_and_normalize_kernel(
    S_ptr,              # *const float, shape [M, N]
    GROUP_IDX_ptr,      # *const int32, shape [M, 4]
    TOPK_IDX_ptr,       # *int32, shape [M, 8]
    TOPK_WEIGHT_ptr,    # *float, shape [M, 8]
    M, N,
    stride_sm, stride_sn,
    stride_gi_m, stride_gi_n,
    stride_tki_m, stride_tki_n,
    stride_tkw_m, stride_tkw_n,
    routed_scaling_factor: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # We select final top-8 per row using iterative argmax across all columns
    for m in range(M):
        s_row = tl.zeros((N,), dtype=tl.float32)
        for n in range(N):
            ptr = S_ptr + (m * stride_sm + n * stride_sn)
            s_row[n] = tl.load(ptr)
        selected_idx = tl.zeros((8,), dtype=tl.int32)
        selected_val = tl.zeros((8,), dtype=tl.float32)
        for t in range(8):
            maxv = -1e20
            idx = -1
            for n in range(N):
                if s_row[n] > maxv:
                    maxv = s_row[n]
                    idx = n
            selected_idx[t] = idx
            selected_val[t] = maxv
            s_row[idx] = -1e20
        # Normalize and scale: sum = L1 norm of selected_val
        sum_val = 0.0
        for t in range(8):
            sum_val += selected_val[t]
        inv_sum = 1.0 / (sum_val + 1e-20)
        for t in range(8):
            selected_val[t] = selected_val[t] * inv_sum * routed_scaling_factor
            ptr = TOPK_WEIGHT_ptr + (m * stride_tkw_m + t * stride_tkw_n)
            tl.store(ptr, selected_val[t])
            # also store index
            idx_ptr = TOPK_IDX_ptr + (m * stride_tki_m + t * stride_tki_n)
            tl.store(idx_ptr, selected_idx[t])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Prepare inputs: ensure float32 on same device
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = 256  # num_experts
        # weight is [N, K] from nn.Linear; we need B = weight.T [K, N]
        B = weight.transpose(0, 1).contiguous()
        A = hidden_states.contiguous()
        # 1) Compute logits via Triton GEMM: C[M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        _matmul_AxB_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            A, B, logits,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )
        # 2) Compute scores = sigmoid(logits) + expert_bias (bias is [N])
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        _sigmoid_add_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(N, 64))](
            logits, expert_bias,
            scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )
        # 3) Compute group_scores [M, 8] by summing top-2 per group
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=device)
        _group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            scores.stride(0), scores.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            BLOCK_M=1, BLOCK_N=32,
            num_warps=2, num_stages=1,
        )
        # 4) Select top-4 groups per token
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=device)
        _group_top4_select_kernel[(M,)](
            group_scores, group_idx,
            M, N,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            BLOCK_M=1,
            num_warps=2, num_stages=1,
        )
        # 5) Final top-8 selection and normalization
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=device)
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=device)
        _final_top8_and_normalize_kernel[(M,)](
            scores, group_idx, topk_idx, topk_weight,
            M, N,
            scores.stride(0), scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            topk_idx.stride(0), topk_idx.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor,
            BLOCK_M=1, BLOCK_N=N,
            num_warps=2, num_stages=1,
        )
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
