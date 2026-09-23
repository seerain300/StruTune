import math
import torch
import triton
import triton.language as tl


# Triton GEMM kernel: C[M, N] = A[M, K] @ B[N, K]^T (no bias)
@triton.jit
def matmul_no_bias_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# Triton RMSNorm per head: normalize each row of length D and scale by per-head weight
# X: [M, D], Weight: [D], Output Y: [M, D]
@triton.jit
def rmsnorm_heads_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    M, D,
    stride_xm, stride_xd,
    stride_ym, stride_yd,
    eps,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    x = tl.load(X_ptr + pid_m * stride_xm + offs * stride_xd, mask=offs < D, other=0.0).to(tl.float32)
    x2 = x * x
    mean = tl.sum(x2, axis=0) / D
    inv_rms = tl.rsqrt(mean + eps)
    w = tl.load(Weight_ptr + offs, mask=offs < D, other=1.0).to(tl.float32)
    y = x * inv_rms * w
    tl.store(Y_ptr + pid_m * stride_ym + offs * stride_yd, y, mask=offs < D)


# Triton Rotated Positional Embedding: rotate halves of a 128-dim vector using cos/sin
@triton.jit
def rotate_half_inplace_kernel(
    X_ptr, Cos_ptr, Sin_ptr,
    M, D,
    stride_xm, stride_xd,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    x = tl.load(X_ptr + pid_m * stride_xm + offs * stride_xd, mask=offs < D, other=0.0).to(tl.float32)
    half = D // 2
    q1 = x[:half]
    q2 = x[half:]
    cos = tl.load(Cos_ptr + offs, mask=offs < D, other=1.0).to(tl.float32)
    sin = tl.load(Sin_ptr + offs, mask=offs < D, other=0.0).to(tl.float32)
    q1_new = q1 * cos - q2 * sin
    q2_new = q1 * sin + q2 * cos
    x_new = tl.zeros((D,), dtype=tl.float32)
    x_new[:half] = q1_new
    x_new[half:] = q2_new
    tl.store(X_ptr + pid_m * stride_xm + offs * stride_xd, x_new, mask=offs < D)


class ModelNew:
    def __init__(self, num_attention_heads=96, head_dim=128, num_key_value_heads=8, num_key_value_groups=12):
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_proj_weight: torch.Tensor,
        q_proj_bias: torch.Tensor,
        k_proj_weight: torch.Tensor,
        k_proj_bias: torch.Tensor,
        v_proj_weight: torch.Tensor,
        v_proj_bias: torch.Tensor,
        o_proj_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rms_norm_eps: float,
    ):
        B, S, H = hidden_states.shape
        D = self.head_dim
        Hq = self.num_attention_heads
        Hk = self.num_key_value_heads

        # 1) Triton dense linear for query, key, value (no bias)
        # hidden_states: [B, S, H], projection weights: [H, H] -> outputs: [B, S, H]
        M_q = B * S * H
        N_q = H
        K_q = H
        hidden_flat_q = hidden_states.reshape(M_q, K_q).contiguous()
        query_flat = torch.empty((M_q, N_q), dtype=torch.float32, device=hidden_states.device)

        grid_q = (triton.cdiv(M_q, 128), triton.cdiv(N_q, 128))
        matmul_no_bias_kernel[grid_q](
            hidden_flat_q, q_proj_weight, query_flat,
            M_q, N_q, K_q,
            hidden_flat_q.stride(0), hidden_flat_q.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            query_flat.stride(0), query_flat.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        )
        query = query_flat.view(B, S, H)

        # key
        M_k = B * S * H
        N_k = H
        K_k = H
        hidden_flat_k = hidden_states.reshape(M_k, K_k).contiguous()
        key_flat = torch.empty((M_k, N_k), dtype=torch.float32, device=hidden_states.device)

        grid_k = (triton.cdiv(M_k, 128), triton.cdiv(N_k, 128))
        matmul_no_bias_kernel[grid_k](
            hidden_flat_k, k_proj_weight, key_flat,
            M_k, N_k, K_k,
            hidden_flat_k.stride(0), hidden_flat_k.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            key_flat.stride(0), key_flat.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        )
        key = key_flat.view(B, S, H)

        # value
        M_v = B * S * H
        N_v = H
        K_v = H
        hidden_flat_v = hidden_states.reshape(M_v, K_v).contiguous()
        value_flat = torch.empty((M_v, N_v), dtype=torch.float32, device=hidden_states.device)

        grid_v = (triton.cdiv(M_v, 128), triton.cdiv(N_v, 128))
        matmul_no_bias_kernel[grid_v](
            hidden_flat_v, v_proj_weight, value_flat,
            M_v, N_v, K_v,
            hidden_flat_v.stride(0), hidden_flat_v.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            value_flat.stride(0), value_flat.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
        )
        value = value_flat.view(B, S, H)

        # 2) Reshape to heads and apply RMSNorm per head
        query_heads = query.view(B, S, Hq, D)
        # Triton RMSNorm per head
        query_norm = torch.empty_like(query_heads)
        for b in range(B):
            for s in range(S):
                for h in range(Hq):
                    row = query_heads[b, s, h]
                    weight = q_norm_weight[h]
                    out = query_norm[b, s, h]
                    rmsnorm_heads_kernel[(1,)](
                        row, weight, out,
                        D, D,
                        1, 1,
                        1, 1,
                        float(rms_norm_eps),
                        BLOCK_D=D,
                    )

        key_heads = key.view(B, S, Hk, D)
        key_norm = torch.empty_like(key_heads)
        for b in range(B):
            for s in range(S):
                for h in range(Hk):
                    row = key_heads[b, s, h]
                    weight = k_norm_weight[h]
                    out = key_norm[b, s, h]
                    rmsnorm_heads_kernel[(1,)](
                        row, weight, out,
                        D, D,
                        1, 1,
                        1, 1,
                        float(rms_norm_eps),
                        BLOCK_D=D,
                    )

        # 3) Apply Rotated Positional Embedding (RoPE) for query and key
        for b in range(B):
            for s in range(S):
                for h in range(Hq):
                    x = query_norm[b, s, h]
                    rotate_half_inplace_kernel[(1,)](
                        x, cos, sin,
                        D, D,
                        1, 1,
                        BLOCK_D=D,
                    )
        query_rot = query_norm

        for b in range(B):
            for s in range(S):
                for h in range(Hk):
                    x = key_norm[b, s, h]
                    rotate_half_inplace_kernel[(1,)](
                        x, cos, sin,
                        D, D,
                        1, 1,
                        BLOCK_D=D,
                    )
        key_rot = key_norm

        # 4) Grouped Query Attention expansion: replicate 8 heads to 96 by num_key_value_groups=12
        key_rot_expanded = torch.empty((B, Hq, S, D), dtype=query_rot.dtype, device=query_rot.device)
        value_expanded = torch.empty((B, Hq, S, D), dtype=value.dtype, device=value.device)
        for k in range(Hk):
            start = k * self.num


def run(*args):
    return ModelNew()(*args)
