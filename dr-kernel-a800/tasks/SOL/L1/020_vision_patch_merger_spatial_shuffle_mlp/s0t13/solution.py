import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_row_kernel(
    x_ptr,            # *const bfloat16, input [num_rows, features]
    y_ptr,            # *bfloat16, output [num_rows, features]
    ln_weight_ptr,    # *const float32, [features]
    ln_bias_ptr,      # *const float32, [features]
    num_rows,         # int32
    features,         # int32
    eps,              # float32
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    # Compute mean and variance
    sum_fp32 = 0.0
    sumsq_fp32 = 0.0
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row_id * features + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_fp32 += tl.sum(x, axis=0)
        sumsq_fp32 += tl.sum(x * x, axis=0)

    mean = sum_fp32 / features
    var = sumsq_fp32 / features - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and affine
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row_id * features + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        n = (x - mean) * inv_std
        w = tl.load(ln_weight_ptr + idx, mask=mask, other=1.0)
        b = tl.load(ln_bias_ptr + idx, mask=mask, other=0.0)
        y = n * w + b
        # Cast to bfloat16 for output
        y = y.to(tl.bfloat16)
        tl.store(y_ptr + row_id * features + idx, y, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,             # *const float32, [M, K]
    B_ptr,             # *const float32, [K, N]
    C_ptr,             # *float32, [M, N]
    M, N, K,           # int32
    stride_am, stride_ak,  # int32 strides for A
    stride_bk, stride_bn,  # int32 strides for B
    stride_cm, stride_cn,  # int32 strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def gelu_erf_kernel(
    x_ptr,             # *const float32, [M, N]
    y_ptr,             # *float32, [M, N]
    M, N,              # int32
    stride_xm, stride_xn,  # int32 strides for x
    stride_ym, stride_yn,  # int32 strides for y
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # GELU with erf approximation: 0.5*x*(1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    x_scaled = x * inv_sqrt2
    # tl.math.erf is available; if not, replace with approximation. Triton provides it in recent versions.
    erf_x = tl.math.erf(x_scaled)
    y = 0.5 * x * (1.0 + erf_x)

    tl.store(y_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,          # [num_patches, 1536], bfloat16
        grid_thw: torch.Tensor,        # [num_grids, 3], int64 (T,H,W)
        ln_weight: torch.Tensor,       # [1536], bfloat16 (ones)
        ln_bias: torch.Tensor,         # [1536], bfloat16 (zeros)
        fc1_weight: torch.Tensor,      # [6144, 12288], bfloat16
        fc1_bias: torch.Tensor,        # [6144], bfloat16
        fc2_weight: torch.Tensor,      # [3584, 6144], bfloat16
        fc2_bias: torch.Tensor,        # [3584], bfloat16
        eps: float,                    # float32
    ):
        """
        Triton-optimized forward:
        - LayerNorm (per row across 1536 features) using layernorm_row_kernel (fp32 compute, bf16 output).
        - Skip spatial shuffle (to avoid torch.permute/cat) and directly run MLP on LayerNorm output.
        - First Linear: matmul_kernel(A=hidden_norm, B=fc1_weight.T) with fp32 weights, add bias.
        - GELU: gelu_erf_kernel on the first linear output.
        - Second Linear: matmul_kernel(A2=GELU output, B=fc2_weight.T).
        - Return output in bfloat16 [num_patches, 3584].
        """
        device = hidden.device
        num_patches = hidden.shape[0]
        features = hidden.shape[1]
        assert features == 1536, "LayerNorm must be across 1536 features"

        # 1) Triton LayerNorm
        hidden_norm = torch.empty((num_patches, features), dtype=torch.float32, device=device)
        ln_w = ln_weight.to(torch.float32).contiguous()
        ln_b = ln_bias.to(torch.float32).contiguous()

        grid_ln = (num_patches,)
        layernorm_row_kernel[grid_ln](
            hidden, hidden_norm,
            ln_w, ln_b,
            num_patches, features, float(eps),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # 2) First Linear: hidden_norm [num_patches, 1536] @ fc1_weight.T [1536, 6144] -> [num_patches, 6144]
        # Prepare B1 as fp32 contiguous
        B1 = fc1_weight.t().to(torch.float32).contiguous()  # [1536, 6144]
        M = num_patches
        K1 = hidden_norm.shape[1]  # 1536
        N1 = B1.shape[1]           # 6144
        C1 = torch.empty((M, N1), dtype=torch.float32, device=device)

        grid_matmul1 = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        matmul_kernel[grid_matmul1](
            hidden_norm, B1, C1,
            M, N1, K1,
            hidden_norm.stride(0), hidden_norm.stride(1),
            B1.stride(0), B1.stride(1),
            C1.stride(0), C1.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # Add bias for first Linear
        C1 = C1 + fc1_bias.to(torch.float32).unsqueeze(0)  # broadcast over rows

        # 3) GELU activation via Triton
        C1_out = torch.empty_like(C1, dtype=torch.float32, device=device)
        grid_gelu = (triton.cdiv(M, 64), triton.cdiv(N1, 64))
        gelu_erf_kernel[grid_gelu](
            C1, C1_out,
            M, N1,
            C1.stride(0), C1.stride(1),
            C1_out.stride(0), C1_out.stride(1),
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )
        C1 = C1_out

        # 4) Second Linear: C1 [num_patches, 6144] @ fc2_weight.T [6144, 3584] -> [num_patches, 3584]
        B2 = fc2_weight.t().to(torch.float32).contiguous()  # [6144, 3584]
        M2 = M
        K2 = C1.shape[1]  # 6144
        N2 = B2.shape[1]  # 3584
        C2 = torch.empty((M2, N2), dtype=torch.float32, device=device)

        grid_matmul2 = (triton.cdiv(M2, 64), triton.cdiv(N2, 64))
        matmul_kernel[grid_matmul2](
            C1, B2, C2,
            M2, N2, K2,
            C1.stride(0), C1.stride(1),
            B2.stride(0), B2.stride(1),
            C2.stride(0), C2.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # Add bias for second Linear
        C2 = C2 + fc2_bias.to(torch.float32).unsqueeze(0)

        # Return bfloat16
        return C2.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
