import torch
import math
import triton
import triton.language as tl


@triton.jit
def layernorm_row_kernel(
    x_ptr,            # *const bfloat16, input [num_rows, features]
    y_ptr,            # *bfloat16, output [num_rows, features]
    ln_weight_ptr,    # *const float32, [features]
    ln_bias_ptr,      # *const float32, [features]
    num_rows,         # int
    features,         # int
    eps,              # float32
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    sum_fp32 = 0.0
    sumsq_fp32 = 0.0

    # Compute mean and variance in fp32
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

    # Normalize and apply affine
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row_id * features + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(y_ptr + row_id * features + idx, y.to(tl.bfloat16), mask=mask)


@triton.jit
def gelu_erf_kernel(
    x_ptr,            # *const float32, input [M, N]
    y_ptr,            # *float32, output [M, N]
    M: tl.constexpr,  # rows
    N: tl.constexpr,  # cols
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    for im in range(0, M, BLOCK_M):
        for jn in range(0, N, BLOCK_N):
            row = im + tl.arange(0, BLOCK_M)
            col = jn + tl.arange(0, BLOCK_N)
            mask_row = row < M
            mask_col = col < N
            x = tl.load(
                x_ptr + row[:, None] * stride_xm + col[None, :] * stride_xn,
                mask=mask_row[:, None] & mask_col[None, :],
                other=0.0,
            )
            # GELU erf approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
            inv_sqrt2 = 0.7071067811865476
            z = x * inv_sqrt2
            # erf approximation (Abramowitz & Stegun 7.1.26)
            p = 0.3275911
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429

            abs_z = tl.abs(z)
            t = 1.0 / (1.0 + p * abs_z)
            poly = a5
            poly = poly * t + a4
            poly = poly * t + a3
            poly = poly * t + a2
            poly = poly * t + a1
            poly = poly * t
            erf_abs = 1.0 - poly * tl.exp(-(abs_z * abs_z))
            sign = tl.where(z >= 0.0, 1.0, -1.0)
            erf_z = erf_abs * sign
            y = 0.5 * x * (1.0 + erf_z)
            tl.store(
                y_ptr + row[:, None] * stride_ym + col[None, :] * stride_yn,
                y,
                mask=mask_row[:, None] & mask_col[None, :],
            )


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * stride_am + k_offsets * stride_ak  # shape [BLOCK_K]
        a = tl.load(a_ptrs, mask=(m0 < M) & (k_offsets < K), other=0.0)
        a = a.to(tl.float32)[:, None]  # [BLOCK_K, 1]

        # Load B tile [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[None, :] * stride_bk + n0 * stride_bn  # shape [1, BLOCK_N]
        b = tl.load(b_ptrs, mask=(k_offsets < K) & (n0 < N), other=0.0)
        b = b.to(tl.float32)[None, :]  # [1, BLOCK_N]

        # Outer product and accumulate
        acc += tl.dot(a, b)  # a: [BLOCK_K, 1], b: [1, BLOCK_N] -> [BLOCK_K, BLOCK_N]; sum over K adds to [BLOCK_M, BLOCK_N]

    # Store C
    for im in range(0, BLOCK_M):
        for jn in range(0, BLOCK_N):
            c_ptrs = C_ptr + (m0 + im) * stride_cm + (n0 + jn) * stride_cn
            out_mask = (m0 + im) < M and (n0 + jn) < N
            tl.store(c_ptrs, acc[im, jn], mask=out_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
    ):
        """
        Triton-optimized forward:
        - LayerNorm via Triton
        - First Linear (GEMM) via Triton
        - GELU (erf approx) via Triton
        - Second Linear (GEMM) via Triton
        Avoids torch.permute and torch.cat. All heavy computation in Triton kernels.
        Returns tensor of shape [num_patches, 3584], bfloat16.
        """
        assert hidden.is_cuda, "Tensors must be on CUDA for Triton kernels"
        device = hidden.device
        num_patches = hidden.shape[0]
        features = hidden.shape[1]
        assert features == 1536, "LayerNorm must be across 1536 features"

        # 1) LayerNorm with Triton
        hidden_norm = torch.empty((num_patches, features), dtype=torch.float32, device=device)
        ln_w = ln_weight.to(torch.float32)  # [features]
        ln_b = ln_bias.to(torch.float32)    # [features]
        layernorm_row_kernel[(num_patches,)](
            hidden, hidden_norm,
            ln_w, ln_b,
            num_patches, features, float(eps),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )
        hidden_norm_bf16 = hidden_norm.to(torch.bfloat16)

        # 2) First Linear: hidden_norm_bf16 (M=num_patches, K=1536) @ fc1_weight.T (K=1536, N=6144) + bias
        B1 = fc1_weight.t().to(torch.float32).contiguous()  # [1536, 6144]
        C1 = torch.empty((num_patches, B1.shape[1]), dtype=torch.float32, device=device)

        grid_matmul1 = (triton.cdiv(num_patches, 64), triton.cdiv(B1.shape[1], 64))
        matmul_kernel[grid_matmul1](
            hidden_norm_bf16, B1, C1,
            num_patches, B1.shape[1], hidden_norm_bf16.shape[1],
            hidden_norm_bf16.stride(0), hidden_norm_bf16.stride(1),
            B1.stride(0), B1.stride(1),
            C1.stride(0), C1.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # 3) GELU via Triton
        C1_out = torch.empty_like(C1, dtype=torch.float32, device=device)
        grid_gelu = (triton.cdiv(num_patches, 64), triton.cdiv(C1.shape[1], 64))
        gelu_erf_kernel[grid_gelu](
            C1, C1_out,
            num_patches, C1.shape[1],
            C1.stride(0), C1.stride(1),
            C1_out.stride(0), C1_out.stride(1),
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )
        C1 = C1_out

        # 4) Second Linear: C1 (M=num_patches, K=6144) @ fc2_weight.T (K=6144, N=3584) + bias
        B2 = fc2_weight.t().to(torch.float32).contiguous()  # [6144, 3584]
        C2 = torch.empty((num_patches, B2.shape[1]), dtype=torch.float32, device=device)

        grid_matmul2 = (triton.cdiv(num_patches, 64), triton.cdiv(B2.shape[1], 64))
        matmul_kernel[grid_matmul2](
            C1, B2, C2,
            num_patches, B2.shape[1], C1.shape[1],
            C1.stride(0), C1.stride(1),
            B2.stride(0), B2.stride(1),
            C2.stride(0), C2.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # Return in bfloat16 (original dtype for output)
        return C2.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
