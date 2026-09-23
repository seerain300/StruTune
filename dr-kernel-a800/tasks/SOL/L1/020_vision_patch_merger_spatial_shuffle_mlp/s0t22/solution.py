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
    BLOCK: tl.constexpr,  # tile size for reduction
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return

    sum_fp32 = 0.0
    sumsq_fp32 = 0.0

    # First pass: compute mean and variance across features
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row_id * features + idx, mask=mask, other=0.0).to(tl.float32)
        sum_fp32 += tl.sum(x, axis=0)
        sumsq_fp32 += tl.sum(x * x, axis=0)

    mean = sum_fp32 / features
    var = sumsq_fp32 / features - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row_id * features + idx, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        w = tl.load(ln_weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        out = norm * w + b
        # store as bfloat16
        tl.store(y_ptr + row_id * features + idx, out.to(tl.bfloat16), mask=mask)


@triton.jit
def matmul_rowcol_kernel(
    A_ptr,  # *const float32, [M, K]
    B_ptr,  # *const float32, [K, N]
    C_ptr,  # *float32, [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def gelu_erf_kernel(
    inp_ptr, out_ptr, M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    in_ptrs = inp_ptr + rm[:, None] * stride_im + rn[None, :] * stride_in
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on

    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)
    # erf approximation (Abramowitz & Stegun 7.1.26)
    # erf(z) ≈ sign * (1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) * exp(-z^2)), t = 1 / (1 + p z)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    z = x * 0.7071067811865476  # 1/sqrt(2)
    sign = tl.where(x < 0, -1.0, 1.0)
    xz = tl.abs(z)
    t = 1.0 / (1.0 + p * xz)
    poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
    erf_approx = sign * (1.0 - poly * tl.exp(-xz * xz))
    y = 0.5 * x * (1.0 + erf_approx)
    tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,          # [num_patches, 1536], bfloat16
        grid_thw: torch.Tensor,        # [num_grids, 3], int64 (T,H,W) - not used here
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
        - LayerNorm (per row across 1536 features) in Triton (fp32 compute), store bfloat16.
        - Spatial permute + reshape (metadata-only) using PyTorch.
        - First Linear (fp32 GEMM via Triton).
        - GELU (erf approximation) in Triton.
        - Second Linear (fp32 GEMM via Triton).
        - Return output in bfloat16.
        """
        device = hidden.device
        num_patches = hidden.shape[0]
        features = hidden.shape[1]
        assert features == 1536, "LayerNorm must be across 1536 features"

        # 1) LayerNorm with Triton, output fp32, but store bfloat16 to match input dtype
        hidden_norm = torch.empty((num_patches, features), dtype=torch.float32, device=device)
        ln_w_fp32 = ln_weight.to(torch.float32)
        ln_b_fp32 = ln_bias.to(torch.float32)
        grid_ln = (num_patches,)
        layernorm_row_kernel[grid_ln](
            hidden, hidden_norm,
            ln_w_fp32, ln_b_fp32,
            num_patches, features, float(eps),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # 2) Spatial permute + reshape (metadata-only) as in original:
        # Reshape to (num_patches, 8, 768) then flatten to (num_merged_patches, 12288)
        num_merged_patches = hidden_norm.shape[0] * 8
        hidden_perm = hidden_norm.view(num_patches, 8, 768).reshape(num_merged_patches, 12288)

        # 3) First Linear: A = hidden_perm [num_merged_patches, 12288], B = fc1_weight.T [12288, 6144]
        B1 = fc1_weight.t().to(torch.float32).contiguous()  # [12288, 6144]
        C1 = torch.empty((num_merged_patches, B1.shape[1]), dtype=torch.float32, device=device)

        grid_matmul1 = (triton.cdiv(num_merged_patches, 128), triton.cdiv(B1.shape[1], 128))
        matmul_rowcol_kernel[grid_matmul1](
            hidden_perm, B1, C1,
            num_merged_patches, B1.shape[1], B1.shape[0],
            hidden_perm.stride(0), hidden_perm.stride(1),
            B1.stride(0), B1.stride(1),
            C1.stride(0), C1.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # 4) GELU via Triton erf approximation, fp32 output
        C1_gelu = torch.empty_like(C1, dtype=torch.float32, device=device)
        grid_gelu = (triton.cdiv(num_merged_patches, 128), triton.cdiv(C1.shape[1], 128))
        gelu_erf_kernel[grid_gelu](
            C1, C1_gelu,
            num_merged_patches, C1.shape[1],
            C1.stride(0), C1.stride(1),
            C1_gelu.stride(0), C1_gelu.stride(1),
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # 5) Second Linear: A = GELU_output [num_merged_patches, 6144], B = fc2_weight.T [6144, 3584]
        B2 = fc2_weight.t().to(torch.float32).contiguous()  # [6144, 3584]
        output = torch.empty((num_merged_patches, B2.shape[1]), dtype=torch.float32, device=device)

        grid_matmul2 = (triton.cdiv(num_merged_patches, 128), triton.cdiv(B2.shape[1], 128))
        matmul_rowcol_kernel[grid_matmul2](
            C1_gelu, B2, output,
            num_merged_patches, B2.shape[1], B2.shape[0],
            C1_gelu.stride(0), C1_gelu.stride(1),
            B2.stride(0), B2.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # Return in bfloat16 (to match original)
        return output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
