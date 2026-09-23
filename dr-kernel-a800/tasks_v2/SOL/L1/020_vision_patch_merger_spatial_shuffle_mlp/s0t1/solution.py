import torch
import math
import triton
import triton.language as tl


@triton.jit
def layernorm_row_kernel(
    x_ptr,            # *const bfloat16, shape [num_rows, features]
    y_ptr,            # *bfloat16, shape [num_rows, features]
    ln_weight_ptr,    # *const float32, shape [features]
    ln_bias_ptr,      # *const float32, shape [features]
    num_rows,         # int
    features,         # int
    eps,              # float
    stride_x_row,     # int
    stride_x_col,     # int
    stride_y_row,     # int
    stride_y_col,     # int
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    # First pass: compute sum and sum of squares (fp32 accumulation)
    s = 0.0
    ss = 0.0
    for off in range(0, features, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < features
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)

    mean = s / features
    var = ss / features - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for off in range(0, features, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < features
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * w + b
        tl.store(y_ptr + row * stride_y_row + cols * stride_y_col, y.to(tl.bfloat16), mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid: (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Cast to fp32 for accumulation
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def gelu_erf_kernel(
    inp_ptr,      # *float32, 1D flattened input
    out_ptr,      # *float32, 1D flattened output
    length,       # int
    inv_sqrt2,    # float, 1/sqrt(2)
    # Constants for erf approximation (Abramowitz & Stegun 7.1.26)
    p: tl.constexpr, a1: tl.constexpr, a2: tl.constexpr, a3: tl.constexpr, a4: tl.constexpr, a5: tl.constexpr
):
    idx = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = idx < length
    x = tl.load(inp_ptr + idx, mask=mask, other=0.0)
    z = x * inv_sqrt2
    az = tl.abs(z)
    t = 1.0 / (1.0 + p * az)
    # Horner's method for polynomial
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erf_approx = 1.0 - poly * tl.exp(-az * az)
    erf_approx = tl.where(z >= 0, erf_approx, -erf_approx)  # apply sign
    y = 0.5 * x * (1.0 + erf_approx)
    tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps = args

        device = hidden.device

        # 1) LayerNorm over 1536 features per vector (bf16 input, fp32 compute)
        hidden_bf16 = hidden.contiguous()
        ln_weight_fp32 = ln_weight.to(torch.float32).contiguous()
        ln_bias_fp32 = ln_bias.to(torch.float32).contiguous()

        hidden_fp32 = hidden_bf16.to(torch.float32)
        hidden_ln_fp32 = torch.empty_like(hidden_fp32, dtype=torch.float32, device=device)

        num_rows, features = hidden_fp32.shape
        assert features == 1536, "LayerNorm must be across 1536 features"

        layernorm_row_kernel[(num_rows,)](
            hidden_fp32, hidden_ln_fp32,
            ln_weight_fp32, ln_bias_fp32,
            num_rows, features,
            float(eps),
            hidden_fp32.stride(0), hidden_fp32.stride(1),
            hidden_ln_fp32.stride(0), hidden_ln_fp32.stride(1),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # 2) Spatial shuffle via PyTorch view/permute/reshape (metadata, no compute)
        # Compute num_merged_patches from grid_thw: for each grid, num_patches_this = t * (H//2) * (W//2)
        offset = 0
        shuffled_patches = []
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())

            h_merged = h // 2
            w_merged = w // 2
            num_patches_this = t * h_merged * w_merged

            patches = hidden_ln_fp32[offset:offset + num_patches_this]  # [num_patches_this, 1536], fp32

            patches = patches.view(t, h_merged, 2, w_merged, 2, 1536)
            patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(num_patches_this, 2 * 2 * 1536)
            # patches: [num_patches_this, 12288], fp32

            offset += num_patches_this
            shuffled_patches.append(patches)

        hidden_shuffled_fp32 = torch.cat(shuffled_patches, dim=0)  # [num_merged_patches, 12288], fp32

        # 3) First linear (Triton GEMM): A[M,K], B[K,N] -> C[M,N]
        # Prepare B1 = fc1_weight^T [K, N] = [12288, 6144], bf16
        B1 = fc1_weight.t().contiguous()  # [6144, 12288] original, we need [12288, 6144]
        B1_t = fc1_weight.t().contiguous()  # already [12288, 6144], bfloat16
        M = hidden_shuffled_fp32.shape[0]
        K = hidden_shuffled_fp32.shape[1]  # 12288
        N1 = B1_t.shape[1]  # 6144

        C1_fp32 = torch.empty((M, N1), dtype=torch.float32, device=device)

        grid_matmul1 = (triton.cdiv(M, 128), triton.cdiv(N1, 128))
        matmul_kernel[grid_matmul1](
            hidden_shuffled_fp32, B1_t,
            C1_fp32,
            M, N1, K,
            hidden_shuffled_fp32.stride(0), hidden_shuffled_fp32.stride(1),
            B1_t.stride(0), B1_t.stride(1),
            C1_fp32.stride(0), C1_fp32.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # 4) GELU (erf-based) in Triton, elementwise
        length = M * N1
        inp_flat = C1_fp32.reshape(-1)
        out_flat = torch.empty_like(inp_flat, dtype=torch.float32, device=device)

        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        # Constants for erf approximation (Abramowitz & Stegun 7.1.26)
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        gelu_grid = (triton.cdiv(length, 1024),)
        gelu_erf_kernel[gelu_grid](
            inp_flat, out_flat, length, inv_sqrt2,
            p, a1, a2, a3, a4, a5,
            num_warps=4, num_stages=2,
        )

        C1_gelu_fp32 = out_flat.reshape(M, N1)

        # 5) Second linear (Triton GEMM): A[M,K], B[K,N] -> C[M,N]
        # Prepare B2 = fc2_weight^T [K, N] = [6144, 3584]
        B2 = fc2_weight.t().contiguous()  # [3584, 6144]
        N2 = B2.shape[1]  # 3584

        C2_fp32 = torch.empty((M, N2), dtype=torch.float32, device=device)

        grid_matmul2 = (triton.cdiv(M, 128), triton.cdiv(N2, 128))
        matmul_kernel[grid_matmul2](
            C1_gelu_fp32, B2,
            C2_fp32,
            M, N2, N1,
            C1_gelu_fp32.stride(0), C1_gelu_fp32.stride(1),
            B2.stride(0), B2.stride(1),
            C2_fp32.stride(0), C2_fp32.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # Return bfloat16 to match original behavior (the original uses bf16 throughout)
        output_bf16 = C2_fp32.to(torch.bfloat16)
        return output_bf16


def run(*args):
    return ModelNew()(*args)
