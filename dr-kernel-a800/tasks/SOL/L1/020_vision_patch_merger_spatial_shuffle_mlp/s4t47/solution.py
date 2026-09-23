import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_rows_kernel(
    hidden_ptr, out_ptr,
    ln_weight_ptr, ln_bias_ptr,
    N,  # hidden_size (last dim)
    eps,
    stride_row,
    BLOCK_SIZE: tl.constexpr
):
    # One program per row
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    hidden_row_ptr = hidden_ptr + row * stride_row
    out_row_ptr = out_ptr + row * stride_row

    # Load row as bfloat16 and convert to fp32 for mean/var
    x = tl.load(hidden_row_ptr + cols, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Compute mean
    mean = tl.sum(x_f32, axis=0) / N
    # Compute variance
    x_centered = x_f32 - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    y = x_centered * rstd

    # Apply layer norm weight and bias (loaded as fp32)
    w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    z = y * w + b  # fp32
    # Store back as bf16
    tl.store(out_row_ptr + cols, z.to(tl.bfloat16), mask=mask)


@triton.jit
def _pack_rows_to_1d_kernel(
    hidden_ptr, out_ptr,
    num_patches, hidden_size,
    hidden_expanded,  # 4 * hidden_size
    stride_row,
    BLOCK: tl.constexpr
):
    # Each program handles one original row and writes it into a 1D output at linearized positions.
    pid = tl.program_id(0)
    if pid >= num_patches:
        return

    # Output index for row pid: out[pid * hidden_expanded : (pid+1) * hidden_expanded]
    out_start = pid * hidden_expanded
    src_row_ptr = hidden_ptr + pid * stride_row

    # Copy entire row (hidden_expanded = 4 * hidden_size). We do it in tiles of BLOCK.
    for off in range(0, hidden_expanded, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < hidden_expanded
        val = tl.load(src_row_ptr + cols, mask=mask, other=0.0)  # bf16
        tl.store(out_ptr + out_start + cols, val, mask=mask)


@triton.jit
def _gemm_rows_cols_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bias_numel, bias_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        off_k = k + tl.arange(0, BLOCK_K)
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + (off_m[:, None] * stride_am + off_k[None, :] * stride_ak)
        a_mask = (off_m[:, None] < M) & (off_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: shape (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + (off_k[:, None] * stride_bk + off_n[None, :] * stride_bn)
        b_mask = (off_k[:, None] < K) & (off_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # Add bias (broadcast over columns)
    if bias_numel > 0:
        bias_vec = tl.load(bias_ptr + off_n, mask=(off_n < N), other=0.0).to(tl.float32)  # (BLOCK_N,)
        acc += bias_vec[None, :]

    # Store to C in bf16
    c_ptrs = C_ptr + (off_m[:, None] * stride_cm + off_n[None, :] * stride_cn)
    c_mask = (off_m[:, None] < M) & (off_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _gelu_tanh_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Grid: (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)

    x_ptrs = X_ptr + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    y_ptrs = Y_ptr + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(inner))

    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        device = hidden.device
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_expanded = hidden_size * 4  # per merged position features

        # 1) LayerNorm per row, last dim = hidden_size
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        grid_ln = (num_patches,)
        _layernorm_rows_kernel[grid_ln](
            hidden, hidden_norm,
            ln_weight, ln_bias,
            hidden_size, eps,
            hidden_norm.stride(0),
            BLOCK_SIZE=hidden_size
        )

        # 2) Spatial packing: create 1D vector of length num_patches * 4 * hidden_size
        hidden_pack = torch.empty(num_patches * hidden_expanded, dtype=torch.bfloat16, device=device)
        grid_pack = (num_patches,)
        _pack_rows_to_1d_kernel[grid_pack](
            hidden_norm, hidden_pack,
            num_patches, hidden_size,
            hidden_expanded,
            hidden_norm.stride(0),
            BLOCK=hidden_expanded  # single tile for the row
        )

        # 3) First Linear: (num_patches // 4, 6144) @ (6144, 6144) -> (num_patches // 4, 6144)
        num_merged_patches = num_patches // 4
        hidden_linear1 = hidden_pack.view(num_merged_patches, hidden_expanded)

        B1 = torch.empty((num_merged_patches, hidden_expanded), dtype=torch.bfloat16, device=device)
        grid_gemm1 = (triton.cdiv(num_merged_patches, 128), triton.cdiv(hidden_expanded, 128))
        _gemm_rows_cols_kernel[grid_gemm1](
            hidden_linear1, fc1_weight, B1,
            num_merged_patches, hidden_expanded, hidden_expanded,
            hidden_linear1.stride(0), hidden_linear1.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            B1.stride(0), B1.stride(1),
            fc1_bias.numel(), fc1_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        # 4) GELU activation
        B1_gelu = torch.empty_like(B1, dtype=torch.bfloat16, device=device)
        grid_gelu = (triton.cdiv(num_merged_patches, 64), triton.cdiv(hidden_expanded, 128))
        _gelu_tanh_kernel[grid_gelu](
            B1, B1_gelu,
            num_merged_patches, hidden_expanded,
            B1.stride(0), B1.stride(1),
            B1_gelu.stride(0), B1_gelu.stride(1),
            BLOCK_M=64, BLOCK_N=128
        )

        # 5) Second Linear: (num_merged_patches, 6144) @ (3584, 6144) -> (num_merged_patches, 3584)
        out_hidden_size = fc2_weight.shape[0]
        output = torch.empty((num_merged_patches, out_hidden_size), dtype=torch.bfloat16, device=device)
        grid_gemm2 = (triton.cdiv(num_merged_patches, 128), triton.cdiv(out_hidden_size, 128))
        _gemm_rows_cols_kernel[grid_gemm2](
            B1_gelu, fc2_weight, output,
            num_merged_patches, out_hidden_size, hidden_expanded,
            B1_gelu.stride(0), B1_gelu.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1),
            fc2_bias.numel(), fc2_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
