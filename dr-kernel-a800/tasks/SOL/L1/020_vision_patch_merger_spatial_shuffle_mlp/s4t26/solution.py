import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_rows_kernel(
    in_ptr, out_ptr, ln_weight_ptr, ln_bias_ptr,
    num_rows, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)
    in_row = in_ptr + row_id * hidden_size
    out_row = out_ptr + row_id * hidden_size

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    # Load row as bf16, cast to fp32 for compute
    x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / hidden_size
    var = tl.sum(x * x, axis=0) / hidden_size - mean * mean
    inv_std = tl.math.rsqrt(var + eps)

    # Load layer norm params (weight, bias) and apply
    ln_w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y_fp32 = (x - mean) * inv_std
    y_fp32 = y_fp32 * ln_w + ln_b

    # Store as bf16
    y_bf16 = y_fp32.to(tl.bfloat16)
    tl.store(out_row + cols, y_bf16, mask=mask)


@triton.jit
def _copy_rows_to_pack_kernel(
    src_ptr, dst_ptr,
    num_rows, hidden_size,
    hidden_size_expanded,
    BLOCK: tl.constexpr,
):
    # One program per row, write to consecutive slots in dst
    row_id = tl.program_id(0)
    in_row = src_ptr + row_id * hidden_size
    start = row_id * hidden_size_expanded
    out_row = dst_ptr + start

    cols = tl.arange(0, BLOCK)
    mask = cols < hidden_size_expanded

    x = tl.load(in_row + cols, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(out_row + cols, x, mask=mask)


@triton.jit
def _gemm_rows_cols_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride0, A_stride1,
    B_stride0, B_stride1,
    C_stride0, C_stride1,
    bias_ptr,  # bias per column (length N), can be None but we always pass a tensor
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)

        mask_a = (rm[:, None] < M) & (rk[None, :] < K)
        mask_b = (rk[:, None] < K) & (rn[None, :] < N)

        A_sub = tl.load(
            A_ptr + rm[:, None] * A_stride0 + rk[None, :] * A_stride1,
            mask=mask_a, other=0.0
        ).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        B_sub = tl.load(
            B_ptr + rk[:, None] * B_stride0 + rn[None, :] * B_stride1,
            mask=mask_b, other=0.0
        ).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(A_sub, B_sub)

    # Add bias per column
    bias_vec = tl.load(bias_ptr + rn, mask=(rn < N), other=0.0).to(tl.float32)  # [BLOCK_N]
    acc = acc + bias_vec[None, :]

    # Store C as bf16
    C_ptrs = C_ptr + rm[:, None] * C_stride0 + rn[None, :] * C_stride1
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.bfloat16), mask=mask_c)


@triton.jit
def _gelu_tanh_kernel(
    in_ptr, out_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    x = tl.load(in_ptr + rm[:, None] * 0 + rn[None, :], mask=mask, other=0.0).to(tl.float32)  # contiguous assumption for simplicity
    # GELU via tanh approximation
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    inner = c0 * (x + c1 * x3)
    t = tl.math.tanh(inner)
    y = 0.5 * x * (1.0 + t)

    tl.store(out_ptr + rm[:, None] * 0 + rn[None, :], y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        hidden = args[0]  # [num_patches, hidden_size], bf16
        grid_thw = args[1]  # [num_grids, 3], int64
        ln_weight = args[2]  # [hidden_size], bf16
        ln_bias = args[3]  # [hidden_size], bf16
        fc1_weight = args[4]  # [hidden_size_expanded, hidden_size_expanded], bf16
        fc1_bias = args[5]  # [hidden_size_expanded], bf16
        fc2_weight = args[6]  # [out_hidden_size, hidden_size_expanded], bf16
        fc2_bias = args[7]  # [out_hidden_size], bf16
        eps = args[8]  # float

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_size_expanded = hidden_size * 4  # 2x2 per position -> 4 groups

        # 1) LayerNorm per row using Triton
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
        grid_ln = (num_patches,)
        _layernorm_rows_kernel[grid_ln](
            hidden, hidden_norm, ln_weight, ln_bias,
            num_patches, hidden_size,
            eps=1e-6,
            BLOCK_SIZE=hidden_size  # 1536
        )

        # 2) Pack rows into 1D vector: length = num_patches * hidden_size_expanded
        #    Each original row i is copied into dst[i * hidden_size_expanded : (i+1) * hidden_size_expanded].
        packed = torch.empty(num_patches * hidden_size_expanded, dtype=torch.bfloat16, device=hidden.device)
        grid_pack = (num_patches,)
        _copy_rows_to_pack_kernel[grid_pack](
            hidden_norm, packed,
            num_patches, hidden_size,
            hidden_size_expanded,
            BLOCK=hidden_size_expanded  # 6144
        )

        # 3) First Linear: (M=num_patches, K=6144) @ (K=6144, N=6144) -> (M=num_patches, N=6144)
        M = num_patches
        K = hidden_size_expanded
        N = hidden_size_expanded

        B1 = torch.empty((M, N), dtype=torch.bfloat16, device=hidden.device)
        grid_gemm1 = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        _gemm_rows_cols_kernel[grid_gemm1](
            packed.view(M, K), fc1_weight, B1,
            M, N, K,
            packed.view(M, K).stride(0), packed.view(M, K).stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            B1.stride(0), B1.stride(1),
            fc1_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        # 4) GELU activation (Triton)
        B1_gelu = torch.empty_like(B1, dtype=torch.bfloat16, device=hidden.device)
        grid_gelu = (triton.cdiv(M, 64), triton.cdiv(N, 128))
        _gelu_tanh_kernel[grid_gelu](
            B1, B1_gelu,
            M, N,
            BLOCK_M=64, BLOCK_N=128
        )

        # 5) Second Linear: (M=num_patches, K=6144) @ (K=6144, N=3584) -> (M=num_patches, N=3584)
        out_hidden_size = fc2_weight.shape[0]  # 3584
        output = torch.empty((M, out_hidden_size), dtype=torch.bfloat16, device=hidden.device)
        grid_gemm2 = (triton.cdiv(M, 128), triton.cdiv(out_hidden_size, 128))
        _gemm_rows_cols_kernel[grid_gemm2](
            B1_gelu, fc2_weight, output,
            M, out_hidden_size, K,
            B1_gelu.stride(0), B1_gelu.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1),
            fc2_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
