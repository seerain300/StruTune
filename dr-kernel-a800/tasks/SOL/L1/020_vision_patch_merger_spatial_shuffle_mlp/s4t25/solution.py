import torch
import math
import triton
import triton.language as tl

# LayerNorm per row: in bfloat16, compute in float32, output bfloat16
@triton.jit
def _layernorm_rows_kernel(
    in_ptr, out_ptr, ln_weight_ptr, ln_bias_ptr,
    num_rows, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)  # one program per row
    in_row = in_ptr + row_id * hidden_size
    out_row = out_ptr + row_id * hidden_size

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    x = tl.load(in_row + cols, mask=mask, other=0.0)  # bfloat16
    x_fp32 = x.to(tl.float32)

    mean = tl.sum(x_fp32, axis=0) / hidden_size
    var = tl.sum(x_fp32 * x_fp32, axis=0) / hidden_size - mean * mean
    inv_std = tl.math.rsqrt(var + eps)

    ln_w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y_fp32 = (x_fp32 - mean) * inv_std
    y_fp32 = y_fp32 * ln_w + ln_b

    y = y_fp32.to(tl.bfloat16)
    tl.store(out_row + cols, y, mask=mask)


# Triton kernel: copy normalized rows into packed 1D vector of length num_patches * hidden_size_expanded
@triton.jit
def _copy_rows_to_pack_kernel(
    in_ptr, out_ptr,
    num_rows, hidden_size, hidden_size_expanded,
    BLOCK_OUT: tl.constexpr,
):
    row_id = tl.program_id(0)  # one program per row
    cols = tl.arange(0, hidden_size)
    mask_in = cols < hidden_size
    x = tl.load(in_ptr + row_id * hidden_size + cols, mask=mask_in, other=0.0)  # bfloat16

    dest_base = row_id * hidden_size_expanded
    out_cols = tl.arange(0, BLOCK_OUT)
    mask_out = out_cols < hidden_size_expanded
    tl.store(out_ptr + dest_base + out_cols, x, mask=mask_out)


# GEMM: A[M,K] @ B[K,N] -> C[M,N], bf16 I/O, fp32 accumulation
@triton.jit
def _gemm_rows_cols_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride0, A_stride1,
    B_stride0, B_stride1,
    C_stride0, C_stride1,
    bias_ptr,  # 1D of length N (can be None -> bias=0)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
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

        acc += tl.dot(A_sub, B_sub)  # [BLOCK_M, BLOCK_N]

    # bias add if provided
    bias_vec = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)  # [BLOCK_N]
    acc += bias_vec[None, :]

    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    C_sub = acc.to(tl.bfloat16)
    tl.store(
        C_ptr + rm[:, None] * C_stride0 + rn[None, :] * C_stride1,
        C_sub,
        mask=mask_c
    )


# Elementwise GELU (tanh approximation) in Triton
@triton.jit
def _gelu_tanh_kernel(
    X_ptr, Y_ptr,
    M, N,
    X_stride0, X_stride1,
    Y_stride0, Y_stride1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (rm[:, None] < M) & (rn[None, :] < N)

    x = tl.load(
        X_ptr + rm[:, None] * X_stride0 + rn[None, :] * X_stride1,
        mask=mask, other=0.0
    ).to(tl.float32)

    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = x + c * x3
    gelu = 0.5 * x * (1.0 + tl.math.tanh(sqrt_2_over_pi * inner))

    y = gelu.to(tl.bfloat16)
    tl.store(
        Y_ptr + rm[:, None] * Y_stride0 + rn[None, :] * Y_stride1,
        y,
        mask=mask
    )


# Entry point model
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, device):
        # Ensure contiguity and dtype
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous().to(torch.bfloat16)
        ln_bias = ln_bias.contiguous().to(torch.bfloat16)
        fc1_weight = fc1_weight.contiguous().to(torch.bfloat16)
        fc1_bias = fc1_bias.contiguous().to(torch.bfloat16)
        fc2_weight = fc2_weight.contiguous().to(torch.bfloat16)
        fc2_bias = fc2_bias.contiguous().to(torch.bfloat16)

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        hidden_size_expanded = hidden_size * 4  # merge_size=2 => 2x2 => 4 features per position

        # 1) LayerNorm per row
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        grid_ln = (num_patches,)
        _layernorm_rows_kernel[grid_ln](
            hidden, hidden_norm, ln_weight, ln_bias,
            num_patches, hidden_size,
            self.eps, BLOCK_SIZE=hidden_size
        )

        # 2) Pack rows into 1D vector of length num_patches * hidden_size_expanded
        packed = torch.empty(num_patches * hidden_size_expanded, dtype=torch.bfloat16, device=device)
        grid_pack = (num_patches,)
        _copy_rows_to_pack_kernel[grid_pack](
            hidden_norm, packed,
            num_patches, hidden_size, hidden_size_expanded,
            BLOCK_OUT=hidden_size_expanded
        )

        # Reshape into [num_merged_patches, hidden_size_expanded]
        num_merged_patches = num_patches // 4
        hidden_linear1 = packed.view(num_merged_patches, hidden_size_expanded)

        # 3) First Linear: (num_merged_patches, 6144) @ (6144, 6144) -> (num_merged_patches, 6144)
        B1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=device)
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid_gemm1 = (triton.cdiv(num_merged_patches, BLOCK_M), triton.cdiv(hidden_size_expanded, BLOCK_N))
        _gemm_rows_cols_kernel[grid_gemm1](
            hidden_linear1, fc1_weight, B1,
            num_merged_patches, hidden_size_expanded, hidden_size_expanded,
            hidden_linear1.stride(0), hidden_linear1.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            B1.stride(0), B1.stride(1),
            1, fc1_bias,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # 4) GELU activation in Triton
        B1_gelu = torch.empty_like(B1, dtype=torch.bfloat16, device=device)
        BLOCK_M_G, BLOCK_N_G = 128, 128
        grid_gelu = (triton.cdiv(num_merged_patches, BLOCK_M_G), triton.cdiv(hidden_size_expanded, BLOCK_N_G))
        _gelu_tanh_kernel[grid_gelu](
            B1, B1_gelu,
            num_merged_patches, hidden_size_expanded,
            B1.stride(0), B1.stride(1),
            B1_gelu.stride(0), B1_gelu.stride(1),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G
        )

        # 5) Second Linear: (num_merged_patches, 6144) @ (3584, 6144) -> (num_merged_patches, 3584)
        out_hidden_size = fc2_weight.shape[0]
        B2 = torch.empty((num_merged_patches, out_hidden_size), dtype=torch.bfloat16, device=device)
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_gemm2 = (triton.cdiv(num_merged_patches, BLOCK_M2), triton.cdiv(out_hidden_size, BLOCK_N2))
        _gemm_rows_cols_kernel[grid_gemm2](
            B1_gelu, fc2_weight, B2,
            num_merged_patches, out_hidden_size, hidden_size_expanded,
            B1_gelu.stride(0), B1_gelu.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            B2.stride(0), B2.stride(1),
            1, fc2_bias,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2
        )

        return B2


def run(*args):
    return ModelNew()(*args)
