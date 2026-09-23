import torch
import triton
import triton.language as tl

# 1) Triton LayerNorm per row: normalize over last dim = hidden_size (1536), apply ln_weight and ln_bias.
@triton.jit
def _layernorm_rows_kernel(
    inp_ptr,          # *bfloat16, input rows: [num_rows, hidden_size]
    weight_ptr,       # *bfloat16, ln weight: [hidden_size]
    bias_ptr,         # *bfloat16, ln bias: [hidden_size]
    out_ptr,          # *bfloat16, output rows: [num_rows, hidden_size]
    num_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    # Load row and compute mean/var in fp32
    row_in = tl.load(inp_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
    x_fp32 = row_in.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) / hidden_size
    x_centered = x_fp32 - mean
    var = tl.sum(x_centered * x_centered, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + 1e-6)
    x_norm = x_centered * rstd

    # Load ln weight/bias and apply
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * weight + bias

    # Store in bf16
    y_bf16 = y.to(tl.bfloat16)
    tl.store(out_ptr + row_id * hidden_size + cols, y_bf16, mask=mask)


# 2) Triton pack kernel: copy each hidden row into 1D output at slot i * hidden_expanded
@triton.jit
def _pack_rows_kernel(
    rows_ptr,         # *bfloat16, input rows: [num_rows, hidden_size]
    out_ptr,          # *bfloat16, output 1D: [num_rows * hidden_expanded]
    num_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    hidden_expanded: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    row_in = tl.load(rows_ptr + row_id * hidden_size + cols, mask=mask, other=0.0)
    # Output slot is row_id * hidden_expanded
    out_start = row_id * hidden_expanded
    tl.store(out_ptr + out_start + cols, row_in, mask=mask)


# 3) Triton GEMM for linear: A[M,K] @ B[K,N] -> C[M,N]
@triton.jit
def _gemm_rows_cols_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    A_stride_row, A_stride_col,
    B_stride_row, B_stride_col,
    C_stride_row, C_stride_col,
    bias_ptr,         # *bfloat16, [N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * A_stride_row) + (offs_k[None, :] * A_stride_col)
        b_ptrs = B_ptr + (offs_k[:, None] * B_stride_row) + (offs_n[None, :] * B_stride_col)

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store to C (bf16)
    c_ptrs = C_ptr + (offs_m[:, None] * C_stride_row) + (offs_n[None, :] * C_stride_col)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# 4) Triton GELU (tanh approximation) over 2D: Y[M,N] = GELU(X[M,N])
@triton.jit
def _gelu_tanh_kernel(
    X_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(X_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    gelu = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x3)))
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], gelu.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int = 1536, merge_size: int = 2, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.merge_size = merge_size
        self.eps = eps
        self.hidden_size_expanded = hidden_size * (merge_size ** 2)  # 6144
        self.out_hidden_size = 3584

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor):
        """
        hidden: [num_patches, hidden_size] bfloat16
        grid_thw: [num_grids, 3] int64 (T, H, W) - not used in Triton path but passed for API compatibility
        ln_weight, ln_bias: [hidden_size] bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] bfloat16
        fc1_bias: [hidden_size_expanded] bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded] bfloat16
        fc2_bias: [out_hidden_size] bfloat16
        """
        device = hidden.device
        num_patches = hidden.shape[0]
        hidden_size = self.hidden_size
        hidden_expanded = self.hidden_size_expanded
        out_hidden_size = self.out_hidden_size

        # 1) LayerNorm (per row) using Triton
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        grid_ln = (num_patches,)
        _layernorm_rows_kernel[grid_ln](
            hidden, ln_weight, ln_bias, hidden_norm,
            num_rows=num_patches, hidden_size=hidden_size, BLOCK_SIZE=hidden_size,
            num_warps=4, num_stages=2
        )

        # 2) Pack rows into 1D vector: length = num_patches * hidden_expanded
        packed = torch.empty(num_patches * hidden_expanded, dtype=torch.bfloat16, device=device)
        _pack_rows_kernel[(num_patches,)](
            hidden_norm, packed,
            num_rows=num_patches, hidden_size=hidden_size, hidden_expanded=hidden_expanded,
            BLOCK_SIZE=hidden_size,
            num_warps=4, num_stages=2
        )

        # 3) First Linear: (num_patches, hidden_expanded) @ (hidden_expanded, hidden_expanded) -> (num_patches, hidden_expanded)
        B1 = torch.empty((num_patches, hidden_expanded), dtype=torch.bfloat16, device=device)
        grid_gemm1 = (triton.cdiv(num_patches, 128), triton.cdiv(hidden_expanded, 128))
        _gemm_rows_cols_kernel[grid_gemm1](
            packed, fc1_weight, B1,
            M=num_patches, K=hidden_expanded, N=hidden_expanded,
            A_stride_row=hidden_expanded, A_stride_col=1,
            B_stride_row=hidden_expanded, B_stride_col=1,
            C_stride_row=hidden_expanded, C_stride_col=1,
            bias_ptr=fc1_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation in Triton
        B1_gelu = torch.empty_like(B1, dtype=torch.bfloat16, device=device)
        grid_gelu = (triton.cdiv(num_patches, 64), triton.cdiv(hidden_expanded, 128))
        _gelu_tanh_kernel[grid_gelu](
            B1, B1_gelu,
            M=num_patches, N=hidden_expanded,
            BLOCK_M=64, BLOCK_N=128,
            num_warps=4, num_stages=2
        )

        # 5) Second Linear: (num_patches, hidden_expanded) @ (out_hidden_size, hidden_expanded) -> (num_patches, out_hidden_size)
        output = torch.empty((num_patches, out_hidden_size), dtype=torch.bfloat16, device=device)
        grid_gemm2 = (triton.cdiv(num_patches, 128), triton.cdiv(out_hidden_size, 128))
        _gemm_rows_cols_kernel[grid_gemm2](
            B1_gelu, fc2_weight, output,
            M=num_patches, K=hidden_expanded, N=out_hidden_size,
            A_stride_row=hidden_expanded, A_stride_col=1,
            B_stride_row=hidden_expanded, B_stride_col=1,
            C_stride_row=out_hidden_size, C_stride_col=1,
            bias_ptr=fc2_bias,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
