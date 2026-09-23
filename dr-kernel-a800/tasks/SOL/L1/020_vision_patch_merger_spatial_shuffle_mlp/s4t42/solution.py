import torch
import triton
import triton.language as tl

# Triton LayerNorm: one program per row, normalize across hidden_size (compile-time BLOCK_SIZE must be >= hidden_size)
@triton.jit
def _layernorm_rows_kernel(
    hidden_ptr,        # *bf16, shape (num_patches, hidden_size)
    hidden_norm_ptr,   # *bf16, shape (num_patches, hidden_size)
    ln_weight_ptr,     # *bf16, shape (hidden_size,)
    ln_bias_ptr,       # *bf16, shape (hidden_size,)
    num_patches,       # int32
    hidden_size,       # int32 (compile-time for masking)
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr,  # set to 1536
):
    pid = tl.program_id(axis=0)  # row index
    if pid >= num_patches:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size
    # Load row in fp32
    x = tl.load(hidden_ptr + pid * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
    # Mean
    mean = tl.sum(x, axis=0) / hidden_size
    # Variance
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / hidden_size
    rstd = 1.0 / tl.sqrt(var + eps)
    # Normalize and scale
    y = xc * rstd
    w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = y * w + b
    # Store as bf16
    tl.store(hidden_norm_ptr + pid * hidden_size + cols, z.to(tl.bfloat16), mask=mask)


# Triton "pack" kernel: map original rows to a 1D destination vector of length num_patches * hidden_size_expanded
# Destination length must equal num_patches * hidden_size_expanded (4 * hidden_size)
@triton.jit
def _pack_rows_to_1d_kernel(
    hidden_ptr,          # *bf16, shape (num_patches, hidden_size_expanded)
    out_ptr,             # *bf16, 1D vector of length num_patches * hidden_size_expanded
    num_patches,         # int32
    hidden_size_expanded,  # int32 (6144)
    BLOCK_SIZE: tl.constexpr,  # e.g., 6144
):
    pid = tl.program_id(axis=0)  # row index
    if pid >= num_patches:
        return
    start = pid * hidden_size_expanded
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size_expanded
    row = tl.load(hidden_ptr + pid * hidden_size_expanded + cols, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + start + cols, row, mask=mask)


# Triton GEMM: C[M, N] = A[M, K] @ B[K, N]
# We use 2D tiling with BLOCK_M/BLOCK_N/BLOCK_K. Accumulate in fp32, store bf16.
@triton.jit
def _gemm_rows_cols_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)  # tile id along M
    pid_n = tl.program_id(axis=1)  # tile id along N
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    # Store C
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton GELU (tanh approximation): elementwise on tensor B, store bf16
@triton.jit
def _gelu_tanh_kernel(
    B_ptr, Out_ptr, M, N,
    stride_bm, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=mask, other=0.0).to(tl.float32)
    # tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,  # unused in this Triton-only path, kept for signature
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,   # unused in this Triton-only path
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,   # unused in this Triton-only path
        eps: float,
    ):
        """
        Triton-optimized forward that performs:
        - LayerNorm (per-row) over last dim
        - Spatial packing into 1D vector of length num_patches * 4 * hidden_size
        - First linear (GEMM) with GELU
        - Second linear (GEMM)
        """
        assert hidden.is_cuda, "Input hidden must be on CUDA for Triton."
        assert ln_weight.is_cuda and ln_bias.is_cuda and fc1_weight.is_cuda and fc2_weight.is_cuda, "All tensors must be on CUDA for Triton."

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        hidden_size_expanded = hidden_size * 4  # 6144
        out_hidden_size = fc2_weight.shape[0]  # 3584

        # 1) Triton LayerNorm per row
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
        _layernorm_rows_kernel[(num_patches,)](
            hidden, hidden_norm,
            ln_weight, ln_bias,
            num_patches, hidden_size,
            eps,
            BLOCK_SIZE=1536
        )

        # 2) Triton pack rows into 1D: destination length = num_patches * hidden_size_expanded
        #    This is a simple linear copy: out[i * 6144 : (i+1) * 6144] = hidden_norm[i, :]
        hidden_pack = torch.empty(num_patches * hidden_size_expanded, dtype=torch.bfloat16, device=hidden.device)
        _pack_rows_to_1d_kernel[(num_patches,)](
            hidden_norm, hidden_pack,
            num_patches, hidden_size_expanded,
            BLOCK_SIZE=hidden_size_expanded
        )

        # 3) First Linear: reshape to [num_patches, hidden_size_expanded] and GEMM
        A1 = hidden_pack.view(num_patches, hidden_size_expanded)
        B1 = torch.empty((num_patches, hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)
        # GEMM: (num_patches, 6144) @ (6144, 6144) -> (num_patches, 6144)
        grid_gemm1 = (triton.cdiv(num_patches, 64), triton.cdiv(hidden_size_expanded, 128))
        _gemm_rows_cols_kernel[grid_gemm1](
            A1, fc1_weight, B1,
            num_patches, hidden_size_expanded, hidden_size_expanded,
            A1.stride(0), A1.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            B1.stride(0), B1.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        # 4) GELU activation via tanh approximation (elementwise)
        B1_gelu = torch.empty_like(B1, dtype=torch.bfloat16, device=hidden.device)
        grid_gelu = (triton.cdiv(num_patches, 64), triton.cdiv(hidden_size_expanded, 128))
        _gelu_tanh_kernel[grid_gelu](
            B1, B1_gelu,
            num_patches, hidden_size_expanded,
            B1.stride(0), B1.stride(1),
            B1_gelu.stride(0), B1_gelu.stride(1),
            BLOCK_M=64, BLOCK_N=128
        )

        # 5) Second Linear: (num_patches, hidden_size_expanded) @ (out_hidden_size, hidden_size_expanded)
        output = torch.empty((num_patches, out_hidden_size), dtype=torch.bfloat16, device=hidden.device)
        grid_gemm2 = (triton.cdiv(num_patches, 64), triton.cdiv(out_hidden_size, 128))
        _gemm_rows_cols_kernel[grid_gemm2](
            B1_gelu, fc2_weight, output,
            num_patches, out_hidden_size, hidden_size_expanded,
            B1_gelu.stride(0), B1_gelu.stride(1),
            fc2_weight.stride(0), fc2_weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64
        )

        return output


def run(*args):
    return ModelNew()(*args)
