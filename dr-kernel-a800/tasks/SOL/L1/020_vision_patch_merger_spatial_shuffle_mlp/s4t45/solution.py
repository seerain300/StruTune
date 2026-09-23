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

    # Load row as bf16, convert to fp32
    x = tl.load(hidden_row_ptr + cols, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Mean
    mean = tl.sum(x_f32, axis=0) / N
    # Variance
    x_centered = x_f32 - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    y = x_centered * rstd

    # Load ln_weight and ln_bias as fp32
    w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    z = y * w + b  # fp32
    z_bf16 = z.to(tl.bfloat16)
    tl.store(out_row_ptr + cols, z_bf16, mask=mask)


@triton.jit
def _pack_rows_linear_kernel(
    hidden_ptr, out_ptr,
    num_patches, hidden_size, hidden_size_expanded,
    BLOCK: tl.constexpr
):
    # One program per original row
    r = tl.program_id(0)
    dest_idx = r * hidden_size_expanded
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden_size
    src_ptr = hidden_ptr + r * hidden_size
    dst_ptr = out_ptr + dest_idx
    x = tl.load(src_ptr + cols, mask=mask, other=0.0).to(tl.bfloat16)
    tl.store(dst_ptr + cols, x, mask=mask)


@triton.jit
def _gemm_rows_cols_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    bias_ptr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D launch: programs over (pid_m, pid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        A_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        B_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)

        acc += tl.dot(A_tile, B_tile)

    # Add bias (per-column)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    C_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.bfloat16), mask=C_mask)


@triton.jit
def _gelu_tanh_kernel(
    X, Y,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    Y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    x = tl.load(X_ptrs, mask=mask, other=0.0).to(tl.float32)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    u = c0 * (x + c1 * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(u))
    tl.store(Y_ptrs, gelu.to(tl.bfloat16), mask=mask)


def _launch_layernorm(hidden: torch.Tensor, out: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float):
    hidden = hidden.contiguous()
    out = out.contiguous()
    N = hidden.shape[1]
    grid = (hidden.shape[0],)
    _layernorm_rows_kernel[grid](
        hidden, out,
        ln_weight, ln_bias,
        N, eps,
        hidden.stride(0),
        BLOCK_SIZE=N  # cover full row
    )


def _launch_packing_kernel(hidden: torch.Tensor, out: torch.Tensor, num_patches: int, hidden_size: int, hidden_size_expanded: int):
    hidden = hidden.contiguous()
    grid = (hidden.shape[0],)
    BLOCK = hidden.shape[1]  # hidden_size
    _pack_rows_linear_kernel[grid](
        hidden, out,
        num_patches, hidden_size, hidden_size_expanded,
        BLOCK=BLOCK
    )


def _launch_gemm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, bias: torch.Tensor):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "GEMM shape mismatch"
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    _gemm_rows_cols_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        bias,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
    )


def _launch_gelu(X: torch.Tensor, Y: torch.Tensor):
    M, N = X.shape
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    _gelu_tanh_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_M=64, BLOCK_N=128
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        Triton-optimized forward:
        - LayerNorm per row in bf16, computed in fp32 and stored bf16.
        - Spatial pack into 1D vector (per-row copy to consecutive slots).
        - First Linear: A[M, K] @ B[K, N] -> C[M, N] (GEMM Triton).
        - GELU activation (Triton tanh approximation).
        - Second Linear: GEMM Triton.
        """
        device = hidden.device
        assert hidden.is_cuda, "Inputs must be CUDA tensors for Triton."
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        # 1) LayerNorm
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        _launch_layernorm(hidden, hidden_norm, ln_weight, ln_bias, eps)

        # 2) Spatial packing: per-row copy into 1D vector of length num_patches * 4 * hidden_size
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_size_expanded = hidden_size * 4  # 6144
        hidden_shuffled = torch.empty(num_patches * hidden_size_expanded, dtype=torch.bfloat16, device=device)
        _launch_packing_kernel(hidden_norm, hidden_shuffled, num_patches, hidden_size, hidden_size_expanded)

        # 3) Reshape for first linear: [num_merged_patches, hidden_size_expanded]
        num_merged_patches = num_patches // 4
        hidden_linear1 = hidden_shuffled.view(num_merged_patches, hidden_size_expanded)

        # 4) First Linear
        B1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=device)
        _launch_gemm(hidden_linear1, fc1_weight, B1, fc1_bias)

        # 5) GELU
        B1_gelu = torch.empty_like(B1, dtype=torch.bfloat16, device=device)
        _launch_gelu(B1, B1_gelu)

        # 6) Second Linear
        out_hidden_size = fc2_weight.shape[0]
        output = torch.empty((num_merged_patches, out_hidden_size), dtype=torch.bfloat16, device=device)
        _launch_gemm(B1_gelu, fc2_weight, output, fc2_bias)

        return output


def run(*args):
    return ModelNew()(*args)
