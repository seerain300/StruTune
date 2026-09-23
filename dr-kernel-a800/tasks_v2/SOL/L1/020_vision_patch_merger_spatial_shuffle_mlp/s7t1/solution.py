import math
import torch
import triton
import triton.language as tl

# Triton kernel: LayerNorm over the last dimension with affine
# Input: hidden [N, C], N = num_patches, C = hidden_size (1536)
# Output: out [N, C], bf16 stored, compute in fp32
@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,      # *bf16, input
    out_ptr,         # *bf16, output
    weight_ptr,      # *bf16, per-channel weight (C elements)
    bias_ptr,        # *bf16, per-channel bias (C elements)
    N,               # int: number of rows
    C,               # int: number of features
    eps,             # float32: epsilon
    BLOCK_SIZE: tl.constexpr,  # tile size along C
):
    row_id = tl.program_id(axis=0)
    if row_id >= N:
        return
    offs = tl.arange(0, BLOCK_SIZE)

    # First pass: compute mean in fp32
    sum_x = 0.0
    x_row = hidden_ptr + row_id * C
    for c in range(0, C, BLOCK_SIZE):
        mask = (c + offs) < C
        x = tl.load(x_row + c + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
    mean = sum_x / C

    # Second pass: compute variance in fp32
    sum_sq = 0.0
    for c in range(0, C, BLOCK_SIZE):
        mask = (c + offs) < C
        x = tl.load(x_row + c + offs, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        sum_sq += tl.sum(diff * diff, axis=0)
    var = sum_sq / C
    rstd = 1.0 / tl.sqrt(var + eps)

    # Third pass: normalize, affine, store
    for c in range(0, C, BLOCK_SIZE):
        mask = (c + offs) < C
        x = tl.load(x_row + c + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        w = tl.load(weight_ptr + (c + offs), mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + (c + offs), mask=mask, other=0.0).to(tl.float32)
        y = y * w + b
        tl.store(out_ptr + row_id * C + c + offs, y.to(tl.bfloat16), mask=mask)


# Triton GEMM kernel: X[M, K] @ W[K, N] -> Out[M, N], add bias B[N] in epilogue
@triton.jit
def matmul_bias_kernel(
    X_ptr,            # *bf16 or *fp16, [M, K]
    W_ptr,            # *bf16 or *fp16, [K, N]
    B_ptr,            # *bf16 or *fp32, [N] bias
    Out_ptr,          # *bf16, [M, N]
    M, N, K,          # sizes
    stride_xm, stride_xk,  # strides for X
    stride_wk, stride_wn,  # strides for W
    stride_om, stride_on,  # strides for Out
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float16)

        # W tile [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float16)

        # Accumulate (fp16 dot -> fp32 acc)
        acc += tl.dot(x, w)

    # Add bias [N] to each column
    b = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += b[None, :]

    # Store result as bf16
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# Triton elementwise GELU (tanh approximation)
# gelu(x) ~ 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, N, alpha: tl.constexpr, beta: tl.constexpr):
    # One program per element for simplicity; N can be large, so we use grid (N,)
    idx = tl.program_id(axis=0)
    if idx >= N:
        return
    x = tl.load(X_ptr + idx)
    x3 = x * x * x
    inner = alpha * (x + beta * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(Y_ptr + idx, y)


def triton_layernorm_affine(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    """
    Triton LayerNorm with affine over last dim (features).
    hidden: [num_patches, hidden_size], bfloat16, CUDA
    weight, bias: [hidden_size], bfloat16, CUDA
    returns normalized tensor of same shape and dtype
    """
    assert hidden.is_cuda, "Triton kernel requires CUDA tensor"
    N, C = hidden.shape
    out = torch.empty_like(hidden)
    hidden_c = hidden.contiguous()
    weight_c = weight.contiguous()
    bias_c = bias.contiguous()
    # Use a BLOCK_SIZE that is a power-of-two close to C; 1024 works well for C=1536
    BLOCK_SIZE = 1024
    grid = (N,)
    layernorm_affine_kernel[grid](
        hidden_c, out, weight_c, bias_c, N, C, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


def triton_matmul_bias(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, out: torch.Tensor, BLOCK_M: int, BLOCK_N: int, BLOCK_K: int):
    """
    Triton GEMM: X[M, K] @ W[K, N] -> Out[M, N], add bias B[N] in epilogue.
    X, W, B: CUDA tensors. X and W can be bf16/fp16; Out is bf16.
    """
    M, Kx = X.shape
    Kw, N = W.shape
    assert Kx == Kw, f"Inner dims must match: X.shape={X.shape}, W.shape={W.shape}"
    # Ensure contiguity
    Xc = X.contiguous()
    Wc = W.contiguous()
    Bc = B.contiguous()
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_bias_kernel[grid](
        Xc, Wc, Bc, out, M, N, Kx,
        Xc.stride(0), Xc.stride(1),
        Wc.stride(0), Wc.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )


def triton_gelu_tanh_approx(inp: torch.Tensor) -> torch.Tensor:
    """
    Apply GELU using tanh approximation via Triton elementwise kernel.
    inp: CUDA tensor, any shape
    returns: same shape, same dtype as inp
    """
    assert inp.is_cuda, "Triton kernel requires CUDA tensor"
    N = inp.numel()
    out = torch.empty_like(inp)
    # We flatten for 1D kernel, then reshape back
    inp_flat = inp.view(-1).contiguous()
    out_flat = out.view(-1)
    # Constants for tanh-approx GELU
    alpha = 0.7978845608028654  # sqrt(2/pi)
    beta = 0.044715
    gelu_tanh_kernel[(N,)](
        inp_flat, out_flat, N, alpha, beta,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        """
        ModelNew implements the full computation in Triton:
        - LayerNorm (pre-shuffle) with affine in Triton
        - Spatial shuffle via PyTorch view/permute/reshape (data reordering, no compute)
        - First linear (GEMM) in Triton
        - GELU (tanh-approx) in Triton
        - Second linear (GEMM) in Triton
        All tensors are on CUDA and in bfloat16, eps is float.
        """
        device = hidden.device

        # Step 1: Triton LayerNorm (pre-shuffle, on hidden_size dimension)
        hidden_norm = triton_layernorm_affine(hidden, ln_weight, ln_bias, eps)

        # Step 2: Spatial shuffle to merge 2x2 patches (PyTorch reordering)
        # Compute offset and reshape for each grid:
        shuffled_patches = []
        offset = 0
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patches = hidden_norm[offset:offset + num_patches_this]
            # Reshape to (T, H//2, 2, W//2, 2, C)
            patches = patches.view(t, h // 2, 2, w // 2, 2, patches.shape[-1])
            # Permute to (T, H//2, W//2, 2, 2, C)
            patches = patches.permute(0, 1, 3, 2, 4, 5)
            # Flatten spatial merge groups: (T * H//2 * W//2, 4*C)
            hidden_expanded = patches.reshape(t * (h // 2) * (w // 2), 4 * patches.shape[-1])
            shuffled_patches.append(hidden_expanded)
            offset += num_patches_this
        hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # shape: [num_merged_patches, hidden_size_expanded]

        # Step 3: First linear (GEMM) in Triton: [M,K] @ [K,N] -> [M,N]
        M = hidden_shuffled.shape[0]
        K = hidden_shuffled.shape[1]
        N1 = fc1_weight.shape[1]  # hidden_size_expanded (6144)
        out1 = torch.empty((M, N1), dtype=torch.bfloat16, device=device)
        triton_matmul_bias(hidden_shuffled, fc1_weight, fc1_bias, out1, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)

        # Step 4: GELU (tanh approximation) in Triton
        out1_gelu = triton_gelu_tanh_approx(out1)

        # Step 5: Second linear (GEMM) in Triton: [M,N1] @ [N2,M] -> [M,N2] where N2=out_hidden_size (3584)
        M2 = out1_gelu.shape[0]
        K2 = out1_gelu.shape[1]
        N2 = fc2_weight.shape[0]  # out_hidden_size (3584)
        out2 = torch.empty((M2, N2), dtype=torch.bfloat16, device=device)
        triton_matmul_bias(out1_gelu, fc2_weight, fc2_bias, out2, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)

        return out2


def run(*args):
    return ModelNew()(*args)
