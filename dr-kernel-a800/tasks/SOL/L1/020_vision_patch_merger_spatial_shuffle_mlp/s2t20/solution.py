import torch
import triton
import triton.language as tl

# LayerNorm kernel: per-row mean/var and affine
@triton.jit
def layernorm_affine_kernel(
    x_ptr,              # *float32, input [num_patches, hidden_size]
    out_ptr,            # *float32, output [num_patches, hidden_size]
    ln_weight_ptr,      # *float32, [hidden_size]
    ln_bias_ptr,        # *float32, [hidden_size]
    hidden_size: tl.constexpr,
    NUM_PATCHES: tl.constexpr,
    eps,                # float32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x = tl.load(x_ptr + pid * hidden_size + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / hidden_size
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / hidden_size
    inv_std = tl.math.rsqrt(var + eps)
    norm = diff * inv_std
    w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0)
    b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0)
    out = norm * w + b
    tl.store(out_ptr + pid * hidden_size + offs, out, mask=mask)


# GEMM kernel: C[M, N] = A[M, K] @ B[K, N] (+ bias)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    HAS_BIAS: tl.constexpr,
    eps,  # unused, but kept for signature symmetry
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = A_ptr + offs_m[:, None] * A_stride_m + (k + offs_k[None, :]) * A_stride_k
        b_ptrs = B_ptr + (k + offs_k[:, None]) * B_stride_k + offs_n[None, :] * B_stride_n
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BLOCK_N]
        acc += bias[None, :]
    c_ptrs = C_ptr + offs_m[:, None] * C_stride_m + offs_n[None, :] * C_stride_n
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# GELU kernel: tanh approximation applied elementwise
@triton.jit
def gelu_kernel(inp_ptr, out_ptr, M, NUM_EXPANDED: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, NUM_EXPANDED)
    mask = offs < NUM_EXPANDED
    x = tl.load(inp_ptr + pid * NUM_EXPANDED + offs, mask=mask, other=0.0)
    # tanh approximation
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    u = c0 * (x + c1 * x3)
    t = tl.tanh(u)
    y = 0.5 * x * (1.0 + t)
    tl.store(out_ptr + pid * NUM_EXPANDED + offs, y, mask=mask)


def _cdiv(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        # Triton-only forward: compute LayerNorm, fc1 (GEMM + bias), GELU, fc2 (GEMM + bias)
        device = hidden.device
        # 1) LayerNorm in fp32 via Triton
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_norm = torch.empty(num_patches, hidden_size, dtype=torch.float32, device=device)

        ln_weight_f32 = ln_weight.to(torch.float32)
        ln_bias_f32 = ln_bias.to(torch.float32)

        layernorm_affine_kernel[(num_patches,)](
            hidden, hidden_norm,
            ln_weight_f32, ln_bias_f32,
            hidden_size, num_patches,
            float(eps),
            1024,  # BLOCK_SIZE >= hidden_size
            num_warps=4,
        )

        # 2) fc1: A=MxK = hidden_norm, B=KxK = fc1_weight.T, bias=fc1_bias
        M = num_patches
        K = hidden_size
        N_fc1 = fc1_weight.shape[0]  # 6144
        A = hidden_norm  # [M, K]
        B = fc1_weight    # [K, K] bfloat16
        bias_fc1 = fc1_bias.to(torch.float32)  # [K]

        # Prepare B as [K, N_fc1] for matmul: we need fc1_weight.T
        # Triton GEMM expects B as [K, N], we will pass fc1_weight as [K, N] (already)
        # Create output C1 [M, N_fc1]
        C1 = torch.empty((M, N_fc1), dtype=torch.float32, device=device)

        # Launch GEMM + bias
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (_cdiv(M, BLOCK_M), _cdiv(N_fc1, BLOCK_N))
        gemm_bias_kernel[grid](
            A, fc1_weight, C1, bias_fc1,
            M, N_fc1, K,
            A.stride(0), A.stride(1),
            fc1_weight.stride(0), fc1_weight.stride(1),
            C1.stride(0), C1.stride(1),
            True,
            float(eps),
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4,
        )

        # 3) GELU activation in Triton
        M2 = M
        NUM_EXPANDED = N_fc1
        C1_out = torch.empty_like(C1, dtype=torch.float32, device=device)
        gelu_kernel[(M2,)](
            C1, C1_out,
            M2, NUM_EXPANDED,
            num_warps=4,
        )

        # 4) fc2: A=MxK, B=KxN_fc2 (fc2_weight), bias=fc2_bias
        N_fc2 = fc2_weight.shape[0]  # 3584
        B_fc2 = fc2_weight  # [N_fc2, K], we need [K, N_fc2] for GEMM
        bias_fc2 = fc2_bias.to(torch.float32)  # [N_fc2]

        C2 = torch.empty((M, N_fc2), dtype=torch.float32, device=device)
        grid2 = (_cdiv(M, BLOCK_M), _cdiv(N_fc2, BLOCK_N))
        gemm_bias_kernel[grid2](
            C1_out, B_fc2, C2, bias_fc2,
            M, N_fc2, K,
            C1_out.stride(0), C1_out.stride(1),
            B_fc2.stride(1), B_fc2.stride(0),  # B2 is [N, K]; we need [K, N] => take transpose strides
            C2.stride(0), C2.stride(1),
            True,
            float(eps),
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4,
        )

        # Return final output
        return C2


def run(*args):
    return ModelNew()(*args)
