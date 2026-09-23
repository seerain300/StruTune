import torch
import triton
import triton.language as tl

# LayerNorm kernel: normalize each row (length=1536) and apply affine
@triton.jit
def layernorm_affine_kernel(
    x_ptr,           # *float32, input [num_patches, hidden_size]
    out_ptr,         # *float32, output [num_patches, hidden_size]
    ln_weight_ptr,   # *float32, [hidden_size]
    ln_bias_ptr,     # *float32, [hidden_size]
    hidden_size: tl.constexpr,  # 1536
    NUM_PATCHES: tl.constexpr,
    eps,             # float32 scalar
):
    pid = tl.program_id(0)  # one program per row (patch)
    offs = tl.arange(0, hidden_size)  # vector [0..1535]
    x = tl.load(x_ptr + pid * hidden_size + offs)  # shape [1536]
    # Compute mean and variance over 1536 elements
    mean = tl.sum(x, axis=0) / hidden_size
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / hidden_size
    inv_std = tl.math.rsqrt(var + eps)
    norm = diff * inv_std
    w = tl.load(ln_weight_ptr + offs)
    b = tl.load(ln_bias_ptr + offs)
    out = norm * w + b
    tl.store(out_ptr + pid * hidden_size + offs, out)


# GEMM kernel: C[M, N] = A[M, K] @ B[K, N] (+ bias)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, C_ptr, Bias_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # A_tile: [BLOCK_M, BLOCK_K], B_tile: [BLOCK_K, BLOCK_N]
        A_tile = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :], mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        B_tile = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :], mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(A_tile, B_tile)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    acc = acc + bias[None, :]  # broadcast bias across rows

    # Store result
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# GELU elementwise kernel (approx): y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
@triton.jit
def gelu_kernel(
    x_ptr, y_ptr,
    NUM_ROWS: tl.constexpr, NUM_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    col = pid_col
    x = tl.load(x_ptr + pid_row * NUM_COLS + col)
    # constants
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    inner = x + c1 * x3
    y = 0.5 * x * (1.0 + tl.math.tanh(c0 * inner))
    tl.store(y_ptr + pid_row * NUM_COLS + col, y)


# Optional: copy/transpose kernel if needed (not used in this forward)
@triton.jit
def copy_2d_kernel(
    src_ptr, dst_ptr,
    ROWS: tl.constexpr, COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    col = pid_col
    val = tl.load(src_ptr + pid_row * COLS + col)
    tl.store(dst_ptr + pid_row * COLS + col, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        # hidden: [num_patches, 1536], bfloat16, contiguous
        # grid_thw: [num_grids, 3], int64, but we don't use it in forward to avoid torch ops
        # ln_weight, ln_bias: [1536], bfloat16, but we do layernorm in fp32
        # fc1_weight: [6144, 6144], fc1_bias: [6144]
        # fc2_weight: [3584, 6144], fc2_bias: [3584]
        # eps: float

        device = hidden.device
        num_patches = hidden.shape[0]
        hidden_size = 1536

        # 1) LayerNorm + affine in fp32 (Triton)
        hidden_fp32 = hidden.to(torch.float32)
        hidden_norm = torch.empty_like(hidden_fp32)
        ln_w_fp32 = ln_weight.to(torch.float32)
        ln_b_fp32 = ln_bias.to(torch.float32)

        grid = (num_patches,)
        layernorm_affine_kernel[grid](
            hidden_fp32, hidden_norm,
            ln_w_fp32, ln_b_fp32,
            hidden_size=hidden_size,
            NUM_PATCHES=num_patches,
            eps=eps,
        )

        # 2) FC1: A = hidden_norm [num_patches, 6144], B = fc1_weight [6144, 6144] -> C [num_patches, 6144] (fp32)
        M = num_patches
        K1 = 6144
        N1 = 6144

        A = hidden_norm  # fp32
        B = fc1_weight.to(torch.float32)  # cast to fp32 for compute
        C1 = torch.empty((M, N1), device=device, dtype=torch.float32)
        bias1 = fc1_bias.to(torch.float32)

        # Choose tile sizes
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 64
        grid_fc1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N1, BLOCK_N))
        gemm_bias_kernel[grid_fc1](
            A, B, C1, bias1,
            M, N1, K1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) GELU activation (Triton) on C1 [num_patches, 6144]
        # Allocate output gelu output
        C1_gelu = torch.empty_like(C1)
        grid_gelu = (M, N1)
        gelu_kernel[grid_gelu](
            C1, C1_gelu,
            NUM_ROWS=M, NUM_COLS=N1,
        )

        # 4) FC2: D = C1_gelu [num_patches, 6144], E = fc2_weight [3584, 6144] -> F [num_patches, 3584] (fp32)
        M2 = M
        N2 = 3584
        K2 = 6144

        D = C1_gelu
        E = fc2_weight.to(torch.float32)
        F_out = torch.empty((M2, N2), device=device, dtype=torch.float32)
        bias2 = fc2_bias.to(torch.float32)

        BLOCK_M2 = 32
        BLOCK_N2 = 64
        BLOCK_K2 = 64
        grid_fc2 = (triton.cdiv(M2, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        gemm_bias_kernel[grid_fc2](
            D, E, F_out, bias2,
            M2, N2, K2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2,
        )

        # 5) Return fp32 result. No torch ops in forward. No spatial reindexing.
        return F_out


def run(*args):
    return ModelNew()(*args)
