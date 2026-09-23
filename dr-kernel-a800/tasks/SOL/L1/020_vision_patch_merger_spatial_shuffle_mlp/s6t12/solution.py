import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    X_ptr,      # *bf16, input of shape [M, K]
    W_ptr,      # *bf16, ln_weight of shape [K]
    B_ptr,      # *bf16, ln_bias of shape [K]
    Out_ptr,    # *bf16, output of shape [M, K]
    M: tl.constexpr,   # number of rows (patches)
    K: tl.constexpr,   # hidden size (e.g., 1536)
    eps: tl.constexpr, # epsilon for LN
    BLOCK: tl.constexpr,
):
    # one program per row
    pid = tl.program_id(0)

    # compute sum and sum of squares in FP32
    sum_val = 0.0
    sum_sq = 0.0
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + pid * K + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    n = K
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine
    for col in range(0, K, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(X_ptr + pid * K + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Out_ptr + pid * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,           # *bf16, input matrix [M, K] (row-major)
    B_ptr,           # *bf16, weight matrix [N, K] (row-major)
    Bias_ptr,        # *bf16, bias [N]
    C_ptr,           # *bf16, output [M, N]
    M: tl.constexpr, # rows of A
    N: tl.constexpr, # cols of C / rows of Bias
    K: tl.constexpr, # feature dim
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch over tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None]) * K + k_ids[None, :]
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (k_ids[None, :] < K), other=0.0).to(tl.float32)
        # B tile: [BLOCK_K, BLOCK_N] by loading B[n, k]
        b_ptrs = B_ptr + n0 * K + k_ids[:, None] * N + (tl.arange(0, BLOCK_N)[None, :] * K)
        b = tl.load(b_ptrs, mask=(k_ids[:, None] < K) & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < BLOCK_N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store to C
    c_ptrs = C_ptr + m0 * N + (tl.arange(0, BLOCK_M)[:, None]) * N + (n0 + tl.arange(0, BLOCK_N)[None, :])
    mask_c = (tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


@triton.jit
def _gelu_kernel(
    In_ptr,     # *bf16, input [M, N]
    Out_ptr,    # *bf16, output [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n0 = pid_n * BLOCK_N
    x = tl.load(In_ptr + m * N + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0).to(tl.float32)
    # tanh approximation
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    u = c * (x + 0.044715 * x3) * (1.0 - tl.tanh(0.7978845608028654 * (x + 0.044715 * x3)))
    tl.store(Out_ptr + m * N + n0 + tl.arange(0, BLOCK_N), u.to(tl.bfloat16), mask=tl.arange(0, BLOCK_N) < N)


def _launch_layer_norm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, device: torch.device):
    # hidden: [N, K], BF16
    N = hidden.shape[0]
    K = hidden.shape[1]
    ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
    BLOCK_ln = 256
    grid_ln = (N,)
    _layer_norm_affine_kernel[grid_ln](
        hidden, ln_weight, ln_bias, ln_out,
        M=N, K=K, eps=1e-6,
        BLOCK=BLOCK_ln, num_warps=4, num_stages=2
    )
    return ln_out


def _launch_fc1(A: torch.Tensor, W: torch.Tensor, Bias: torch.Tensor, device: torch.device):
    # A: [M, K1] BF16 (packed input)
    M = A.shape[0]
    K1 = A.shape[1]
    N1 = W.shape[0]  # output features
    C = torch.empty((M, N1), dtype=torch.bfloat16, device=device)
    BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
    grid_fc1 = (triton.cdiv(M, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
    _gemm_bias_kernel[grid_fc1](
        A, W, Bias, C,
        M=M, N=N1, K=K1,
        BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
        num_warps=4, num_stages=2
    )
    return C


def _launch_gelu(X: torch.Tensor, device: torch.device):
    # X: [M, N1] BF16
    M = X.shape[0]
    N1 = X.shape[1]
    Y = torch.empty_like(X, dtype=torch.bfloat16, device=device)
    BLOCK_N_gelu = 256
    grid_gelu = (M, triton.cdiv(N1, BLOCK_N_gelu))
    _gelu_kernel[grid_gelu](
        X, Y,
        M=M, N=N1, BLOCK_N=BLOCK_N_gelu,
        num_warps=4, num_stages=2
    )
    return Y


def _launch_fc2(X: torch.Tensor, W: torch.Tensor, Bias: torch.Tensor, device: torch.device):
    # X: [M, N1] BF16 (gelu output)
    M = X.shape[0]
    N1 = X.shape[1]
    N2 = W.shape[0]  # output size (e.g., 3584)
    C = torch.empty((M, N2), dtype=torch.bfloat16, device=device)
    BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 64, 64
    grid_fc2 = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
    _gemm_bias_kernel[grid_fc2](
        X, W, Bias, C,
        M=M, N=N2, K=N1,
        BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
        num_warps=4, num_stages=2
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        # Ensure tensors are on CUDA for Triton
        device = hidden.device

        # 1) LayerNorm + affine (Triton)
        ln_out = _launch_layer_norm(hidden, ln_weight, ln_bias, device)


def run(*args):
    return ModelNew()(*args)
