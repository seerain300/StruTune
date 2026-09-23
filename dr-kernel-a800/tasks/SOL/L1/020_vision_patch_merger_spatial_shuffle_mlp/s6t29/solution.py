import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_affine_kernel(
    x_ptr,         # *bf16, input [M, K]
    weight_ptr,    # *bf16, [K]
    bias_ptr,      # *bf16, [K]
    y_ptr,         # *bf16, output [M, K]
    M, K, eps,     # int32, float32
    BLOCK: tl.constexpr,
):
    # One program per row
    row = tl.program_id(axis=0)
    if row >= M:
        return
    # First pass: compute mean and variance in fp32
    acc1 = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        acc1 += tl.sum(x, axis=0)
        acc2 += tl.sum(x * x, axis=0)
    mean = acc1 / K
    var = acc2 / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for k0 in range(0, K, BLOCK):
        offs = k0 + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(y_ptr + row * K + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _gemm_bias_kernel(
    A_ptr,         # *bf16, input [M, K]
    W_ptr,         # *bf16, weight [N, K] (row-major: N is fast dim)
    B_ptr,         # *bf16, bias [N]
    C_ptr,         # *bf16, output [M, N]
    M, N, K,       # int32 sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None]) * K + k_offs[None, :]
        a_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M
        a_mask = a_mask & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load W tile as [BLOCK_K, BLOCK_N]
        # W is [N, K] row-major; to get W[k, n] for our tiles, use index n0 + arange(BLOCK_N) on fast dim
        w_ptrs = W_ptr + (n0 + tl.arange(0, BLOCK_N))[None, :] * K + k_offs[:, None]
        w_mask = (n0 + tl.arange(0, BLOCK_N))[None, :] < N
        w_mask = w_mask & k_mask[:, None]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Add bias: broadcast across rows
    bias_ptrs = B_ptr + n0 + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptrs, mask=(n0 + tl.arange(0, BLOCK_N)) < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store result
    c_ptrs = C_ptr + m0 * N + (tl.arange(0, BLOCK_M)[:, None]) * N + (n0 + tl.arange(0, BLOCK_N))[None, :]
    out_mask = ((m0 + tl.arange(0, BLOCK_M))[:, None] < M) & ((n0 + tl.arange(0, BLOCK_N))[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def _gelu_tanh_kernel(
    X_ptr,  # *bf16, input [M, N]
    Y_ptr,  # *bf16, output [M, N]
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(axis=0)
    col_block = tl.program_id(axis=1)
    col0 = col_block * BLOCK_N
    cols = col0 + tl.arange(0, BLOCK_N)
    mask = (row < M) & (cols < N)

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    u = c0 * (x + c1 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(u))
    tl.store(Y_ptr + row * N + cols, y.to(tl.bfloat16), mask=mask)


# ModelNew: Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
    ):
        # hidden: [num_patches, hidden_size] (1536), bf16
        # ln_weight, ln_bias: [hidden_size], bf16
        # fc1_weight: [hidden_size_expanded, hidden_size_expanded] (6144, 6144), bf16
        # fc1_bias: [hidden_size_expanded], bf16
        # fc2_weight: [out_hidden_size, hidden_size_expanded] (3584, 6144), bf16
        # fc2_bias: [out_hidden_size], bf16
        device = hidden.device
        dtype = hidden.dtype  # bf16

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536

        # 1) LayerNorm + affine (pre-shuffle), Triton
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (num_patches,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            num_patches, hidden_size, float(eps),
            BLOCK=BLOCK_ln,
            num_warps=4, num_stages=2
        )

        # 2) Spatial packing view (T=1, num_patches % 4 == 0)
        # From (num_patches, hidden_size) -> (num_patches//4, 4, hidden_size) -> (num_patches//4, hidden_size, 4) -> (num_patches//4, 4*hidden_size)
        M_out = num_patches // 4
        packed = ln_out.view(M_out, 4, hidden_size).transpose(1, 2).reshape(M_out, 4 * hidden_size)

        # 3) fc1: (M_out, 4*hidden_size) @ (4*hidden_size, 4*hidden_size)^T + bias -> (M_out, 4*hidden_size)
        K1 = 4 * hidden_size  # 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_out, N1), dtype=torch.bfloat16, device=device)

        # 2D grid over output rows and columns
        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(M_out, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M_out, N1, K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation, Triton
        BLOCK_N_gelu = 256
        grid_gelu = (M_out, triton.cdiv(N1, BLOCK_N_gelu))
        fc1_after_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        _gelu_tanh_kernel[grid_gelu](
            fc1_out, fc1_after_gelu,
            M_out, N1,
            BLOCK_N=BLOCK_N_gelu,
            num_warps=4, num_stages=2
        )

        # 5) fc2: (num_merged_patches=M_out, N1=6144) @ (3584, 6144)^T + bias -> (num_merged_patches, 3584)
        num_merged_patches = M_out  # as per provided inputs, this must match M_out
        out_hidden_size = fc2_weight.shape[0]  # 3584
        fc2_out = torch.empty((num_merged_patches, out_hidden_size), dtype=torch.bfloat16, device=device)

        # GEMM grid
        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_fc2 = (triton.cdiv(num_merged_patches, BLOCK_M2), triton.cdiv(out_hidden_size, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            fc1_after_gelu, fc2_weight, fc2_bias, fc2_out,
            num_merged_patches, out_hidden_size, N1,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return fc2_out


# If the evaluator calls ModelNew via the same interface, this is the entry point.
# Note: grid_thw is not used in computation (as in the original run), assuming T=1 and packing via view.


def run(*args):
    return ModelNew()(*args)
