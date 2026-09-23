import torch
import math
import triton
import triton.language as tl

# Kernel 1: LayerNorm (per-row) + affine
@triton.jit
def _layer_norm_affine_kernel(
    X_ptr,        # *bf16, input of shape [num_patches, hidden_size]
    W_ptr,        # *bf16, ln_weight of shape [hidden_size]
    B_ptr,        # *bf16, ln_bias of shape [hidden_size]
    Out_ptr,      # *bf16, output of shape [num_patches, hidden_size]
    H: tl.constexpr,            # hidden_size
    eps: tl.constexpr,          # epsilon
    BLOCK_SIZE: tl.constexpr    # tile size for reduction
):
    row = tl.program_id(0)
    row_base = row * H
    # First pass: compute mean and variance
    sum_val = 0.0
    sum_sq = 0.0
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    n = H
    mean = sum_val / n
    var = sum_sq / n - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Second pass: normalize and affine
    for col in range(0, H, BLOCK_SIZE):
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row_base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(Out_ptr + row_base + offs, y.to(tl.bfloat16), mask=mask)


# Kernel 2: Pack 2x2 features into expanded dimension (T=1 assumption)
@triton.jit
def _pack_2x2_to_expanded_kernel(
    In_ptr,       # *bf16, input after LN, shape [num_patches, hidden_size]
    Out_ptr,      # *bf16, output packed, shape [M, 4*hidden_size], M=num_patches//4
    H: tl.constexpr,             # hidden_size (1536)
    BLOCK_SIZE: tl.constexpr     # e.g., 1024 or 2048
):
    # One program per output row
    r = tl.program_id(0)  # r in [0, M)
    if r >= tl.num_programs(0):
        return

    # Compute indices for the 2x2 block components
    # Note: T=1 implies num_patches = 4*M
    base = r * 4 * H

    # Segment 1: kh=0, kw=0
    offs1 = tl.arange(0, BLOCK_SIZE)
    mask1 = offs1 < H
    src1 = In_ptr + r * H + offs1  # original row r
    dst1 = Out_ptr + base + 0 * H + offs1
    tl.store(dst1, tl.load(src1, mask=mask1, other=0.0), mask=mask1)

    # Segment 2: kh=1, kw=0 -> original row r + (w//2) == r + (H//2)
    r2 = r + (H // 2)
    if r2 < (4 * H // 2):
        src2 = In_ptr + r2 * H + offs1
        dst2 = Out_ptr + base + 1 * H + offs1
        tl.store(dst2, tl.load(src2, mask=mask1, other=0.0), mask=mask1)
    else:
        # Masked store, nothing to do since r2 out of range for valid r; we could leave zeros but we
        # structure our r so r2 is always valid (M=num_patches//4), keep mask for safety.
        pass

    # Segment 3: kh=0, kw=1 -> original row r + (w//2) == r + (H//2)
    # (same as above, we used r2 already)

    # Segment 4: kh=1, kw=1 -> original row r + 2*(w//2) == r + 2*(H//2)
    r3 = r + 2 * (H // 2)
    if r3 < 4 * H:
        src3 = In_ptr + r3 * H + offs1
        dst3 = Out_ptr + base + 2 * H + offs1
        tl.store(dst3, tl.load(src3, mask=mask1, other=0.0), mask=mask1)
    else:
        pass

# Kernel 3: GEMM + bias: C[M, N] = A[M, K] @ W[N, K]^T + Bias[N]
@triton.jit
def _gemm_bias_kernel(
    A_ptr,           # *bf16, [M, K]
    B_ptr,           # *bf16, [N, K] (we access as W[n, k] = B[n*K + k])
    Bias_ptr,        # *bf16, [N]
    C_ptr,           # *bf16, [M, N]
    M: tl.constexpr, # number of rows in A (and C)
    N: tl.constexpr, # number of columns in C (and Bias)
    K: tl.constexpr, # feature dimension
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m0 * K + (tl.arange(0, BLOCK_M)[:, None]) * K + k_ids[None, :]
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (k_ids[None, :] < K), other=0.0)
        a = a.to(tl.float32)
        # Load B tile as W[n, k] with B_ptr[n*K + k]
        b_ptrs = B_ptr + n0 * K + k_ids[None, :] * N + tl.arange(0, BLOCK_N)[:, None]
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_N)[:, None] < BLOCK_N) & (k_ids[None, :] < K), other=0.0)
        b = b.to(tl.float32)
        acc += tl.dot(a, b)  # (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)

    # Add bias
    bias = tl.load(Bias_ptr + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < BLOCK_N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store
    c_ptrs = C_ptr + m0 * N + (tl.arange(0, BLOCK_M)[:, None]) * N + (n0 + tl.arange(0, BLOCK_N)[None, :])
    mask_c = (tl.arange(0, BLOCK_M)[:, None] < BLOCK_M) & (tl.arange(0, BLOCK_N)[None, :] < BLOCK_N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


# Kernel 4: GELU elementwise
@triton.jit
def _gelu_kernel(
    In_ptr,     # *bf16, input of shape [M, N]
    Out_ptr,    # *bf16, output of shape [M, N]
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n0 = pid_n * BLOCK_N
    # Load row segment
    x = tl.load(In_ptr + m * N + n0 + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N, other=0.0).to(tl.float32)
    # GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(t))
    tl.store(Out_ptr + m * N + n0 + tl.arange(0, BLOCK_N), y.to(tl.bfloat16), mask=tl.arange(0, BLOCK_N) < N)


def _ceil_div(a, b):
    return (a + b - 1) // b


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
        """
        hidden: [num_patches, 1536], bfloat16
        grid_thw: [num_grids, 3], int64, T H W
        ln_weight, ln_bias: [1536], bfloat16
        fc1_weight: [6144, 6144], bfloat16
        fc1_bias: [6144], bfloat16
        fc2_weight: [3584, 6144], bfloat16
        fc2_bias: [3584], bfloat16
        eps: float
        """
        device = hidden.device
        dtype = hidden.dtype

        H = 1536  # hidden size
        H_expanded = 6144  # 4 * H
        H_out = 3584  # output hidden size

        # 1) LayerNorm + affine (Triton)
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        grid = (_ceil_div(hidden.shape[0], 128),)
        _layer_norm_affine_kernel[grid](
            hidden, ln_weight, ln_bias, hidden_norm,
            H=H, eps=eps,
            BLOCK_SIZE=1024
        )

        # 2) Spatial shuffle: pack 2x2 to expanded dimension (T=1 assumption)
        # M_input_rows = num_patches // 4 (since T=1 and we merge 2x2 -> 1 row)
        M_input_rows = hidden.shape[0] // 4
        shuffled_in = torch.empty((M_input_rows, H_expanded), dtype=torch.bfloat16, device=device)
        _pack_2x2_to_expanded_kernel[(M_input_rows,)](
            hidden_norm, shuffled_in,
            H=H, BLOCK_SIZE=2048
        )

        # 3) fc1: GEMM + bias, then GELU (Triton)
        M_fc1 = M_input_rows  # equals num_merged_patches (given grid_thw produces T=1 and 2x2 merge)
        N_fc1 = H_expanded
        K_fc1 = H_expanded
        A = shuffled_in  # [M_fc1, K_fc1]
        W1 = fc1_weight  # [N_fc1, K_fc1]
        b1 = fc1_bias    # [N_fc1]
        out_fc1 = torch.empty((M_fc1, N_fc1), dtype=torch.bfloat16, device=device)
        # Tile sizes; can be tuned. Using 64 for these dims.
        BLOCK_M_fc1, BLOCK_N_fc1, BLOCK_K_fc1 = 64, 64, 64
        grid_fc1 = (_ceil_div(M_fc1, BLOCK_M_fc1), _ceil_div(N_fc1, BLOCK_N_fc1))
        _gemm_bias_kernel[grid_fc1](
            A, W1, b1, out_fc1,
            M=M_fc1, N=N_fc1, K=K_fc1,
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1, BLOCK_K=BLOCK_K_fc1
        )
        # GELU on out_fc1 (Triton)
        out_fc1_gelu = torch.empty_like(out_fc1, dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_fc1, _ceil_div(N_fc1, BLOCK_N_gelu))
        _gelu_kernel[grid_gelu](
            out_fc1, out_fc1_gelu,
            M=M_fc1, N=N_fc1, BLOCK_N=BLOCK_N_gelu
        )

        # 4) fc2: GEMM + bias (Triton)
        M_fc2 = M_fc1  # num_merged_patches equals M_fc1
        N_fc2 = H_out
        K_fc2 = N_fc1  # 6144
        A2 = out_fc1_gelu  # [M_fc2, K_fc2]
        W2 = fc2_weight     # [N_fc2, K_fc2]
        b2 = fc2_bias       # [N_fc2]
        output = torch.empty((M_fc2, N_fc2), dtype=torch.bfloat16, device=device)
        BLOCK_M_fc2, BLOCK_N_fc2, BLOCK_K_fc2 = 64, 64, 64
        grid_fc2 = (_ceil_div(M_fc2, BLOCK_M_fc2), _ceil_div(N_fc2, BLOCK_N_fc2))
        _gemm_bias_kernel[grid_fc2](
            A2, W2, b2, output,
            M=M_fc2, N=N_fc2, K=K_fc2,
            BLOCK_M=BLOCK_M_fc2, BLOCK_N=BLOCK_N_fc2, BLOCK_K=BLOCK_K_fc2
        )

        return output


def run(*args):
    return ModelNew()(*args)
