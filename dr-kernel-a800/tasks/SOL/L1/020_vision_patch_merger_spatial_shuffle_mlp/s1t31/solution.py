import torch
import triton
import triton.language as tl


# Triton LayerNorm per row: C[N, C] = (A[N, C] - mean) / sqrt(var + eps) * ln_weight + ln_bias
# Input A: [N, C] bfloat16 (flattened row-major), Output C: [N, C] bfloat16
@triton.jit
def layernorm_kernel(
    A_ptr,            # *bf16, [N, C] flattened
    LN_out_ptr,       # *bf16, [N, C] flattened
    ln_weight_ptr,    # *bf16, [C]
    ln_bias_ptr,      # *bf16, [C]
    N, C, eps,        # int32, int32, float32
    BLOCK_M: tl.constexpr,  # process BLOCK_M rows per program
    BLOCK_N: tl.constexpr,  # process BLOCK_N columns (C) per program
):
    pid = tl.program_id(0)
    start_row = pid * BLOCK_M
    offs_m = start_row + tl.arange(0, BLOCK_M)
    # Accumulate mean and variance in float32
    sum_vec = tl.zeros((BLOCK_M,), dtype=tl.float32)
    sumsq_vec = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # First pass: mean
    for c in range(0, C, BLOCK_N):
        offs_n = c + tl.arange(0, BLOCK_N)
        # mask for valid rows and columns
        mask = (offs_m[:, None] < N) & (offs_n[None, :] < C)
        # load A as bf16, cast to float32
        a = tl.load(A_ptr + (offs_m[:, None] * C) + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
        sum_vec += tl.sum(a, axis=1)
        sumsq_vec += tl.sum(a * a, axis=1)
    # compute mean and var
    mean = sum_vec / C
    var = sumsq_vec / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and affine
    for c in range(0, C, BLOCK_N):
        offs_n = c + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < N) & (offs_n[None, :] < C)
        a = tl.load(A_ptr + (offs_m[:, None] * C) + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
        ln_w = tl.load(ln_weight_ptr + offs_n, mask=(offs_n < C), other=0.0).to(tl.float32)
        ln_b = tl.load(ln_bias_ptr + offs_n, mask=(offs_n < C), other=0.0).to(tl.float32)
        norm = (a - mean[:, None]) * inv_std[:, None]
        out = norm * ln_w[None, :] + ln_b[None, :]
        # store as bfloat16
        tl.store(LN_out_ptr + (offs_m[:, None] * C) + offs_n[None, :],
                 out.to(tl.bfloat16), mask=mask)


# Triton Spatial Shuffle:
# Input: hidden_norm: [N, C] (bfloat16 flattened)
# Output: shuffled: [M, 4*C] (bfloat16)
# T, H, W are known from get_inputs; M=num_merged_patches.
@triton.jit
def spatial_shuffle_kernel(
    hidden_norm_ptr,   # *bf16, [N, C] flattened
    shuffled_ptr,      # *bf16, [M, 4*C] flattened
    num_merged,        # int32
    num_patches,       # int32
    T, H, W, C,        # int32 (hidden_size = C = 1536)
    BLOCK_N: tl.constexpr,  # process BLOCK_N columns of C per program (usually 64 or 128)
):
    # Each program handles one grid row j in [0, num_merged)
    j = tl.program_id(0)
    if j >= num_merged:
        return

    N_per_grid = num_patches // num_merged
    start = j * N_per_grid

    # We need to map each row i in [start, start + N_per_grid) to its (t,h,w) indices.
    # Then compute r = h*W*4 + w*4 + c*8 for c in [0, C).
    # First, iterate over i to collect all rows; for each i, compute (t, h, w).
    # We'll build the output rows in blocks of BLOCK_N features.
    for k in range(0, C, BLOCK_N):
        offs_c = k + tl.arange(0, BLOCK_N)
        # For each i in this grid's slice, compute normalized indices (since i is strictly within grid)
        # t = i // (H*W), h = (i // W) % H, w = i % W
        # But i is not known; instead we process each i via host launching; here we assume N_per_grid == T*H*W, so we can decode.
        # However, Triton kernel can't loop over N_per_grid with dynamic variable; we instead assume host sets grid size and do per-j complete mapping.
        # Better: pass N_per_grid and loop over i inside kernel? Triton allows loops, but dynamic bound is not ideal. We will instead rely on host setting grid as (num_merged,) and compute all rows for j inside one program using N_per_grid.
        # Compute N_per_grid in host and pass. Here we compute N_per_grid as runtime integer.
        # We will iterate over i from 0 to N_per_grid-1 and map to (t,h,w), then write to shuffled.
        # This requires nested loop; Triton can handle it.
        for i_local in range(0, N_per_grid):
            i = start + i_local
            # Compute t, h, w from i
            t = i // (H * W)
            rem = i % (H * W)
            h = rem // W
            w = rem % W
            # For each c in [k, k+BLOCK_N), compute shuffled index r = (h*W + w)*4 + c*2, since merge_size=2 -> 4*C
            # r_vec = (h*W + w) * 4 + offs_c * 2
            r_vec = ((h * W + w) * 4) + (offs_c * 2)
            # Load hidden_norm row i, columns [k : k+BLOCK_N)
            a = tl.load(hidden_norm_ptr + (i * C) + (k + tl.arange(0, BLOCK_N)), mask=(k + tl.arange(0, BLOCK_N) < C), other=0.0).to(tl.float32)
            # Store to shuffled[j, r_vec]
            tl.store(shuffled_ptr + (j * (4 * C)) + r_vec, a.to(tl.bfloat16), mask=(k + tl.arange(0, BLOCK_N) < C))


# Triton GEMM without bias: C[M, N] = A[M, K] @ W[K, N], store FP32
@triton.jit
def matmul_kernel_nobias(
    A_ptr,             # *bf16, [M, K]
    W_ptr,             # *bf16, [K, N]
    C_ptr,             # *bf32, [M, N]
    M, K, N,           # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + (offs_m[:, None] * K) + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(W_ptr + (offs_k[:, None] * N) + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    tl.store(C_ptr + (offs_m[:, None] * N) + offs_n[None, :],
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton GEMM with bias: C[M, N] = A[M, K] @ W[K, N] + bias[N], store FP32
@triton.jit
def matmul_bias_kernel(
    A_ptr,             # *bf16, [M, K]
    W_ptr,             # *bf16, [K, N]
    bias_ptr,          # *bf32, [N]
    C_ptr,             # *bf32, [M, N]
    M, K, N,           # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + (offs_m[:, None] * K) + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(W_ptr + (offs_k[:, None] * N) + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
    # add bias
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += bias[None, :]
    tl.store(C_ptr + (offs_m[:, None] * N) + offs_n[None, :],
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise GELU on FP32 input, store FP32
@triton.jit
def gelu_kernel(
    x_ptr,             # *bf16, [M, N]
    y_ptr,             # *bf32, [M, N]
    M, N,              # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    # GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + x^3 / 3)))
    # Constants
    c0 = 0.5
    c1 = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c1 * (x + x3 * (1.0 / 3.0))
    y = c0 * x * (1.0 + tl.tanh(inner))
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,  # [num_grids, 3] (T,H,W) but not needed in computation
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float,
                # axes provided by evaluator (optional; we use them as kernel args):
                num_patches: int,
                num_merged_patches: int,
                T: int, H: int, W: int):
        """
        hidden: [num_patches, 1536], bfloat16
        ln_weight, ln_bias: [1536], bfloat16
        fc1_weight: [6144, 6144], bfloat16
        fc1_bias: [6144], bfloat16
        fc2_weight: [3584, 6144], bfloat16
        fc2_bias: [3584], bfloat16
        eps: float
        num_patches: int
        num_merged_patches: int
        T, H, W: int from grid_thw per-evaluation config
        """
        assert hidden.is_cuda and hidden.dtype == torch.bfloat16, "hidden must be CUDA bfloat16"
        C = hidden.shape[1]
        device = hidden.device

        # 1) LayerNorm per row
        hidden_fp32 = hidden.to(torch.float32)
        ln_out = torch.empty_like(hidden_fp32, dtype=torch.bfloat16)  # we'll compute in fp32 and store as bfloat16
        # Launch layernorm kernel
        N = hidden_fp32.shape[0]
        # Choose block sizes
        BLOCK_M = 32
        BLOCK_N = 128
        grid_ln = (triton.cdiv(N, BLOCK_M),)
        layernorm_kernel[grid_ln](
            hidden_fp32, ln_out, ln_weight, ln_bias,
            N, C, float(eps),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 2) Spatial shuffle: hidden_norm is ln_out (fp32), but we need bfloat16? The original model uses bfloat16 for shuffle and MLP. We'll produce shuffled in bfloat16.
        # We need to produce [num_merged_patches, 4*C] bfloat16 tensor via Triton.
        M = num_merged_patches
        N_per_grid = num_patches // num_merged_patches  # runtime integer passed as args
        # Create output shuffled tensor
        shuffled = torch.empty((M, 4 * C), dtype=torch.bfloat16, device=device)

        # Launch spatial_shuffle_kernel
        BLOCK_N_SHUFFLE = 128
        grid_ss = (M,)
        spatial_shuffle_kernel[grid_ss](
            ln_out,  # input as fp32, we'll read as fp32 and cast as needed
            shuffled,
            M, num_patches, T, H, W, C,
            BLOCK_N=BLOCK_N_SHUFFLE,
        )

        # 3) fc1: linear (no bias) -> GELU -> add bias
        M_out = M
        K = 4 * C  # 6144
        N_fc1 = 6144
        A_fc1 = shuffled  # bfloat16
        W_fc1 = fc1_weight  # bfloat16
        C_fc1 = torch.empty((M_out, N_fc1), dtype=torch.float32, device=device)

        BLOCK_M_fc1 = 64
        BLOCK_N_fc1 = 64
        BLOCK_K_fc1 = 64
        grid_fc1 = (triton.cdiv(M_out, BLOCK_M_fc1), triton.cdiv(N_fc1, BLOCK_N_fc1))
        matmul_kernel_nobias[grid_fc1](
            A_fc1, W_fc1,
            C_fc1,
            M_out, K, N_fc1,
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1, BLOCK_K=BLOCK_K_fc1,
        )

        # Add fc1 bias
        C_fc1 = C_fc1 + fc1_bias.to(torch.float32)

        # GELU activation
        C_fc1_gelu = torch.empty_like(C_fc1, dtype=torch.float32)
        grid_gelu = (triton.cdiv(M_out, BLOCK_M_fc1), triton.cdiv(N_fc1, BLOCK_N_fc1))
        gelu_kernel[grid_gelu](
            C_fc1, C_fc1_gelu,
            M_out, N_fc1,
            BLOCK_M=BLOCK_M_fc1, BLOCK_N=BLOCK_N_fc1,
        )

        # 4) fc2: linear with bias
        N_fc2 = 3584
        A_fc2 = C_fc1_gelu  # fp32
        W_fc2 = fc2_weight  # bfloat16
        output_fp32 = torch.empty((M_out, N_fc2), dtype=torch.float32, device=device)

        BLOCK_M_fc2 = 64
        BLOCK_N_fc2 = 64
        BLOCK_K_fc2 = 64
        grid_fc2 = (triton.cdiv(M_out, BLOCK_M_fc2), triton.cdiv(N_fc2, BLOCK_N_fc2))
        matmul_bias_kernel[grid_fc2](
            A_fc2, W_fc2, fc2_bias,
            output_fp32,
            M_out, K, N_fc2,
            BLOCK_M=BLOCK_M_fc2, BLOCK_N=BLOCK_N_fc2, BLOCK_K=BLOCK_K_fc2,
        )

        # Cast output to bfloat16 as per original model's output dtype
        output = output_fp32.to(torch.bfloat16)
        return output


def run(*args):
    return ModelNew()(*args)
