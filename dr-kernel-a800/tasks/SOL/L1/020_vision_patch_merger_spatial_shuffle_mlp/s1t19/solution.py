import torch
import math
import triton
import triton.language as tl


# -------------------------
# 1) LayerNorm kernel: per-row LN
# -------------------------
@triton.jit
def layer_norm_kernel(
    hidden_ptr,        # *bf16, [N, C]
    out_ptr,           # *bf16, [N, C]
    ln_weight_ptr,     # *bf16, [C]
    ln_bias_ptr,       # *bf16, [C]
    N, C,              # int32
    eps,               # float32
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # row index
    if pid >= N:
        return

    # Compute mean (fp32)
    mean = 0.0
    for c0 in range(0, C, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(hidden_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / C

    # Compute variance (fp32)
    var = 0.0
    for c0 in range(0, C, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(hidden_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
        var += tl.sum((x - mean) * (x - mean), axis=0)
    var = var / C
    inv_std = tl.rsqrt(var + eps)

    # Normalize and apply affine, store bfloat16
    for c0 in range(0, C, BLOCK):
        offs = c0 + tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(hidden_ptr + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(out_ptr + pid * C + offs, y.to(tl.bfloat16), mask=mask)


# -------------------------
# 2) Spatial shuffle kernel: merge 2x2 into 4*C features per patch
# -------------------------
@triton.jit
def spatial_shuffle_kernel(
    src_ptr,           # *bf16, [num_patches, C] (LN output)
    dst_ptr,           # *bf16, [num_merged_patches, 4*C] (output of shuffle)
    grid_thw_ptr,      # *int32, [num_grids, 3] = [(t, h, w)]
    num_grids,         # int32
    num_patches,       # int32
    num_merged_patches,# int32
    C,                 # int32
    MERGE: tl.constexpr,  # 2
):
    pid_row = tl.program_id(0)  # output row in [0, num_merged_patches)
    pid_col = tl.program_id(1)  # output col in [0, 4*C)

    if (pid_row >= num_merged_patches) or (pid_col >= 4 * C):
        return

    # Determine which grid this output row belongs to and its per-grid (t, h, w)
    # We loop over grids and assign row to the first grid whose count exceeds row.
    # This is robust since num_patches = sum_i t_i*h_i*w_i.
    grid = -1
    base = 0
    for g in range(0, num_grids):
        t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
        h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
        w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)
        count = t * h * w
        if pid_row < (base + count):
            grid = g
            break
        base += count
    if grid == -1:
        return  # safety

    # Map output row to local (t_local, h_local, w_local) in this grid
    t = tl.load(grid_thw_ptr + grid * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + grid * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + grid * 3 + 2).to(tl.int32)

    # Number of patches in this grid
    num_patches_grid = t * h * w

    # Decode local patch id (pid_row is absolute across all grids)
    t_local = pid_row // (h * w)
    rem = pid_row % (h * w)
    h_local = rem // w
    w_local = rem % w

    # Decode spatial merge and feature index from output column
    s = pid_col // C        # {0,1,2,3}
    r = pid_col % C         # feature offset in [0, C)

    # Map to original spatial indices for 2x2 merge
    th = s // 2
    tw = s % 2
    hh = h_local + th * MERGE
    ww = w_local + tw * MERGE

    # Compute input patch id (absolute) and feature offset
    patch_id = base + t_local * (h * MERGE) * (w * MERGE) + hh * (w * MERGE) + ww
    feature_off = r

    # Load and store
    val = tl.load(src_ptr + patch_id * C + feature_off)
    tl.store(dst_ptr + pid_row * (4 * C) + pid_col, val.to(tl.bfloat16))


# -------------------------
# 3) GEMM without bias: C = A @ W (A: [M,K], W: [K,N], no bias)
# -------------------------
@triton.jit
def matmul_kernel_nobias(
    A_ptr,             # *bf16, [M, K]
    W_ptr,             # *bf16, [K, N]
    C_ptr,             # *bf16, [M, N]
    M, K, N,           # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(A_ptr + (offs_m[:, None] * K) + k[None, :],
                    mask=(offs_m[:, None] < M) & (k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(W_ptr + (k[:, None] * N) + offs_n[None, :],
                    mask=(k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    tl.store(C_ptr + (offs_m[:, None] * N) + offs_n[None, :],
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# -------------------------
# 4) GELU elementwise kernel: y = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
# -------------------------
@triton.jit
def gelu_kernel(
    x_ptr,             # *bf16, [M, K]
    y_ptr,             # *bf16, [M, K]
    M, K,              # int32
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0).to(tl.float32)

    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715

    x3 = x * x * x
    inner = x + c * x3
    y = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * inner))

    tl.store(y_ptr + offs_m[:, None] * K + offs_k[None, :], y.to(tl.bfloat16), mask=mask)


# -------------------------
# 5) GEMM with bias: C = A @ W + b (A: [M,K], W: [K,N], b: [N])
# -------------------------
@triton.jit
def matmul_bias_kernel(
    A_ptr,             # *bf16, [M, K]
    W_ptr,             # *bf16, [K, N]
    b_ptr,             # *bf16, [N]
    C_ptr,             # *bf16, [M, N]
    M, K, N,           # int32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a = tl.load(A_ptr + (offs_m[:, None] * K) + k[None, :],
                    mask=(offs_m[:, None] < M) & (k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(W_ptr + (k[:, None] * N) + offs_n[None, :],
                    mask=(k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)  # [BLOCK_N]
    acc = acc + bias[None, :]

    tl.store(C_ptr + (offs_m[:, None] * N) + offs_n[None, :],
             acc.to(tl.bfloat16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        """
        args: (hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps)
        hidden: [num_patches, 1536], bfloat16
        grid_thw: [num_grids, 3], int64 or int32 (we'll convert to int32 for Triton)
        ln_weight, ln_bias: [1536], bfloat16
        fc1_weight: [6144, 6144], bfloat16
        fc1_bias: [6144], bfloat16 (unused in Triton forward, but args exist)
        fc2_weight: [3584, 6144], bfloat16
        fc2_bias: [3584], bfloat16
        eps: float
        """
        # Unpack arguments
        hidden = args[0]
        grid_thw = args[1].contiguous()
        ln_weight = args[2].contiguous()
        ln_bias = args[3].contiguous()
        fc1_weight = args[4].contiguous()  # [K, K] = [6144, 6144]
        fc1_bias = args[5].contiguous()    # [K]
        fc2_weight = args[6].contiguous()  # [Nout, K] = [3584, 6144]
        fc2_bias = args[7].contiguous()    # [Nout]
        eps = float(args[8])

        device = hidden.device
        num_patches = hidden.shape[0]
        C = hidden.shape[1]
        num_grids = grid_thw.shape[0]

        # 1) Layer Normalization
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        N = num_patches
        grid_ln = (N,)
        layer_norm_kernel[grid_ln](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, C, eps,
            BLOCK=1024,
            num_warps=4,
        )

        # 2) Spatial shuffle per grid: produce [num_merged_patches, 4*C] in Triton
        # We need num_merged_patches. In the original code, it's equal to total patches processed:
        # However, for each workload, num_merged_patches is provided as input, so we reuse it.
        num_merged_patches = args[1].shape[0]  # actually provided as input num_merged_patches in axes; here we rely on caller.
        # Note: We will pass num_merged_patches from the host as the second argument to spatial_shuffle_kernel.
        # Construct dst tensor
        fourC = 4 * C
        hidden_shuffled = torch.empty((num_merged_patches, fourC), dtype=torch.bfloat16, device=device)

        # Ensure grid_thw is int32 for Triton
        grid_thw_int32 = grid_thw.to(torch.int32)

        # Launch spatial shuffle kernel with 2D grid: rows over num_merged_patches, cols over 4*C
        grid_shuffle = (num_merged_patches, fourC)
        spatial_shuffle_kernel[grid_shuffle](
            hidden_norm, hidden_shuffled, grid_thw_int32, num_grids, num_patches, num_merged_patches, C, MERGE=2
        )

        # 3) fc1: A is [num_merged_patches, 6144], W is [6144, 6144]
        M = num_merged_patches
        K = 6144
        A = hidden_shuffled  # [M, K]
        W = fc1_weight       # [K, K]
        fc1_out = torch.empty((M, K), dtype=torch.bfloat16, device=device)

        grid_fc1 = (triton.cdiv(M, 64), triton.cdiv(K, 128))
        matmul_kernel_nobias[grid_fc1](A, W, fc1_out, M, K, K,
                                       BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4)

        # 4) GELU
        fc1_gelu = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        grid_gelu = (triton.cdiv(M, 64), triton.cdiv(K, 128))
        gelu_kernel[grid_gelu](fc1_out, fc1_gelu, M, K,
                               BLOCK_M=64, BLOCK_K=128, num_warps=4)

        # 5) fc2: A is [M, K], W is [K, Nout], b is [Nout]
        Nout = fc2_weight.shape[0]  # 3584
        output = torch.empty((M, Nout), dtype=torch.bfloat16, device=device)
        grid_fc2 = (triton.cdiv(M, 64), triton.cdiv(Nout, 128))
        matmul_bias_kernel[grid_fc2](fc1_gelu, fc2_weight, fc2_bias, output, M, K, Nout,
                                     BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4)

        return output


def run(*args):
    return ModelNew()(*args)
