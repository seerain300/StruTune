import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,          # *bf16, [num_patches, hidden_size]
    ln_weight_ptr,       # *bf16, [hidden_size]
    ln_bias_ptr,         # *bf16, [hidden_size]
    out_ptr,             # *bf16, [num_patches, hidden_size]
    num_patches: tl.int32,
    hidden_size: tl.int32,
    eps: tl.float32,
    BLOCK_C: tl.constexpr,
):
    # One program per row (patch)
    row = tl.program_id(0)
    if row >= num_patches:
        return

    # Compute mean and variance over C features
    c = 0
    sum_val = 0.0
    sum_sq = 0.0
    while c < hidden_size:
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row * hidden_size + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # masked load: other=0.0 ensures masked elements don't affect sum
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        c += BLOCK_C

    C = hidden_size
    mean = sum_val / C
    var = sum_sq / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    c = 0
    while c < hidden_size:
        offs = c + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        # store as bf16
        tl.store(out_ptr + row * hidden_size + offs, y.to(tl.bfloat16), mask=mask)
        c += BLOCK_C


@triton.jit
def spatial_shuffle_to_X_fc1_kernel(
    ln_out_ptr,           # *bf16, [num_patches, hidden_size]
    grid_thw_ptr,         # *int64, [num_grids, 3]
    X_fc1_ptr,            # *bf16, [num_merged_patches, hidden_size_expanded]
    num_patches: tl.int32,
    num_grids: tl.int32,
    hidden_size: tl.int32,
    hidden_size_expanded: tl.int32,
    BLOCK_P: tl.constexpr,  # number of patches handled per program in a loop
):
    # One program per grid
    grid_id = tl.program_id(0)
    if grid_id >= num_grids:
        return

    # Read t, h, w for this grid
    t = tl.load(grid_thw_ptr + grid_id * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + grid_id * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + grid_id * 3 + 2).to(tl.int32)

    # num patches in this grid
    num_patches_grid = t * h * w

    # Iterate over patches in this grid
    p = 0
    while p < num_patches_grid:
        # Map original 1D patch index to 2D (i, j)
        i0 = p // w
        j0 = p % w

        # Merge spatially: 2x2 -> 1
        i_merged = i0 // 2
        j_merged = j0 // 2

        # Compute original row index in LayerNorm output for this grid
        base_row = grid_id * (t * h * w) + p

        # For each feature c in [0, hidden_size_expanded)
        c = 0
        while c < hidden_size_expanded:
            # Linear index in LayerNorm output: (row * hidden_size + c)
            src_idx = base_row * hidden_size + c

            # Destination grid's merged patches count
            num_merged_patches_grid = t * (h // 2) * (w // 2)

            # Destination row index: grid_id * merged_patches + (i_merged * (w//2) + j_merged)
            dest_row = grid_id * num_merged_patches_grid + (i_merged * (w // 2) + j_merged)

            # Linear index in X_fc1: (dest_row * hidden_size_expanded + c)
            dst_idx = dest_row * hidden_size_expanded + c

            # Load from ln_out_ptr and store to X_fc1_ptr
            val = tl.load(ln_out_ptr + src_idx)
            tl.store(X_fc1_ptr + dst_idx, val.to(tl.bfloat16))
            c += 1
        p += 1


@triton.jit
def matmul_bias_kernel(
    A_ptr,                # *bf16, [M, K]
    B_ptr,                # *bf16, [K, N] (weight)
    Bias_ptr,             # *bf16, [N]
    Out_ptr,              # *bf16, [M, N]
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_om: tl.int32, stride_on: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0).to(tl.float32)

        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def gelu_tanh_kernel(
    x_ptr,  # *bf16, [M, N] flattened
    y_ptr,  # *bf16, [M, N] flattened
    M: tl.int32, N: tl.int32,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total = M * N
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < total

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    x3 = x * x * x
    tanh_arg = 0.7978845608028654 * (x + 0.044715 * x3)
    t = tl.tanh(tanh_arg)
    y = 0.5 * x * (1.0 + t)
    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)


# Define ModelNew with Triton kernels launched in forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args order must match original signature: (hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps)
        # We will create inputs dynamically to match the original get_inputs behavior, but we must not use torch ops in forward beyond data movement.

        # Extract shapes
        hidden = args[0]
        grid_thw = args[1]
        ln_weight = args[2]
        ln_bias = args[3]
        fc1_weight = args[4]
        fc1_bias = args[5]
        fc2_weight = args[6]
        fc2_bias = args[7]
        eps = args[8]

        device = hidden.device

        # Ensure tensors are contiguous and bf16
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        # These constants are provided in original setup
        hidden_size_expanded = 6144
        out_hidden_size = 3584

        # 1) Triton LayerNorm (pre-shuffle) with affine
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)

        BLOCK_C = 128  # feature block
        grid_ln = (num_patches,)
        layernorm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out, num_patches, hidden_size, eps,
            BLOCK_C=BLOCK_C, num_warps=4, num_stages=2,
        )

        # 2) Triton spatial shuffle to produce X_fc1 (no torch)
        # Allocate X_fc1
        num_merged_patches = args[1].shape[0] * (args[1].shape[1] // 2) * (args[1].shape[2] // 2)
        X_fc1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=device)
        num_grids = args[1].shape[0]

        # Launch spatial_shuffle kernel
        spatial_shuffle_to_X_fc1_kernel[(num_grids,)](
            ln_out, args[1], X_fc1, num_patches, num_grids, hidden_size, hidden_size_expanded,
            BLOCK_P=1,  # handle one patch at a time; while loops cover all
            num_warps=4, num_stages=2,
        )

        # 3) Triton Linear 1: X_fc1 @ fc1_weight.T + fc1_bias
        G = torch.empty((X_fc1.shape[0], fc1_weight.shape[1]), dtype=torch.bfloat16, device=device)
        # We need K for GEMM: fc1_weight is [hidden_size_expanded, hidden_size_expanded] => K=hidden_size_expanded, N=hidden_size_expanded
        M = X_fc1.shape[0]  # num_merged_patches
        K = hidden_size_expanded
        N = hidden_size_expanded

        # Strides for A: X_fc1 [M, K]
        stride_am, stride_ak = M, 1
        # Strides for B: fc1_weight [K, N]
        stride_bk, stride_bn = K, 1
        # Strides for Out: G [M, N]
        stride_om, stride_on = M, 1

        # Choose block sizes to cover typical dims; small matrices in given workloads
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        matmul_bias_kernel[(grid_m, grid_n)](
            X_fc1, fc1_weight, fc1_bias, G, M, K, N,
            stride_am, stride_ak, stride_bk, stride_bn,
            stride_om, stride_on,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 4) Triton GELU (tanh approximation)
        G_flat = G.reshape(-1)
        y_flat = torch.empty_like(G_flat, dtype=torch.bfloat16, device=device)
        total = G.numel()
        BLOCK_G = 1024
        grid_g = (triton.cdiv(total, BLOCK_G),)
        gelu_tanh_kernel[grid_g](G_flat, y_flat, total, BLOCK=BLOCK_G, num_warps=4, num_stages=2)
        G = y_flat.reshape(G.shape)

        # 5) Triton Linear 2: G @ fc2_weight.T + fc2_bias
        Out = torch.empty((G.shape[0], fc2_weight.shape[1]), dtype=torch.bfloat16, device=device)
        M2, K2, N2 = G.shape[0], G.shape[1], fc2_weight.shape[1]  # N2 = out_hidden_size = 3584

        stride_am2, stride_ak2 = M2, 1
        stride_bk2, stride_bn2 = K2, 1
        stride_om2, stride_on2 = M2, 1

        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_m2 = (M2 + BLOCK_M2 - 1) // BLOCK_M2
        grid_n2 = (N2 + BLOCK_N2 - 1) // BLOCK_N2

        matmul_bias_kernel[(grid_m2, grid_n2)](
            G, fc2_weight, fc2_bias, Out, M2, K2, N2,
            stride_am2, stride_ak2, stride_bk2, stride_bn2,
            stride_om2, stride_on2,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2,
        )

        return Out


def run(*args):
    return ModelNew()(*args)
