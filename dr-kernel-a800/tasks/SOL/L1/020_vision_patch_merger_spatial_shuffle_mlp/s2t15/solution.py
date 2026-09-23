import math
import torch
import triton
import triton.language as tl

# --------- Triton kernels ----------

@triton.jit
def layernorm_affine_rowwise_kernel(
    x_ptr,                # *bf16, [N, H]
    y_ptr,                # *bf16, [N, H] output
    ln_weight_ptr,        # *bf16, [H]
    ln_bias_ptr,          # *bf16, [H]
    N, H,                 # int32
    eps,                  # float32
    BLOCK_H: tl.constexpr
):
    row_id = tl.program_id(axis=0)  # 0..N-1
    # Pointer to start of this row
    row_x = x_ptr + row_id * H
    row_y = y_ptr + row_id * H

    # Accumulate sum and sum of squares in fp32
    sum_ = tl.zeros((), dtype=tl.float32)
    sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for k in range(0, H, BLOCK_H):
        cols = k + tl.arange(0, BLOCK_H)
        mask = cols < H
        x = tl.load(row_x + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sum_ / H
    var = sumsq / H - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for k in range(0, H, BLOCK_H):
        cols = k + tl.arange(0, BLOCK_H)
        mask = cols < H
        x = tl.load(row_x + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        # Cast back to bfloat16
        y = y.to(tl.bfloat16)
        tl.store(row_y + cols, y, mask=mask)


@triton.jit
def spatial_shuffle_kernel(
    norm_ptr,          # *bf16, [num_patches, hidden_size]
    out_ptr,           # *bf16, [num_merged_patches, hidden_size_expanded]
    grid_thw_ptr,      # *int64, [num_grids, 3]
    offsets_ptr,       # *int64, [num_grids]
    N_MERGED, H_EXP,   # int32
    NUM_GRIDS: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # 2D grid: axis 0 over rows (merged patches), axis 1 over columns (expanded hidden)
    r = tl.program_id(axis=0)  # merged patch row index
    j = tl.program_id(axis=1)  # expanded hidden column index

    # Decode j into (merge_h, merge_w, c)
    # hidden_size_expanded = (2*2) * hidden_size = 4 * 1536 = 6144
    C = 1536
    merge2 = 2
    # j = merge_h * (C * merge2) + merge_w * C + c
    merge_h = j // (C * merge2)
    rem = j % (C * merge2)
    merge_w = rem // C
    c = rem % C

    # For each grid, compute if this j maps to this grid
    # total_per_grid[i] = T[i] * H[i] * W[i]
    # We need grid index g for which r falls in [offsets[g], offsets[g] + total[g])
    found = False
    for i in range(NUM_GRIDS):
        total_i = tl.load(grid_thw_ptr + i * 3 + 0) * tl.load(grid_thw_ptr + i * 3 + 1) * tl.load(grid_thw_ptr + i * 3 + 2)
        off_i = tl.load(offsets_ptr + i)
        if (r >= off_i) & (r < off_i + total_i):
            found = True
            break

    if not found:
        # If r does not fall into any grid (defensive), write zeros
        zero = tl.zeros((), dtype=tl.bfloat16)
        tl.store(out_ptr + r * H_EXP + j, zero)
        return

    # Now we have grid i. Decode T, H, W from grid_thw_ptr[i,:]
    T_i = tl.load(grid_thw_ptr + i * 3 + 0)
    H_i = tl.load(grid_thw_ptr + i * 3 + 1)
    W_i = tl.load(grid_thw_ptr + i * 3 + 2)

    # Compute base offset in normalized hidden for this grid
    # Each grid contributes t * h * w rows. We need which group (t), and within that (h,w).
    per_group = H_i * W_i
    group_id = (r - off_i) // per_group
    within = (r - off_i) % per_group
    h_idx = within // W_i
    w_idx = within % W_i
    t_idx = group_id  # since per_group == H_i * W_i, group_id is t

    # Final input row index = t_idx * (H_i * W_i) + h_idx * W_i + w_idx
    in_row = t_idx * (H_i * W_i) + h_idx * W_i + w_idx

    # Finally, map to normalized hidden row and column c
    in_row_idx = in_row * hidden_size + c
    val = tl.load(norm_ptr + in_row_idx).to(tl.float32)  # load as fp32
    zero = tl.zeros((), dtype=tl.bfloat16)
    tl.store(out_ptr + r * H_EXP + j, val.to(tl.bfloat16), mask=(j < H_EXP))


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    A_stride0, A_stride1,
    B_stride0, B_stride1,
    C_stride0, C_stride1,
    eps,  # not used, placeholder
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    m0 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n0 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + m0[:, None] * A_stride0 + k[None, :] * A_stride1
        b_ptrs = B_ptr + k[:, None] * B_stride0 + n0[None, :] * B_stride1
        a = tl.load(a_ptrs, mask=(m0[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (n0[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + n0, mask=(n0 < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + m0[:, None] * C_stride0 + n0[None, :] * C_stride1
    tl.store(c_ptrs, acc.to(tl.float32), mask=(m0[:, None] < M) & (n0[None, :] < N))


@triton.jit
def gelu_kernel(
    x_ptr, y_ptr, M, N,
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(axis=0)  # iterate over rows
    pid_n = tl.program_id(axis=1)  # iterate over columns in tiles
    m = pid_m
    n0 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n0 < N

    x = tl.load(x_ptr + m * N + n0, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    # Constants
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(y_ptr + m * N + n0, y, mask=mask)


# --------- ModelNew (forward) ----------

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        args: (hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps)
              hidden: [num_patches, hidden_size], bfloat16, CUDA
              grid_thw: [num_grids, 3], int64 (T, H, W per grid), CUDA
              ln_weight, ln_bias: [hidden_size], bfloat16, CUDA
              fc1_weight: [hidden_size_expanded, hidden_size_expanded], bfloat16, CUDA
              fc1_bias: [hidden_size_expanded], bfloat16, CUDA
              fc2_weight: [out_hidden_size, hidden_size_expanded], bfloat16, CUDA
              fc2_bias: [out_hidden_size], bfloat16, CUDA
              eps: float
        Returns: output [num_merged_patches, out_hidden_size], bfloat16
        """
        # Extract inputs (don't create torch tensors in forward)
        hidden = args[0]
        grid_thw = args[1]
        ln_weight = args[2]
        ln_bias = args[3]
        fc1_weight = args[4]
        fc1_bias = args[5]
        fc2_weight = args[6]
        fc2_bias = args[7]
        eps = args[8]

        # Dimensions
        hidden_size = 1536
        hidden_expanded = hidden_size * 4  # 6144
        num_patches = hidden.shape[0]
        num_merged_patches = args[1][0, 0].numel()  # num_grids * per_grid counts
        # We can derive num_merged_patches from args more robustly:
        # Since grid_thw is [num_grids, 3], and we construct out based on T*H*W per grid,
        # num_merged_patches is sum(grid_thw[:, 0]*grid_thw[:, 1]*grid_thw[:, 2]).
        num_merged_patches = int(((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum()).item())
        out_hidden_size = fc2_bias.shape[0]

        # 1) LayerNorm + affine (normalized_hidden) in Triton
        normalized_hidden = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)

        # Launch per-row LN kernel
        # Grid: (num_patches,)
        BLOCK_H = 1024
        grid_layernorm = (num_patches,)
        layernorm_affine_rowwise_kernel[grid_layernorm](
            hidden, normalized_hidden,
            ln_weight, ln_bias,
            num_patches, hidden_size,
            eps,
            BLOCK_H=BLOCK_H,
            num_warps=4
        )

        # 2) Spatial shuffle (2x2 merge) in Triton
        # Compute total_per_grid and offsets on host using provided grid_thw
        total_per_grid = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.int64).cpu().tolist()
        offsets = []
        cumulative = 0
        for t in total_per_grid:
            offsets.append(cumulative)
            cumulative += t
        offsets = torch.tensor(offsets, dtype=torch.int64, device=hidden.device)

        hidden_shuffled = torch.empty((num_merged_patches, hidden_expanded), dtype=torch.bfloat16, device=hidden.device)

        # Launch 2D shuffle kernel
        BLOCK_N = 128
        grid_shuffle = (num_merged_patches, hidden_expanded)
        spatial_shuffle_kernel[grid_shuffle](
            normalized_hidden, hidden_shuffled,
            grid_thw, offsets,
            num_merged_patches, hidden_expanded,
            NUM_GRIDS=grid_thw.shape[0],
            BLOCK_N=BLOCK_N,
            num_warps=4
        )

        # 3) FC1: hidden_shuffled [M, K] @ fc1_weight [K, K] (+ fc1_bias)
        M = hidden_shuffled.shape[0]  # num_merged_patches
        K = fc1_weight.shape[1]  # 6144
        N = fc1_weight.shape[0]  # 6144

        # Output buffer for fc1
        fc1_out = torch.empty((M, K), dtype=torch.float32, device=hidden.device)

        # Set strides (row-major)
        A = hidden_shuffled
        B = fc1_weight
        C = fc1_out

        grid_fc1 = (triton.cdiv(M, 64), triton.cdiv(K, 64))
        gemm_kernel[grid_fc1](
            A, B, fc1_bias, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            eps,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4
        )

        # 4) GELU activation in Triton
        fc1_gelu = torch.empty_like(fc1_out, dtype=torch.float32, device=hidden.device)
        BLOCK_N_GELU = 256
        grid_gelu = (M, triton.cdiv(K, BLOCK_N_GELU))
        gelu_kernel[grid_gelu](
            fc1_out, fc1_gelu,
            M, K,
            BLOCK_N=BLOCK_N_GELU,
            num_warps=4
        )

        # 5) FC2: [M, K] @ fc2_weight [out_hidden_size, K] (+ fc2_bias)
        out_hidden_size = fc2_bias.shape[0]
        output = torch.empty((M, out_hidden_size), dtype=torch.float32, device=hidden.device)

        A2 = fc1_gelu
        B2 = fc2_weight
        C2 = output

        grid_fc2 = (triton.cdiv(M, 64), triton.cdiv(out_hidden_size, 64))
        gemm_kernel[grid_fc2](
            A2, B2, fc2_bias, C2,
            M, out_hidden_size, K,
            A2.stride(0), A2.stride(1),
            B2.stride(0), B2.stride(1),
            C2.stride(0), C2.stride(1),
            eps,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4
        )

        # Return in bfloat16 (as original uses bfloat16 inputs and outputs)
        return output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
