import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,       # *bf16, [num_patches, hidden_size]
    ln_weight_ptr,    # *bf16, [hidden_size]
    ln_bias_ptr,      # *bf16, [hidden_size]
    out_ptr,          # *bf16, [num_patches, hidden_size]
    num_patches: tl.constexpr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r = tl.program_id(0)  # row id
    # Accumulate sum and sum of squares in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, hidden_size, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        mask = cols < hidden_size
        vals = tl.load(hidden_ptr + r * hidden_size + cols, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        # Sum and sum of squares
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val / hidden_size
    var = sum_sq / hidden_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and affine, write back
    for c0 in range(0, hidden_size, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        mask = cols < hidden_size
        vals = tl.load(hidden_ptr + r * hidden_size + cols, mask=mask, other=0.0).to(tl.float32)
        ln_w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        ln_b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (vals - mean) * inv_std
        y = y * ln_w + ln_b
        # Cast back to bf16 and store
        y = y.to(tl.bfloat16)
        tl.store(out_ptr + r * hidden_size + cols, y, mask=mask)


@triton.jit
def spatial_shuffle_to_fc1_kernel(
    ln_out_ptr,        # *bf16, [num_patches, hidden_size]
    X_ptr,             # *bf16, [num_merged_patches, hidden_size_expanded] (output buffer)
    grid_thw_ptr,      # *int64, [num_grids, 3]
    num_patches: tl.constexpr,
    hidden_size: tl.constexpr,
    num_merged_patches: tl.constexpr,
    hidden_size_expanded: tl.constexpr,
    num_grids: tl.constexpr,
):
    grid_id = tl.program_id(0)
    # Read T, H, W for this grid
    t = tl.load(grid_thw_ptr + grid_id * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + grid_id * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + grid_id * 3 + 2).to(tl.int32)
    h_merged = h // 2
    w_merged = w // 2

    # offset: total patches contributed by previous grids
    # Compute total = sum of previous grids' patches
    total = 0
    for g in range(0, num_grids):
        if g == grid_id:
            break
        total += tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32) * tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32) * tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    # Iterate over all original patches in this grid
    # For each original (i, j) in 2x2 tile
    for p in range(0, t * h * w):
        # src_row in original LayerNorm output
        src_row = p  # hidden_out is [num_patches, hidden_size], and patches are contiguous rows
        # Map to original coordinates within this grid
        i = p // w
        j = p % w

        # Compute merged coordinates
        i_merged = i // 2
        j_merged = j // 2
        row_in_merged = i_merged * w_merged + j_merged

        # Feature loop
        for c0 in range(0, hidden_size_expanded, 8):
            cols = c0 + tl.arange(0, 8)
            mask = cols < hidden_size_expanded
            vals = tl.load(ln_out_ptr + src_row * hidden_size + cols, mask=mask, other=0.0).to(tl.bfloat16)
            # Write to global X at offset + row_in_merged
            global_row = total + row_in_merged * hidden_size_expanded + src_row * hidden_size_expanded
            tl.store(X_ptr + global_row + cols, vals, mask=mask)


@triton.jit
def matmul_bias_kernel(
    A_ptr,             # *bf16, [M, K]
    B_ptr,             # *bf16, [K, N]  (note: actual storage is [N, K], we index as [K, N] logically)
    bias_ptr,          # *bf16, [N]
    C_ptr,             # *bf16, [M, N]  (output)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        off_k = k0
        a_ptrs = A_ptr + off_m[:, None] * K + off_k[None, :]
        b_ptrs = B_ptr + off_n[None, :] * K + off_k  # B is [K, N] so we index as row=off_k, col=off_n

        a = tl.load(a_ptrs, mask=(off_m[:, None] < M) & (off_k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(off_n[None, :] < N) & (off_k[None, :] < K), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + off_n, mask=off_n < N, other=0.0).to(tl.float32)  # [BLOCK_N]
    acc += bias[None, :]

    # Write back (cast to bf16)
    c_ptrs = C_ptr + off_m[:, None] * N + off_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=(off_m[:, None] < M) & (off_n[None, :] < N))


@triton.jit
def gelu_tanh_kernel(
    X_ptr,             # *bf16, [M, N]
    Y_ptr,             # *bf16, [M, N] (output)
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    for m0 in range(0, M, BLOCK_M):
        for n0 in range(0, N, BLOCK_N):
            rows = m0 + tl.arange(0, BLOCK_M)
            cols = n0 + tl.arange(0, BLOCK_N)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            x = tl.load(X_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(tl.float32)
            # tanh-based GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
            x3 = x * x * x
            c = 0.7978845608028654  # sqrt(2/pi)
            inner = c * (x + 0.044715 * x3)
            t = tl.tanh(inner)
            y = 0.5 * x * (1.0 + t)
            tl.store(Y_ptr + rows[:, None] * N + cols[None, :], y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        # Ensure bf16 dtype
        device = hidden.device
        dtype = torch.bfloat16

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        # Create output buffer for LayerNorm
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)

        # Launch LayerNorm Triton kernel: one program per row
        grid_layernorm = (num_patches,)
        layernorm_affine_kernel[grid_layernorm](
            hidden, ln_weight, ln_bias, hidden_norm,
            num_patches, hidden_size, eps,
            BLOCK_C=64,
            num_warps=4,
        )

        # Allocate first linear input X [num_merged_patches, hidden_size_expanded] (we'll fill via Triton reorder)
        # Note: We need to compute num_merged_patches from grid_thw: sum over grids of t*h*w.
        num_merged_patches = 0
        for g in range(grid_thw.shape[0]):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            num_merged_patches += t * h * (h // 2) * (w // 2)
        hidden_size_expanded = fc1_weight.shape[0]  # 6144
        out_fc1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=device)

        # Launch spatial shuffle to fill first linear input
        grid_reorder = (grid_thw.shape[0],)
        spatial_shuffle_to_fc1_kernel[grid_reorder](
            hidden_norm, out_fc1, grid_thw,
            num_patches, hidden_size, num_merged_patches, hidden_size_expanded, grid_thw.shape[0],
            num_warps=4,
        )

        # First Linear: GEMM + bias using Triton
        # We pass fc1_weight as [hidden_size_expanded, hidden_size_expanded], bias as fc1_bias
        M_fc1 = out_fc1.shape[0]
        N_fc1 = out_fc1.shape[1]
        K_fc1 = N_fc1  # since fc1_weight is [N_fc1, N_fc1]
        BLOCK_M_fc1 = 64
        BLOCK_N_fc1 = 128
        BLOCK_K_fc1 = 32
        grid_fc1 = (triton.cdiv(M_fc1, BLOCK_M_fc1), triton.cdiv(N_fc1, BLOCK_N_fc1))
        matmul_bias_kernel[grid_fc1](
            out_fc1, fc1_weight, fc1_bias,
            out_fc1,  # output overwrites input for first linear
            M_fc1, N_fc1, K_fc1,
            BLOCK_M_fc1, BLOCK_N_fc1, BLOCK_K_fc1,
            num_warps=4,
        )

        # GELU activation Triton
        M_gelu = M_fc1
        N_gelu = N_fc1
        BLOCK_M_gelu = 64
        BLOCK_N_gelu = 128
        Y_fc1 = torch.empty((M_fc1, N_fc1), dtype=torch.bfloat16, device=device)
        gelu_tanh_kernel[(triton.cdiv(M_fc1, BLOCK_M_gelu), triton.cdiv(N_fc1, BLOCK_N_gelu))](
            out_fc1, Y_fc1,
            M_fc1, N_fc1,
            BLOCK_M_gelu, BLOCK_N_gelu,
            num_warps=4,
        )

        # Second Linear: GEMM + bias using Triton
        # We pass fc2_weight as [out_hidden_size, hidden_size_expanded]; B_ptr will index as [hidden_size_expanded, out_hidden_size]
        out_hidden_size = fc2_weight.shape[0]  # 3584
        M_fc2 = Y_fc1.shape[0]
        N_fc2 = out_hidden_size
        K_fc2 = Y_fc1.shape[1]  # hidden_size_expanded
        BLOCK_M_fc2 = 64
        BLOCK_N_fc2 = 128
        BLOCK_K_fc2 = 32
        grid_fc2 = (triton.cdiv(M_fc2, BLOCK_M_fc2), triton.cdiv(N_fc2, BLOCK_N_fc2))
        matmul_bias_kernel[grid_fc2](
            Y_fc1, fc2_weight, fc2_bias,
            Y_fc1,  # output
            M_fc2, N_fc2, K_fc2,
            BLOCK_M_fc2, BLOCK_N_fc2, BLOCK_K_fc2,
            num_warps=4,
        )

        return Y_fc1


def run(*args):
    return ModelNew()(*args)
