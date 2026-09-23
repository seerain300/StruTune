import math
import torch
import triton
import triton.language as tl

# Triton LayerNorm: one program per row. Two passes: reduce -> normalize -> apply affine.
@triton.jit
def _layernorm_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                      N, C, eps,
                      BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C]
    ln_weight_ptr, ln_bias_ptr: *bf16, shape [C]
    eps: float32
    """
    row = tl.program_id(0)
    if row >= N:
        return
    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    # Pass 1: mean and variance in fp32
    sum_val = 0.0
    sum_sq = 0.0
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK_SIZE
    mean = sum_val / C
    var = sum_sq / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize, affine, store
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: spatial 2x2 shuffle per grid.
# Input x is the normalized hidden per grid: shape [T * H * W, C] (we pass it as flattened).
# Output out_grid is [T * (H//2) * (W//2), 4*C], contiguous per row, columns ordered as:
# col = (h_merged*2 + rh) * (W//2)*C + (w_merged*2 + rw) * C + cdim
@triton.jit
def _shuffle_2x2_grid_kernel(x_ptr, grid_thw_ptr, out_grid_ptr,
                             T, H, W, C,
                             BLOCK_M: tl.constexpr):
    """
    x_ptr: *bf16, flattened [T*H*W, C]
    grid_thw_ptr: *int64, shape [1, 3] but we can generalize; here only one program used per grid.
    out_grid_ptr: *bf16, flattened [T*(H//2)*(W//2), 4*C]
    Launch one program per grid via grid=(NUM_GRIDS,)
    """
    # This kernel is per-grid. grid_thw_ptr should be provided per grid (we'll pass NUM_GRIDS program and reuse input shapes).
    # However, to keep simple, we assume the host launches this with grid=(NUM_GRIDS,) and computes T/H/W from arguments.
    g = tl.program_id(0)
    if g >= 1:
        return  # single program per grid, host controls grid size

    # Recompute T/H/W for this grid from arguments
    Tg = T
    Hg = H
    Wg = W

    T = Tg
    H = Hg
    W = Wg

    h_merged = H // 2
    w_merged = W // 2
    num_merged_rows = T * h_merged * w_merged

    # For each original patch m in [0, T*H*W):
    # t_index = m // (H*W), rem = m % (H*W)
    # h2 = rem // W, w2 = rem % W
    # out_row = t_index * (h_merged * w_merged) + (h2//2) * w_merged + (w2//2)
    m = 0
    while m < T * H * W:
        t_index = m // (H * W)
        rem = m % (H * W)
        h2 = rem // W
        w2 = rem % W

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_grid_ptr + out_row * (4 * C)

        # idx=0: (r=0,c=0)
        idx = 0
        col0 = (h2 * 2 + 0) * (w_merged * C) + (w2 * 2 + 0) * C
        src0 = t_index * (H * W) + h2 * W + w2
        val0 = tl.load(x_ptr + src0 * C + 0, mask=(h2 < H) & (w2 < W), other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val0)

        # idx=1: (r=0,c=1)
        idx = 1
        col1 = (h2 * 2 + 0) * (w_merged * C) + (w2 * 2 + 1) * C
        src1 = t_index * (H * W) + h2 * W + (w2 + 1)
        val1 = tl.load(x_ptr + src1 * C + 0, mask=(h2 < H) & (w2 + 1 < W), other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val1)

        # idx=2: (r=1,c=0)
        idx = 2
        col2 = (h2 * 2 + 1) * (w_merged * C) + (w2 * 2 + 0) * C
        src2 = t_index * (H * W) + (h2 + 1) * W + w2
        val2 = tl.load(x_ptr + src2 * C + 0, mask=((h2 + 1) < H) & (w2 < W), other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val2)

        # idx=3: (r=1,c=1)
        idx = 3
        col3 = (h2 * 2 + 1) * (w_merged * C) + (w2 * 2 + 1) * C
        src3 = t_index * (H * W) + (h2 + 1) * W + (w2 + 1)
        val3 = tl.load(x_ptr + src3 * C + 0, mask=((h2 + 1) < H) & (w2 + 1 < W), other=0.0).to(tl.bfloat16)
        tl.store(base_out + idx * C, val3)

        m += 1


# Triton kernel: Linear (row-wise) y_row[i_out] = sum_k x_row[i_out, k] * W[k, i_out] + bias[i_out]
# We implement two specializations: K=6144, K_out=6144 and K=6144, K_out=3584.
@triton.jit
def _linear_rowwise_6144_6144(x_ptr, W_ptr, b_ptr, y_ptr,
                              N, C, BLOCK_K: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    W_ptr: *bf16, shape [C, C], row-major (i.e., W[k, j] contiguous for fixed k across j)
    b_ptr: *bf16, shape [C]
    y_ptr: *bf16, shape [N, C]
    Each program computes one row i_out in [0, N).
    """
    i_out = tl.program_id(0)
    if i_out >= N:
        return

    y_row_ptr = y_ptr + i_out * C

    # Iterate k over C in blocks
    k_start = 0
    while k_start < C:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < C

        # Load x_row[i_out, offs_k]
        x_vec = tl.load(x_ptr + i_out * C + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        # Load W[k, i_out] for k in offs_k -> W is [C, C], row-major so stride along j is 1
        W_vec = tl.load(W_ptr + offs_k * C + i_out, mask=mask_k, other=0.0).to(tl.float32)

        # Accumulate dot
        acc = tl.sum(x_vec * W_vec, axis=0)

        # Add bias
        b = tl.load(b_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        acc = acc + b

        # Store
        tl.store(y_row_ptr + offs_k, acc.to(tl.bfloat16), mask=mask_k)

        k_start += BLOCK_K


@triton.jit
def _linear_rowwise_6144_3584(x_ptr, W_ptr, b_ptr, y_ptr,
                              N, C_OUT, BLOCK_K: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    W_ptr: *bf16, shape [C, C_OUT], row-major (i.e., W[k, j] contiguous for fixed k across j)
    b_ptr: *bf16, shape [C_OUT]
    y_ptr: *bf16, shape [N, C_OUT]
    Each program computes one row i_out in [0, N).
    """
    i_out = tl.program_id(0)
    if i_out >= N:
        return

    y_row_ptr = y_ptr + i_out * C_OUT

    k_start = 0
    while k_start < C:
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < C

        x_vec = tl.load(x_ptr + i_out * C + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        W_vec = tl.load(W_ptr + offs_k * C_OUT + i_out, mask=mask_k, other=0.0).to(tl.float32)

        acc = tl.sum(x_vec * W_vec, axis=0)

        b = tl.load(b_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        acc = acc + b

        tl.store(y_row_ptr + offs_k, acc.to(tl.bfloat16), mask=mask_k)

        k_start += BLOCK_K


# Triton kernel: GELU elementwise on [N, C] row-major, one program per row.
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, y_ptr: *bf16, shapes [N, C]
    """
    row = tl.program_id(0)
    if row >= N:
        return
    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        z = x * inv_sqrt2
        # Use tanh-based GELU approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3)))
        k = 0.7978845608028654  # sqrt(2/pi)
        z3 = z * z * z
        t = k * (z + 0.044715 * z3)
        gelu = 0.5 * x * (1.0 + tl.tanh(t))
        tl.store(y_row_ptr + offs, gelu.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters here; everything is computed via Triton kernels

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
        hidden: [num_patches, hidden_size], bfloat16
        grid_thw: [num_grids, 3] int64, each row [t, h, w]
        ln_weight, ln_bias: [hidden_size], bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded], bfloat16
        fc1_bias: [hidden_size_expanded], bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded], bfloat16
        fc2_bias: [out_hidden_size], bfloat16
        eps: float
        Returns output of shape [num_merged_patches, hidden_size_expanded], bfloat16.
        """
        assert hidden.is_cuda and grid_thw.is_cuda, "Triton requires CUDA tensors"
        device = hidden.device
        N = hidden.shape[0]  # num_patches
        C = hidden.shape[1]  # hidden_size = 1536
        HES = fc1_weight.shape[0]  # hidden_size_expanded = 6144
        assert fc1_weight.shape[1] == C, "fc1_weight must be [HES, C]"
        assert fc2_weight.shape[1] == HES, "fc2_weight second dim must be HES"
        assert fc2_weight.shape[0] == 3584, "Expected out_hidden_size = 3584"

        # 1) LayerNorm per row: hidden_norm
        hidden_norm = torch.empty_like(hidden, device=device, dtype=torch.bfloat16)
        # Triton kernel launch: one program per row
        grid = (N,)
        # Choose BLOCK_SIZE as 1024 (C=1536, so two passes)
        _layernorm_kernel[grid](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, C, eps,
            BLOCK_SIZE=1024,
            num_warps=4
        )

        # 2) Spatial shuffle per grid: produce per-grid output and concatenate
        # We'll pre-allocate per-grid outputs in a list of tensors, then concatenate.
        grids = []
        NUM_GRIDS = grid_thw.shape[0]
        # For each grid, compute t,h,w and launch the shuffle kernel to produce [T*(H//2)*(W//2), 4*C]
        for g in range(NUM_GRIDS):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())

            Tg = t
            Hg = h
            Wg = w

            # Number of rows in output for this grid
            num_merged_rows = Tg * (Hg // 2) * (Wg // 2)
            out_grid = torch.empty((num_merged_rows, 4 * C), device=device, dtype=torch.bfloat16)

            # Launch shuffle kernel: grid=(1,) because one program per grid in host loop
            _shuffle_2x2_grid_kernel[(1,)](
                hidden_norm, grid_thw, out_grid,
                Tg, Hg, Wg, C,
                BLOCK_M=1024,
                num_warps=4
            )

            grids.append(out_grid)

        # Concatenate per-grid outputs to form hidden_shuffled
        hidden_shuffled = torch.cat([g for g in grids], dim=0)

        # 3) MLP: Linear1 -> GELU -> Linear2, all via Triton kernels
        N2 = hidden_shuffled.shape[0]  # num_merged_patches
        HES1 = fc1_weight.shape[0]     # 6144
        C_OUT = fc2_weight.shape[0]    # 3584

        # Linear1: y1 = hidden_shuffled @ fc1_weight.T + fc1_bias
        y1 = torch.empty((N2, HES1), device=device, dtype=torch.bfloat16)
        _linear_rowwise_6144_6144[(N2,)](
            hidden_shuffled, fc1_weight, fc1_bias, y1,
            N2, HES1,
            BLOCK_K=1024,
            num_warps=4
        )

        # GELU
        y1_gelu = torch.empty_like(y1, device=device, dtype=torch.bfloat16)
        _gelu_kernel[(N2,)](
            y1, y1_gelu, N2, HES1,
            BLOCK_SIZE=1024,
            num_warps=4
        )

        # Linear2: y = y1_gelu @ fc2_weight.T + fc2_bias
        y = torch.empty((N2, C_OUT), device=device, dtype=torch.bfloat16)
        _linear_rowwise_6144_3584[(N2,)](
            y1_gelu, fc2_weight, fc2_bias, y,
            N2, C_OUT,
            BLOCK_K=1024,
            num_warps=4
        )

        return y


def run(*args):
    return ModelNew()(*args)
