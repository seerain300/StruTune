import torch
import math
import triton
import triton.language as tl

# Triton kernel: LayerNorm per row. One program per row.
@triton.jit
def _layer_norm_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                        N, C, eps,
                        BLOCK_SIZE: tl.constexpr):
    """
    x_ptr: *bf16, shape [N, C], row-major
    y_ptr: *bf16, shape [N, C], output
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

    # Pass 2: normalize and apply affine
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


# Triton kernel: Exact spatial shuffle from hidden_norm (shape [num_patches, C]) into
# output shaped [total_num_merged_patches, 4*C], where total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2).
@triton.jit
def _shuffle_2x2_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                        N, C, NUM_GRIDS,
                        BLOCK_ROWS: tl.constexpr):
    """
    hidden_ptr: *bf16, shape [N, C] row-major
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], where each row is [t, h, w]
    out_ptr: *bf16, shape [TOTAL_MERGED_ROWS, 4*C], row-major
    """
    # We will process all grids inside this kernel. Triton program id can be 0 only for this kernel.
    # But to make it general, we can rely on host to pass total merged rows; we compute TOTAL_MERGED_ROWS
    # via Python code and launch with proper grid. Here, we compute per program for each grid using
    # grid_thw_ptr and write into out_ptr. We use a simple loop across grids, but Triton doesn't support
    # Python for-loops over NUM_GRIDS; instead, we compute per program using tl.load on grid_thw_ptr.
    # However, to avoid indexing issues, we design this kernel as one program per grid, since NUM_GRIDS
    # is small in provided workloads. So we launch grid size = NUM_GRIDS.

    g = tl.program_id(0)  # program id equals grid index
    if g >= NUM_GRIDS:
        return

    # Load grid dimensions
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    # Compute h_merged and w_merged
    h_merged = h // 2
    w_merged = w // 2
    num_merged_rows = t * h_merged * w_merged

    # For each original patch m in [0, t*h*w):
    # Compute (t_index, h2, w2), then out_row = t_index * (h_merged * w_merged) + h2//2 * w_merged + w2//2
    # Copy hidden[row=m, cdim] into four columns of out[out_row, :]
    total_patches = t * h * w
    base = g * (t * h * w)  # each grid's base offset in hidden_ptr when flattened
    # We will iterate m in tiles for performance. Triton prefers constexpr loops; since total_patches
    # is dynamic, we implement a while loop.
    m = 0
    while m < total_patches:
        # decode m -> t_index, h2, w2
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        # Compute out base for this row
        out_base = out_ptr + out_row * (4 * C)

        # Copy four positions corresponding to the 2x2 merge
        # col mapping for original C dimension: for each cdim in [0, C), write to 4 columns:
        # idx=0: (r=0,c=0) -> hidden[t_index, h2, w2]
        # idx=1: (r=0,c=1) -> hidden[t_index, h2, w2+1]
        # idx=2: (r=1,c=0) -> hidden[t_index, h2+1, w2]
        # idx=3: (r=1,c=1) -> hidden[t_index, h2+1, w2+1]
        # We need to compute the original row index within hidden_ptr for the grid:
        # hidden_ptr is flattened: offset = base + t_index*(h*w) + h2*w + w2
        src_offset = base + t_index * (h * w) + h2 * w + w2
        # Load one element as a vector across C
        offs_c = tl.arange(0, BLOCK_ROWS)  # we'll iterate cdim=0..C-1 with BLOCK_ROWS as 1 vector for simplicity
        # Since we need per-cdim copies, we perform scalar stores by iterating cdim. Triton supports scalar loads/stores.
        # To keep it efficient, we compute addresses and load/store scalar per cdim. This is acceptable for C=1536.
        # We'll implement per-cdim in a while loop over cdim. Triton supports while loops for runtime counts.
        cdim = 0
        while cdim < C:
            # idx=0
            idx0 = 0
            col0 = (h2 * 2 + 0) * w_merged * C + (w2 * 2 + 0) * C + cdim
            val0 = tl.load(hidden_ptr + src_offset + cdim).to(tl.bfloat16)
            tl.store(out_base + idx0 * C + cdim, val0)

            # idx=1
            idx1 = 1
            col1 = (h2 * 2 + 0) * w_merged * C + (w2 * 2 + 1) * C + cdim
            if w2 + 1 < w:
                val1 = tl.load(hidden_ptr + src_offset + (w2 + 1) * C + cdim).to(tl.bfloat16) if w2 + 1 < w else tl.zeros((), dtype=tl.bfloat16)
            else:
                val1 = tl.load(hidden_ptr + src_offset + (w2 + 1) * C + cdim).to(tl.bfloat16)
            tl.store(out_base + idx1 * C + cdim, val1)

            # idx=2
            idx2 = 2
            col2 = (h2 * 2 + 1) * w_merged * C + (w2 * 2 + 0) * C + cdim
            if h2 + 1 < h:
                val2 = tl.load(hidden_ptr + src_offset + (h2 + 1) * W * C + cdim).to(tl.bfloat16)  # W is not in scope, compute properly
            # Fix: we need to compute proper row offset when reading next row. Use base + t_index*(h*w) + (h2+1)*w + w2
            row_offset = base + t_index * (h * w) + (h2 + 1) * w + w2
            val2 = tl.load(hidden_ptr + row_offset + cdim).to(tl.bfloat16)
            tl.store(out_base + idx2 * C + cdim, val2)

            # idx=3
            idx3 = 3
            col3 = (h2 * 2 + 1) * w_merged * C + (w2 * 2 + 1) * C + cdim
            if (h2 + 1) < h and (w2 + 1) < w:
                row_offset2 = base + t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
                val3 = tl.load(hidden_ptr + row_offset2 + cdim).to(tl.bfloat16)
            else:
                val3 = tl.zeros((), dtype=tl.bfloat16)
            tl.store(out_base + idx3 * C + cdim, val3)

            cdim += 1

        m += BLOCK_ROWS


# Triton kernel: GELU elementwise. One program per row.
@triton.jit
def _gelu_kernel(x_ptr, y_ptr, N, C, BLOCK_SIZE: tl.constexpr):
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
        # GELU approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        k = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        t = k * (x + 0.044715 * x3)
        y = 0.5 * x * (1.0 + tl.tanh(t))
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


# Triton kernel: Linear GEMM row-wise, output one row vector. Specialized for Linear1 (K=6144, OUT=6144).
@triton.jit
def _linear_row_gemm1(x_ptr, w_ptr, b_ptr, y_ptr,
                      N_OUT, C_IN, C_OUT, K,
                      BLOCK_K: tl.constexpr, BLOCK_CO: tl.constexpr):
    """
    x_ptr: *bf16, shape [N_OUT, C_IN]
    w_ptr: *bf16, shape [C_OUT, C_IN] (note: PyTorch weight is [C_OUT, C_IN], we index as w[co, ci])
    b_ptr: *bf16, shape [C_OUT]
    y_ptr: *bf16, shape [N_OUT, C_OUT]
    """
    row_id = tl.program_id(0)
    if row_id >= N_OUT:
        return
    x_row_ptr = x_ptr + row_id * C_IN
    y_row_ptr = y_ptr + row_id * C_OUT
    acc = tl.zeros([C_OUT], dtype=tl.float32)

    k = 0
    while k < K:
        k_offs = k + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K
        # load x[k_offs] vector
        x_vec = tl.load(x_row_ptr + k_offs, mask=mask_k, other=0.0).to(tl.float32)
        # load w[co, k_offs] tile of size [BLOCK_CO, BLOCK_K]
        co = 0
        w_tile = tl.zeros([BLOCK_CO, BLOCK_K], dtype=tl.float32)
        while co < BLOCK_CO:
            co_offs = co + tl.arange(0, BLOCK_CO)
            # Addressing: w_ptr indexed as w[co, k] -> w_ptr + co*stride_w_co + k*stride_w_ci
            # PyTorch strides: w.stride(0)=C_IN, w.stride(1)=1
            w_tile_sub = tl.load(w_ptr + co_offs[:, None] * C_IN + k_offs[None, :], mask=(co_offs[:, None] < C_OUT) & (mask_k[None, :]), other=0.0)
            w_tile = w_tile + w_tile_sub
            co += BLOCK_CO

        # accumulate dot: sum over K for each co
        dot_vec = tl.zeros([BLOCK_CO], dtype=tl.float32)
        kk = 0
        while kk < BLOCK_K:
            # sum w_tile[:, kk] * x_vec[kk] across kk
            # For each kk, select x_vec[kk] scalar and multiply each row of w_tile by it, then sum rows
            # Here, we implement the reduction manually over kk.
            mul_row = w_tile[:, kk] * x_vec[kk]
            dot_vec += mul_row
            kk += 1

        # add to acc
        acc += dot_vec
        k += BLOCK_K

    # add bias
    co = 0
    while co < C_OUT:
        co_offs = co + tl.arange(0, BLOCK_CO)
        mask_co = co_offs < C_OUT
        b_vec = tl.load(b_ptr + co_offs, mask=mask_co, other=0.0).to(tl.float32)
        acc += b_vec
        co += BLOCK_CO

    # store result
    co = 0
    while co < C_OUT:
        co_offs = co + tl.arange(0, BLOCK_CO)
        mask_co = co_offs < C_OUT
        tl.store(y_row_ptr + co_offs, acc[co_offs].to(tl.bfloat16), mask=mask_co)
        co += BLOCK_CO


# Triton kernel: Linear GEMM row-wise, output one row vector. Specialized for Linear2 (K=6144, OUT=3584).
@triton.jit
def _linear_row_gemm2(x_ptr, w_ptr, b_ptr, y_ptr,
                      N_OUT, C_IN, C_OUT, K,
                      BLOCK_K: tl.constexpr, BLOCK_CO: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= N_OUT:
        return
    x_row_ptr = x_ptr + row_id * C_IN
    y_row_ptr = y_ptr + row_id * C_OUT
    acc = tl.zeros([C_OUT], dtype=tl.float32)

    k = 0
    while k < K:
        k_offs = k + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K
        x_vec = tl.load(x_row_ptr + k_offs, mask=mask_k, other=0.0).to(tl.float32)

        co = 0
        w_tile = tl.zeros([BLOCK_CO, BLOCK_K], dtype=tl.float32)
        while co < BLOCK_CO:
            co_offs = co + tl.arange(0, BLOCK_CO)
            w_tile_sub = tl.load(w_ptr + co_offs[:, None] * C_IN + k_offs[None, :], mask=(co_offs[:, None] < C_OUT) & (mask_k[None, :]), other=0.0)
            w_tile = w_tile + w_tile_sub
            co += BLOCK_CO

        dot_vec = tl.zeros([BLOCK_CO], dtype=tl.float32)
        kk = 0
        while kk < BLOCK_K:
            mul_row = w_tile[:, kk] * x_vec[kk]
            dot_vec += mul_row
            kk += 1

        acc += dot_vec
        k += BLOCK_K

    co = 0
    while co < C_OUT:
        co_offs = co + tl.arange(0, BLOCK_CO)
        mask_co = co_offs < C_OUT
        b_vec = tl.load(b_ptr + co_offs, mask=mask_co, other=0.0).to(tl.float32)
        acc += b_vec
        co += BLOCK_CO

    co = 0
    while co < C_OUT:
        co_offs = co + tl.arange(0, BLOCK_CO)
        mask_co = co_offs < C_OUT
        tl.store(y_row_ptr + co_offs, acc[co_offs].to(tl.bfloat16), mask=mask_co)
        co += BLOCK_CO


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                eps: float):
        """
        hidden: [num_patches, hidden_size] (bf16)
        grid_thw: [num_grids, 3] int64, [t, h, w] per grid
        ln_weight, ln_bias: [hidden_size] (bf16)
        fc1_weight, fc1_bias: [6144, 6144] (bf16)
        fc2_weight, fc2_bias: [3584, 6144] (bf16)
        eps: float
        """
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda \
               and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."

        device = hidden.device
        hidden_size = hidden.shape[1]
        num_patches = hidden.shape[0]
        num_grids = grid_thw.shape[0]
        # Step 1: LayerNorm over last dim (hidden_size), per row
        hidden_norm = torch.empty_like(hidden, device=device, dtype=torch.bfloat16)
        # Cast ln_weight/bias to fp32 for compute
        ln_w = ln_weight.to(torch.float32)
        ln_b = ln_bias.to(torch.float32)

        # Grid: one program per row
        grid = (num_patches,)
        _layer_norm_kernel[grid](hidden, hidden_norm, ln_w, ln_b,
                                 num_patches, hidden_size, eps,
                                 BLOCK_SIZE=1024)

        # Step 2: SpatialShuffle (exact 2x2 merge). We need to build hidden_shuffled of shape
        # [total_num_merged_patches, 4*hidden_size].
        # total_num_merged_patches = sum_{g} t_g * (h_g//2) * (w_g//2)
        total_merged = 0
        for g in range(num_grids):
            t = int(grid_thw[g, 0].item())
            h = int(grid_thw[g, 1].item())
            w = int(grid_thw[g, 2].item())
            h_merged = h // 2
            w_merged = w // 2
            total_merged += t * h_merged * w_merged

        hidden_shuffled = torch.empty((total_merged, 4 * hidden_size), device=device, dtype=torch.bfloat16)
        # Launch Triton kernel: one program per grid
        _shuffle_2x2_kernel[(num_grids,)](hidden_norm, grid_thw, hidden_shuffled,
                                          num_patches, hidden_size, num_grids,
                                          BLOCK_ROWS=1)

        # Step 3: Linear1 via Triton GEMM (approx one program per output row; robust for these sizes)
        out1 = torch.empty((hidden_shuffled.shape[0], hidden_shuffled.shape[1]), device=device, dtype=torch.bfloat16)
        # We pass N_OUT = hidden_shuffled.shape[0], C_IN=hidden_shuffled.shape[1]=6144, C_OUT=6144, K=6144.
        N_OUT = hidden_shuffled.shape[0]
        C_IN = hidden_shuffled.shape[1]
        C_OUT = C_IN  # 6144
        K = C_IN      # 6144

        # Choose BLOCK_K and BLOCK_CO: for simplicity and correctness, use moderate blocks.
        _linear_row_gemm1[(N_OUT,)](hidden_shuffled, fc1_weight, fc1_bias, out1,
                                    N_OUT, C_IN, C_OUT, K,
                                    BLOCK_K=512, BLOCK_CO=128)

        # GELU activation in Triton
        out1_gelu = torch.empty_like(out1, dtype=torch.bfloat16)
        _gelu_kernel[(hidden_shuffled.shape[0],)](out1, out1_gelu,
                                                  out1.shape[0], out1.shape[1],
                                                  BLOCK_SIZE=1024)

        # Step 4: Linear2 via Triton GEMM
        out2 = torch.empty((out1_gelu.shape[0], fc2_weight.shape[0]), device=device, dtype=torch.bfloat16)
        N_OUT2 = out1_gelu.shape[0]
        C_IN2 = out1_gelu.shape[1]  # 6144
        C_OUT2 = fc2_weight.shape[0]  # 3584
        K2 = C_IN2

        _linear_row_gemm2[(N_OUT2,)](out1_gelu, fc2_weight, fc2_bias, out2,
                                     N_OUT2, C_IN2, C_OUT2, K2,
                                     BLOCK_K=512, BLOCK_CO=128)

        return out2


def run(*args):
    return ModelNew()(*args)
