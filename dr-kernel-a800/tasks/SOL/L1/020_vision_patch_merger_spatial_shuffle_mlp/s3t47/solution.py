import torch
import math
import triton
import triton.language as tl


@triton.jit
def _layer_norm_rows_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                             N, C, eps,
                             BLOCK_SIZE: tl.constexpr):
    """
    LayerNorm over the last dimension C for each row (size N).
    x_ptr: *bf16, [N, C]
    y_ptr: *bf16, [N, C]
    ln_weight_ptr, ln_bias_ptr: *bf16, [C]
    """
    row = tl.program_id(0)
    if row >= N:
        return
    x_row = x_ptr + row * C
    y_row = y_ptr + row * C

    # Pass 1: compute mean/variance in fp32
    sum_val = 0.0
    sum_sq = 0.0
    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
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
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(y_row + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


@triton.jit
def _shuffle_2x2_per_grid_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                 C, NUM_GRIDS):
    """
    hidden_ptr: *bf16, flattened [total_patches, C] (concatenation of per-grid patches)
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], rows are [t, h, w]
    out_ptr: *bf16, flattened [total_merged_rows, 4*C]
    Each program handles one grid, computes mapping (t,h,w) -> (t, h_merged, w_merged, 2,2) -> [4*C].
    """
    g = tl.program_id(0)
    if g >= NUM_GRIDS:
        return

    # Load grid dims
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)
    h_merged = h // 2
    w_merged = w // 2
    num_merged_rows = t * h_merged * w_merged

    # Iterate over all original patches in this grid
    m = 0
    while m < t * h * w:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_ptr + out_row * (4 * C)

        # Four positions in the 2x2 merge
        # r=0,c=0: hidden[t_index, h2, w2]
        src0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + src0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + 0 * C, val0)

        # r=0,c=1: hidden[t_index, h2, w2+1]
        if w2 + 1 < w:
            val1 = tl.load(hidden_ptr + src0 * C + C, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 1 * C, val1)

        # r=1,c=0: hidden[t_index, h2+1, w2]
        if h2 + 1 < h:
            src2 = t_index * (h * w) + (h2 + 1) * w + w2
            val2 = tl.load(hidden_ptr + src2 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 2 * C, val2)

        # r=1,c=1: hidden[t_index, h2+1, w2+1]
        if (h2 + 1 < h) and (w2 + 1 < w):
            src3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
            val3 = tl.load(hidden_ptr + src3 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 3 * C, val3)

        m += 1


@triton.jit
def _linear_row_kernel(x_ptr, W_ptr, b_ptr, y_ptr,
                        K_in, K_out, eps):
    """
    Row-wise linear: compute y[row] = x[row] @ W.T + b.
    x_ptr: *bf16, length K_in (row vector)
    W_ptr: *bf16, shape [K_out, K_in]
    b_ptr: *bf16, length K_out
    y_ptr: *bf16, length K_out
    eps is not used here (placeholder for signature symmetry).
    """
    row = tl.program_id(0)
    if row != 0:
        return

    # Load x row as vector in fp32
    x_row = x_ptr
    x_fp32 = tl.load(x_row + 0 + tl.arange(0, K_in), mask=True, other=0.0).to(tl.float32)  # single row
    # Initialize output accumulator
    y_acc = tl.zeros((K_out,), dtype=tl.float32)

    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, 64)  # BLOCK_K=64, loop over k anyway
        mask_k = offs_k < K_in
        x_vec = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        w_vec = tl.load(W_ptr + offs_k * K_out, mask=mask_k, other=0.0).to(tl.float32)
        y_acc += x_vec * w_vec  # elementwise multiply then sum over k done by loop
        k += 64

    # Add bias and store
    b = tl.load(b_ptr + 0 + tl.arange(0, K_out), mask=True, other=0.0).to(tl.float32)
    y_out = y_acc + b
    tl.store(y_ptr + 0 + tl.arange(0, K_out), y_out.to(tl.bfloat16), mask=True)


@triton.jit
def _gelu_row_kernel(x_ptr, y_ptr, C):
    """
    Elementwise GELU over a row of length C (1D).
    x_ptr: *bf16, length C
    y_ptr: *bf16, length C
    """
    row = tl.program_id(0)
    if row != 0:
        return
    x = tl.load(x_ptr + 0 + tl.arange(0, C), mask=True, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    z = x * inv_sqrt2
    # erf approximation
    # erf(z) ≈ sign(z) * (1 - exp(-z^2) * (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5)),
    # t = 1 / (1 + p*z), p=0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = tl.where(z >= 0, 1.0, -1.0)
    az = tl.abs(z)
    t = 1.0 / (1.0 + p * az)
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erf_approx = sign * (1.0 - poly * tl.exp(-az * az))
    gelu = 0.5 * x * (1.0 + erf_approx)
    tl.store(y_ptr + 0 + tl.arange(0, C), gelu.to(tl.bfloat16), mask=True)


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
        Triton-only implementation of the original forward:
        1) LayerNorm (per-row) on hidden, with affine ln_weight/ln_bias
        2) Spatial shuffle per grid: 2x2 merge into 4*C columns, concatenate per-grid into hidden_shuffled
        3) Linear1: hidden_shuffled @ fc1_weight.T + fc1_bias
        4) GELU activation
        5) Linear2: result @ fc2_weight.T + fc2_bias
        """
        assert hidden.is_cuda and grid_thw.is_cuda and fc1_weight.is_cuda and fc2_weight.is_cuda
        N = hidden.shape[0]
        C = hidden.shape[1]
        total_patches = N  # as per provided get_inputs, hidden is [num_patches, C]
        # 1) LayerNorm in Triton
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
        BLOCK = 256 if C >= 256 else 128
        grid_ln = (N,)
        _layer_norm_rows_kernel[grid_ln](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, C, eps,
            BLOCK_SIZE=BLOCK
        )

        # 2) Spatial shuffle per grid into one output [total_merged_rows, 4*C]
        # Build per-grid hidden_norm and concatenate: total_patches is sum of per-grid patches.
        # However, in the provided get_inputs, hidden is already the concatenated tensor over num_grids.
        # We will shuffle using the provided grid_thw over total_patches.
        total_merged_rows = 0
        for i in range(grid_thw.shape[0]):
            # Read dims for this grid
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            h_merged = h // 2
            w_merged = w // 2
            total_merged_rows += t * h_merged * w_merged

        hidden_shuffled = torch.empty((total_merged_rows, 4 * C), dtype=torch.bfloat16, device=hidden.device)

        # Launch one program per grid to fill hidden_shuffled
        grid_shuffle = (grid_thw.shape[0],)
        _shuffle_2x2_per_grid_kernel[grid_shuffle](
            hidden_norm, grid_thw, hidden_shuffled,
            C, grid_thw.shape[0]
        )

        # 3) Linear1 (row-wise Triton kernel): input rows = total_merged_rows
        X1 = hidden_shuffled
        K_in1 = 4 * C  # hidden_expanded after shuffle
        K_out1 = fc1_weight.shape[0]  # 6144
        W1 = fc1_weight  # [K_out1, K_in1]
        b1 = fc1_bias    # [K_out1]
        Y1 = torch.empty((K_out1,), dtype=torch.bfloat16, device=hidden.device)
        grid_lin1 = (1,)
        _linear_row_kernel[grid_lin1](
            X1, W1, b1, Y1,
            K_in1, K_out1, eps
        )

        # 4) GELU (row-wise Triton kernel)
        Y1_fp32 = Y1.to(torch.float32)
        Y1_gelu = torch.empty_like(Y1_fp32, dtype=torch.float32, device=hidden.device)
        grid_gelu = (1,)
        _gelu_row_kernel[grid_gelu](
            Y1_fp32, Y1_gelu, K_out1
        )

        # 5) Linear2 (row-wise Triton kernel): input rows = K_out1
        K_in2 = K_out1
        K_out2 = fc2_weight.shape[0]  # 3584
        W2 = fc2_weight  # [K_out2, K_in2]
        b2 = fc2_bias    # [K_out2]
        Y2 = torch.empty((K_out2,), dtype=torch.bfloat16, device=hidden.device)
        grid_lin2 = (1,)
        _linear_row_kernel[grid_lin2](
            Y1_gelu, W2, b2, Y2,
            K_in2, K_out2, eps
        )

        return Y2


# Optional: keep get_inputs helper (not used in ModelNew, but shown for parity with original)
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_patches = axes_and_scalars["num_patches"]
    num_merged_patches = axes_and_scalars["num_merged_patches"]
    num_grids = axes_and_scalars["num_grids"]
    hidden_size = 1536
    hidden_size_expanded = 6144
    out_hidden_size = 3584
    merge_size = 2
    eps = 1e-6

    # Try to create grid_thw such that total patches matches num_patches
    # Each grid contributes T * H * W patches, and H, W must be divisible by merge_size
    patches_per_grid = num_patches // num_grids
    sqrt_patches = int(math.sqrt(patches_per_grid))
    h = (sqrt_patches // merge_size) * merge_size
    if h == 0:
        h = merge_size
    w = (patches_per_grid // h // merge_size) * merge_size
    if w == 0:
        w = merge_size
    t = patches_per_grid // (h * w)
    if t == 0:
        t = 1

    grid_thw = torch.zeros((num_grids, 3), dtype=torch.int64, device=device)
    remaining_patches = num_patches
    for i in range(num_grids):
        if i == num_grids - 1:
            patches_for_this = remaining_patches
        else:
            patches_for_this = t * h * w
        sqrt_p = int(math.sqrt(patches_for_this))
        h_i = (sqrt_p // merge_size) * merge_size
        if h_i == 0:
            h_i = merge_size
        w_i = (patches_for_this // h_i // merge_size) * merge_size
        if w_i == 0:
            w_i = merge_size
        t_i = patches_for_this // (h_i * w_i)
        if t_i == 0:
            t_i = 1
        grid_thw[i, 0] = t_i
        grid_thw[i, 1] = h_i
        grid_thw[i, 2] = w_i
        remaining_patches -= t_i * h_i * w_i

    hidden = torch.randn(num_patches, hidden_size, dtype=torch.bfloat16, device=device)
    ln_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    ln_bias = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)
    fc1_weight = torch.randn(hidden_size_expanded, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc1_bias = torch.randn(hidden_size_expanded, dtype=torch.bfloat16, device=device)
    fc2_weight = torch.randn(out_hidden_size, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc2_bias = torch.randn(out_hidden_size, dtype=torch.bfloat16, device=device)

    return {
        "hidden": hidden,
        "grid_thw": grid_thw,
        "ln_weight": ln_weight,
        "ln_bias": ln_bias,
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
        "eps": eps,
    }


# Example usage:
# model = ModelNew().cuda()
# inputs = get_inputs({'num_patches': 4096, 'num_merged_patches': 1024, 'num_grids': 4}, torch.device('cuda'))
# out = model(
#     inputs['hidden'],
#     inputs['grid_thw'],
#     inputs['ln_weight'],
#     inputs['ln_bias'],
#     inputs['fc1_weight'],
#     inputs['fc1_bias'],
#     inputs['fc2_weight'],
#     inputs['fc2_bias'],
#     inputs['eps'],
# )


def run(*args):
    return ModelNew()(*args)
