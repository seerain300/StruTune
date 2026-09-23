import torch
import math
import triton
import triton.language as tl


# Triton kernel: LayerNorm over last dimension (C) for each row (N).
# We use two passes: first reduce mean/variance in fp32, second normalize and apply affine.
@triton.jit
def _layer_norm_rows_kernel(x_ptr, y_ptr, ln_weight_ptr, ln_bias_ptr,
                             N, C, eps,
                             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= N:
        return

    x_row = x_ptr + row * C
    y_row = y_ptr + row * C

    # Pass 1: mean and variance
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

    # Pass 2: normalize + affine
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


# Triton kernel: Per-grid exact 2x2 spatial merge.
# Input is the concatenation of per-grid hidden_norm tensors (shape: [total_patches, C]).
# Output is a single tensor of shape [total_num_merged_rows, 4*C], where total_num_merged_rows = sum over grids of t * (h//2) * (w//2).
# We launch one program per grid. For each original patch m in [0, t*h*w):
#   - t_index = m // (h*w), rem = m % (h*w), h2 = rem // w, w2 = rem % w
#   - out_row = t_index * (h_merged * w_merged) + (h2//2) * w_merged + (w2//2)
#   - We write four values into contiguous columns: (0,1,2,3) corresponding to r=0,c=0; r=0,c=1; r=1,c=0; r=1,c=1.
@triton.jit
def _shuffle_2x2_per_grid_kernel(hidden_ptr, grid_thw_ptr, out_ptr,
                                 total_patches, C, NUM_GRIDS,
                                 h_list_ptr, w_list_ptr,
                                 BLOCK_M: tl.constexpr):
    """
    hidden_ptr: *bf16, flattened [total_patches, C]
    grid_thw_ptr: *int64, shape [NUM_GRIDS, 3], each row is [t, h, w]
    out_ptr: *bf16, flattened [total_merged_rows, 4*C]
    h_list_ptr, w_list_ptr: *int64, shape [NUM_GRIDS], precomputed h/w per grid
    """
    g = tl.program_id(0)
    if g >= NUM_GRIDS:
        return

    # Load grid dimensions
    t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

    h_merged = h // 2
    w_merged = w // 2
    num_merged_rows = t * h_merged * w_merged

    # For each original patch m in [0, t*h*w):
    m = 0
    while m < t * h * w:
        t_index = m // (h * w)
        rem = m % (h * w)
        h2 = rem // w
        w2 = rem % w

        out_row = t_index * (h_merged * w_merged) + (h2 // 2) * w_merged + (w2 // 2)
        base_out = out_ptr + out_row * (4 * C)

        # idx 0: r=0,c=0 -> hidden[t_index, h2, w2]
        col0 = (h2 * 2 + 0) * w * C + (w2 * 2 + 0) * C
        src0 = t_index * (h * w) + h2 * w + w2
        val0 = tl.load(hidden_ptr + src0 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
        tl.store(base_out + 0 * C, val0)

        # idx 1: r=0,c=1 -> hidden[t_index, h2, w2+1]
        if w2 + 1 < w:
            col1 = col0 + C
            val1 = tl.load(hidden_ptr + src0 * C + C, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 1 * C, val1)

        # idx 2: r=1,c=0 -> hidden[t_index, h2+1, w2]
        if h2 + 1 < h:
            src2 = t_index * (h * w) + (h2 + 1) * w + w2
            col2 = (h2 * 2 + 1) * w * C + (w2 * 2 + 0) * C
            val2 = tl.load(hidden_ptr + src2 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 2 * C, val2)

        # idx 3: r=1,c=1 -> hidden[t_index, h2+1, w2+1]
        if (h2 + 1 < h) and (w2 + 1 < w):
            src3 = t_index * (h * w) + (h2 + 1) * w + (w2 + 1)
            col3 = (h2 * 2 + 1) * w * C + (w2 * 2 + 1) * C
            val3 = tl.load(hidden_ptr + src3 * C + 0, mask=True, other=0.0).to(tl.bfloat16)
            tl.store(base_out + 3 * C, val3)

        m += 1


# Triton row-wise matmul + bias: y = x @ W.T + b
# x is 1D row vector of length K_in; W is [K_out, K_in]; b is [K_out]; y is [K_out]
# We specialize this for hidden_size_expanded where K_in=6144 and K_out=6144 or 3584.
@triton.jit
def _row_matmul_bias_kernel(x_ptr, W_ptr, b_ptr, y_ptr,
                             K_in, K_out, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    # Compute one output row per program
    x_row_ptr = x_ptr + row * K_in
    y_row_ptr = y_ptr + row * K_out

    acc = tl.zeros((K_out,), dtype=tl.float32)

    k = 0
    while k < K_in:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_in
        x = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        Wk = tl.load(W_ptr + offs_k * K_out, mask=mask_k, other=0.0).to(tl.float32)
        acc += x * Wk
        k += BLOCK_K

    # add bias
    offs = tl.arange(0, K_out)
    b = tl.load(b_ptr + offs, mask=True, other=0.0).to(tl.float32)
    acc += b

    # store result as bfloat16
    tl.store(y_row_ptr + offs, acc.to(tl.bfloat16), mask=True)


# Triton elementwise GELU for a row vector of length C (we launch one program per output row after Linear1)
@triton.jit
def _gelu_row_kernel(x_ptr, y_ptr, C, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * C
    y_row_ptr = y_ptr + row * C

    col = 0
    while col < C:
        offs = col + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865476
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        z = x * inv_sqrt2
        # Approximate erf(z) using a well-known polynomial; Triton has no erf, so use approximation.
        # erf(z) ≈ sign(z) * (1 - t * (a1 + t*(a2 + t*(a3 + t*(a4 + t*a5)))) * exp(-z^2)), t=1/(1+p*|z|)
        # Use p=0.3275911 and coefficients from Abramowitz & Stegun.
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        sign = tl.where(z >= 0.0, 1.0, -1.0)
        az = tl.abs(z)
        t = 1.0 / (1.0 + 0.3275911 * az)
        poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
        erf_approx = sign * (1.0 - poly * tl.exp(-az * az))
        y = 0.5 * x * (1.0 + erf_approx)
        tl.store(y_row_ptr + offs, y.to(tl.bfloat16), mask=mask)
        col += BLOCK_SIZE


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
        hidden: [num_patches, hidden_size] (bfloat16), row-major
        grid_thw: [num_grids, 3] int64, each row [t, h, w]
        ln_weight, ln_bias: [hidden_size] bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded] bfloat16
        fc1_bias: [hidden_size_expanded] bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded] bfloat16
        fc2_bias: [out_hidden_size] bfloat16
        eps: float
        Returns: [num_merged_patches, out_hidden_size] bfloat16
        """
        device = hidden.device
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        hidden_size_expanded = fc1_weight.shape[0]  # 6144
        out_hidden_size = fc2_weight.shape[0]       # 3584

        # 1) LayerNorm (per-row) on hidden
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        N = num_patches
        C = hidden_size
        # We choose BLOCK_SIZE as 1024 (constexpr), mask handles C not divisible by 1024
        _layer_norm_rows_kernel[(N,)](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, C, float(eps),
            BLOCK_SIZE=1024,
        )

        # 2) Spatial shuffle: per-grid 2x2 merge into [total_num_merged_rows, 4*C]
        total_merged_rows = 0
        # Compute per-grid h and w for input pointer mapping
        NUM_GRIDS = grid_thw.shape[0]
        # We don't have per-grid h/w readily; however, we can reconstruct the output based on total hidden_norm.
        # The original code builds per-grid hidden tensors and then performs reshape. Here we simulate the same
        # by launching one program per grid and using total hidden_norm buffer with per-grid t,h,w.
        # Allocate output buffer for all grids
        # We need to know total_num_merged_rows to allocate out. But Triton kernels work on tensors already created.
        # We will re-compute total_num_merged_rows from grid_thw. The original function also creates grid_thw.
        # We'll do it in two steps: first compute total_merged_rows, then allocate, then run kernels.

        # Compute total_num_merged_rows from grid_thw (T=grid_thw[...,0], H=grid_thw[...,1], W=grid_thw[...,2])
        total_merged_rows = 0
        for i in range(NUM_GRIDS):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            total_merged_rows += t * (h // 2) * (w // 2)

        hidden_shuffled = torch.empty((total_merged_rows, 4 * hidden_size), dtype=torch.bfloat16, device=device)

        # Prepare per-grid h,w (we don't need precomputed lists; we can read from grid_thw inside the kernel by index g.
        # But Triton cannot index by program_id into a tensor. So we pass h_list and w_list as tensors of shape [NUM_GRIDS].
        # We'll compute h and w per grid from grid_thw inside the launcher by reading rows. Triton kernel expects pointers.
        # Simpler: we don't need h_list/w_list since the kernel reads grid_thw_ptr directly. We pass them as dummy tensors if needed.
        # We can pass grid_thw_ptr and total_patches to the kernel; it uses total_patches only for m-loop bound via while.

        _shuffle_2x2_per_grid_kernel[(NUM_GRIDS,)](
            hidden_norm, grid_thw, hidden_shuffled,
            num_patches, hidden_size, NUM_GRIDS,
            # h_list_ptr, w_list_ptr: not used, we read from grid_thw inside; Triton can't index by program_id into grid_thw,
            # so we rely on pointer arithmetic and no extra lists. We pass dummy tensors if required; for simplicity, Triton
            # accepts these pointers; the kernel only needs grid_thw.
        )

        # 3) Linear1: hidden_shuffled [total_merged_rows, 6144] @ fc1_weight.T [6144, 6144] + fc1_bias
        # Output after Linear1: [total_merged_rows, 6144]
        hidden_fc1 = torch.empty((total_merged_rows, hidden_size_expanded), dtype=torch.bfloat16, device=device)
        # Launch one program per output row
        _row_matmul_bias_kernel[(total_merged_rows,)](
            hidden_shuffled, fc1_weight, fc1_bias, hidden_fc1,
            hidden_size_expanded, hidden_size_expanded,
            BLOCK_K=256,
        )

        # 4) GELU activation on hidden_fc1
        hidden_gelu = torch.empty_like(hidden_fc1, dtype=torch.bfloat16, device=device)
        _gelu_row_kernel[(total_merged_rows,)](
            hidden_fc1, hidden_gelu, hidden_size_expanded,
            BLOCK_SIZE=1024,
        )

        # 5) Linear2: hidden_gelu [total_merged_rows, 6144] @ fc2_weight.T [3584, 6144] + fc2_bias -> [total_merged_rows, 3584]
        output = torch.empty((total_merged_rows, out_hidden_size), dtype=torch.bfloat16, device=device)
        # We need to reshape total_merged_patches == num_merged_patches. The original pipeline expects output [num_merged_patches, out_hidden_size].
        # We can compute num_merged_patches as total_merged_rows (since we concatenated all grids).
        _row_matmul_bias_kernel[(total_merged_rows,)](
            hidden_gelu, fc2_weight, fc2_bias, output,
            hidden_size_expanded, out_hidden_size,
            BLOCK_K=256,
        )

        return output


# Example helper to generate inputs (not used in evaluation, kept for parity with original):
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_patches = axes_and_scalars["num_patches"]
    num_merged_patches = axes_and_scalars["num_merged_patches"]
    num_grids = axes_and_scalars["num_grids"]
    hidden_size = 1536
    hidden_size_expanded = 6144
    out_hidden_size = 3584
    eps = 1e-6

    # Random grid sizes (to match num_patches, with T*H*W == num_patches per grid)
    # This mirrors the original helper creation; exact shapes may vary, but the forward will handle dynamically.
    patches_per_grid = num_patches // num_grids if num_grids > 0 else num_patches
    # Construct t, h, w such that t*h*w == patches_per_grid
    # Simple heuristic using sqrt
    sqrt_p = int(math.sqrt(patches_per_grid))
    h = (sqrt_p // 2) * 2  # ensure divisible by 2 (merge_size=2)
    if h == 0:
        h = 2
    w = (patches_per_grid // h // 2) * 2
    if w == 0:
        w = 2
    t = patches_per_grid // (h * w)
    if t == 0:
        t = 1

    grid_thw = torch.zeros((num_grids, 3), dtype=torch.int64, device=device)
    # Fill grid_thw with (t,h,w). For generality, we can set all grids to same (t,h,w) if num_patches divisible; otherwise distribute.
    # Here we assume uniform grid sizes if num_patches % num_grids == 0.
    if num_patches % num_grids == 0:
        for i in range(num_grids):
            grid_thw[i] = torch.tensor([t, h, w], dtype=torch.int64, device=device)
    else:
        # Simple distribution: keep t=h=1, w=num_patches
        # Note: this deviates from original helper but ensures total_patches == num_patches
        for i in range(num_grids):
            grid_thw[i] = torch.tensor([1, 1, num_patches], dtype=torch.int64, device=device)

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


# Example run (not used in evaluation, but demonstrates usage):
if __name__ == "__main__":
    device = "cuda"
    model = ModelNew().to(device)
    axes_and_scalars = {
        "num_patches": 4096,
        "num_merged_patches": 1024,
        "num_grids": 4,
    }
    inputs = get_inputs(axes_and_scalars, torch.device(device))
    output = model(
        inputs["hidden"],
        inputs["grid_thw"],
        inputs["ln_weight"],
        inputs["ln_bias"],
        inputs["fc1_weight"],
        inputs["fc1_bias"],
        inputs["fc2_weight"],
        inputs["fc2_bias"],
        inputs["eps"],
    )
    print("Output shape:", output.shape)


def run(*args):
    return ModelNew()(*args)
