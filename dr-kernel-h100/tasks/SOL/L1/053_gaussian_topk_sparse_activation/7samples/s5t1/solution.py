import torch
import triton
import triton.language as tl


# Triton kernels: reduce sum and sum of squares along the last dimension (features)
@triton.jit
def reduce_sum_and_sumsq_kernel(
    X_ptr,            # *const float32
    N_rows,           # int32
    N_cols,           # int32 (H, the last dimension length)
    STRIDE_ROW,       # int32 (distance between rows in elements)
    STRIDE_COL,       # int32 (distance between columns, typically 1)
    SUM_ptr,          # *float32 (1-element, output: sum per row)
    SUMSQ_ptr,        # *float32 (1-element, output: sum of squares per row)
    BLOCK_SIZE: tl.constexpr,  # how many columns to process per loop
):
    pid = tl.program_id(axis=0)  # 0 .. (B*L - 1)
    # Bounds check (grid is exactly N_rows)
    # Initialize accumulators
    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    # Loop over columns in chunks
    for col_start in range(0, N_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_cols
        row_offset = pid * STRIDE_ROW
        x = tl.load(X_ptr + row_offset + cols * STRIDE_COL, mask=mask, other=0.0)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)

    # Write out one scalar per row
    tl.store(SUM_ptr + pid, acc_sum)
    tl.store(SUMSQ_ptr + pid, acc_sumsq)


# Triton scalar kernel: compute inverse-normal quantile z = _ndtri(sp) for sp in (0,1)
# Uses Abramowitz-Stegun approximation with piecewise regions.
@triton.jit
def ndtri_scalar_kernel(
    SP_ptr,           # *const float32 (1-element, input sparsity target)
    Z_ptr,            # *float32 (1-element output)
    p_low: tl.constexpr,  # float32
    a1, a2, a3, a4, a5, a6,  # float32
    b1, b2, b3, b4, b5,      # float32
    c1, c2, c3, c4, c5, c6,  # float32
    d1, d2, d3, d4,          # float32
):
    sp = tl.load(SP_ptr)  # scalar float32
    p_high = 1.0 - p_low
    # lower region
    mask_low = sp < p_low
    q_low = tl.sqrt(-2.0 * tl.log(sp))
    z_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    # central region
    mask_mid = (sp >= p_low) & (sp <= p_high)
    q_mid = sp - 0.5
    r_mid = q_mid * q_mid
    z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    # upper region
    mask_high = sp > p_high
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - sp))
    z_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select z based on mask. Triton supports scalar if/else on masks.
    z = tl.where(mask_low, z_low, 0.0)
    z = tl.where(mask_mid, z_mid, z)
    z = tl.where(mask_high, z_high, z)

    tl.store(Z_ptr, z)


# Triton elementwise kernel: apply y = max(0, X - threshold), where threshold per row
# is provided as a vector THRESH_ptr of length N_cols, loaded with stride 1.
@triton.jit
def sparsify_elementwise_kernel(
    X_ptr,               # *const float32
    OUT_ptr,             # *float32
    THRESH_ptr,          # *const float32, shape [N_cols] (broadcast per row)
    N_rows,              # int32
    N_cols,              # int32
    STRIDE_ROW,          # int32
    STRIDE_COL,          # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # row index in [0, N_rows)
    # Guard (normally grid == N_rows)
    # Load threshold vector for this row (broadcast)
    # We'll process columns in chunks of BLOCK_SIZE
    # Create an output accumulator for the row
    # For elementwise write, iterate over columns
    for col_start in range(0, N_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_cols
        row_offset = pid * STRIDE_ROW
        x = tl.load(X_ptr + row_offset + cols * STRIDE_COL, mask=mask, other=0.0)
        # Load threshold for this row; THRESH is 1D contiguous vector of length N_cols
        thr = tl.load(THRESH_ptr + cols, mask=mask, other=0.0)
        y = tl.maximum(x - thr, 0.0)  # ReLU
        tl.store(OUT_ptr + row_offset + cols * STRIDE_COL, y, mask=mask)


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-only implementation of the original run function.
    Returns a bfloat16 tensor of the same shape as inputs, sparsified via
    per-(batch, seq) mean/std and Gaussian-based threshold.
    """
    assert inputs.is_cuda, "inputs must be on CUDA for Triton kernels"
    assert inputs.dim() == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
    B, L, H = inputs.shape
    # Compute stats in float32
    x = inputs.contiguous().to(torch.float32)

    # Prepare output (float32) and threshold vectors
    out = torch.empty_like(x, dtype=torch.float32)
    # We need mean and std per row (B*L). Allocate vectors of length B*L
    sums = torch.empty(B * L, dtype=torch.float32, device=x.device)
    sums_sq = torch.empty(B * L, dtype=torch.float32, device=x.device)

    # Launch reduction kernels
    BLOCK = 1024  # handle up to 16384 features; tuneable
    grid = (B * L,)
    # Assume contiguous last dimension; STRIDE_COL = 1, STRIDE_ROW = H
    reduce_sum_and_sumsq_kernel[grid](
        x,
        B * L,
        H,
        H,  # STRIDE_ROW in elements = H for contiguous [B,L,H]
        1,  # STRIDE_COL in elements = 1 for contiguous last dim
        sums,
        sums_sq,
        BLOCK_SIZE=BLOCK,
    )

    # Compute per-row mean and std
    # mean = sum / H; var = sumsq/H - mean^2
    mean = (sums / H)
    var = (sums_sq / H) - mean * mean
    # Numerical safety: clamp var to >= 0
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)

    # Compute z = _ndtri(target_sparsity) via Triton scalar kernel
    sp_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=x.device)
    z_tensor = torch.empty((), dtype=torch.float32, device=x.device)
    ndtri_scalar_kernel[(1,)](
        sp_tensor,
        z_tensor,
        0.02425,  # p_low
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    )

    # Compute per-row threshold: mean + std * z
    threshold = mean + std * z_tensor  # shape [B*L], float32
    # Build a 2D threshold tensor of shape [B*L, H] for broadcasting in Triton
    # We'll pass it as a contiguous 1D vector of length H for each row program
    # However, in the kernel, we load THRESH_ptr per row. We can create a [B*L, H] tensor here.
    threshold_2d = threshold.view(B * L, 1).expand(B * L, H).contiguous()  # shape [B*L, H]

    # Launch elementwise sparsification kernel
    # We'll write directly into out; inputs are float32, output is float32, then cast to bfloat16
    sparsify_elementwise_kernel[grid](
        x,                           # X_ptr
        out,                         # OUT_ptr
        threshold_2d.reshape(B * L * H),  # THRESH_ptr: pass as 1D vector of length H per row
        B * L,
        H,
        H,                           # STRIDE_ROW = H
        1,                           # STRIDE_COL = 1 (contiguous last dim)
        BLOCK_SIZE=BLOCK,
    )

    # Cast back to bfloat16 to match original behavior
    return out.to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D tensor input: [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        # If input is not on CUDA, move it. The evaluation harness provides CUDA tensors.
        if not inputs.is_cuda:
            inputs = inputs.cuda()
        # Default target sparsity as in the original code; could be made configurable
        return _run_triton(inputs, target_sparsity=0.1)


def run(*args):
    return ModelNew()(*args)
