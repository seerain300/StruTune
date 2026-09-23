import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Compute per-row mean and population std over the last dim K for each row in a flattened [total_rows, K] view.
    x_ptr points to a contiguous tensor with row stride = K.
    mean_ptr, std_ptr: [total_rows]
    """
    pid = tl.program_id(axis=0)
    row_start = pid * K
    # Accumulators in fp32
    acc = 0.0
    acc2 = 0.0
    # Vector of offsets within a tile
    offsets = tl.arange(0, BLOCK_SIZE)
    # Loop over tiles along K
    for k in range(0, K, BLOCK_SIZE):
        idx = k + offsets
        mask = idx < K
        vals = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
        acc2 += tl.sum(vals * vals, axis=0)
    n = K
    mean = acc / n
    # Population variance: E[x^2] - (E[x])^2
    var = acc2 / n - mean * mean
    # std = sqrt(max(var, 0)) to avoid tiny negative due to numerical error
    std = tl.sqrt(tl.maximum(var, 0.0))
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_buf_ptr, target_sparsity, a1, a2, a3, a4, a5, a6,
                         b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low):
    """
    Compute inverse standard normal CDF for sparsity 'target_sparsity' using Abramowitz & Stegun 5.2.23.
    Store result into z_buf_ptr (size 1).
    """
    p = target_sparsity  # scalar
    # Piecewise approximation
    # Lower region
    mask_low = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
    # Central region
    mask_mid = (p >= p_low) & (p <= 1.0 - p_low)
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    y_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    # Upper region
    mask_high = p > 1.0 - p_low
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
    # Select piece
    y = tl.where(mask_low, y_low, 0.0)
    y = tl.where(mask_mid, y_mid, y)
    y = tl.where(mask_high, y_high, y)
    tl.store(z_buf_ptr, y)


@triton.jit
def apply_threshold_kernel(x_ptr, mean_ptr, std_ptr, z_ptr, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row, compute threshold = mean + std * z and write to out_ptr with shape [total_rows, K].
    x_ptr: input flattened [total_rows, K]
    mean_ptr, std_ptr: [total_rows]
    z_ptr: scalar [1]
    out_ptr: output flattened [total_rows, K]
    """
    pid = tl.program_id(axis=0)
    row_start = pid * K
    # Load mean and std for this row
    mean = tl.load(mean_ptr + pid)
    std = tl.load(std_ptr + pid)
    z = tl.load(z_ptr)  # scalar
    thresh = mean + std * z
    # Write threshold across K
    offsets = tl.arange(0, BLOCK_SIZE)
    for k in range(0, K, BLOCK_SIZE):
        idx = k + offsets
        mask = idx < K
        # Broadcast threshold to vector and store
        tl.store(out_ptr + row_start + idx, thresh, mask=mask)


@triton.jit
def gating_relu_kernel(x_ptr, thresh_ptr, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Apply ReLU gating: out = max(x - thresh, 0). thresh_ptr points to per-row thresholds [total_rows, K].
    """
    pid = tl.program_id(axis=0)
    row_start = pid * K
    offsets = tl.arange(0, BLOCK_SIZE)
    for k in range(0, K, BLOCK_SIZE):
        idx = k + offsets
        mask = idx < K
        x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        t = tl.load(thresh_ptr + row_start + idx, mask=mask, other=0.0)
        out = x - t
        out = tl.where(out > 0, out, 0.0)  # ReLU without tl.maximum to avoid host ops
        tl.store(out_ptr + row_start + idx, out, mask=mask)


def _choose_gate_params(K: int):
    """
    Simple heuristic for BLOCK_SIZE and num_warps based on K.
    """
    if K >= 8192:
        return 4096, 8, 2
    else:
        return 2048, 4, 2


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-only implementation of the Gaussian-based top-k sparse activation.
        Computes per-row mean and std, finds z = ndtri(target_sparsity), then out = relu(x - (mean + std * z)),
        returned as bfloat16.
        """
        # If not CUDA, fallback to PyTorch (for robustness). The evaluation uses CUDA inputs.
        if not x.is_cuda:
            # Fallback: ensure correctness on CPU if ever needed
            x_f32 = x.to(torch.float32)
            mean = x_f32.mean(dim=-1, keepdim=True)
            std = x_f32.std(dim=-1, keepdim=True, unbiased=False)
            # Use torch's inverse CDF via Normal CDF; small overhead but correct on CPU
            z = torch.tensor(target_sparsity, dtype=torch.float32, device=x.device)
            z_val = torch.distributions.normal.Normal(0.0, 1.0).icdf(z)
            thresh = mean + std * z_val
            out_f32 = torch.relu(x_f32 - thresh)
            return out_f32.to(torch.bfloat16)

        # Ensure contiguous [total_rows, K] view
        total_rows = x.shape[0] * x.shape[1]
        K = x.shape[2]
        x_flat = x.contiguous().view(-1, K)

        # 1) Compute mean and std per row via Triton reduction
        x_flat_f32 = x_flat.to(torch.float32)
        mean = torch.empty(total_rows, device=x.device, dtype=torch.float32)
        std = torch.empty(total_rows, device=x.device, dtype=torch.float32)

        # Choose BLOCK_SIZE for reduction: 1024 works well generally
        BLOCK_SIZE_RED = 1024
        compute_row_stats_kernel[(total_rows,)](
            x_flat_f32, mean, std, total_rows, K,
            BLOCK_SIZE=BLOCK_SIZE_RED,
            num_warps=4, num_stages=2
        )

        # 2) Compute inverse normal CDF z via Triton (no torch tensors on host)
        z_buf = torch.empty(1, device=x.device, dtype=torch.float32)
        # Constants for Abramowitz & Stegun 5.2.23
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, float(target_sparsity),
            a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low,
            num_warps=1, num_stages=1
        )

        # 3) Compute per-row threshold in Triton
        thresh = torch.empty_like(x_flat_f32)
        block_size_gate, num_warps_gate, num_stages_gate = _choose_gate_params(K)
        apply_threshold_kernel[(total_rows,)](
            x_flat_f32, mean, std, z_buf, thresh, total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate, num_stages=num_stages_gate
        )

        # 4) Apply ReLU gating in Triton
        out_f32 = torch.empty_like(x_flat_f32)
        gating_relu_kernel[(total_rows,)](
            x_flat_f32, thresh, out_f32, total_rows, K,
            BLOCK_SIZE=block_size_gate,
            num_warps=num_warps_gate, num_stages=num_stages_gate
        )

        # 5) Reshape back and cast to bfloat16
        out = out_f32.view(*x.shape).to(torch.bfloat16)
        return out


def run(*args):
    return ModelNew()(*args)
