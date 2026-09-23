import torch
import triton
import triton.language as tl


@triton.jit
def compute_row_stats_kernel(x_ptr, mean_ptr, std_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    For each row index pid in [0, total_rows), compute mean and std over K elements.
    x_ptr points to a flattened [total_rows, K] region; stride per row is K.
    Writes mean and std to mean_ptr[pid] and std_ptr[pid].
    """
    pid = tl.program_id(0)
    row_offset = pid * K
    acc_sum = 0.0
    acc_sum2 = 0.0

    # Process the row in tiles of BLOCK_SIZE
    for k in range(0, K, BLOCK_SIZE):
        offs = k + tl.arange(0, BLOCK_SIZE)
        mask = offs < K
        x = tl.load(x_ptr + row_offset + offs, mask=mask, other=0.0)
        # Accumulate in fp32
        x = x.to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sum2 += tl.sum(x * x, axis=0)

    n = K  # population std
    mean = acc_sum / n
    # population variance: E[x^2] - (E[x])^2
    var = acc_sum2 / n - mean * mean
    std = tl.sqrt(var)

    # Store per-row mean and std (scalars per row)
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_ndtri_kernel(z_buf, p, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, p_low, p_high):
    """
    Compute inverse standard normal CDF at p (scalar) using Abramowitz & Stegun 5.2.23 approximation.
    Store result into z_buf (1-element tensor).
    """
    # Lower tail
    q = tl.sqrt(-2.0 * tl.log(p))
    z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    # Upper tail
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_up = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Piecewise selection
    if p < p_low:
        z = z_low
    elif p > (1.0 - p_low):
        z = z_up
    else:
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        denom = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        z = poly * q / denom

    tl.store(z_buf, z)


@triton.jit
def apply_gating_1d_kernel(in_ptr, mean_ptr, std_ptr, z_buf, out_ptr, total_rows, K, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise gating over flattened input:
    For each index i, compute row r = i // K, threshold t = mean[r] + std[r] * z,
    out[i] = max(0, in[i] - t). Writes directly to out_ptr with 3D indexing mapping.
    """
    pid = tl.program_id(0)
    n_elems = total_rows * K

    for i in range(pid * BLOCK_SIZE, n_elems, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elems

        # Map flattened index to (row, col)
        r = offs // K  # row index [0, total_rows)
        c = offs % K   # column index [0, K)

        # Load input and per-row stats
        x = tl.load(in_ptr + offs, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + r, mask=mask, other=0.0)
        std = tl.load(std_ptr + r, mask=mask, other=0.0)
        z = tl.load(z_buf)  # scalar

        # Compute threshold and gating
        t = mean + std * z
        y = x - t
        y = tl.maximum(y, 0.0)

        # Write output (direct 3D write: out[r, 0, c])
        tl.store(out_ptr + r * K + c, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable parameters for Triton kernels
        self.block_size_stats = 1024
        self.num_warps_stats = 4
        self.num_stages_stats = 2

        self.block_size_gate = 1024
        self.num_warps_gate = 4
        self.num_stages_gate = 2

    def forward(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Early return if no sparsity
        if target_sparsity == 0.0:
            return x

        assert x.is_cuda, "Input must be on CUDA for Triton kernels."

        B, S, K = x.shape
        total_rows = B * S

        # Ensure contiguous and use float32 for compute
        x_contig = x.contiguous()

        # 1) Compute per-row mean and std
        mean = torch.empty((total_rows,), dtype=torch.float32, device=x.device)
        std = torch.empty((total_rows,), dtype=torch.float32, device=x.device)

        grid_stats = (total_rows,)
        compute_row_stats_kernel[grid_stats](
            x_contig.view(-1), mean, std, total_rows, K,
            BLOCK_SIZE=self.block_size_stats,
            num_warps=self.num_warps_stats,
            num_stages=self.num_stages_stats
        )

        # 2) Compute z = _ndtri(target_sparsity) via Triton
        z_buf = torch.empty((1,), dtype=torch.float32, device=x.device)
        # Constants for A&S approximation
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02; a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02; b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00; c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00; d4 = 3.754408661907416e+00
        p_low = 0.02425

        compute_ndtri_kernel[(1,)](
            z_buf, target_sparsity,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, 1.0 - p_low,
            num_warps=1,
            num_stages=1
        )

        # 3) Apply gating with 1D kernel; write directly to 3D output
        in_f32 = x_contig.to(torch.float32)
        out_f32 = torch.empty((B, S, K), dtype=torch.float32, device=x.device)

        n_elems = total_rows * K
        grid_gate = (triton.cdiv(n_elems, self.block_size_gate),)
        apply_gating_1d_kernel[grid_gate](
            in_f32.view(-1), mean, std, z_buf, out_f32.view(-1), total_rows, K,
            BLOCK_SIZE=self.block_size_gate,
            num_warps=self.num_warps_gate,
            num_stages=self.num_stages_gate
        )

        # Cast back to bfloat16 to match original behavior
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
