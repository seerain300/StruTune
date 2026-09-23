import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_reduce_kernel(x_ptr, cos_ptr, out_ptr,
                             n_rows, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for all j=0..seqlen:
      real_out[j] = sum_{k=0..2*seqlen-1} x[k] * cos(j*k/(2*seqlen)) / (2*seqlen)
    x_ptr: flat padded input of length n_rows * (2*seqlen), row-major.
    cos_ptr: 2D grid [n_rows, (seqlen+1)*(2*seqlen)] of cos(j*k/(2*seqlen)).
    out_ptr: flat real output of length n_rows * (seqlen+1), row-major.
    """
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    n = 2 * seqlen
    base_x = row_id * n
    base_out = row_id * (seqlen + 1)

    # Accumulator for real output per j
    acc = tl.zeros((seqlen + 1,), dtype=tl.float32)

    # Loop over k in chunks
    for k_start in range(0, n, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        mask_k = k < n

        # Load x[k] for this row
        x_vals = tl.load(x_ptr + base_x + k, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # For each j, load cos(j, k) and accumulate
        for j in range(0, seqlen + 1):
            col_idx = j * n + k
            cos_vals = tl.load(cos_ptr + row_id * ((seqlen + 1) * n) + col_idx, mask=mask_k, other=0.0)
            prod = x_vals * cos_vals  # [BLOCK_K]
            acc[j] += tl.sum(prod, axis=0)

    # Normalize by 2*seqlen
    acc = acc / (2.0 * seqlen)

    # Store results to out_ptr row
    out_row_start = base_out
    for j in range(0, seqlen + 1):
        tl.store(out_ptr + out_row_start + j, acc[j])


@triton.jit
def rfft_imag_reduce_kernel(x_ptr, sin_ptr, out_ptr,
                             n_rows, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for j=1..seqlen-1:
      imag_out[j] = sum_{k=0..2*seqlen-1} x[k] * sin(j*k/(2*seqlen)) / (2*seqlen)
    x_ptr: same as above.
    sin_ptr: 2D grid [n_rows, (seqlen-1)*(2*seqlen)] of sin(j*k/(2*seqlen)) for j in 1..seqlen-1.
    out_ptr: flat imag output of length n_rows * (seqlen+1), row-major; imag_out[0] and [seqlen] are zero.
    """
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return

    n = 2 * seqlen
    base_x = row_id * n
    base_out = row_id * (seqlen + 1)

    # Accumulator for imag output per j, only for j in 1..seqlen-1
    acc = tl.zeros((seqlen + 1,), dtype=tl.float32)  # will fill j=1..seqlen-1

    for k_start in range(0, n, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        mask_k = k < n
        x_vals = tl.load(x_ptr + base_x + k, mask=mask_k, other=0.0)

        for j in range(1, seqlen):  # j=1..seqlen-1
            col_idx = j * n + k
            sin_vals = tl.load(sin_ptr + row_id * ((seqlen - 1) * n) + col_idx, mask=mask_k, other=0.0)
            prod = x_vals * sin_vals
            acc[j] += tl.sum(prod, axis=0)

    acc = acc / (2.0 * seqlen)

    # Store imag_out[1..seqlen-1]
    out_row_start = base_out
    for j in range(1, seqlen):
        tl.store(out_ptr + out_row_start + j, acc[j])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen), float32 on CUDA
        Returns: (batch, channels, seqlen+1) real and imag parts.
        """
        assert x.is_cuda, "ModelNew.forward expects CUDA tensors"
        assert x.dtype == torch.float32, "Input must be float32"
        batch, channels, seqlen = x.shape
        n_rows = batch * channels
        n = 2 * seqlen

        # Build padded inputs flat (device-side vector)
        x_padded_flat = x.reshape(n_rows, seqlen).reshape(-1)  # assuming x is contiguous float32 on device

        # Build j and k vectors and trig grids on device
        j_vec = torch.arange(seqlen + 1, dtype=torch.float32, device=x.device)
        k_vec = torch.arange(n, dtype=torch.float32, device=x.device)
        # Compute cos grid: cos(j*k/n), shape [seqlen+1, n] for each row
        cos_grid = torch.cos((j_vec[:, None] * k_vec[None, :]) * (2.0 * torch.pi / float(n)))
        # Expand to [n_rows, (seqlen+1)*n]
        cos_flat = cos_grid.unsqueeze(0).expand(n_rows, -1, -1).reshape(n_rows, (seqlen + 1) * n).contiguous()

        # Compute sin grid for imag (exclude j=0 and j=seqlen)
        j_vec_imag = torch.arange(1, seqlen, dtype=torch.float32, device=x.device)
        sin_grid = torch.sin((j_vec_imag[:, None] * k_vec[None, :]) * (2.0 * torch.pi / float(n)))
        sin_flat = sin_grid.unsqueeze(0).expand(n_rows, -1, -1).reshape(n_rows, (seqlen - 1) * n).contiguous()

        # Output buffers (flat)
        out_real_flat = torch.empty(n_rows * (seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag_flat = torch.empty(n_rows * (seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag_flat.zero_()  # imag_out[0] and imag_out[seqlen] will remain zero

        # Launch Triton kernels: one program per row
        BLOCK_K = 256
        rfft_real_reduce_kernel[(n_rows,)](
            x_padded_flat, cos_flat, out_real_flat,
            n_rows, seqlen, BLOCK_K=BLOCK_K, num_warps=4
        )

        if seqlen > 1:
            rfft_imag_reduce_kernel[(n_rows,)](
                x_padded_flat, sin_flat, out_imag_flat,
                n_rows, seqlen, BLOCK_K=BLOCK_K, num_warps=4
            )

        # Reshape outputs to (batch, channels, seqlen+1)
        out_real = out_real_flat.view(batch, channels, seqlen + 1)
        out_imag = out_imag_flat.view(batch, channels, seqlen + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
