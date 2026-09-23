import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      n_rows, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for each row (one program per (batch, channel) row).
    x_ptr: base pointer to input padded vectors (length = n_rows * (2*seqlen), contiguous).
    out_ptr: base pointer to output real vectors (length = n_rows * (seqlen + 1), contiguous).
    """
    row_id = tl.program_id(0)  # 0 .. n_rows-1

    # Base offsets for this row
    base_x = row_id * (2 * seqlen)
    base_out = row_id * (seqlen + 1)

    # Loop over bins j = 0 .. seqlen
    j = 0
    while j <= seqlen:
        acc = 0.0
        k = 0
        while k < (2 * seqlen):
            xk = tl.load(x_ptr + base_x + k)
            angle = 2.0 * 3.141592653589793 * j * k / (2.0 * seqlen)
            ck = tl.cos(angle)
            acc += xk * ck
            k += BLOCK_K
        # Normalize by 2*seqlen
        acc = acc / (2.0 * seqlen)
        tl.store(out_ptr + base_out + j, acc)
        j += 1


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      n_rows, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for each row: j in 1..seqlen-1
    x_ptr: base pointer to input padded vectors (length = n_rows * (2*seqlen), contiguous).
    out_ptr: base pointer to output imaginary vectors (length = n_rows * (seqlen + 1), contiguous).
    """
    row_id = tl.program_id(0)  # 0 .. n_rows-1

    base_x = row_id * (2 * seqlen)
    base_out = row_id * (seqlen + 1)

    j = 1
    while j < seqlen:
        acc = 0.0
        k = 0
        while k < (2 * seqlen):
            xk = tl.load(x_ptr + base_x + k)
            angle = 2.0 * 3.141592653589793 * j * k / (2.0 * seqlen)
            sk = tl.sin(angle)
            acc += xk * sk
            k += BLOCK_K
        acc = acc / (2.0 * seqlen)
        tl.store(out_ptr + base_out + j, acc)
        j += 1


def _build_padded_inputs(x, seqlen, device):
    """
    Build a flat padded input tensor of shape (n_rows * (2*seqlen)) with zeros appended.
    x: (batch, channels, seqlen), float32 on device.
    Returns: 1D tensor of length n_rows * (2*seqlen).
    """
    batch, channels = x.shape[0], x.shape[1]
    n_rows = batch * channels
    n = 2 * seqlen
    x_padded_flat = torch.zeros(n_rows * n, dtype=torch.float32, device=device)
    # For each (b, c), copy x[b, c, :] into the first seqlen entries of its row
    for b in range(batch):
        for c in range(channels):
            row_id = b * channels + c
            row_start = row_id * n
            row_vec = x[b, c, :]
            x_padded_flat[row_start : row_start + seqlen] = row_vec
    return x_padded_flat


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

        # Build padded inputs (no torch math in forward, only allocations)
        x_padded_flat = _build_padded_inputs(x, seqlen, x.device)

        # Output buffers (flat): real and imag, each length n_rows * (seqlen + 1)
        out_real_flat = torch.empty(n_rows * (seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag_flat = torch.empty(n_rows * (seqlen + 1), dtype=torch.float32, device=x.device)
        # Initialize imag output with zeros; we will fill 1..seqlen-1 in kernel
        out_imag_flat.zero_()

        # Launch Triton kernels: one program per row
        grid = (n_rows,)

        # Choose BLOCK_K; 1024 is a good default
        rfft_real_kernel[grid](
            x_padded_flat, out_real_flat,
            n_rows, seqlen,
            BLOCK_K=1024,
            num_warps=4
        )

        rfft_imag_kernel[grid](
            x_padded_flat, out_imag_flat,
            n_rows, seqlen,
            BLOCK_K=1024,
            num_warps=4
        )

        # Assemble outputs into (batch, channels, seqlen+1)
        out_real = out_real_flat.view(batch, channels, seqlen + 1)
        out_imag = out_imag_flat.view(batch, channels, seqlen + 1)

        # Ensure imag_out[0] and imag_out[seqlen] are zero (imag_out initialized to zero)
        # No need to set; already correct.

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
