import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      N, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N, for j in 0..seqlen.
    x_ptr points to the padded input vector of length N.
    out_ptr points to the output real vector of length seqlen+1 (we write j in 0..seqlen).
    """
    pid = tl.program_id(0)
    # j loop
    for j in range(0, seqlen + 1):
        acc = 0.0
        # k loop over blocks
        for k_start in range(0, N, BLOCK_K):
            k_idx = k_start + tl.arange(0, BLOCK_K)
            mask = k_idx < N
            # load x[k]
            xk = tl.load(x_ptr + k_idx, mask=mask, other=0.0)
            # cos term: cos(2*pi*j*k/N)
            angle = 2.0 * 3.141592653589793 * j * k_idx / N
            cos_term = tl.cos(angle)
            # accumulate
            acc += tl.sum(xk * cos_term, axis=0)
        # normalize
        acc = acc / N
        # store result
        tl.store(out_ptr + j, acc)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      N, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N, for j in 1..seqlen-1.
    x_ptr points to the padded input vector of length N.
    out_ptr points to the output imag vector of length seqlen (we write j in 1..seqlen-1).
    """
    pid = tl.program_id(0)
    # j loop starting from 1
    for j in range(1, seqlen):
        acc = 0.0
        # k loop over blocks
        for k_start in range(0, N, BLOCK_K):
            k_idx = k_start + tl.arange(0, BLOCK_K)
            mask = k_idx < N
            xk = tl.load(x_ptr + k_idx, mask=mask, other=0.0)
            angle = 2.0 * 3.141592653589793 * j * k_idx / N
            sin_term = tl.sin(angle)
            acc += tl.sum(xk * sin_term, axis=0)
        acc = acc / N
        tl.store(out_ptr + j, acc)


def _run_triton_rfft(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Compute real and imaginary parts of rfft(x, n=2*seqlen) using Triton, returning
    real and imag tensors of shape (batch, channels, seqlen+1).
    """
    batch, channels, seqlen = x.shape
    N = 2 * seqlen
    total_rows = batch * channels

    # Prepare padded inputs: for each (b, c), take x[b, c, :] and pad zeros to N.
    # Allocate a list of 1D padded vectors for simplicity; Triton will read them.
    # We use PyTorch for allocation; no torch math in forward.
    x_rows = []
    for b in range(batch):
        for c in range(channels):
            row = x[b, c, :].contiguous().to(torch.float32)
            # pad zeros: [row, zeros]
            pad = torch.zeros(N - seqlen, device=row.device, dtype=row.dtype)
            padded = torch.cat([row, pad], dim=0)
            x_rows.append(padded)

    # Output real and imag vectors, shape (total_rows, seqlen+1) and (total_rows, seqlen)
    real_out_flat = torch.empty((total_rows, seqlen + 1), device=x.device, dtype=torch.float32)
    imag_out_flat = torch.empty((total_rows, seqlen), device=x.device, dtype=torch.float32)

    # Launch Triton kernels: one program per row
    grid = (total_rows,)
    # Use a reasonable BLOCK_K; Triton loop handles N generically.
    BLOCK_K = 128
    rfft_real_kernel[grid](x_rows[0], real_out_flat, N, seqlen, BLOCK_K, num_warps=4)
    rfft_imag_kernel[grid](x_rows[0], imag_out_flat, N, seqlen, BLOCK_K, num_warps=4)

    # Reshape back to (batch, channels, seqlen+1)
    real_out = real_out_flat.view(batch, channels, seqlen + 1)
    imag_out = imag_out_flat.view(batch, channels, seqlen)
    # imag_out[0] and imag_out[seqlen] are zero by symmetry; set them explicitly
    # Create zeros tensors and concat to imag_out
    zero_a = torch.zeros((batch, channels, 1), device=x.device, dtype=torch.float32)
    zero_b = torch.zeros((batch, channels, 1), device=x.device, dtype=torch.float32)
    imag_out = torch.cat([zero_a, imag_out, zero_b], dim=2)

    return real_out, imag_out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Cast to float32 for numerical stability (original code does x.to(torch.float32))
        x_f32 = x.to(torch.float32)
        real_out, imag_out = _run_triton_rfft(x_f32)
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
