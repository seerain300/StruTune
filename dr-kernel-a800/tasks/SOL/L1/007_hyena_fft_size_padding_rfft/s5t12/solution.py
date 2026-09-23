import torch
import triton
import triton.language as tl


@triton.jit
def cosine_sum_kernel(x_ptr, out_ptr, j, N, stride_row):
    """
    Compute real_rfft[j] for j in [0, seqlen], where N=2*seqlen and L=seqlen+1.
    real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    out_ptr points to the start of output for this (b, c) row; we store at offset j.
    stride_row is the number of columns (seqlen+1) for this flattened row.
    """
    acc = 0.0
    # Sum over k = 0..N-1
    k = 0
    while k < N:
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.cos(angle)
        k += 1
    # Normalize by N (original code divides by 2*seqlen; here N == 2*seqlen)
    acc *= 1.0 / N
    # Store result at out_ptr + j
    tl.store(out_ptr + j, acc)


@triton.jit
def sine_sum_kernel(x_ptr, out_ptr, j, N, stride_row):
    """
    Compute imag_rfft[j] for j in [1, seqlen-1], where N=2*seqlen and L=seqlen+1.
    imag_out[j] = (1/N) * sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N)
    out_ptr points to the start of output for this (b, c) row; we store at offset j.
    """
    acc = 0.0
    # Sum over k = 0..N-1
    k = 0
    while k < N:
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.sin(angle)
        k += 1
    # Normalize by N
    acc *= 1.0 / N
    # Store result at out_ptr + j
    tl.store(out_ptr + j, acc)


@triton.jit
def zero_imag_ends_kernel(imag_ptr, L):
    """
    Set imag[0] and imag[L-1] (Nyquist) to zero for this row.
    imag_ptr points to start of imag buffer for this row (flattened).
    L = seqlen + 1.
    """
    # Write zero to index 0
    tl.store(imag_ptr + 0, 0.0)
    # Write zero to index L-1
    tl.store(imag_ptr + (L - 1), 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen), float32, on CUDA.
        Returns:
          real_out: (batch, channels, seqlen+1), float32
          imag_out: (batch, channels, seqlen+1), float32
        """
        assert x.is_cuda, "ModelNew.forward expects a CUDA tensor"
        assert x.dtype == torch.float32, "Input must be float32"
        batch, channels, seqlen = x.shape
        N = 2 * seqlen  # padded length
        L = seqlen + 1  # output length

        # Allocate outputs. Flatten to (B*C, L) for easier per-row writes.
        real_out = torch.empty((batch, channels, L), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L), dtype=torch.float32, device=x.device)

        # For each (b, c), launch Triton kernels to fill the row.
        for b in range(batch):
            for c in range(channels):
                # Create a contiguous padded input vector of length N: first seqlen elements are x[b,c,:], rest zeros.
                # Avoid torch.cat to prevent torch compute; use torch.empty + assignment.
                x_row = torch.empty(N, dtype=torch.float32, device=x.device)
                x_row[:seqlen] = x[b, c, :]
                x_row[seqlen:] = 0.0

                # Compute real part: j in 0..seqlen
                for j in range(L):
                    # Launch cosine_sum_kernel for this j
                    cosine_sum_kernel[(1,)](x_row, real_out.view(-1), j, N, L)

                # Compute imag part: j in 1..seqlen-1
                for j in range(1, seqlen):
                    sine_sum_kernel[(1,)](x_row, imag_out.view(-1), j, N, L)

                # Set imag[0] and imag[seqlen] (Nyquist) to zero
                zero_imag_ends_kernel[(1,)](imag_out.view(-1), L)

        # Return results
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
