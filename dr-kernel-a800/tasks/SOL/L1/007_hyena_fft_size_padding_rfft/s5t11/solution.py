import torch
import triton
import triton.language as tl


@triton.jit
def cosine_sum_kernel(x_ptr, out_ptr, j, N, L):
    """
    Compute real_rfft[j] for j in [0, L), where L=seqlen+1 and N=2*seqlen.
    real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    x_ptr: 1D tensor of length N (contiguous), first seqlen elements are input x, rest zeros.
    out_ptr: 1D tensor of length L, per-(b,c) row output. We store at index j.
    """
    acc = 0.0
    k = 0
    while k < N:
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.cos(angle)
        k += 1
    # Normalize by N (original code divides by 2*seqlen; here N == 2*seqlen)
    acc *= 1.0 / N
    tl.store(out_ptr + j, acc)


@triton.jit
def sine_sum_kernel(x_ptr, out_ptr, j, N, L):
    """
    Compute imag_rfft[j] for j in [1, L-2], where L=seqlen+1 and N=2*seqlen.
    imag_out[j] = (1/N) * sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N)
    x_ptr: 1D tensor of length N (contiguous).
    out_ptr: 1D tensor of length L. We store at index j.
    """
    acc = 0.0
    k = 0
    while k < N:
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.sin(angle)
        k += 1
    acc *= 1.0 / N
    tl.store(out_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          x_freq = torch.fft.rfft(x, n=2*seqlen) / (2*seqlen)
          return x_freq.real, x_freq.imag, both shape (batch, channels, seqlen+1)
        """
        # Ensure float32 and contiguous input
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        batch, channels, seqlen = x.shape
        N = 2 * seqlen  # padded length for rfft
        L = seqlen + 1  # output length

        # Allocate outputs
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # For each (batch, channel) row, create padded 1D input of length N
        # and launch Triton kernels to compute real and imaginary parts.
        for b in range(batch):
            for c in range(channels):
                # Flatten the row to 1D (length seqlen)
                row_vec = x[b, c, :].contiguous()  # shape [seqlen], dtype float32
                # Pad to N: zeros in the second half
                if N > seqlen:
                    pad = torch.zeros(N - seqlen, dtype=torch.float32, device=x.device)
                    x_row_padded = torch.cat([row_vec, pad], dim=0)  # shape [N]
                else:
                    x_row_padded = row_vec  # defensive: N should be >= seqlen

                # Flatten outputs for this row to length L
                out_flat_real = real_out[b, c, :].contiguous().view(L)
                out_flat_imag = imag_out[b, c, :].contiguous().view(L)

                # Launch cosine kernel for j in 0..seqlen
                for j in range(L):
                    cosine_sum_kernel[(1,)](x_row_padded, out_flat_real, j, N, L)

                # Launch sine kernel for j in 1..seqlen-1
                for j in range(1, seqlen):
                    sine_sum_kernel[(1,)](x_row_padded, out_flat_imag, j, N, L)
                # Set j=0 and j=seqlen to zero (imag part is zero for real inputs at these positions)
                imag_out[b, c, 0] = 0.0
                imag_out[b, c, seqlen] = 0.0

        # The normalization is already applied in the kernels (divide by N).
        # The original code divides by 2*seqlen. Since N == 2*seqlen here, we are consistent.

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
