import torch
import triton
import triton.language as tl


@triton.jit
def real_cosine_kernel(x_ptr, out_ptr, j, N: tl.constexpr, M: tl.constexpr):
    """
    Compute real_rfft[j] for j in [0, M], where M=seqlen and N=2*seqlen.
    real_rfft[j] = sum_{k=0..2N-1} x[k] * cos(2*pi*j*k/(2N)) / (2N)
    This kernel writes a single scalar to out_ptr (temporary buffer for the row).
    """
    acc = 0.0
    TWO_N = 2 * N
    for k in range(0, TWO_N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / TWO_N
        ck = tl.cos(angle)
        acc += xk * ck
    # Store the result (host will divide by 2*N for normalization)
    tl.store(out_ptr, acc)


@triton.jit
def imag_sine_kernel(x_ptr, out_ptr, j, N: tl.constexpr, M: tl.constexpr):
    """
    Compute imaginary rfft for j in [1, M-1]:
    imag_rfft[j] = sum_{k=0..2N-1} x[k] * sin(2*pi*j*k/(2N)) / (2N)
    This kernel writes a single scalar to out_ptr (temporary buffer for the row).
    """
    acc = 0.0
    TWO_N = 2 * N
    for k in range(0, TWO_N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / TWO_N
        sk = tl.sin(angle)
        acc += xk * sk
    # Store the result (host will divide by 2*N for normalization)
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation that computes the real and imaginary parts of
        the FFT for each (batch, channel) row, with n=2*seqlen, normalized by 2*seqlen,
        and returns them separately. Entry point: ModelNew.
        """
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        N = 2 * seqlen  # original code uses n=2*seqlen
        M = seqlen      # number of bins (output length = M+1)

        # Ensure float32 and contiguous
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        # Flatten to (B*C, N)
        B = batch
        C = channels
        x_flat = x.reshape(B * C, N).contiguous()

        # Allocate outputs (B*C, seqlen+1)
        real_out = torch.empty((B * C, M + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B * C, M + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels to compute real parts for j=0..M
        # For each row (b, c), compute and store to real_out[row, j]
        for row in range(B * C):
            x_row_ptr = x_flat[row]
            real_row_ptr = real_out[row]
            # Compute real parts
            for j in range(0, M + 1):
                # Temporary scalar output for this j
                tmp = torch.empty(1, dtype=torch.float32, device=x.device)
                real_cosine_kernel[(1,)](x_row_ptr, tmp, j, N, M)
                # Normalize by 2*N here (matching original code)
                real_row_ptr[j] = tmp[0] / (2.0 * N)

            # Compute imaginary parts for j=1..M-1
            imag_row_ptr = imag_out[row]
            for j in range(1, M):
                tmp = torch.empty(1, dtype=torch.float32, device=x.device)
                imag_sine_kernel[(1,)](x_row_ptr, tmp, j, N, M)
                imag_row_ptr[j] = tmp[0] / (2.0 * N)

            # Set j=0 and j=M imaginary parts to 0 (imaginary part is zero at 0 and Nyquist for real inputs)
            imag_row_ptr[0] = 0.0
            imag_row_ptr[M] = 0.0

        # Reshape back to (batch, channels, seqlen+1)
        real_out = real_out.view(batch, channels, seqlen + 1)
        imag_out = imag_out.view(batch, channels, seqlen + 1)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
