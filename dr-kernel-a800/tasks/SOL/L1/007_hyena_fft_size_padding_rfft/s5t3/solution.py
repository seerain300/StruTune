import torch
import triton
import triton.language as tl


@triton.jit
def cosine_sum_kernel(x_ptr, out_ptr, j, N, M, stride_row):
    """
    Compute real_rfft[j] for j in [0, M], where M=seqlen and N=2*seqlen.
    real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    Writes to out_ptr at index base + j, where base = b * stride_row + c * (M+1).
    """
    acc = 0.0
    # Sum over k = 0..N-1 (x_ptr is length N, padded zeros already)
    for k in range(0, N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.cos(angle)
    # Normalize by N (original code divides by 2*seqlen, here N == 2*seqlen)
    acc *= 1.0 / N
    # Store result at out_ptr[base + j]
    tl.store(out_ptr + (j + 0), acc)  # out_ptr is per-(b,c) row; we'll pass base via index math in forward


@triton.jit
def sine_sum_kernel(x_ptr, out_ptr, j, N, M, stride_row):
    """
    Compute imag_rfft[j] for j in [1, M-1], where M=seqlen and N=2*seqlen.
    imag_out[j] = (1/N) * sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N)
    Writes to out_ptr at index base + j.
    """
    acc = 0.0
    for k in range(0, N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.sin(angle)
    acc *= 1.0 / N
    tl.store(out_ptr + (j + 0), acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: input tensor of shape (batch, channels, seqlen), float32 on CUDA.
        Returns:
          real_out: (batch, channels, seqlen+1), float32
          imag_out: (batch, channels, seqlen+1), float32
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().to(torch.float32)

        B, C, M = x.shape
        N = 2 * M
        L = M + 1

        # Allocate outputs as (B, C, L) and flatten to (B*C, L) for Triton writes
        real_out = torch.empty((B, C, L), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B, C, L), dtype=torch.float32, device=x.device)
        real_out_flat = real_out.view(B * C, L)
        imag_out_flat = imag_out.view(B * C, L)

        # For each (batch, channel) row, compute real and imag parts via Triton
        for b in range(B):
            for c in range(C):
                base = b * C * L + c * L
                # Zero-pad x[b, c, :] to N
                row = x[b, c, :].contiguous().to(torch.float32)  # shape (M,)
                x_row_padded = torch.zeros(N, dtype=torch.float32, device=x.device)
                x_row_padded[:M] = row  # first M elements are original row; rest zeros

                # Compute real part j in 0..M
                for j in range(L):
                    cosine_sum_kernel[(1,)](x_row_padded, real_out_flat, j, N, M, base)

                # Compute imag part j in 1..M-1; set j=0 and j=M to zero after
                for j in range(1, M):
                    sine_sum_kernel[(1,)](x_row_padded, imag_out_flat, j, N, M, base)

        # Set imag_out[0] and imag_out[M] to zeros (imaginary part at 0 and Nyquist for real inputs)
        imag_out[:, :, 0] = 0
        imag_out[:, :, M] = 0

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
