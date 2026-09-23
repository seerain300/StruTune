import torch
import triton
import triton.language as tl


@triton.jit
def cosine_j_kernel(x_ptr, real_out_ptr, j, M: tl.constexpr, N: tl.constexpr):
    """
    Compute rfft real part for a single index j, for a given row.
    Input x_ptr has length M (original seqlen).
    real_out_ptr points to the per-(b, c) row of length L_out = M + 1.
    """
    acc = 0.0
    # Sum over original input length k=0..M-1. Note: original input has length M, no padding needed in k-summation
    # because rfft implicitly pads zeros in frequency domain, but here we compute bins from original signal only.
    for k in range(0, M):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        # symmetry term with k -> M - k
        angle_sym = 2.0 * 3.141592653589793 * j * (M - k) / N
        parity = (k % 2)  # 0 or 1
        parity_factor = 1.0 - 2.0 * parity  # +1 if k even, -1 if k odd
        acc += xk * (tl.cos(angle) + parity_factor * tl.cos(angle_sym))
    # Normalize by N (which equals 2*seqlen)
    acc *= 1.0 / N
    # Store into real_out_ptr[j]
    tl.store(real_out_ptr + j, acc)


@triton.jit
def sine_j_kernel(x_ptr, imag_out_ptr, j, M: tl.constexpr, N: tl.constexpr):
    """
    Compute rfft imag part for a single index j, for a given row.
    imag_out_ptr points to the per-(b, c) row of length L_out = M + 1.
    For real input, imag[0] and imag[M] should be set to zero in host.
    """
    acc = 0.0
    for k in range(0, M):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        angle_sym = 2.0 * 3.141592653589793 * j * (M - k) / N
        parity = (k % 2)
        parity_factor = 1.0 - 2.0 * parity
        acc += xk * (tl.sin(angle) - parity_factor * tl.sin(angle_sym))
    acc *= 1.0 / N
    tl.store(imag_out_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation that computes rfft over the last dimension for each (batch, channel),
        normalizes by 2*seqlen, and returns real and imaginary parts separately as float32 tensors
        of shape (batch, channels, seqlen+1).
        """
        # Ensure float32 and contiguous for consistent math
        x = x.to(torch.float32).contiguous()
        batch, channels, seqlen = x.shape
        N = 2 * seqlen
        M = seqlen
        L_out = M + 1

        # Flatten (batch, channels) into rows; each row has length M
        x_flat = x.view(batch * channels, M).contiguous()

        # Allocate outputs for each (b, c) row
        real_out_row = torch.empty((batch * channels, L_out), dtype=torch.float32, device=x.device)
        imag_out_row = torch.empty((batch * channels, L_out), dtype=torch.float32, device=x.device)

        # Launch Triton kernels: one program per (b, c) row, iterate j inside the kernel
        # For real part: compute all j=0..M
        for row in range(0, batch * channels):
            # cosine_j_kernel writes into real_out_row[row, :]
            for j in range(0, M + 1):
                cosine_j_kernel[(1,)](x_flat[row], real_out_row[row], j, M=M, N=N)
            # imag part: compute j=1..M-1; imag[0] and imag[M] will be set to zero in host
            for j in range(1, M):
                sine_j_kernel[(1,)](x_flat[row], imag_out_row[row], j, M=M, N=N)

        # Set imag[0] and imag[M] to zero for real inputs
        imag_out_row[:, 0] = 0.0
        imag_out_row[:, M] = 0.0

        # Reshape back to (batch, channels, seqlen+1)
        real_out = real_out_row.view(batch, channels, L_out)
        imag_out = imag_out_row.view(batch, channels, L_out)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
