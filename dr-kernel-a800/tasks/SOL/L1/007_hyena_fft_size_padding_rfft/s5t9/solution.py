import torch
import triton
import triton.language as tl


@triton.jit
def cosine_sum_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    """
    For each j in 0..M, compute:
      real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    out_ptr points to a 1D buffer of length M+1; we write to index j.
    """
    for j in range(0, M + 1):
        acc = 0.0
        # Sum over k = 0..N-1
        for k in range(0, N):
            xk = tl.load(x_ptr + k)
            angle = 2.0 * 3.141592653589793 * j * k / N
            acc += xk * tl.cos(angle)
        acc *= 1.0 / N
        # Store into out_ptr[j]
        tl.store(out_ptr + j, acc)


@triton.jit
def sine_sum_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    """
    For each j in 1..M-1, compute:
      imag_out[j] = (1/N) * sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N)
    out_ptr points to a 1D buffer of length M+1; we write to index j.
    """
    for j in range(1, M):  # We will set j=0 and j=M separately after kernel
        acc = 0.0
        # Sum over k = 0..N-1
        for k in range(0, N):
            xk = tl.load(x_ptr + k)
            angle = 2.0 * 3.141592653589793 * j * k / N
            acc += xk * tl.sin(angle)
        acc *= 1.0 / N
        # Store into out_ptr[j]
        tl.store(out_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Computes real and imaginary parts of rfft(x, n=2*seqlen) normalized by 2*seqlen.
        - Returns (batch, channels, seqlen+1) real and imag tensors.
        """
        # Ensure dtype and flatten
        B, C, M = x.shape
        x_f32 = x.to(torch.float32)
        BC = B * C

        # We need to compute per (batch, channel) row. To avoid strided addressing in Triton,
        # we concatenate all rows into one long vector of length BC*M and launch kernels per row.
        # Create a flat view without copying (since we will launch per-row).
        # We'll create a flat source and dest buffers. But each Triton launch must read x_row
        # from the original tensor. Triton kernels can read from x_ptr directly if we pass it.
        # So we will use x_f32.contiguous() and for each row, copy that row into a temp vector
        # (but to avoid extra memory, we read directly from x_f32). Triton can accept pointer
        # arithmetic, but looping with dynamic pointers is not supported; hence we copy per row.
        # To keep it simple and correct, we make a 2D contiguous tensor [BC, M] for rows.

        # Make 2D contiguous: [BC, M]
        x2d = x_f32.reshape(BC, M).contiguous()

        # Allocate outputs for each row: real_out[M+1], imag_out[M+1]
        real_out_rows = torch.empty((BC, M + 1), dtype=torch.float32, device=x.device)
        imag_out_rows = torch.empty((BC, M + 1), dtype=torch.float32, device=x.device)

        # For each row, invoke Triton kernels
        for r in range(BC):
            x_row_ptr = x2d[r, :].contiguous()  # pointer to this row's data
            # N = 2 * M
            N = 2 * M

            # Launch cosine_sum_kernel to compute real_out for j in 0..M
            # We pass the pointer to the row data; inside kernel, it iterates k over N.
            # Note: Triton loops require compile-time bounds; here M and N are tl.constexpr via arguments.
            cosine_sum_kernel[(1,)](x_row_ptr, real_out_rows[r, :], M=M, N=N)

            # Launch sine_sum_kernel to compute imag_out for j in 1..M-1
            sine_sum_kernel[(1,)](x_row_ptr, imag_out_rows[r, :], M=M, N=N)

            # Set imag[0] = 0 and imag[M] = 0 (rfft for real inputs has zero at those bins)
            imag_out_rows[r, 0] = 0.0
            imag_out_rows[r, M] = 0.0

        # Reshape back to (B, C, M+1)
        real_out = real_out_rows.view(B, C, M + 1)
        imag_out = imag_out_rows.view(B, C, M + 1)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
