import torch
import triton
import triton.language as tl


@triton.jit
def real_bin_kernel(x_row_ptr, out_ptr, j, N):
    """
    Compute real_rfft[j] for a single row:
    real_rfft[j] = sum_{k=0..N-1} x_row[k] * cos(2*pi*j*k/N) / N
    x_row_ptr: pointer to a 1D padded row of length N (float32), zeros beyond M.
    out_ptr: pointer to a single-element buffer where the result will be stored.
    """
    acc = 0.0
    # Sum over k = 0..N-1
    for k in range(0, N):
        xk = tl.load(x_row_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.cos(angle)
    # Normalize by N (original divides by 2*seqlen == N)
    acc *= 1.0 / N
    tl.store(out_ptr, acc)


@triton.jit
def imag_bin_kernel(x_row_ptr, out_ptr, j, N):
    """
    Compute imag_rfft[j] for a single row:
    imag_rfft[j] = sum_{k=0..N-1} x_row[k] * sin(2*pi*j*k/N) / N
    x_row_ptr: pointer to a 1D padded row of length N (float32), zeros beyond M.
    out_ptr: pointer to a single-element buffer where the result will be stored.
    """
    acc = 0.0
    for k in range(0, N):
        xk = tl.load(x_row_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.sin(angle)
    acc *= 1.0 / N
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation that computes real and imaginary parts of
        rfft(x, n=2*seqlen), normalized by 2*seqlen, and returns them as float32
        tensors of shape (batch, channels, seqlen+1).
        """
        # Cast to float32 (original code casts)
        x_f32 = x.to(torch.float32)
        B, C, M = x_f32.shape  # M = seqlen
        N = 2 * M  # implicit zero-padding length for rfft
        half = N // 2  # equals M

        # Prepare padded input for each (b, c) row: length N, zeros beyond M
        x_row_padded = torch.zeros((B * C, N), dtype=torch.float32, device=x.device)
        # Fill first M elements with x[b, c, :]
        x_flat = x_f32.view(B * C, M)
        x_row_padded[:, :M] = x_flat

        # Output tensors: (B, C, M+1)
        real_out = torch.empty((B, C, M + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B, C, M + 1), dtype=torch.float32, device=x.device)

        # Compute real parts for j = 0..M
        for j in range(0, M + 1):
            for b in range(B):
                for c in range(C):
                    base = b * C + c
                    x_row_1d = x_row_padded[base]  # get 1D view of the row
                    out_buf = torch.empty(1, dtype=torch.float32, device=x.device)
                    real_bin_kernel[(1,)](x_row_1d, out_buf, j, N)
                    real_out[b, c, j] = out_buf[0]

        # Compute imaginary parts for j = 1..M-1
        for j in range(1, M):
            for b in range(B):
                for c in range(C):
                    base = b * C + c
                    x_row_1d = x_row_padded[base]
                    out_buf = torch.empty(1, dtype=torch.float32, device=x.device)
                    imag_bin_kernel[(1,)](x_row_1d, out_buf, j, N)
                    imag_out[b, c, j] = out_buf[0]

        # Imaginary parts at j=0 and j=M are zero for real inputs
        imag_out[:, :, 0] = 0
        imag_out[:, :, M] = 0

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
