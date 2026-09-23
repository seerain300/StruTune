import torch
import triton
import triton.language as tl


@triton.jit
def cosine_sum_kernel(x_ptr, out_ptr, j, N, stride_row):
    """
    Compute real_rfft[j] for j in [0, M], where M=seqlen and N=2*seqlen.
    real_out[j] = (1/N) * sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N)
    x_ptr: points to a vector of length N (padded zeros)
    out_ptr: points to a flattened output buffer; we write at out_ptr[base + j],
             where base is the row start offset in the flattened (B*C, M+1) view.
    stride_row: number of columns per row (M+1).
    """
    acc = 0.0
    for k in range(0, N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / N
        acc += xk * tl.cos(angle)
    # Normalize by N (original code divides by 2*seqlen; here N == 2*seqlen)
    acc *= 1.0 / N
    # Write to out_ptr at index base + j. Triton receives out_ptr pointing to row start.
    tl.store(out_ptr + j, acc)


@triton.jit
def sine_sum_kernel(x_ptr, out_ptr, j, N, stride_row):
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
    tl.store(out_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
            x_freq = torch.fft.rfft(x, n=2*seqlen)
            x_freq = x_freq / (2*seqlen)
            return real(x_freq), imag(x_freq)
        Outputs:
            real_out: (B, C, seqlen+1), float32
            imag_out: (B, C, seqlen+1), float32
        """
        assert x.is_cuda, "ModelNew requires CUDA tensors"
        B, C, M = x.shape  # batch, channels, seqlen
        N = 2 * M  # padded length

        # Cast to float32 for numerical stability and consistency with original code
        x_in = x.to(torch.float32)

        # Pad to N with zeros (implicit padding in torch.fft.rfft when n > input_len)
        # x_padded has shape (B, C, N)
        x_padded = torch.zeros((B, C, N), dtype=torch.float32, device=x.device)
        x_padded[:, :, :M] = x_in

        # Allocate outputs (B, C, M+1), zero-initialized
        real_out = torch.zeros((B, C, M + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.zeros((B, C, M + 1), dtype=torch.float32, device=x.device)

        # Flatten for pointer arithmetic: total rows = B * C
        rows = B * C
        stride_row = M + 1  # columns per row

        # Launch cosine sum kernels for j in 0..M (inclusive)
        for j in range(M + 1):  # j = 0..seqlen
            for r in range(rows):
                b = r // C
                c = r % C
                # Compute base offset for this (b,c) row in flattened outputs
                base = b * C * stride_row + c * stride_row

                # Prepare per-row pointers: out tensors flattened to (rows, stride_row)
                out_ptr_real = real_out.view(-1) + base
                out_ptr_imag = imag_out.view(-1) + base

                # x_ptr per row: x_padded[b, c, :]
                x_row_ptr = x_padded[b, c, :]

                # Launch cosine kernel
                cosine_sum_kernel[(1,)](x_row_ptr, out_ptr_real, j, N, stride_row)

                # Imaginary part: only write for j in 1..M-1; imag[0] and imag[M] are zeros (already)
                if 1 <= j <= M - 1:
                    sine_sum_kernel[(1,)](x_row_ptr, out_ptr_imag, j, N, stride_row)

        # Normalize by N = 2 * seqlen (original code divides by 2*seqlen)
        real_out = real_out / float(N)
        imag_out = imag_out / float(N)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
