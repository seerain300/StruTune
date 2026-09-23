import torch
import triton
import triton.language as tl


@triton.jit
def cosine_kernel(x_ptr, out_real_ptr, j, N, M, stride_row):
    """
    Compute real_rfft[j] for j in [0, M], where M=seqlen and N=2*seqlen.
    real_out[j] = (1/(2N)) * sum_{k=0..2N-1} x[k] * cos(2*pi*j*k/(2N))
    x_ptr points to a padded vector of length N (float32), where x[k] = x[orig_k] if orig_k<M else 0.
    out_real_ptr points to a flattened output buffer of length (B*C)*(M+1), base index is pid*(M+1).
    """
    acc = 0.0
    TWO_N = 2 * N
    for k in range(0, N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / TWO_N
        acc += xk * tl.cos(angle)
    # Normalize by 2N (original code divides by 2*seqlen)
    acc *= 1.0 / (2.0 * N)
    base = tl.program_id(0) * (M + 1)
    tl.store(out_real_ptr + (base + j), acc)


@triton.jit
def sine_kernel(x_ptr, out_imag_ptr, j, N, M, stride_row):
    """
    Compute imag_rfft[j] for j in [1, M-1], where M=seqlen and N=2*seqlen.
    imag_out[j] = (1/(2N)) * (-) * sum_{k=0..2N-1} x[k] * sin(2*pi*j*k/(2N))
    x_ptr points to a padded vector of length N (float32), where x[k] = x[orig_k] if orig_k<M else 0.
    out_imag_ptr points to a flattened output buffer of length (B*C)*(M+1), base index is pid*(M+1).
    Note: imag[0] and imag[M] must be set to zero by host code.
    """
    acc = 0.0
    TWO_N = 2 * N
    for k in range(0, N):
        xk = tl.load(x_ptr + k)
        angle = 2.0 * 3.141592653589793 * j * k / TWO_N
        acc += xk * tl.sin(angle)
    # Normalize and apply negative sign: imag = (-sum) / (2N)
    acc = -acc * (1.0 / (2.0 * N))
    base = tl.program_id(0) * (M + 1)
    tl.store(out_imag_ptr + (base + j), acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation that matches the original behavior:
        - Compute rfft(x, n=2*seqlen) via sums (with zero-padding to 2*seqlen).
        - Normalize by 2*seqlen.
        - Return real and imaginary parts separately as float32 tensors of shape (B, C, seqlen+1).
        """
        # Ensure float32 for numerical stability
        x = x.to(torch.float32)
        batch, channels, seqlen = x.shape
        N = 2 * seqlen  # FFT length per original code
        M = seqlen

        # Prepare padded input per (batch, channel) row: length N with zeros after seqlen
        rows = batch * channels
        x_padded = []  # list of 1D tensors per row
        for b in range(batch):
            for c in range(channels):
                x_row = x[b, c, :].contiguous()  # (seqlen,)
                pad = torch.zeros(N - seqlen, dtype=torch.float32, device=x.device)
                x_padded.append(torch.cat([x_row, pad], dim=0))  # length N

        # Allocate outputs
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels: one program per (b,c) row
        grid = (rows,)

        # Compute real parts: j = 0..seqlen
        for j in range(M + 1):
            cosine_kernel[grid](x_padded[j], real_out.view(-1), j, N, M, 0)

        # Compute imag parts: j = 1..seqlen-1; set imag[0] and imag[seqlen] to zero after
        for j in range(1, M):
            sine_kernel[grid](x_padded[j], imag_out.view(-1), j, N, M, 0)

        # Set imag[0] and imag[seqlen] to zero for all rows
        imag_out[:, :, 0].zero_()
        imag_out[:, :, M].zero_()

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
