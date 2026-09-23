import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size
    C: tl.constexpr,        # channels
    L: tl.constexpr,        # seqlen (input last dim)
):
    # One program per (batch, channel) slice
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Compute N and M
    N = 2 * L
    M = L + 1

    # Base offset for input and output for this (b, c) slice
    base_in = (b * C + c) * L
    base_out = (b * C + c) * M

    # Prepare accumulators for real and imaginary parts
    re = tl.zeros([M], dtype=tl.float32)
    im = tl.zeros([M], dtype=tl.float32)

    # Loop over t from 0 to N-1; read x[t] and accumulate
    for t in range(0, N):
        x_val = tl.load(x_ptr + base_in + t)  # x_val is float32
        for j in range(0, M):
            angle = 2.0 * 3.141592653589793 * (j * t) / N
            re_j = x_val * tl.cos(angle)
            im_j = x_val * tl.sin(angle)
            re[j] += re_j
            im[j] += im_j

    # Normalize by N
    invN = 1.0 / N
    re = re * invN
    im = im * invN

    # Store results (no torch ops)
    for j in range(0, M):
        tl.store(out_real_ptr + base_out + j, re[j])
        tl.store(out_imag_ptr + base_out + j, im[j])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of fused FFT size padding and real FFT computation.
        Computes equivalent of torch.fft.rfft(x, n=2*seqlen) normalized by 2*seqlen and returns real and imaginary parts
        as separate tensors of shape (batch, channels, seqlen+1).
        """
        # x: (B, C, L)
        B, C, L = x.shape

        # Allocate outputs (B, C, L+1) on the same device as x
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B, C)
        _rfft_real_imag_triton_kernel[grid](
            x, out_real, out_imag,
            B=B, C=C, L=L,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
