import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_kernel(
    x_ptr,            # *const float32, input pointer to x with shape (B, C, L)
    out_real_ptr,     # *float32, output pointer to real part (B, C, L+1) flattened
    out_imag_ptr,     # *float32, output pointer to imag part (B, C, L+1) flattened
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    L: tl.constexpr,  # seqlen
    N: tl.constexpr,  # N = 2 * L
    M: tl.constexpr,  # M = L + 1
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Base offsets for this (b, c) slice
    base_in = (b * C + c) * L
    base_out = (b * C + c) * M

    # Loop over j = 0..M-1
    j = 0
    while j < M:
        # Accumulators for this j
        acc_real = tl.zeros((), dtype=tl.float32)
        acc_imag = tl.zeros((), dtype=tl.float32)

        # Sum over t = 0..N-1
        t = 0
        while t < N:
            # Load x[b, c, t] as scalar
            index_in = base_in + t
            x_val = tl.load(x_ptr + index_in)

            # Compute angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * (j * t) / N

            # Accumulate real and imaginary parts
            acc_real += x_val * tl.cos(angle)
            acc_imag += x_val * tl.sin(angle)

            t += 1

        # Normalize by 1/N (i.e., 1/(2*seqlen))
        inv_N = 1.0 / N
        acc_real *= inv_N
        acc_imag *= inv_N

        # Store results to out[b, c, j]
        tl.store(out_real_ptr + base_out + j, acc_real)
        tl.store(out_imag_ptr + base_out + j, acc_imag)

        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Ensure x is contiguous and on CUDA for Triton
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()

        # Allocate outputs; flatten to 1D contiguous for kernel
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B, C)
        _rfft_real_imag_kernel[grid](
            x, out_real, out_imag,
            B, C, L, N, M,
        )

        # Reshape back to (B, C, M)
        out_real = out_real.view(B, C, M)
        out_imag = out_imag.view(B, C, M)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
