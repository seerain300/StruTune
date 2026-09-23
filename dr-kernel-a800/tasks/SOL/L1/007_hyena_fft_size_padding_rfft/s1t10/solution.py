import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (unused in kernel body, passed for clarity)
    C: tl.constexpr,        # channels (unused in kernel body, passed for clarity)
    L: tl.constexpr,        # seqlen (input length)
    N: tl.constexpr,        # total length for FFT (2 * L)
    M: tl.constexpr,        # output length (L + 1)
):
    # Each program handles one (b, c) slice
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Base offsets for this (b, c) slice
    # Input x is contiguous with layout (B, C, L)
    base_in = (b * C + c) * L

    # Output tensors are contiguous with layout (B, C, M)
    base_out = b * C * M + c * M

    # Constants
    inv_N = 1.0 / N
    pi = 3.141592653589793

    # Compute real and imaginary parts for j = 0..M-1
    for j in range(0, M):
        re = 0.0
        im = 0.0
        # Sum over t = 0..N-1
        for t in range(0, N):
            # Load x[t] for this slice
            x_val = tl.load(x_ptr + base_in + t)  # x is float32 by design
            # angle = 2*pi*j*t/N
            angle = (2.0 * pi) * (j * t) * inv_N
            cosv = tl.cos(angle)
            sinv = tl.sin(angle)
            # Accumulate
            re += x_val * cosv
            im += x_val * sinv

        # Normalize by N (as per original: divide by n=2*L)
        re *= inv_N
        im *= inv_N

        # Store to outputs
        tl.store(out_real_ptr + base_out + j, re)
        tl.store(out_imag_ptr + base_out + j, im)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only forward that computes real and imaginary parts of rfft(x, n=2*L),
        normalized by 2*L, and returns two float32 tensors of shape (B, C, L+1).
        """
        # x shape: (B, C, L)
        assert x.dim() == 3, "Input must be 3D (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Ensure input is contiguous and float32 on CUDA
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        assert x.is_cuda, "Input tensor must be on CUDA for Triton execution"

        # Allocate outputs
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: grid over (B, C)
        grid = (B, C)
        _rfft_real_imag_triton_kernel[grid](
            x, out_real, out_imag,
            B, C, L, N, M,
            num_warps=1,  # simple kernel; tune as needed
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
