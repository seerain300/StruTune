import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,           # *const float32, input pointer to x (B, C, L) contiguous linear
    out_real_ptr,    # *float32, output pointer to real part (B, C, L+1) contiguous linear
    out_imag_ptr,    # *float32, output pointer to imag part (B, C, L+1) contiguous linear
    B: tl.constexpr, # batch size (for grid only)
    C: tl.constexpr, # channels (for grid only)
    L: tl.constexpr, # seqlen (runtime value)
    N: tl.constexpr, # N = 2 * L (compile-time loop bound for t)
    M: tl.constexpr, # M = L + 1 (compile-time output length)
):
    # Map program_id(0) -> b, program_id(1) -> c
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Base linear offset for this (b, c) slice in x: x is contiguous with linear index = b*(C*L) + c*L + t
    base_x = b * (C * L) + c * L

    # Precompute scaling factor 1/N
    invN = 1.0 / N

    # For each output index j in 0..M-1
    for j in range(0, M):
        # Accumulators for real and imaginary parts
        re = 0.0
        im = 0.0

        # Sum over t = 0..N-1
        # Compute factor = 2*pi*j/N once, reuse in the loop
        factor = 2.0 * 3.141592653589793 * j / N
        for t in range(0, N):
            x_val = tl.load(x_ptr + base_x + t)  # x[b, c, t]
            angle = factor * t
            re += x_val * tl.cos(angle)
            im += x_val * tl.sin(angle)

        # Normalize by N
        re = re * invN
        im = im * invN

        # Compute base linear offset for output (b, c, j): out is contiguous with linear index = b*(C*M) + c*M + j
        base_out = b * (C * M) + c * M + j

        # Store results
        tl.store(out_real_ptr + base_out, re)
        tl.store(out_imag_ptr + base_out, im)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation: compute real and imaginary parts of rfft
        for each (batch, channel) slice, normalized by 2*seqlen, and return
        real and imaginary parts separately.
        """
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Ensure contiguous input and dtype float32
        x_f32 = x.contiguous().to(torch.float32)

        # Allocate outputs (B, C, M) float32
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x_f32.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x_f32.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B, C)
        _rfft_real_imag_triton_kernel[grid](
            x_f32, out_real, out_imag,
            B, C, L, N, M
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
