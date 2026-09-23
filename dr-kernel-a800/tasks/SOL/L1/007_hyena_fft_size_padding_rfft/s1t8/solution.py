import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                   # *float32, input pointer to x (B, C, L)
    out_real_ptr,            # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,            # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,         # batch size
    C: tl.constexpr,         # channels
    L: tl.constexpr,         # original seqlen
    stride_b: tl.constexpr,  # stride for batch in x
    stride_c: tl.constexpr,  # stride for channel in x
    stride_t: tl.constexpr,  # stride for time in x
    out_stride_b: tl.constexpr,  # stride for batch in outputs
    out_stride_c: tl.constexpr,  # stride for channel in outputs
    out_stride_m: tl.constexpr    # stride for m (last dim) in outputs
):
    # One program per (b, c) slice
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    # Output length
    M = L + 1

    # Normalization factor: 1 / (2 * L)
    scale = 1.0 / (2.0 * L)

    # Loop over j = 0..M-1
    for j in range(0, M):
        sum_re = 0.0
        sum_im = 0.0

        # Sum over t = 0..2*L-1 (pad length)
        # Note: x only has L elements; the original code constructs the complex FFT for a real input of length 2*L.
        # Here, we explicitly sum over 2*L positions as if zero-padded from the original L.
        # However, since the original x is length L, we treat x beyond L as zero. We can extend x logically to 2*L
        # by padding zeros. To avoid torch ops, we simply skip loading for t >= L and set those contributions to zero.
        # This matches the behavior of rfft on real input of length 2*L, since padding zeros contributes nothing.

        # Sum over t = 0..L-1 (original data)
        for t in range(0, L):
            x_off = b * stride_b + c * stride_c + t * stride_t
            x_val = tl.load(x_ptr + x_off)
            angle = 2.0 * 3.141592653589793 * (j * t) / (2.0 * L)
            sum_re += x_val * tl.cos(angle)
            sum_im += x_val * tl.sin(angle)

        # Apply normalization
        sum_re = sum_re * scale
        sum_im = sum_im * scale

        # Store to outputs
        out_off_real = b * out_stride_b + c * out_stride_c + j * out_stride_m
        out_off_imag = b * out_stride_b + c * out_stride_c + j * out_stride_m
        tl.store(out_real_ptr + out_off_real, sum_re)
        tl.store(out_imag_ptr + out_off_imag, sum_im)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute fused FFT size padding and real FFT computation for Hyena convolution,
        but entirely in Triton. Returns real and imaginary parts separately.

        Input x: (batch, channels, seqlen), any dtype; we cast to float32 for computation.
        Output: (batch, channels, seqlen+1) for both real and imaginary parts.
        """
        # Ensure x is contiguous and cast to float32 for numerical stability
        x = x.contiguous()
        B, C, L = x.shape

        # Allocate outputs
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Get strides (in elements)
        stride_b, stride_c, stride_t = x.stride()
        out_stride_b, out_stride_c, out_stride_m = out_real.stride()

        # Launch one program per (b, c) slice
        grid = (B * C,)

        # Run Triton kernel: all computation inside Triton, no torch ops
        _rfft_real_imag_triton_kernel[grid](
            x, out_real, out_imag,
            B, C, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            num_warps=1,
            num_stages=1
        )

        # Return outputs (already normalized to 1/(2*L))
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
