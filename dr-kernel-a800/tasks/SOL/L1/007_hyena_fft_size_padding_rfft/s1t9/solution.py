import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (for grid only)
    C: tl.constexpr,        # channels (for grid only)
    L: tl.constexpr,        # seqlen
    stride_b: tl.constexpr, # stride for batch in x
    stride_c: tl.constexpr, # stride for channel in x
    stride_t: tl.constexpr, # stride for time (last dim) in x
    out_stride_b: tl.constexpr,  # stride for batch in outputs
    out_stride_c: tl.constexpr,  # stride for channel in outputs
    out_stride_m: tl.constexpr,  # stride for m (0..L) in outputs
    # N is runtime, but we pass 2*L explicitly:
    N,                      # int32, 2 * L (FFT length)
):
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # channel index

    # Compute base offsets for this (b, c) slice
    x_base = pid_b * stride_b + pid_c * stride_c
    out_base = pid_b * out_stride_b + pid_c * out_stride_c

    # We will compute re_j and im_j for j = 0 .. L
    # Direct DFT for real input:
    # re_j = (1/N) * sum_{t=0}^{N-1} x[t] * cos(2*pi*j*t/N)
    # im_j = (1/N) * sum_{t=0}^{N-1} x[t] * sin(2*pi*j*t/N)
    # Note: N = 2*L, M = L+1.

    # Precompute inv_N
    inv_N = 1.0 / N

    # Loop over j from 0 to L; we will store to out_real/out_imag at index j
    for j in range(0, L + 1):
        # Initialize accumulators
        acc_re = 0.0
        acc_im = 0.0

        # Loop over t from 0 to N-1
        # Triton supports runtime loops; this is simple and robust.
        for t in range(0, N):
            # Load x[b, c, t]; x is laid out as (B, C, L) contiguous => offset = x_base + t * stride_t
            # x_ptr is float32*
            x_val = tl.load(x_ptr + x_base + t * stride_t)
            # Compute cos/sin
            # angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * float(j) * float(t) / float(N)
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)
            acc_re += x_val * cos_term
            acc_im += x_val * sin_term

        # Apply normalization
        acc_re = acc_re * inv_N
        acc_im = acc_im * inv_N

        # Store to outputs: out[b, c, j]
        # out strides: out[b, c, m] => base + m * out_stride_m
        tl.store(out_real_ptr + out_base + j * out_stride_m, acc_re)
        tl.store(out_imag_ptr + out_base + j * out_stride_m, acc_im)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only forward: compute rfft for real inputs of length 2*seqlen,
        return real and imaginary parts, both (batch, channels, seqlen+1).
        """
        # x: (B, C, L)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L  # FFT length per original code

        # Ensure input is contiguous for simple pointer arithmetic
        # Note: No torch operations in forward beyond .shape and .contiguous()
        x = x.contiguous()

        # Prepare outputs of shape (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Get strides (in elements)
        stride_b, stride_c, stride_t = x.stride()
        out_stride_b, out_stride_c, out_stride_m = out_real.stride()

        # Launch one program per (b, c) slice
        grid = (B, C)

        _rfft_real_imag_triton_kernel[grid](
            x, out_real, out_imag,
            B, C, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            N,
            num_warps=1,  # simple kernel; 1 warp is sufficient
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
