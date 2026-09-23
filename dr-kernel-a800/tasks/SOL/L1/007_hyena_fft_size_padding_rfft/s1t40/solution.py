import triton
import triton.language as tl


@triton.jit
def _compute_rfft_real_imag_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (for grid only)
    C: tl.constexpr,        # channels (for grid only)
    L: tl.constexpr,        # seqlen (runtime int)
    stride_x_b, stride_x_c, stride_x_l,         # strides for x
    stride_out_b, stride_out_c, stride_out_l,   # strides for out
    N: tl.constexpr,        # nfft = 2 * L (runtime int, loop bound)
    M: tl.constexpr,        # output length = L + 1 (runtime int, loop bound)
):
    # Each program handles one (b, c) slice
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over output index j from 0 to M-1
    j = 0
    while j < M:
        # Accumulators for real and imaginary parts
        sum_real = 0.0
        sum_imag = 0.0

        # Loop over input index t from 0 to N-1
        t = 0
        while t < N:
            # Load x[b, c, t]
            x_offset = b * stride_x_b + c * stride_x_c + t * stride_x_l
            x_val = tl.load(x_ptr + x_offset)

            # Compute cos and sin for this j and t
            # angle = 2*pi*j*t / N
            angle = (2.0 * 3.141592653589793) * j * t / N
            c_term = tl.cos(angle)
            s_term = tl.sin(angle)

            # Accumulate
            sum_real += x_val * c_term
            sum_imag += x_val * s_term

            t += 1

        # Normalize by 1/N
        norm = 1.0 / N
        sum_real = sum_real * norm
        sum_imag = sum_imag * norm

        # Store to out[b, c, j] as real and imag
        out_offset = b * stride_out_b + c * stride_out_c + j * stride_out_l
        tl.store(out_real_ptr + out_offset, sum_real)
        tl.store(out_imag_ptr + out_offset, sum_imag)

        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Allocate outputs (float32 to match original .to(torch.float32))
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Strides for input and output (tensors are contiguous by default)
        stride_x_b = x.stride(0)
        stride_x_c = x.stride(1)
        stride_x_l = x.stride(2)

        stride_out_b = out_real.stride(0)
        stride_out_c = out_real.stride(1)
        stride_out_l = out_real.stride(2)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B, C)
        _compute_rfft_real_imag_kernel[grid](
            x, out_real, out_imag,
            B, C, L,
            stride_x_b, stride_x_c, stride_x_l,
            stride_out_b, stride_out_c, stride_out_l,
            N, M,
            num_warps=1,  # keep warps low for small per-program work
            num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
