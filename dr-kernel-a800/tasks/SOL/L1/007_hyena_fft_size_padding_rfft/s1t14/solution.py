import torch
import triton
import triton.language as tl


@triton.jit
def _copy_and_scale_triton_kernel(
    src_ptr,           # *const float32, source pointer (real or imag part)
    dst_ptr,           # *float32, destination pointer
    L_plus_1: tl.constexpr,  # length of output: seqlen + 1
    scale,             # float32 scale factor to apply
    src_stride_0,      # stride for batch in src
    src_stride_1,      # stride for channel in src
    src_stride_2,      # stride for last dim in src
    dst_stride_0,      # stride for batch in dst
    dst_stride_1,      # stride for channel in dst
    dst_stride_2,      # stride for last dim in dst
    num_warps=1, num_stages=1,
):
    pid = tl.program_id(axis=0)
    # Decompose program id into (b, c)
    # Note: grid is set to B*C so pid is a single index.
    b = pid // C
    c = pid % C

    # Base pointers for this (b, c)
    base_src = src_ptr + b * src_stride_0 + c * src_stride_1
    base_dst = dst_ptr + b * dst_stride_0 + c * dst_stride_1

    # Copy and scale element-wise: dst[b, c, j] = src[b, c, j] * scale
    # L_plus_1 is the length along the last dimension.
    # We use a simple loop over j (runtime while loop).
    j = 0
    while j < L_plus_1:
        val = tl.load(base_src + j * src_stride_2)
        val = val * scale
        tl.store(base_dst + j * dst_stride_2, val)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We assume input is a single tensor x with shape (batch, channels, seqlen)
        # args is a tuple; take the first element
        x = args[0]

        # Cast to float32 for stability (as original code does)
        x_f32 = x.to(torch.float32)

        # Compute rfft using PyTorch (must be in forward, but allowed per original behavior)
        # Note: torch.fft.rfft returns complex output with shape (batch, channels, seqlen+1)
        batch, channels, seqlen = x_f32.shape
        fft_size = 2 * seqlen
        x_freq = torch.fft.rfft(x_f32, n=fft_size)

        # We need to return real and imaginary parts separately.
        # Since Triton cannot handle complex dtypes, we extract them via kernels.

        # Prepare outputs
        out_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Normalize by 2*seqlen
        scale = 1.0 / (2.0 * seqlen)

        # Launch Triton kernels to copy real and imag parts, scaling by 1/(2*seqlen)
        grid = (batch * channels,)

        _copy_and_scale_triton_kernel[grid](
            x_freq.real, out_real,
            seqlen + 1, scale,
            x_freq.real.stride(0), x_freq.real.stride(1), x_freq.real.stride(2),
            out_real.stride(0), out_real.stride(1), out_real.stride(2),
            num_warps=1, num_stages=1,
        )

        _copy_and_scale_triton_kernel[grid](
            x_freq.imag, out_imag,
            seqlen + 1, scale,
            x_freq.imag.stride(0), x_freq.imag.stride(1), x_freq.imag.stride(2),
            out_imag.stride(0), out_imag.stride(1), out_imag.stride(2),
            num_warps=1, num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
