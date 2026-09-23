import torch
import triton
import triton.language as tl

@triton.jit
def real_fft_normalize_kernel_2d(x_ptr, out_real_ptr, out_imag_ptr,
                                 seqlen: tl.int32, N: tl.int32, M: tl.int32, C: tl.int32):
    """
    Triton kernel computing real FFT for each (batch, channel) slice and normalizing by 1/(2*seqlen).
    It writes both real and imaginary parts into out_real_ptr and out_imag_ptr of shape (B, C, M),
    where M = seqlen + 1. Grid: (B, C), one program per (b, c) slice.
    """
    b = tl.program_id(0)  # batch index
    c = tl.program_id(1)  # channel index

    # Base index along the last dimension for this (b, c) slice
    base = (b * C + c) * seqlen

    # Precompute normalization factor: original divides by 2*seqlen
    inv_n = 1.0 / (2.0 * seqlen)

    # Loop over k from 0 to M-1
    k = 0
    while k < M:
        re_k = 0.0
        im_k = 0.0

        # Vectorize over t in chunks of BLOCK_T
        BLOCK_T = 256
        t_start = 0
        while t_start < N:
            t_vec = t_start + tl.arange(0, BLOCK_T)
            mask = t_vec < N

            # Load x[t] for this (b, c) slice as a vector
            x_vec = tl.load(x_ptr + base + t_vec, mask=mask, other=0.0)

            # Compute angles for all t in this chunk: angle = 2*pi*k*t/N
            angle = 2.0 * 3.141592653589793 * float(k) * t_vec / float(N)

            # Accumulate contributions: re_k += sum(x_vec * cos(angle)), im_k += sum(x_vec * sin(angle))
            re_k += tl.sum(x_vec * tl.cos(angle), axis=0)
            im_k += tl.sum(x_vec * tl.sin(angle), axis=0)

            t_start += BLOCK_T

        # Apply normalization: original divides by 2*seqlen, so multiply by 1/(2*seqlen)
        re_k = re_k * inv_n
        im_k = im_k * inv_n

        # Compute linear index for out[b, c, k] in a contiguous (B, C, M) tensor:
        # offset = b*C*M + c*M + k, where M = seqlen + 1
        out_base = b * C * M + c * M + k

        # Store results
        tl.store(out_real_ptr + out_base, re_k)
        tl.store(out_imag_ptr + out_base, im_k)

        k += 1

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape (batch, channels, seqlen)
        x = args[0]
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Cast to float32 and ensure contiguous
        x_f32 = x.to(torch.float32).contiguous()

        # Compute N and M according to original: n = 2*seqlen, M = seqlen + 1
        N = 2 * seqlen
        M = seqlen + 1

        # Allocate outputs: real and imaginary parts, shape (batch, channels, seqlen+1)
        out_real = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel with 2D grid: (batch, channels), one program per (b, c) slice
        grid = (batch, channels)
        real_fft_normalize_kernel_2d[grid](
            x_f32, out_real, out_imag,
            seqlen=seqlen, N=N, M=M, C=channels,
            num_warps=4, num_stages=2
        )

        # Return real and imaginary parts as per original function
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
