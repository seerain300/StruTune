import torch
import triton
import triton.language as tl


@triton.jit
def _rfft_real_kernel(x_ptr, out_ptr,
                      B, C, N, M,  # N = 2*seqlen, M = seqlen + 1
                      stride_b, stride_c, stride_t,
                      out_stride_b, out_stride_c, out_stride_m,
                      BLOCK_J: tl.constexpr):
    # One program per (b, c) slice
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    base_x = b * stride_b + c * stride_c

    pi = 3.141592653589793
    inv_N = 1.0 / N

    # Accumulator for re
    re_vec = tl.zeros([M], dtype=tl.float32)

    # Sum over t from 0 to N-1
    for t in range(0, N):
        x_offset = base_x + t * stride_t
        x_val = tl.load(x_ptr + x_offset).to(tl.float32)

        # Compute angles for all j in 0..M-1
        j = tl.arange(0, M)  # vector of j indices
        angle = (2.0 * pi) * (j.to(tl.float32)) * (t.to(tl.float32)) / N
        cosv = tl.cos(angle)

        re_vec += x_val * cosv

    re_vec *= inv_N

    # Store to out_real[b, c, j]
    out_base = b * out_stride_b + c * out_stride_c
    j64 = j.to(tl.int64)
    out_offset = out_base + j64 * out_stride_m
    tl.store(out_ptr + out_offset, re_vec)


@triton.jit
def _rfft_imag_kernel(x_ptr, out_ptr,
                      B, C, N, M,  # N = 2*seqlen, M = seqlen + 1
                      stride_b, stride_c, stride_t,
                      out_stride_b, out_stride_c, out_stride_m,
                      BLOCK_J: tl.constexpr):
    # One program per (b, c) slice
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    base_x = b * stride_b + c * stride_c

    pi = 3.141592653589793
    inv_N = 1.0 / N

    im_vec = tl.zeros([M], dtype=tl.float32)

    for t in range(0, N):
        x_offset = base_x + t * stride_t
        x_val = tl.load(x_ptr + x_offset).to(tl.float32)

        j = tl.arange(0, M)
        angle = (2.0 * pi) * (j.to(tl.float32)) * (t.to(tl.float32)) / N
        sinv = tl.sin(angle)

        im_vec += x_val * sinv

    im_vec *= inv_N

    out_base = b * out_stride_b + c * out_stride_c
    j64 = j.to(tl.int64)
    out_offset = out_base + j64 * out_stride_m
    tl.store(out_ptr + out_offset, im_vec)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        assert x.ndim == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L
        M = L + 1  # rfft length for real input of length N

        # Ensure input is float32 and contiguous along last dim
        x = x.contiguous()
        x = x.to(torch.float32)

        # Allocate outputs: shape (B, C, M)
        out_real = torch.empty((B, C, M), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, M), device=x.device, dtype=torch.float32)

        # Strides (in elements)
        stride_b, stride_c, stride_t = x.stride()
        out_stride_b, out_stride_c, out_stride_m = out_real.stride()

        # Launch one program per (b, c) slice
        grid = (B * C,)

        # Choose BLOCK_J = M (compile-time constant per launch)
        # Triton requires constexpr; passing M is fine since it's known at runtime but Triton will handle this.
        _rfft_real_kernel[grid](
            x, out_real,
            B, C, N, M,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_J=M,
        )

        _rfft_imag_kernel[grid](
            x, out_imag,
            B, C, N, M,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_J=M,
        )

        # The original code divides by N (=2*seqlen). We applied 1/N inside the kernels.
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
