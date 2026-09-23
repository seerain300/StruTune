import math
import torch
import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    L: tl.constexpr,        # original seqlen
    invN: tl.constexpr,     # 1.0 / (2*L), normalization factor
    C: tl.constexpr,        # number of channels
    B: tl.constexpr,        # batch size
):
    # Each program handles one (b, c) slice: pid = b*C + c
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Compute base offsets for this slice
    # Input x is expected to be contiguous in (B, C, L) so base_in = pid * L
    base_in = pid * L
    M = L + 1

    # We will compute outputs (real and imag) for j = 0..M-1
    # Use a simple loop over j with scalar work; avoids Triton tensor/indexing pitfalls
    for j in range(0, M):
        re_sum = 0.0  # float32 accumulators
        im_sum = 0.0

        # Sum over t from 0 to 2*L - 1 (zero-padding to N = 2*L)
        for t in range(0, 2 * L):
            x_val = tl.load(x_ptr + base_in + t)
            # angle = 2*pi*j*t/(2*L) = pi*j*t/L
            angle = (j * t) * (math.pi * invN)
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)
            re_sum += x_val * cos_term
            im_sum += x_val * sin_term

        # Normalize by 1/(2*L)
        re_sum = re_sum * invN
        im_sum = im_sum * invN

        # Store results for this (b, c) slice at position j
        # Output is contiguous (B, C, M); linear index for (b, c, j) is b*(C*M) + c*M + j
        out_real_index = b * (C * M) + c * M + j
        out_imag_index = b * (C * M) + c * M + j
        tl.store(out_real_ptr + out_real_index, re_sum)
        tl.store(out_imag_ptr + out_imag_index, im_sum)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
          - Implicit zero-padding to 2*seqlen on last dim
          - rfft for real input along last dim
          - normalization by (2*seqlen)
          - return real and imaginary parts separately (float32), shape (B, C, seqlen+1)
        """
        # Input shape: (batch, channels, seqlen)
        batch, channels, seqlen = x.shape

        # Ensure float32; the original code also casts to float32
        # Important: do not use any torch operations like .contiguous() or .view() in forward.
        x_f32 = x.to(torch.float32)

        # We assume x is contiguous along the last dimension in the original code; here we rely
        # on Triton pointer arithmetic with strides. Since x is already in (B, C, L), we treat it
        # as contiguous in that layout. The kernel uses base = pid * L which matches contiguous (B, C, L).
        # Allocate outputs: (batch, channels, seqlen+1) contiguous
        M = seqlen + 1
        out_real = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)

        # Grid: one program per (b, c) slice
        grid = (batch * channels,)

        # Normalization factor: 1 / (2 * seqlen)
        invN = 1.0 / (2 * seqlen)

        # Invoke Triton kernel
        _rfft_real_imag_triton_kernel[grid](
            x_f32, out_real, out_imag,
            seqlen, invN, channels, batch,
            num_warps=1,  # keep simple; kernel is scalar-heavy
            num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
