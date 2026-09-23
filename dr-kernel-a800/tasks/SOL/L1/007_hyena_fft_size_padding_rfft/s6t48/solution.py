import math
import torch
import triton
import triton.language as tl

@triton.jit
def rfft_real_direct_kernel(
    x_ptr,            # *float32, input x flattened over (b,c) with elements per (b,c) = S
    out_real_ptr,     # *float32, output real part (B,C,S+1) flattened over (b,c)
    out_imag_ptr,     # *float32, output imag part (B,C,S+1) flattened over (b,c)
    S: tl.int32,      # length of x per (b,c)
):
    bc = tl.program_id(0)  # one program per (b,c)
    # Compute sum_x once
    sum_x = 0.0
    i = 0
    while i < S:
        xi = tl.load(x_ptr + bc * S + i)
        sum_x += xi
        i += 1

    N = 2 * S

    # k = 0 term: y[0] = sum_x / N, imag = 0
    tl.store(out_real_ptr + bc * (S + 1) + 0, sum_x / N)
    tl.store(out_imag_ptr + bc * (S + 1) + 0, 0.0)

    # k from 1..S
    k = 1
    while k <= S:
        theta = math.pi * k / N
        cos_t = tl.cos(theta)
        sin_t = tl.sin(theta)

        # Determine parity of k
        is_even = (k % 2) == 0

        if is_even:
            # y[k] = (sum_x * cos - sum_x * sin) / N, purely real
            val = (sum_x * cos_t - sum_x * sin_t) / N
            tl.store(out_real_ptr + bc * (S + 1) + k, val)
            tl.store(out_imag_ptr + bc * (S + 1) + k, 0.0)
        else:
            # y[k] = purely imaginary: 0 + i * (-sum_x * sin) / N
            val = -(sum_x * sin_t) / N
            tl.store(out_real_ptr + bc * (S + 1) + k, 0.0)
            tl.store(out_imag_ptr + bc * (S + 1) + k, val)

        k += 1

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, S) float32 tensor on CUDA.
        Returns:
          real_out: (B, C, S+1) float32
          imag_out: (B, C, S+1) float32
        """
        assert x.is_cuda, "Input must be a CUDA tensor for Triton."
        assert x.dtype == torch.float32, "Input must be float32."
        B, C, S = x.shape
        # Ensure contiguous for simple stride math
        x = x.contiguous()

        # Allocate outputs
        out_real = torch.empty((B, C, S + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, S + 1), device=x.device, dtype=torch.float32)

        # Launch one program per (b, c)
        grid = (B * C,)
        rfft_real_direct_kernel[grid](
            x.view(B * C, S),               # flatten per (b,c)
            out_real.view(B * C, S + 1),   # flatten per (b,c)
            out_imag.view(B * C, S + 1),   # flatten per (b,c)
            S,
        )
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
