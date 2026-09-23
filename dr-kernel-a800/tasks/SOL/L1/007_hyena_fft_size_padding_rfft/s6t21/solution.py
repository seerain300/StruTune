import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_imag_kernel(
    out_real_ptr, out_imag_ptr,
    S: tl.int32,
    sum_all: tl.float32,
    stride_out_bc: tl.int32,
):
    # One program per (b, c) row; grid size must be B*C in the launcher.
    # Note: Triton does not support dynamic loops like 'for k in range(...)'.
    # We use vectorized operations to compute all outputs at once.

    M = S + 1  # output length
    k = tl.arange(0, M)  # vector of indices [0..S]
    scale = 1.0 / (2.0 * S)
    sum_scaled = sum_all * scale

    # ang = pi * k / (2*S)
    ang = tl.math.pi * k * scale

    # For rfft of real inputs:
    # y_real[k] = sum_scaled * cos(pi*k/(2*S)) if k is even, else 0
    # y_imag[k] = 0 if k is even, else -sum_scaled * sin(pi*k/(2*S))
    is_even = (k % 2) == 0

    y_real = tl.where(is_even, sum_scaled * tl.cos(ang), 0.0)
    y_imag = tl.where(is_even, 0.0, -sum_scaled * tl.sin(ang))

    # Compute base address for this (b, c) row
    bc = tl.program_id(0)  # 0..(B*C-1)
    base = bc * stride_out_bc

    # Store outputs (contiguous per row of length M)
    tl.store(out_real_ptr + base + k, y_real)
    tl.store(out_imag_ptr + base + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, S)
        x = x.to(torch.float32)
        B, C, S = x.shape

        # Compute sum over all elements per (b, c), using general tensor ops (not torch.fft)
        sum_all = x.reshape(B * C, S).sum(dim=1)  # shape: (B*C,), dtype=float32

        # Allocate outputs
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        rfft_real_imag_kernel[grid](
            out_real, out_imag,
            S,
            sum_all,            # per (b, c) scalar
            out_real.stride(0) * out_real.shape[1]  # stride between (b, c) rows (in elements)
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
