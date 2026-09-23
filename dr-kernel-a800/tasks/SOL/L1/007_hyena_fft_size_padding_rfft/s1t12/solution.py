import torch
import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, flattened input pointer (B*C*L elements)
    out_real_ptr,           # *float32, flattened output pointer (B*C*(L+1) elements)
    out_imag_ptr,           # *float32, flattened output pointer (B*C*(L+1) elements)
    L: tl.constexpr,        # seqlen (last dim length)
    N,                      # int, total input length (2*seqlen)
    M,                      # int, output length (L+1)
    invN,                   # float32, 1.0 / N
):
    # Each Triton program handles one (b, c) slice. We launch grid=(B*C,).
    pid = tl.program_id(axis=0)
    # Base offsets for this slice in flattened views
    base_in = pid * L
    base_out = pid * M

    # Iterate over j = 0..M-1
    j = 0
    while j < M:
        acc_re = 0.0
        acc_im = 0.0

        # Iterate over t = 0..N-1
        t = 0
        while t < N:
            # Load x[t] for this (b, c) slice
            x_val = tl.load(x_ptr + base_in + t)
            # Compute angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * j * t * invN
            # Accumulate DFT contributions
            acc_re += x_val * tl.cos(angle)
            acc_im += x_val * tl.sin(angle)
            t += 1

        # Normalize by 1/N
        acc_re = acc_re * invN
        acc_im = acc_im * invN

        # Store results to outputs at index j
        tl.store(out_real_ptr + base_out + j, acc_re)
        tl.store(out_imag_ptr + base_out + j, acc_im)

        j += 1


def run_triton(x: torch.Tensor):
    """
    Triton-only implementation: compute real and imaginary parts of rfft along last dim
    for each (batch, channels) slice, with zero-padding to 2*seqlen and normalization by 2*seqlen.
    No torch operations are used.
    """
    # Input x: shape (batch, channels, seqlen)
    batch, channels, seqlen = x.shape
    N = 2 * seqlen
    M = seqlen + 1

    # Ensure x is contiguous and flatten to 1D per (b, c) slice. Grid = (batch*channels,).
    x_contig = x.contiguous()
    grid = (batch * channels,)

    # Allocate outputs: (batch, channels, seqlen+1), float32
    out_real = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)
    out_imag = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)

    # Launch Triton kernel on flattened views
    _rfft_real_imag_triton_kernel[grid](
        x_contig.view(-1),
        out_real.view(-1),
        out_imag.view(-1),
        seqlen,                 # constexpr L
        N,
        M,
        1.0 / float(N),
        num_warps=1, num_stages=1,
    )

    # Return real and imaginary parts separately
    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor argument x of shape (batch, channels, seqlen)
        x = args[0]
        return run_triton(x)


def run(*args):
    return ModelNew()(*args)
