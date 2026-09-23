import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_rfft_real_imag_triton(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) flattened
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size
    C: tl.constexpr,        # channels
    L: tl.constexpr,        # original seqlen
    N: tl.constexpr,        # N = 2 * L, length for rfft
    M: tl.constexpr,        # M = L + 1, output length
    BLOCK_J: tl.constexpr,  # chunk size for j
):
    # program id: one program per (batch, channel)
    pid = tl.program_id(axis=0)
    # Compute which (b, c) slice this program handles
    # Note: pid in [0, B*C)
    c = pid % C
    b = pid // C
    # Base linear index into x (flattened across B, C, L)
    # We flatten as (B, C, L) -> 1D of length B*C*L
    # For a given (b, c), linear index starts at b*C*L
    k = b * C + c
    base = k * L

    # Precompute invN for scaling (1 / (2*L))
    invN = 1.0 / N

    # Process j in chunks of BLOCK_J
    for j_start in range(0, M, BLOCK_J):
        j_offsets = tl.arange(0, BLOCK_J)
        j_idx = j_start + j_offsets  # shape [BLOCK_J]
        mask_j = j_idx < M

        # Accumulate sums for re and im
        re_sum = tl.zeros([BLOCK_J], dtype=tl.float32)
        im_sum = tl.zeros([BLOCK_J], dtype=tl.float32)

        # Loop over t = 0 .. N-1 (zero-padded real input implicitly)
        for t in range(0, N):
            # Load x[t] for this (b, c) slice. Address is x_ptr + base + t
            x_val = tl.load(x_ptr + base + t, mask=True, other=0.0)

            # Compute angles for all j in chunk
            # angle = 2*pi * j * t / N
            # Note: t and N are scalars; j_offsets is vector.
            angle = 2.0 * 3.141592653589793 * (j_offsets * (t / N))
            # cos and sin for the chunk
            cos_j = tl.cos(angle)
            sin_j = tl.sin(angle)

            # Accumulate
            re_sum += x_val * cos_j
            im_sum += x_val * sin_j

        # Scale by 1/N (normalization)
        re_sum *= invN
        im_sum *= invN

        # Store to outputs at positions (b, c, j_idx)
        # We flatten output as (B, C, M) -> 1D of length B*C*M
        out_base = b * C * M + c * M
        out_offsets = j_idx
        out_mask = mask_j
        tl.store(out_real_ptr + out_base + out_offsets, re_sum, mask=out_mask)
        tl.store(out_imag_ptr + out_base + out_offsets, im_sum, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
        - Implicit zero-padding to 2*seqlen
        - torch.fft.rfft along last dim
        - normalization by 2*seqlen
        - return real and imaginary parts separately (float32), shape (batch, channels, seqlen+1)
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Ensure dtype float32 for numerical stability
        x = x.to(torch.float32)

        # Make input contiguous and flatten to 1D across (B, C, L)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Flatten input: (B, C, L) -> 1D length B*C*L
        x_flat = x.contiguous().view(B * C * L)

        # Allocate outputs (B, C, M) each flattened to 1D of length B*C*M
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        _compute_rfft_real_imag_triton[grid](
            x_flat,
            out_real.view(-1),  # flatten output for pointer arithmetic
            out_imag.view(-1),
            B, C, L, N, M,
            BLOCK_J=64,
            num_warps=1,
            num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
