import math
import torch
import triton
import triton.language as tl

# Triton kernel: zero-pad x[b, c, :] into a flattened vector of length two_L.
# Output out_ptr is a flat buffer of length M = (B*C) * two_L, laid out as:
# out_ptr[base + t] = x[b, c, t] for t in [0, L-1], and zeros for t in [L, 2*L-1].
@triton.jit
def pad_kernel(
    x_ptr,            # *float32, input x of shape (B, C, L), contiguous
    out_ptr,          # *float32, output flattened padded vectors
    B: tl.int32,      # batch size
    C: tl.int32,      # channels
    L: tl.int32,      # seqlen
    two_L: tl.int32,  # 2 * L
):
    b = tl.program_id(0)  # batch index
    c = tl.program_id(1)  # channel index
    # Compute base linear index in the flattened output for this (b, c) slice:
    # Each (b, c) slice occupies a contiguous segment of length two_L in the flat buffer.
    base = (b * C + c) * two_L

    # Copy x[b, c, :] into out_ptr[base + t] for t in [0, L)
    # x_ptr is laid out as ((b*C)*L) + c*L + t
    for t in range(0, L):
        val = tl.load(x_ptr + (b * C + c) * L + t)
        tl.store(out_ptr + base + t, val)

    # Fill zeros for t in [L, 2*L)
    for t in range(L, two_L):
        tl.store(out_ptr + base + t, 0.0)


# Triton kernel: compute real-valued DFT over the padded vector for each (b, c).
# Produces out_real and out_imag of shape (B, C, L+1), normalized by two_L.
# Note: This kernel assumes out_ptr points to the padded vectors laid out as (B*C)*two_L.
@triton.jit
def rfft_real_kernel(
    out_ptr,          # *float32, flattened padded vectors
    out_real_ptr,     # *float32, output real part (B*C*(L+1))
    out_imag_ptr,     # *float32, output imag part (B*C*(L+1))
    B: tl.int32,
    C: tl.int32,
    L: tl.int32,
    two_L: tl.int32,
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Base offset in the flattened padded vector for this (b, c)
    base = (b * C + c) * two_L

    # Precompute 2*pi / two_L as float32
    two_pi_over_two_L = 2.0 * 3.141592653589793 / two_L

    # Compute DFT for k = 0..L
    for k in range(0, L + 1):
        real_acc = 0.0
        imag_acc = 0.0
        # Sum over t = 0..2*L-1
        for t in range(0, two_L):
            x_t = tl.load(out_ptr + base + t)
            angle = two_pi_over_two_L * k * t
            real_contrib = x_t * tl.cos(angle)
            imag_contrib = x_t * tl.sin(angle)
            real_acc += real_contrib
            imag_acc += imag_contrib

        # Normalize by 2*L
        real_acc = real_acc / two_L
        imag_acc = imag_acc / two_L

        # Store to output: layout is (B*C*(L+1)), contiguous
        out_index = (b * C + c) * (L + 1) + k
        tl.store(out_real_ptr + out_index, real_acc)
        tl.store(out_imag_ptr + out_index, imag_acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Pads input to length 2*L per (b, c) using a Triton kernel.
        - Computes rfft real DFT over padded vector using a Triton kernel.
        - Returns real and imaginary parts of shape (B, C, L+1), normalized by 2*L.
        """
        # Ensure dtype and contiguity
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, C, L = x.shape
        two_L = 2 * L

        # Allocate flattened padded input: M = (B*C) * two_L
        M = B * C * two_L
        out_flat = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (b, c)
        grid = (B, C)
        pad_kernel[grid](
            x, out_flat,
            B, C, L, two_L,
            num_warps=1,
        )

        # Allocate outputs: real and imag, shape (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Launch DFT kernel: one program per (b, c)
        rfft_real_kernel[grid](
            out_flat, out_real, out_imag,
            B, C, L, two_L,
            num_warps=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
