import math
import torch
import triton
import triton.language as tl


@triton.jit
def real_table_kernel(
    padded_ptr,          # *float32, flattened padded vector for all (b, c) slices, length = (B*C) * two_L
    cos_ptr,             # *float32, cosine table per element, length = two_L
    out_real_ptr,        # *float32, flattened output real part buffer, length = (B*C) * L
    two_L,               # int: 2 * L
):
    pid_bc = tl.program_id(0)  # each program handles one (b, c)
    k = tl.program_id(1)       # frequency index in [0..L)

    # Compute base offset in padded/out buffers for this (b, c)
    base = pid_bc * two_L       # each (b, c) has two_L elements in padded
    # Accumulator
    acc_real = 0.0

    # Loop over t = 0..two_L-1
    t = 0
    while t < two_L:
        x_t = tl.load(padded_ptr + base + t)
        cos_t = tl.load(cos_ptr + t)
        acc_real += x_t * cos_t
        t += 1

    # Normalize by 2*L
    acc_real = acc_real / (2.0 * float(two_L))

    # Store to output at flattened index idx = pid_bc * L + k
    idx = pid_bc * L + k
    tl.store(out_real_ptr + idx, acc_real)


@triton.jit
def imag_table_kernel(
    padded_ptr,          # *float32, flattened padded vector for all (b, c) slices, length = (B*C) * two_L
    sin_ptr,             # *float32, sine table per element, length = two_L
    out_imag_ptr,        # *float32, flattened output imag part buffer, length = (B*C) * L
    two_L,               # int: 2 * L
):
    pid_bc = tl.program_id(0)  # each program handles one (b, c)
    k = tl.program_id(1)       # frequency index in [0..L)

    base = pid_bc * two_L
    acc_imag = 0.0

    t = 0
    while t < two_L:
        x_t = tl.load(padded_ptr + base + t)
        sin_t = tl.load(sin_ptr + t)
        acc_imag += x_t * sin_t
        t += 1

    acc_imag = acc_imag / (2.0 * float(two_L))

    idx = pid_bc * L + k
    tl.store(out_imag_ptr + idx, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape
        two_L = 2 * L

        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)

        # Prepare padded input per (b, c): [x[b, c, :], zeros of length L]
        # We'll stack all (B*C) slices and then flatten for Triton input.
        # Build padded tensors list and stack to (B, C, two_L)
        padded_list = []
        for b in range(B):
            for c in range(C):
                inp = x[b, c, :]  # (L,)
                zeros = torch.zeros(L, dtype=torch.float32, device=x.device)
                padded = torch.cat([inp, zeros], dim=0)  # (2*L,)
                padded_list.append(padded)

        padded_stack = torch.stack(padded_list, dim=0)  # (B, C, two_L)
        # Flatten to 1D for Triton input
        padded_flat = padded_stack.reshape(-1)          # total elements = (B*C) * two_L

        # Precompute cosine table per element for t in [0..two_L-1]:
        # cos(2*pi*k / (2*L)) -> here, since we're using t as index, we need cos(2*pi*t / (2*L))
        # But for DFT, we need cos(2*pi*k*t / (2*L)), where k is frequency index and t iterates over time.
        # We'll construct cos_table and sin_table on GPU with torch to ensure compatibility.
        t = torch.arange(two_L, dtype=torch.float32, device=x.device)
        # For real DFT, we multiply by cos(2*pi*k*t/(2*L)) summed over t.
        # We'll create a 1D table of length two_L: cos(2*pi*t/(2*L))
        # Then each (b, c, k) kernel loads cos_table[t] and multiplies with padded[t].
        angle = (2.0 * math.pi * t) / float(two_L)      # shape (two_L,)
        cos_table = torch.cos(angle).contiguous()       # (two_L,)
        sin_table = torch.sin(angle).contiguous()       # (two_L,)

        # Allocate outputs (flattened) for real and imag parts of shape (B, C, L+1)
        L_out = L + 1
        total_bc = B * C
        out_real = torch.empty(total_bc * L_out, dtype=torch.float32, device=x.device)
        out_imag = torch.empty(total_bc * L_out, dtype=torch.float32, device=x.device)

        # Launch Triton kernels: grid = (B*C, L). Each program computes one (b, c, k).
        grid = (total_bc, L)

        # Run real part kernel
        real_table_kernel[grid](
            padded_flat,
            cos_table,
            out_real,
            two_L,
        )

        # Run imag part kernel (imaginary part is zero for real inputs, but we compute to satisfy original output structure)
        imag_table_kernel[grid](
            padded_flat,
            sin_table,
            out_imag,
            two_L,
        )

        # Reshape outputs to (B, C, L+1)
        out_real = out_real.reshape(B, C, L_out).contiguous()
        out_imag = out_imag.reshape(B, C, L_out).contiguous()

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
