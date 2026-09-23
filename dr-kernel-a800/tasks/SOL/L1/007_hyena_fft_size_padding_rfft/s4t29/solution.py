import math
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    # One program per (b, c) row flattened by BC = B*C
    pid_bc = tl.program_id(0)
    # Copy x[pid_bc, :L] into out[pid_bc, 0:L]
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + pid_bc * L + i)
        tl.store(out_ptr + pid_bc * N + i, val)
    # Fill the rest with zeros
    for i in tl.static_range(L, N):
        tl.store(out_ptr + pid_bc * N + i, 0.0)


@triton.jit
def _compute_real_kernel(padded_ptr, out_real_ptr, L: tl.int32, N: tl.int32, inv2L: tl.float32):
    # One program per (b, c, k) where k is the second program_id(1)
    pid_bc = tl.program_id(0)
    k = tl.program_id(1)  # k in [0, L]

    acc = 0.0
    # Sum over j=0..N-1
    # Loop explicitly to avoid complex vectorized non-constant index loads.
    for j in tl.static_range(0, N):
        val = tl.load(padded_ptr + pid_bc * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        cosv = tl.cos(angle)
        acc += val * cosv

    # Normalize by 2*L
    acc = acc * inv2L

    # Store to out_real[pid_bc, k]
    # out_real has shape (BC, L+1) flattened; for (b, c), index k is linear offset.
    out_index = pid_bc * (L + 1) + k
    tl.store(out_real_ptr + out_index, acc)


@triton.jit
def _compute_imag_kernel(padded_ptr, out_imag_ptr, L: tl.int32, N: tl.int32, inv2L: tl.float32):
    # One program per (b, c, k)
    pid_bc = tl.program_id(0)
    k = tl.program_id(1)  # k in [0, L]

    acc = 0.0
    # Sum over j=0..N-1
    for j in tl.static_range(0, N):
        val = tl.load(padded_ptr + pid_bc * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        sinv = tl.sin(angle)
        acc += val * sinv

    # Normalize by 2*L
    acc = acc * inv2L

    # Store to out_imag[pid_bc, k]
    out_index = pid_bc * (L + 1) + k
    tl.store(out_imag_ptr + out_index, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x shape: (batch, channels, seqlen) in float32
        B, C, L = x.shape
        N = 2 * L  # padded length for rfft

        # Ensure input is float32
        x_f32 = x.to(torch.float32)

        # Allocate padded buffer of shape (B*C, N)
        padded = torch.empty((B * C, N), dtype=torch.float32, device=x.device)

        # Launch copy kernel to fill padded with x[:, :, :L] and zeros afterwards
        _copy_row_to_padded_kernel[(B * C,)](padded, padded, L, N)

        # Allocate outputs (B*C, L+1) for real and imaginary parts
        out_real = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)

        # Prepare grid for kernels: each k is handled by a separate program
        grid = (B * C, L + 1)

        # Compute inv2L
        inv2L = 1.0 / (2.0 * L)

        # Launch real and imaginary computation kernels
        _compute_real_kernel[grid](padded, out_real, L, N, inv2L)
        _compute_imag_kernel[grid](padded, out_imag, L, N, inv2L)

        # Reshape to (B, C, L+1)
        out_real_bc = out_real.view(B, C, L + 1)
        out_imag_bc = out_imag.view(B, C, L + 1)

        return out_real_bc, out_imag_bc


def run(*args):
    return ModelNew()(*args)
