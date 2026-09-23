import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    # Each program handles one (b*c) row: pid_bc in [0, B*C)
    pid_bc = tl.program_id(0)
    # Copy x[pid_bc, :L] into out[pid_bc, 0:L]
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + pid_bc * L + i)
        tl.store(out_ptr + pid_bc * N + i, val)
    # Fill the rest with zeros
    for i in tl.static_range(L, N):
        tl.store(out_ptr + pid_bc * N + i, 0.0)


@triton.jit
def _accumulate_rfft_kernel(padded_ptr, out_real_ptr, out_imag_ptr, L: tl.int32, N: tl.int32):
    # For each (bc, k), accumulate sum_j padded[j] * cos(2*pi*k*j/N)
    # and padded[j] * sin(2*pi*k*j/N). Normalization happens after.
    pid_bc = tl.program_id(0)
    k = tl.program_id(1)

    # Initialize accumulators
    acc_real = 0.0
    acc_imag = 0.0

    # Sum over j=0..N-1
    for j in tl.static_range(0, N):
        val = tl.load(padded_ptr + pid_bc * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        acc_real += val * tl.cos(angle)
        acc_imag += val * tl.sin(angle)

    # Store accumulated values (we'll normalize in a separate kernel)
    tl.store(out_real_ptr + pid_bc * (L + 1) + k, acc_real)
    tl.store(out_imag_ptr + pid_bc * (L + 1) + k, acc_imag)


@triton.jit
def _divide_kernel(in_ptr, out_ptr, numel: tl.int32, divisor: tl.float32, BLOCK: tl.constexpr):
    # Elementwise division: out[i] = in[i] / divisor
    grid = tl.num_programs(0)
    for i in tl.static_range(0, numel):
        val = tl.load(in_ptr + i)
        val = val / divisor
        tl.store(out_ptr + i, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L), float32
        B, C, L = x.shape
        N = 2 * L  # padding to 2*seqlen

        # Ensure input is contiguous
        x = x.contiguous()

        # Allocate padded buffer: shape (B*C, N)
        BC = B * C
        padded_x = torch.empty((BC, N), device=x.device, dtype=torch.float32)

        # Launch copy kernel
        _copy_row_to_padded_kernel[(BC,)](x.view(BC, L), padded_x, L, N)

        # Allocate outputs: shape (BC, L+1), float32
        out_real = torch.zeros((BC, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.zeros((BC, L + 1), device=x.device, dtype=torch.float32)

        # Accumulate rfft coefficients per k
        _accumulate_rfft_kernel[(BC, L)](padded_x, out_real, out_imag, L, N)

        # Normalize by 2*L using Triton division kernel
        inv_N = 1.0 / (2.0 * L)
        _divide_kernel[(BC * (L + 1),)](out_real, out_real, BC * (L + 1), inv_N, BLOCK=1024)
        _divide_kernel[(BC * (L + 1),)](out_imag, out_imag, BC * (L + 1), inv_N, BLOCK=1024)

        # Reshape back to (B, C, L+1)
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
