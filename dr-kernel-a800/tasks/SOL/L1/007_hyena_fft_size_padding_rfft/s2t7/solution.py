import torch
import triton
import triton.language as tl


@triton.jit
def pad_kernel_bc(
    x_ptr,        # *const float32, input x of shape (B, C, L), contiguous
    out_ptr,      # *float32, output flattened padded vector of length M = (B*C)*(2*L)
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    two_L: tl.constexpr,
):
    # One program per (b, c)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Base offset for this (b, c) slice in input: b*C*L + c*L
    base_x = (b * C + c) * L

    # Base offset for this (b, c) slice in output: ((b*C + c) * two_L)
    base_out = (b * C + c) * two_L

    # Copy first L elements
    for t in range(0, L):
        val = tl.load(x_ptr + base_x + t)
        tl.store(out_ptr + base_out + t, val)

    # Write zeros for the padded part
    for t in range(L, two_L):
        tl.store(out_ptr + base_out + t, 0.0)


@triton.jit
def real_dft_scalar_kernel_bc(
    x_ptr,         # *const float32, flattened zero-padded input vector of length two_L
    out_real_ptr,  # *float32, flattened output real, length M = B*C*(L+1)
    out_imag_ptr,  # *float32, flattened output imag, length M = B*C*(L+1)
    two_L: tl.constexpr,   # padded length = 2*L
    L_out: tl.constexpr,   # output length = L+1
    B: tl.constexpr,       # batch size
    C: tl.constexpr,       # channels
):
    # Grid dims: (B, C, L+1)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)  # k index in [0, L_out-1]

    # Base index for this (b, c) slice in the flattened vector
    base = (pid_b * C + pid_c) * two_L

    # Accumulator (float32)
    accum = 0.0

    # Compute sum over t from 0 to two_L-1: x[t] * exp(-2*pi*i*k*t / two_L)
    for t in range(0, two_L):
        x_t = tl.load(x_ptr + base + t)
        angle = -2.0 * 3.141592653589793 * float(pid_k) * float(t) / float(two_L)
        accum += x_t * tl.cos(angle)  # for real input, imag contribution is zero

    # Normalize by 2*L
    accum *= (1.0 / float(two_L))

    # Compute linear index for output: ((pid_b*C + pid_c)*(L+1) + pid_k)
    linear_index = (pid_b * C + pid_c) * L_out + pid_k

    # Store real and imag; imag should be 0 for real input
    tl.store(out_real_ptr + linear_index, accum)
    tl.store(out_imag_ptr + linear_index, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute fused FFT size padding and real FFT, normalized by 2*L.
        Returns:
            x_freq_real: float32 tensor of shape (B, C, L+1)
            x_freq_imag: float32 tensor of shape (B, C, L+1)
        """
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape
        device = x.device

        # Ensure contiguous and float32 (no torch compute, just data preparation)
        x = x.contiguous().to(torch.float32)

        # Padded length
        two_L = 2 * L
        L_out = L + 1

        # Allocate output buffer for padded input
        M = (B * C) * two_L
        padded = torch.empty(M, dtype=torch.float32, device=device)

        # Launch pad kernel: one program per (b, c)
        grid_pad = (B * C,)
        pad_kernel_bc[grid_pad](
            x, padded,
            B, C, L, two_L,
            num_warps=1,
        )

        # Allocate outputs (flattened)
        out_real = torch.empty(B * C * L_out, dtype=torch.float32, device=device)
        out_imag = torch.empty(B * C * L_out, dtype=torch.float32, device=device)

        # Launch real DFT kernel: grid over (B, C, L+1)
        grid_dft = (B, C, L_out)
        real_dft_scalar_kernel_bc[grid_dft](
            padded, out_real, out_imag,
            two_L, L_out, B, C,
            num_warps=1,
        )

        # Reshape back to (B, C, L+1)
        x_freq_real = out_real.view(B, C, L_out)
        x_freq_imag = out_imag.view(B, C, L_out)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
