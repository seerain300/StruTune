import torch
import triton
import triton.language as tl


@triton.jit
def pad_kernel(
    x_ptr,            # *const float32, input [B, C, L]
    padded_ptr,       # *float32, output flattened vector of length (B*C)*(2*L)
    B: tl.constexpr,  # int
    C: tl.constexpr,  # int
    L: tl.constexpr,  # int
    two_L: tl.constexpr,  # int
    BLOCK: tl.constexpr,  # int, >= 2*L
):
    # One program per (b, c)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    bc = b * C + c
    base = bc * two_L  # offset in flattened padded vector for this (b, c)

    # Create vector of indices t in [0, 2*L)
    t_offsets = tl.arange(0, BLOCK)
    # Load x[b, c, :] values into registers
    x_offsets = bc * L + t_offsets  # index into x_ptr
    mask_x = t_offsets < L
    x_vals = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0)

    # Write x_vals into padded_ptr at [base : base + L)
    out_offsets_x = base + t_offsets
    tl.store(padded_ptr + out_offsets_x, x_vals, mask=mask_x)

    # Write zeros for the padded part [base + L : base + 2*L)
    pad_offsets = base + (t_offsets + L)  # indices where t >= L
    tl.store(padded_ptr + pad_offsets, 0.0, mask=(t_offsets >= L))


@triton.jit
def real_dft_zero_padded_kernel(
    padded_ptr,       # *const float32, zero-padded input vector of length (B*C)*(2*L)
    out_real_ptr,     # *float32, output real part vector of length (B*C)*(2*L)
    out_imag_ptr,     # *float32, output imag part vector of length (B*C)*(2*L)
    B: tl.constexpr,  # int
    C: tl.constexpr,  # int
    two_L: tl.constexpr,  # int
    inv_two_L,        # float32 scalar = 1.0 / (2*L)
    BLOCK: tl.constexpr,  # dummy, not used (we iterate scalar-wise)
):
    # Grid: (B, C, two_L)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)

    bc = pid_b * C + pid_c
    base = bc * two_L

    accum = tl.zeros((), dtype=tl.float32)

    # DFT over padded input of length two_L
    t = 0
    while t < two_L:
        x_t = tl.load(padded_ptr + base + t)  # scalar load
        theta = -2.0 * 3.141592653589793 * pid_k * t * inv_two_L
        cos_theta = tl.cos(theta)
        accum += x_t * cos_theta
        t += 1

    accum = accum * inv_two_L

    tl.store(out_real_ptr + base + pid_k, accum)
    tl.store(out_imag_ptr + base + pid_k, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # x: (B, C, L), float32 or castable
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        B, C, L = x.shape
        two_L = 2 * L

        # Ensure contiguous and float32
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Allocate flattened padded input: length M = (B*C)*(2*L)
        M = (B * C) * two_L
        padded = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (b, c)
        grid_pad = (B * C,)
        pad_kernel[grid_pad](
            x, padded,
            B, C, L, two_L,
            BLOCK=two_L,  # vector length = 2*L; t_offsets < L for x, and we write zeros for padded part
            num_warps=4,
        )

        # Allocate outputs
        out_real = torch.empty(M, dtype=torch.float32, device=x.device)
        out_imag = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch real DFT kernel: grid over (B, C, 2*L)
        grid_dft = (B, C, two_L)
        inv_two_L = 1.0 / float(two_L)
        real_dft_zero_padded_kernel[grid_dft](
            padded, out_real, out_imag,
            B, C, two_L,
            inv_two_L,
            BLOCK=1,  # not used in scalar loop
            num_warps=4,
        )

        # Reshape back to (B, C, 2*L)
        x_freq_real = out_real.view(B, C, two_L)
        x_freq_imag = out_imag.view(B, C, two_L)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
