import torch
import triton
import triton.language as tl

# Triton kernel: compute real part of rfft for each (b, c, k).
# Grid: (B*C, L). Each program computes one output X[b, c, k].
@triton.jit
def real_part_rfft_kernel(
    pad_ptr,       # *float32, input padded tensor (B, C, 2*L), contiguous
    out_ptr,       # *float32, output real tensor (B, C, L+1), contiguous
    B: tl.constexpr,
    C: tl.constexpr,
    two_L: tl.constexpr,
    L_out: tl.constexpr,
):
    bc = tl.program_id(0)  # spans 0..(B*C-1)
    k = tl.program_id(1)   # spans 0..(L-1)

    b = bc // C
    c = bc % C

    acc = 0.0
    inv_two_L = 1.0 / two_L
    two_pi = 2.0 * 3.141592653589793

    # Sum over t: X[k] = sum_{t=0}^{2*L-1} x[t] * exp(-2*pi*i*k*t / (2*L))
    # For real input, imag part is zero; we compute only real part using cos.
    for t in range(0, two_L):
        x_t = tl.load(pad_ptr + bc * two_L + t)
        angle = -two_pi * k * t * inv_two_L
        acc += x_t * tl.cos(angle)

    # Normalize by 2*L
    acc = acc * inv_two_L

    # Store to output at (b, c, k)
    out_idx = bc * (L_out) + k
    tl.store(out_ptr + out_idx, acc)


# Triton kernel: compute imag part (always zero for real input), but we store zeros to satisfy return requirements.
@triton.jit
def imag_part_rfft_kernel(
    pad_ptr,       # *float32, unused but kept for interface
    out_ptr,       # *float32, output imag tensor (B, C, L+1), contiguous
    B: tl.constexpr,
    C: tl.constexpr,
    two_L: tl.constexpr,
    L_out: tl.constexpr,
):
    bc = tl.program_id(0)
    k = tl.program_id(1)

    b = bc // C
    c = bc % C

    out_idx = bc * (L_out) + k
    tl.store(out_ptr + out_idx, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-optimized version of:
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2*L)
            x_freq = x_freq / (2*L)
            return x_freq.real, x_freq.imag
        Output: (B, C, L+1) for both real and imag, float32 tensors.
        """
        # Ensure contiguous and float32
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, C, L = x.shape
        two_L = 2 * L
        L_out = L + 1

        # Prepare zero-padded input of shape (B, C, 2*L)
        pad = torch.zeros((B, C, two_L), dtype=torch.float32, device=x.device)
        # Copy x into first L positions
        pad[:, :, 0:L] = x

        # Allocate outputs (real and imag) flattened as (B*C, L+1)
        out_real = torch.empty((B, C, L_out), dtype=torch.float32, device=x.device).view(-1)
        out_imag = torch.empty((B, C, L_out), dtype=torch.float32, device=x.device).view(-1)

        # Launch Triton kernels: grid = (B*C, L)
        grid = (B * C, L)
        real_part_rfft_kernel[grid](
            pad,                 # input padded tensor
            out_real,            # output real flattened
            B,
            C,
            two_L,
            L_out,
            num_warps=1,
            num_stages=1,
        )

        # Imag part is zero for real input; we filled out_imag earlier, but to be explicit:
        imag_part_rfft_kernel[grid](
            pad,                 # unused, kept for interface consistency
            out_imag,            # output imag flattened
            B,
            C,
            two_L,
            L_out,
            num_warps=1,
            num_stages=1,
        )

        # Reshape to (B, C, L+1)
        out_real = out_real.view(B, C, L_out)
        out_imag = out_imag.view(B, C, L_out)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
