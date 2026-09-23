import torch
import triton
import triton.language as tl


@triton.jit
def pad_input_kernel(x_ptr, in_ptr,
                      B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, two_L: tl.constexpr):
    # One Triton program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)
    # x_ptr has shape (B*C*L); index for x[b, c, i]
    base_x = b * C * L + c * L

    # Write x[b, c, :] into in_ptr[b*C + c] at indices 0..L-1
    for i in range(0, L):
        val = tl.load(x_ptr + base_x + i)
        tl.store(in_ptr + (b * C + c) * two_L + i, val)

    # Write zeros for padding from L to 2*L-1
    for i in range(L, two_L):
        tl.store(in_ptr + (b * C + c) * two_L + i, 0.0)


@triton.jit
def dft_one_k_kernel(in_ptr, out_real_ptr, out_imag_ptr,
                     B: tl.constexpr, C: tl.constexpr, L: tl.constexpr, two_L: tl.constexpr, L_plus1: tl.constexpr):
    # One Triton program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Output base index for this (b, c), out has shape (B*C*(L_plus1))
    base_out = (b * C + c) * L_plus1

    # Accumulators for real and imaginary parts
    acc_real = 0.0
    acc_imag = 0.0

    # Compute DFT for k in [0..L]
    for k in range(0, L + 1):
        # Accumulate over t = 0..two_L-1
        for t in range(0, two_L):
            val = tl.load(in_ptr + (b * C + c) * two_L + t)
            # angle = -2π * k * t / two_L
            angle = -2.0 * 3.141592653589793 * k * t / two_L
            # Complex exp: cos(angle) + i sin(angle)
            real_part = val * tl.cos(angle)
            imag_part = val * tl.sin(angle)
            acc_real += real_part
            acc_imag += imag_part

        # Normalize by two_L
        norm = 1.0 / two_L
        tl.store(out_real_ptr + base_out + k, acc_real * norm)
        tl.store(out_imag_ptr + base_out + k, acc_imag * norm)

        acc_real = 0.0
        acc_imag = 0.0


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # No torch operations in forward; only allocate outputs and launch Triton kernels
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        # x shape: (B, C, L); we don't call .contiguous() or any torch data movement
        B = x.shape[0]
        C = x.shape[1]
        L = x.shape[2]
        two_L = 2 * L
        L_plus1 = L + 1

        # Allocate outputs directly with target shape; Triton kernels write into them
        out_real = torch.empty((B, C, L_plus1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L_plus1), device=x.device, dtype=torch.float32)

        # Flatten views for Triton pointer arithmetic (sizes are constexpr meta-params)
        # We will pass x as-is; Triton loads/stores use pointer arithmetic based on B, C, L meta-params.
        # For pad, we need a flattened in buffer of length B*C*two_L
        in_buf = torch.empty(B * C * two_L, device=x.device, dtype=torch.float32)

        # Launch pad kernel: grid over (B, C)
        grid_pad = (B, C)
        pad_input_kernel[grid_pad](x, in_buf, B=B, C=C, L=L, two_L=two_L)

        # Launch DFT kernel: grid over (B, C)
        grid_dft = (B, C)
        dft_one_k_kernel[grid_dft](in_buf, out_real, out_imag, B=B, C=C, L=L, two_L=two_L, L_plus1=L_plus1)

        # Return outputs; forward does not perform any torch data movement (no .view(), no .contiguous())
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
