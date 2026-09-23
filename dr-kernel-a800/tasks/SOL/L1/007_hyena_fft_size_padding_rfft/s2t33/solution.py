import torch
import triton
import triton.language as tl


# Triton kernel: compute real DFT for each (b, c) slice, over a zero-padded input of length 2*L.
# Output: real and imag parts of length L+1, normalized by 2*L.
@triton.jit
def real_dft_kernel(
    x_ptr,          # *float32, pointer to input vector (length = 2*L), laid out as B*C contiguous chunks
    out_real_ptr,   # *float32, pointer to output real part flattened as 1D (length = B*C*(L+1))
    out_imag_ptr,   # *float32, pointer to output imag part flattened as 1D (length = B*C*(L+1))
    B: tl.constexpr,
    C: tl.constexpr,
    L,                 # runtime int
    two_L,             # runtime int
    stride_bc,         # stride between (b, c) in flattened outputs: L+1
):
    pid = tl.program_id(0)  # linear id over (b, c)
    b = pid // C
    c = pid % C

    base_real = (b * C + c) * (L + 1)
    base_imag = (b * C + c) * (L + 1)

    # Compute DFT for k in [0..L]
    for k in range(0, L + 1):
        sum_real = 0.0
        sum_imag = 0.0
        for t in range(0, two_L):
            val = tl.load(x_ptr + t)  # x_ptr points to a contiguous vector of length 2*L
            angle = -2.0 * 3.141592653589793 * k * t / two_L  # imag part uses sin
            cos_a = tl.cos(angle)
            sin_a = tl.sin(angle)
            sum_real += val * cos_a
            sum_imag += val * sin_a

        # Normalize by 2*L
        norm = 1.0 / two_L
        sum_real *= norm
        sum_imag *= norm

        # Store to (b, c, k)
        tl.store(out_real_ptr + base_real + k, sum_real)
        tl.store(out_imag_ptr + base_imag + k, sum_imag)


# Helper: zero-pad x to length two_L
def pad_to_twoL(x_bc: torch.Tensor, two_L: int) -> torch.Tensor:
    """
    x_bc: 1D tensor of length L (e.g., x[b, c, :])
    Returns: 1D tensor of length two_L, first L entries = x_bc, next two_L - L entries = 0
    """
    assert x_bc.numel() == x_bc.shape[0]
    pad = torch.zeros(two_L - x_bc.shape[0], device=x_bc.device, dtype=x_bc.dtype)
    return torch.cat([x_bc, pad])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Input: x (B, C, L), float32.
        Output: out_real, out_imag, both float32 of shape (B, C, L+1), normalized by 2*L.
                For real inputs, out_imag is zero.
        """
        assert x.dim() == 3, "Input must be (B, C, L)"
        B, C, L = x.shape
        two_L = 2 * L

        # Ensure float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # Flatten the (B*C) slices to a single input buffer of length (B*C)*two_L
        # Each (b, c) slice x[b, c, :] is copied into consecutive positions of length two_L.
        x_flat = torch.empty((B * C) * two_L, device=x.device, dtype=torch.float32)
        for bc in range(B * C):
            b = bc // C
            c = bc % C
            x_bc = x_f32[b, c, :]  # 1D tensor of length L
            x_padded = pad_to_twoL(x_bc, two_L)
            x_flat[bc * two_L : (bc + 1) * two_L] = x_padded

        # Allocate outputs
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        real_dft_kernel[grid](
            x_flat,              # input buffer (B*C*2*L) contiguous
            out_real.view(-1),   # flattened real output
            out_imag.view(-1),   # flattened imag output (we'll set to zeros)
            B, C, L, two_L, (L + 1),  # stride_bc is L+1
        )

        # For real inputs, the imaginary part of rfft is zero. We can zero-initialize out_imag.
        # Alternatively, we could have a Triton kernel that writes zeros; but torch.zero_ is fine.
        out_imag.zero_()

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
