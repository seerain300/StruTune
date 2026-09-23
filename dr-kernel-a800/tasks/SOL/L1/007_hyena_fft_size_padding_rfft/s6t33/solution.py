import torch
import triton
import triton.language as tl

@triton.jit
def real_rfft_triton_kernel(
    x_ptr,             # *f32, input x flattened per (b,c) segment of length S
    out_real_ptr,      # *f32, output real parts (B*C*(S+1))
    out_imag_ptr,      # *f32, output imag parts (B*C*(S+1))
    B: tl.int32,       # batch size
    C: tl.int32,       # channels
    S: tl.constexpr,   # seqlen as compile-time constant for loops
):
    pid = tl.program_id(0)  # one program per (b, c)
    # Derive b and c
    b = pid // C
    c = pid % C

    # Base offset for this (b, c) in x
    base_x = (b * C + c) * S

    # Vector of k indices
    k = tl.arange(0, S)
    # Load x vector
    x_vals = tl.load(x_ptr + base_x + k)

    # Compute sum of x
    sum_x = tl.sum(x_vals, axis=0)

    # Base output offset for this (b, c)
    bc = b * C + c
    out_base = bc * (S + 1)

    # j = 0: y[0] = sum_x / (2*S)
    y0_real = sum_x / (2.0 * S)
    tl.store(out_real_ptr + out_base + 0, y0_real)
    tl.store(out_imag_ptr + out_base + 0, 0.0)

    # For j = 1 .. S-1
    for j in range(1, S):
        # angle = 2*pi*k*j/(2*S) = pi*k*j/S
        angle = (3.141592653589793) * k * j / S
        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)

        # Even j: real component
        if (j % 2) == 0:
            real_val = tl.sum(x_vals * cos_term - x_vals * sin_term, axis=0)
            real_val = real_val / (4.0 * S)  # normalize by 2*S for rfft and divide by 2*N => 4*S
            tl.store(out_real_ptr + out_base + j, real_val)
            tl.store(out_imag_ptr + out_base + j, 0.0)
        else:
            # Odd j: imaginary component only
            sum_sin = tl.sum(x_vals * sin_term, axis=0)
            imag_val = -sum_sin / (2.0 * S)
            tl.store(out_real_ptr + out_base + j, 0.0)
            tl.store(out_imag_ptr + out_base + j, imag_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute rfft along the last dimension for real inputs, normalize by 2*S, and return
        real and imaginary parts as (B, C, S+1) float32 tensors.

        Input: x of shape (B, C, S), float32
        Output: (B, C, S+1) real, (B, C, S+1) imag
        """
        assert x.dim() == 3, "Input must be (B, C, S)"
        B, C, S = x.shape

        # Ensure dtype and contiguity
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        # Allocate outputs
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        real_rfft_triton_kernel[grid](
            x, out_real, out_imag,
            B, C, S,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
