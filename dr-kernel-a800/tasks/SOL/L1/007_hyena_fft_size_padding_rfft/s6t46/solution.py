import math
import torch
import triton
import triton.language as tl

@triton.jit
def rfft_real_input_kernel(
    x_ptr,                    # *float32, input flattened to (BC, S)
    cos_ptr, sin_ptr,         # *float32, length S+1 arrays for cos(pi*k/(2*S)), sin(pi*k/(2*S))
    out_real_ptr, out_imag_ptr,  # *float32, output flattened to (BC, S+1)
    S: tl.int32,              # seqlen
    stride_x_bc: tl.int32,    # stride between (b,c) in x (elements)
    stride_out_bc: tl.int32,  # stride between (b,c) in output (elements)
):
    # One program per (b,c)
    bc = tl.program_id(0)
    base_x = bc * stride_x_bc
    base_out = bc * stride_out_bc

    # Compute sum of x over S elements
    total = 0.0
    for i in range(0, S):
        v = tl.load(x_ptr + base_x + i)
        total += v

    # Scale factor for normalization (divide by 2*S)
    scale = total / (2.0 * S)

    # k=0: real=scale, imag=0
    tl.store(out_real_ptr + base_out + 0, scale)
    tl.store(out_imag_ptr + base_out + 0, 0.0)

    # k=1..S-1: real/imag based on even/odd
    for k in range(1, S):
        c = tl.load(cos_ptr + k)  # cos(pi*k/(2*S))
        s = tl.load(sin_ptr + k)  # sin(pi*k/(2*S))
        even = (k % 2) == 0
        if even:
            real_k = scale * (c - s)
        else:
            real_k = 0.0
        # Imaginary part: only odd k has non-zero imaginary
        imag_k = 0.0
        if not even:
            imag_k = -scale * s
        tl.store(out_real_ptr + base_out + k, real_k)
        tl.store(out_imag_ptr + base_out + k, imag_k)

    # k=S (Nyquist): real = (sum(x) * cos(pi) - sum(x) * sin(pi)) / (2*S)
    # Since sin(pi) = 0, cos(pi) = -1, this is -scale
    # But to be explicit, use cos_ptr[S] and sin_ptr[S]
    cS = tl.load(cos_ptr + S)  # cos(pi*S/(2*S)) = cos(pi) = -1
    sS = tl.load(sin_ptr + S)  # sin(pi*S/(2*S)) = sin(pi) = 0
    real_S = scale * (cS - sS)  # = scale * (-1 - 0) = -scale
    tl.store(out_real_ptr + base_out + S, real_S)
    tl.store(out_imag_ptr + base_out + S, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, S), can be any float dtype; cast to float32
        B, C, S = x.shape
        BC = B * C

        x_f32 = x.to(torch.float32)
        # Flatten x to (BC, S) for kernel
        x_flat = x_f32.reshape(BC, S)

        # Prepare precomputed trig values for k=0..S
        # We compute for k=0..S, length S+1
        k = torch.arange(S + 1, device=x.device, dtype=torch.float32)
        ang = math.pi * k / (2.0 * S)
        cos_vals = torch.cos(ang)  # length S+1
        sin_vals = torch.sin(ang)  # length S+1

        # Allocate outputs (BC, S+1)
        out_real = torch.empty((BC, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((BC, S + 1), dtype=torch.float32, device=x.device)

        # Strides in elements (row-major contiguous)
        stride_x_bc = S
        stride_out_bc = S + 1

        # Launch Triton kernel: one program per (b,c)
        grid = (BC,)
        rfft_real_input_kernel[grid](
            x_flat, cos_vals, sin_vals,
            out_real, out_imag,
            S, stride_x_bc, stride_out_bc,
            num_warps=1,
            num_stages=1,
        )

        # Reshape back to (B, C, S+1)
        out_real = out_real.view(B, C, S + 1)
        out_imag = out_imag.view(B, C, S + 1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
