import math
import torch

import triton
import triton.language as tl


@triton.jit
def rfft_row_kernel(
    x_ptr,                  # *const float, input tensor pointer
    out_real_ptr,           # *float, output real tensor pointer
    out_imag_ptr,           # *float, output imag tensor pointer
    L,                      # int32 length
    stride_b, stride_c, stride_l,        # int64 strides for x
    out_stride_b, out_stride_c, out_stride_l,  # int64 strides for outputs
):
    # One program per (b, c) row
    pid = tl.program_id(0)  # 0..B*C-1
    b = pid // C
    c = pid % C

    # Base pointers for this row
    x_row_ptr = x_ptr + b * stride_b + c * stride_c
    out_real_row_ptr = out_real_ptr + b * out_stride_b + c * out_stride_c
    out_imag_row_ptr = out_imag_ptr + b * out_stride_b + c * out_stride_c

    n = 2 * L

    # Loop over output frequency bins k = 0..L (note: output length is L+1)
    for k in range(0, L + 1):
        real_acc = 0.0
        imag_acc = 0.0

        # Sum over j from 0 to n-1
        for j in range(0, n):
            # Load x[b, c, j] only if j < L; for j >= L, x_val is 0 (implicitly, since we don't write for j>=L)
            x_val = tl.load(x_row_ptr + j * stride_l, mask=(j < L), other=0.0)

            # Compute cosine and sine for this k, j
            # Triton expects float32 math; k and j are integers, but we cast to float for math.
            angle = 2.0 * math.pi * float(k) * float(j) / float(n)
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            real_acc += x_val * cos_term
            imag_acc += x_val * sin_term

        # Normalize by 2*L
        norm = 2.0 * float(L)
        real_acc = real_acc / norm
        imag_acc = imag_acc / norm

        # Store to output at index k
        tl.store(out_real_row_ptr + k * out_stride_l, real_acc)
        tl.store(out_imag_row_ptr + k * out_stride_l, imag_acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x shape: (B, C, L)
        assert x.dim() == 3, "Input must be 3D (batch, channels, seqlen)"
        B, C, L = x.shape

        # Output real/imag of shape (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Ensure x is contiguous for simple stride handling
        x = x.contiguous()

        # Get strides in elements (PyTorch gives strides in elements, which Triton expects for pointer arithmetic)
        stride_b = x.stride(0)
        stride_c = x.stride(1)
        stride_l = x.stride(2)

        out_stride_b = out_real.stride(0)
        out_stride_c = out_real.stride(1)
        out_stride_l = out_real.stride(2)

        # Launch one program per (b, c) row
        grid = (B * C,)
        rfft_row_kernel[grid](
            x, out_real, out_imag,
            L,
            stride_b, stride_c, stride_l,
            out_stride_b, out_stride_c, out_stride_l,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
