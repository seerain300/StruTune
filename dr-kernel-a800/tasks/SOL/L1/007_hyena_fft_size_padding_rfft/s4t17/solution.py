import torch
import triton
import triton.language as tl


@triton.jit
def rfft_row_kernel(
    x_ptr,                # *float32
    out_real_ptr,         # *float32
    out_imag_ptr,         # *float32
    B,                    # batch size (runtime int)
    C,                    # channels (runtime int)
    L,                    # seqlen (runtime int)
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_l: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_l: tl.constexpr,
):
    # One program per (b, c) row
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Base offsets for input row and output row
    x_row_base = b * stride_b + c * stride_c
    out_row_base_real = b * out_stride_b + c * out_stride_c
    out_row_base_imag = b * out_stride_b + c * out_stride_c  # same row for imag

    # Compute k from 0..L-1; output has L+1, but index 0..L
    for k in range(0, L):
        # Accumulators
        real_acc = 0.0
        imag_acc = 0.0

        n = 2 * L  # pad to next power of two length for rfft
        # Sum over j from 0 to n-1; contributions j >= L come from x[j] which is zero-padded
        for jj in range(0, n):
            valid = jj < L
            x_offset = x_row_base + jj * stride_l
            # load x[b, c, jj] if jj < L else 0.0
            x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

            # Compute angle theta = 2*pi*k*jj / n
            theta = 2.0 * 3.141592653589793 * k * jj / n
            c_term = tl.cos(theta)
            s_term = tl.sin(theta)

            real_acc += x_val * c_term
            imag_acc += x_val * s_term

        # Normalize by 2*L
        norm = 2.0 * L
        real_acc = real_acc / norm
        imag_acc = imag_acc / norm

        # Store to output (B, C, L+1); indices [0..L]
        out_offset_real = out_row_base_real + k * out_stride_l
        out_offset_imag = out_row_base_imag + k * out_stride_l

        tl.store(out_real_ptr + out_offset_real, real_acc)
        tl.store(out_imag_ptr + out_offset_imag, imag_acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x shape: (B, C, L)
        assert x.ndim == 3, "Input must be 3D (batch, channels, seqlen)"
        # Ensure float32 for computation
        x = x.to(torch.float32)
        B, C, L = x.shape

        # Outputs: (B, C, L+1), float32
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Get strides (in elements, Triton expects element-wise strides)
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
            B, C, L,
            stride_b, stride_c, stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            num_warps=4,  # reasonable default
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
