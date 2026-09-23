import torch
import triton
import triton.language as tl


@triton.jit
def rfft_row_kernel(
    x_ptr,                      # *f32, input pointer (B, C, L)
    out_real_ptr,               # *f32, output real pointer (B, C, L+1)
    out_imag_ptr,               # *f32, output imag pointer (B, C, L+1)
    B, C, L,                    # int32, sizes
    stride_b, stride_c, stride_l,   # int32, strides for x (in elements)
    out_stride_b, out_stride_c, out_stride_l  # int32, strides for out (in elements)
):
    # program id corresponds to (b, c)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # base offsets for this row
    base_x = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    n = 2 * L  # FFT size for rfft

    # Accumulators for real and imag at bin k
    acc_real = 0.0
    acc_imag = 0.0

    # Loop over output bins k = 0..L
    # We use static K to help Triton generate code; L is runtime but small loop is fine.
    K = tl.max(1, L)  # ensure at least one iteration if L==0 (not expected here)
    for k in range(0, K):
        # Sum over j from 0 to n-1
        for j in range(0, n):
            # load x[b, c, j] only if j < L, else 0
            x_index = base_x + j * stride_l
            x_val = tl.load(x_ptr + x_index, mask=(j < L), other=0.0)
            # angle = 2*pi*k*j / n
            angle = (2.0 * 3.141592653589793 * k * j) / n
            # compute cos and sin
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)
            # accumulate
            acc_real += x_val * cos_term
            acc_imag += x_val * sin_term

    # normalize by 2*L
    norm = 2.0 * L
    acc_real = acc_real / norm
    acc_imag = acc_imag / norm

    # store to output at k = L (we write L+1 index as L+1 elements)
    out_index = base_out + k * out_stride_l
    tl.store(out_real_ptr + out_index, acc_real)
    tl.store(out_imag_ptr + out_index, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute rfft on each (batch, channel) row, normalize by 2*seqlen, and return
        real and imaginary parts of shape (B, C, seqlen+1).
        """
        assert x.ndim == 3, "Input must be of shape (batch, channels, seqlen)"
        B, C, L = x.shape
        # output tensors (float32) for real and imaginary parts
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Ensure x is contiguous in memory for predictable strides
        x_contig = x.contiguous()
        stride_b, stride_c, stride_l = x_contig.stride()
        out_stride_b, out_stride_c, out_stride_l = out_real.stride()

        # Launch one Triton program per (b, c) row
        grid = (B * C,)
        rfft_row_kernel[grid](
            x_contig,
            out_real,
            out_imag,
            B, C, L,
            stride_b, stride_c, stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            num_warps=4, num_stages=2,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
