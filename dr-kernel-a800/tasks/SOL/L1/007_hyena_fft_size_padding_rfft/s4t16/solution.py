import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_imag_row(x_ptr, B, C, L, out_real_ptr, out_imag_ptr,
                        stride_b, stride_c, stride_l,
                        out_stride_b, out_stride_c, out_stride_l):
    # One Triton program per (b, c) row
    b = tl.program_id(0) // C
    c = tl.program_id(0) % C

    # We will compute output for k = 0..L-1; k=L will be left as 0.0.
    k_idx = tl.arange(0, L)  # (L,)
    n = 2 * L

    # Accumulators for real and imaginary parts for k in [0..L-1]
    acc_real = tl.zeros((L,), dtype=tl.float32)
    acc_imag = tl.zeros((L,), dtype=tl.float32)

    # Iterate over input j in chunks and accumulate contributions to all k at once
    # For each chunk, build cos/sin matrices of shape (L, BLOCK_J) for angles 2*pi*k*j/n
    BLOCK_J = 128
    for j0 in range(0, n, BLOCK_J):
        jj = j0 + tl.arange(0, BLOCK_J)  # vector of indices in [0..2*L)
        mask_j = jj < n
        # Load x[b, c, jj] with masking (jj < L -> valid, otherwise 0 since we only keep jj < n)
        # Address: x_ptr + b*stride_b + c*stride_c + jj*stride_l
        xj = tl.load(x_ptr + b * stride_b + c * stride_c + jj * stride_l, mask=mask_j, other=0.0)

        # Compute angles for all k and jj: angles[k, jj] = 2*pi*k*jj / n
        angles = 2.0 * 3.141592653589793 * k_idx[:, None] * jj[None, :] / n  # (L, BLOCK_J)
        cos_chunk = tl.cos(angles)  # (L, BLOCK_J)
        sin_chunk = tl.sin(angles)  # (L, BLOCK_J)

        # Multiply and reduce across jj for each k
        prod_real = xj[None, :] * cos_chunk  # (L, BLOCK_J)
        prod_imag = xj[None, :] * sin_chunk  # (L, BLOCK_J)
        acc_real += tl.sum(prod_real, axis=1)  # (L,)
        acc_imag += tl.sum(prod_imag, axis=1)  # (L,)

    # Normalize by 2*L
    norm = 2.0 * L
    acc_real = acc_real / norm
    acc_imag = acc_imag / norm

    # Store outputs into out[b, c, 0..L-1]
    out_base_real = out_real_ptr + b * out_stride_b + c * out_stride_c
    out_base_imag = out_imag_ptr + b * out_stride_b + c * out_stride_c
    for kk in range(0, L):
        tl.store(out_base_real + kk * out_stride_l, acc_real[kk])
        tl.store(out_base_imag + kk * out_stride_l, acc_imag[kk])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L), float32 expected
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert x.dtype == torch.float32, "Input must be float32."
        B, C, L = x.shape

        # Allocate outputs; we will compute k in [0..L-1], k=L remains 0.0 to match rfft length L+1
        out_real_final = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag_final = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Strides for input
        stride_b = x.stride(0)
        stride_c = x.stride(1)
        stride_l = x.stride(2)

        # Strides for outputs
        out_stride_b = out_real_final.stride(0)
        out_stride_c = out_real_final.stride(1)
        out_stride_l = out_real_final.stride(2)

        # Launch one program per (b, c) row
        grid = (B * C,)
        rfft_real_imag_row[grid](
            x, B, C, L, out_real_final, out_imag_final,
            stride_b, stride_c, stride_l,
            out_stride_b, out_stride_c, out_stride_l,
        )

        return out_real_final, out_imag_final


def run(*args):
    return ModelNew()(*args)
