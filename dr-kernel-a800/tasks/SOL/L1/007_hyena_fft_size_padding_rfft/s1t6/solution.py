import torch
import triton
import triton.language as tl

# Kernel to compute real part: out_real[b, c, j] = (1/N) * sum_t x[b, c, t] * cos(2*pi*j*t/N)
@triton.jit
def _real_to_re_kernel(x_ptr, out_ptr,
                        B, C, N, L,
                        stride_b, stride_c, stride_t,
                        out_stride_b, out_stride_c, out_stride_m,
                        BLOCK_J: tl.constexpr):
    # One program per (b, c) slice
    pid = tl.program_id(0)
    # Map pid to (b, c)
    b = pid // C
    c = pid % C
    # Base offset for this (b, c) slice
    base = b * stride_b + c * stride_c

    # Prepare arrays for j-chunks
    j_vec = tl.zeros((BLOCK_J,), dtype=tl.int32)
    re_vec = tl.zeros((BLOCK_J,), dtype=tl.float32)

    # Loop over j in chunks
    for j_start in range(0, L + 1, BLOCK_J):
        j_vec[:] = j_start + tl.arange(0, BLOCK_J)
        mask_j = j_vec < (L + 1)

        # Accumulate re over all t
        for t in range(0, N):
            # Load x[b, c, t]
            x_val = tl.load(x_ptr + base + t * stride_t)
            # Cast to float32
            x_val = x_val.to(tl.float32)

            # Compute angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * j_vec * t / N

            # cos(angle) vector
            cos_vec = tl.cos(angle)
            # Accumulate sum over j for this t
            re_vec += x_val * cos_vec

        # Normalize by N
        re_vec = re_vec * (1.0 / N)

        # Store results for valid j
        # We need addresses out_ptr + b*out_stride_b + c*out_stride_c + j_vec*out_stride_m
        out_base = b * out_stride_b + c * out_stride_c
        out_addrs = out_ptr + out_base + j_vec * out_stride_m
        tl.store(out_addrs, re_vec, mask=mask_j)

# Kernel to compute imaginary part: out_imag[b, c, j] = (1/N) * sum_t x[b, c, t] * sin(2*pi*j*t/N)
@triton.jit
def _real_to_im_kernel(x_ptr, out_ptr,
                        B, C, N, L,
                        stride_b, stride_c, stride_t,
                        out_stride_b, out_stride_c, out_stride_m,
                        BLOCK_J: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    base = b * stride_b + c * stride_c

    j_vec = tl.zeros((BLOCK_J,), dtype=tl.int32)
    im_vec = tl.zeros((BLOCK_J,), dtype=tl.float32)

    for j_start in range(0, L + 1, BLOCK_J):
        j_vec[:] = j_start + tl.arange(0, BLOCK_J)
        mask_j = j_vec < (L + 1)

        for t in range(0, N):
            x_val = tl.load(x_ptr + base + t * stride_t).to(tl.float32)

            angle = 2.0 * 3.141592653589793 * j_vec * t / N
            sin_vec = tl.sin(angle)

            im_vec += x_val * sin_vec

        im_vec = im_vec * (1.0 / N)

        out_base = b * out_stride_b + c * out_stride_c
        out_addrs = out_ptr + out_base + j_vec * out_stride_m
        tl.store(out_addrs, im_vec, mask=mask_j)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
            - Pad to 2*seqlen implicitly (we consider the whole length of x along last dim)
            - Compute rfft for real inputs producing seqlen+1 outputs
            - Normalize by 2*seqlen
            - Return real and imaginary parts separately, shape (batch, channels, seqlen+1)
        """
        # Expect x shape (B, C, L)
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L  # pad to 2*seqlen as in original

        # Ensure input is contiguous along last dim; we'll handle strides explicitly
        x = x.contiguous()

        # Prepare output tensors (float32)
        out_real = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, L + 1), device=x.device, dtype=torch.float32)

        # Get strides (elements)
        stride_b, stride_c, stride_t = x.stride()
        out_stride_b, out_stride_c, out_stride_m = out_real.stride()

        # Launch one program per (b, c) slice
        grid = (B * C,)

        # Choose a modest chunk size for j to keep register usage reasonable
        BLOCK_J = 64

        _real_to_re_kernel[grid](
            x, out_real,
            B, C, N, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_J=BLOCK_J,
            num_warps=1,  # keep simple and robust
            num_stages=1,
        )

        _real_to_im_kernel[grid](
            x, out_imag,
            B, C, N, L,
            stride_b, stride_c, stride_t,
            out_stride_b, out_stride_c, out_stride_m,
            BLOCK_J=BLOCK_J,
            num_warps=1,
            num_stages=1,
        )

        # out_real and out_imag already normalized by 1/N in the kernels
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
