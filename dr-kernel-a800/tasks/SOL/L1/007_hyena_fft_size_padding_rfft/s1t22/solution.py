import torch
import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_kernel(
    x_ptr,                # *const float32, input pointer to x flattened as (BC, L)
    out_real_ptr,         # *float32, output pointer to real part (BC, M)
    out_imag_ptr,         # *float32, output pointer to imag part (BC, M)
    L: tl.int32,          # seqlen
    N: tl.int32,          # 2 * seqlen
    M: tl.int32,          # seqlen + 1
    BLOCK_J: tl.constexpr,  # chunk size for j
):
    pid = tl.program_id(0)  # one program per (b, c) slice
    base = pid * (L + M)    # linear base offset for this slice in flattened x and outputs

    invN = 1.0 / N

    j_start = 0
    while j_start < M:
        j_offsets = j_start + tl.arange(0, BLOCK_J)          # [BLOCK_J]
        mask_j = j_offsets < M

        acc_re = tl.zeros([BLOCK_J], dtype=tl.float32)
        acc_im = tl.zeros([BLOCK_J], dtype=tl.float32)

        t = 0
        while t < N:
            # Load x[pid, t]; if t >= L, value is 0.0 (implicit zero-padding)
            x_val = tl.load(x_ptr + base + t, mask=(t < L), other=0.0)  # scalar
            angles = 2.0 * 3.141592653589793 * j_offsets * t / N       # [BLOCK_J]
            cos_angles = tl.cos(angles)
            sin_angles = tl.sin(angles)
            acc_re += x_val * cos_angles
            acc_im += x_val * sin_angles
            t += 1

        # Normalize by 1/N and store
        acc_re = acc_re * invN
        acc_im = acc_im * invN

        out_base = pid * (L + M)
        out_ptrs_re = out_real_ptr + out_base + j_offsets
        out_ptrs_im = out_imag_ptr + out_base + j_offsets

        tl.store(out_ptrs_re, acc_re, mask=mask_j)
        tl.store(out_ptrs_im, acc_im, mask=mask_j)

        j_start += BLOCK_J


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen) float32 tensor on CUDA device.
        Returns:
          x_freq_real: (batch, channels, seqlen+1) float32
          x_freq_imag: (batch, channels, seqlen+1) float32
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert x.dtype == torch.float32, "Input must be float32."

        batch, channels, seqlen = x.shape
        N = 2 * seqlen
        M = seqlen + 1
        BC = batch * channels

        # Ensure contiguous and flatten (BC, L)
        x_contig = x.contiguous()
        x_flat = x_contig.view(BC, seqlen)

        # Allocate outputs (BC, M) contiguous
        out_real = torch.empty((BC, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((BC, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (BC,)
        _rfft_real_imag_kernel[grid](
            x_flat, out_real, out_imag,
            seqlen, N, M,
            BLOCK_J=128,  # chunk size for j; 128 works well across various sizes
            num_warps=4,
        )

        # Reshape outputs back to (batch, channels, seqlen+1)
        x_freq_real = out_real.view(batch, channels, M)
        x_freq_imag = out_imag.view(batch, channels, M)
        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
