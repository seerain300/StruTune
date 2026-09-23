import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      total_elems: tl.constexpr, seqlen: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row using contiguous flattened x:
      real_out[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N
      for j in 0..seqlen.
    x_ptr points to the contiguous input vector (length rows * seqlen).
    out_ptr points to real output vector (length rows * (seqlen + 1)).
    total_elems = rows * seqlen.
    """
    pid = tl.program_id(axis=0)
    # j loop over bins
    for j in range(0, seqlen + 1):
        acc = 0.0
        # k loop over padded length
        for k in range(0, N, BLOCK_K):
            idx = k + tl.arange(0, BLOCK_K)
            mask = idx < N
            # Since x_ptr is flattened (rows * seqlen), we can just load from idx.
            # For idx >= seqlen, the original data is zero (implicit padding), so masked load suffices.
            vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
            angle = 2.0 * tl.pi * j * idx / N
            cos_term = tl.cos(angle)
            acc += tl.sum(vals * cos_term, axis=0)
        # normalize by N (2*seqlen) and store
        out_val = acc / N
        tl.store(out_ptr + pid * (seqlen + 1) + j, out_val)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      total_elems: tl.constexpr, seqlen: tl.constexpr, N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row (bins 1..seqlen-1):
      imag_out[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N
      for j in 1..seqlen-1.
    x_ptr points to the contiguous input vector (length rows * seqlen).
    out_ptr points to imag output vector (length rows * (seqlen + 1)).
    total_elems = rows * seqlen.
    """
    pid = tl.program_id(axis=0)
    for j in range(1, seqlen):
        acc = 0.0
        for k in range(0, N, BLOCK_K):
            idx = k + tl.arange(0, BLOCK_K)
            mask = idx < N
            vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
            angle = 2.0 * tl.pi * j * idx / N
            sin_term = tl.sin(angle)
            acc += tl.sum(vals * sin_term, axis=0)
        out_val = acc / N
        tl.store(out_ptr + pid * (seqlen + 1) + j, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute real and imaginary parts of normalized rfft(x, n=2*seqlen) per (batch, channel) row,
        using Triton kernels. Return shape (batch, channels, seqlen+1) as float32 tensors.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Ensure float32 for stable math
        x = x.to(torch.float32)
        batch, channels, seqlen = x.shape
        rows = batch * channels

        # Prepare output buffers (float32) for real and imag parts
        x_freq_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Flatten input to 1D contiguous for kernels
        x_flat = x.reshape(rows * seqlen).contiguous()
        total_elems = rows * seqlen

        # Padded length
        N = 2 * seqlen

        # Launch Triton kernels: one program per row
        grid = (rows,)

        # Choose block size; 1024 is fine for typical seqlen. Triton will loop as needed.
        BLOCK_K = 1024

        # Real part kernel: j=0..seqlen
        rfft_real_kernel[grid](
            x_flat, x_freq_real.reshape(rows * (seqlen + 1)),
            total_elems=total_elems, seqlen=seqlen, N=N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Imaginary part kernel: j=1..seqlen-1; initialize imag to zeros
        x_freq_imag.zero_()
        rfft_imag_kernel[grid](
            x_flat, x_freq_imag.reshape(rows * (seqlen + 1)),
            total_elems=total_elems, seqlen=seqlen, N=N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Set imag_out[0] and imag_out[seqlen] to zero explicitly
        x_freq_imag[:, :, 0] = 0.0
        x_freq_imag[:, :, seqlen] = 0.0

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
