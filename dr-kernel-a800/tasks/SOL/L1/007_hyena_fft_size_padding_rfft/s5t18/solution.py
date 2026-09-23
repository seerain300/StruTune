import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr, N, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for a padded row x_ptr of length N = 2 * seqlen:
      real_out[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N  for j in 0..seqlen.
    x_ptr points to the padded input vector; out_ptr points to real_out vector of length seqlen+1.
    """
    j = tl.program_id(0)  # one program per j
    acc = 0.0
    # Iterate over k in chunks for better codegen; N is typically not huge here.
    for k_start in range(0, N, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask = k_idx < N
        # Load x[k] as float32
        x_k = tl.load(x_ptr + k_idx, mask=mask, other=0.0).to(tl.float32)
        # Compute cos term: cos(2*pi*j*k/N)
        angle = 2.0 * 3.141592653589793 * j * k_idx / N
        cos_term = tl.cos(angle)
        # Accumulate sum over this chunk
        acc += tl.sum(x_k * cos_term, axis=0)
    # Normalize by N and store at index j
    out_val = acc / N
    tl.store(out_ptr + j, out_val)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr, N, seqlen, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for a padded row x_ptr of length N = 2 * seqlen:
      imag_out[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N  for j in 1..seqlen-1.
    x_ptr points to the padded input vector; out_ptr points to imag_out vector of length seqlen-1.
    """
    j = tl.program_id(0) + 1  # j starts from 1
    acc = 0.0
    for k_start in range(0, N, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask = k_idx < N
        x_k = tl.load(x_ptr + k_idx, mask=mask, other=0.0).to(tl.float32)
        angle = 2.0 * 3.141592653589793 * j * k_idx / N
        sin_term = tl.sin(angle)
        acc += tl.sum(x_k * sin_term, axis=0)
    out_val = acc / N
    tl.store(out_ptr + (j - 1), out_val)  # write at index j-1 (since j starts at 1)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen) float32 tensor on CUDA
        Returns:
          x_freq_real: (batch, channels, seqlen+1) float32
          x_freq_imag: (batch, channels, seqlen+1) float32
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Reshape to rows for kernel: each (b, c) row
        rows = batch * channels
        x_rows = x.reshape(rows, seqlen)

        # Allocate padded input: [rows, N], fill with zeros and copy first seqlen elements
        x_padded = torch.zeros((rows, N), dtype=torch.float32, device=x.device)
        x_padded[:, :seqlen] = x_rows

        # Allocate outputs
        x_freq_real = torch.empty((rows, seqlen + 1), dtype=torch.float32, device=x.device)
        # Imaginary output for j=1..seqlen-1; we'll pad later to seqlen+1
        x_freq_imag_partial = torch.empty((rows, seqlen - 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels: one program per j
        BLOCK_K = 1024  # constexpr chunk size for k-loop
        num_warps = 4

        # Real part: j in 0..seqlen -> output length seqlen+1
        grid_real = (seqlen + 1,)
        rfft_real_kernel[grid_real](x_padded, x_freq_real, N, seqlen, BLOCK_K, num_warps=num_warps)

        # Imag part: j in 1..seqlen-1 -> output length seqlen-1; pad to seqlen+1 later
        grid_imag = (seqlen - 1,)
        rfft_imag_kernel[grid_imag](x_padded, x_freq_imag_partial, N, seqlen, BLOCK_K, num_warps=num_warps)

        # Pad imag to include j=0 and j=seqlen with zeros (correct for real rfft)
        # Pad left by 1 zero, then pad right by 1 zero
        pad_left = 1
        pad_right = 1
        x_freq_imag = torch.nn.functional.pad(x_freq_imag_partial, (pad_left, pad_right))

        # Reshape back to (batch, channels, seqlen+1)
        out_real = x_freq_real.view(batch, channels, seqlen + 1)
        out_imag = x_freq_imag.view(batch, channels, seqlen + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
