import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      seqlen, N,  # N = 2 * seqlen
                      BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N
      for j in 0..seqlen, where x_ptr points to a padded input vector of length N.
    """
    pid = tl.program_id(axis=0)  # one program per (batch, channel) row
    # Accumulator for real part
    # Note: we'll store per j in a loop; no need for a vector accumulator.
    # Loop over output bins j
    for j in range(0, seqlen + 1):  # j runs from 0 to seqlen (inclusive)
        acc = 0.0
        # Loop over k in chunks to accumulate
        for k_start in range(0, N, BLOCK_K):
            k = k_start + tl.arange(0, BLOCK_K)
            mask = k < N
            # Load x[k] with zeros for padded part
            x_vals = tl.load(x_ptr + k, mask=mask, other=0.0)
            # angle = 2*pi*j*k/N
            angle = (2.0 * 3.141592653589793 * j * k) / N
            cos_term = tl.cos(angle)
            # Multiply and reduce across the vector chunk
            acc += tl.sum(x_vals * cos_term, axis=0)
        # Normalize
        acc = acc / N
        # Store result to out_ptr[j]
        tl.store(out_ptr + j, acc)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      seqlen, N,  # N = 2 * seqlen
                      BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N
      for j in 1..seqlen-1, where x_ptr points to a padded input vector of length N.
    """
    pid = tl.program_id(axis=0)  # one program per (batch, channel) row
    # Loop over output bins j starting from 1
    for j in range(1, seqlen):
        acc = 0.0
        for k_start in range(0, N, BLOCK_K):
            k = k_start + tl.arange(0, BLOCK_K)
            mask = k < N
            x_vals = tl.load(x_ptr + k, mask=mask, other=0.0)
            angle = (2.0 * 3.141592653589793 * j * k) / N
            sin_term = tl.sin(angle)
            acc += tl.sum(x_vals * sin_term, axis=0)
        acc = acc / N
        tl.store(out_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: input tensor of shape (batch, channels, seqlen)
        Returns:
          x_freq_real: float32 tensor of shape (batch, channels, seqlen+1)
          x_freq_imag: float32 tensor of shape (batch, channels, seqlen+1)
        """
        assert x.is_cuda, "ModelNew requires a CUDA tensor input."
        batch, channels, seqlen = x.shape
        N = 2 * seqlen  # zero-padding length

        # Outputs
        x_freq_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        # Initialize imag[0] and imag[seqlen] to zero
        x_freq_imag[:, :, 0] = 0.0
        x_freq_imag[:, :, seqlen] = 0.0

        # For each row, build padded input vector and launch Triton kernels.
        for b in range(batch):
            for c in range(channels):
                # Row data: convert to float32 and contiguous 1D
                x_row = x[b, c, :].to(torch.float32).contiguous()  # shape (seqlen,)
                # Create padded vector of length N with zeros
                x_padded = torch.zeros((N,), dtype=torch.float32, device=x.device)
                # Copy the original row into the padded vector
                x_padded[:seqlen] = x_row

                # Launch real kernel: j=0..seqlen
                rfft_real_kernel[(1,)](x_padded, x_freq_real[b, c, :],
                                       seqlen, N,
                                       BLOCK_K=1024, num_warps=4)
                # Launch imag kernel: j=1..seqlen-1
                rfft_imag_kernel[(1,)](x_padded, x_freq_imag[b, c, 1:seqlen],
                                       seqlen, N,
                                       BLOCK_K=1024, num_warps=4)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
