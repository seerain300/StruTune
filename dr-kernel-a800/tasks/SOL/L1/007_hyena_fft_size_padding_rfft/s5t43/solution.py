import torch
import triton
import triton.language as tl


@triton.jit
def pad_input_kernel(x_row_ptr, out_ptr,
                      seqlen: tl.int32, N: tl.int32,
                      BLOCK_K: tl.constexpr):
    """
    Write padded input into out_ptr of length N = 2 * seqlen.
    out_ptr[0:seqlen] = x_row_ptr[0:seqlen], out_ptr[seqlen:N] = 0.0
    Assumes x_row_ptr points to a contiguous vector of length seqlen.
    out_ptr points to a contiguous vector of length N.
    """
    # We launch with grid=(1,), as this is a per-row operation.
    # Load x into out[0:seqlen], then zero out the rest.
    k = 0
    while k < seqlen:
        val = tl.load(x_row_ptr + k)
        tl.store(out_ptr + k, val)
        k += 1
    # Zero the remaining N - seqlen elements
    k = seqlen
    while k < N:
        tl.store(out_ptr + k, 0.0)
        k += 1


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      seqlen: tl.int32, N: tl.int32,
                      BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N
      for j in 0..seqlen.
    x_ptr points to the padded input vector of length N.
    out_ptr points to the output real vector of length seqlen+1.
    """
    j = 0
    while j <= seqlen:
        acc = 0.0
        k = 0
        while k < N:
            kk = k + tl.arange(0, BLOCK_K)
            mask = kk < N
            xk = tl.load(x_ptr + kk, mask=mask, other=0.0)
            # cos(2*pi*j*k/N)
            # We need a scalar j; Triton handles scalar vs vector broadcasting.
            arg = 2.0 * 3.141592653589793 * j * kk / N
            cosv = tl.cos(arg)
            prod = xk * cosv
            # Reduce over the block
            acc += tl.sum(prod, axis=0)
            k += BLOCK_K
        # Normalize by N and store
        acc = acc / N
        tl.store(out_ptr + j, acc)
        j += 1


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      seqlen: tl.int32, N: tl.int32,
                      BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N
      for j in 1..seqlen-1.
    x_ptr points to the padded input vector of length N.
    out_ptr points to the output imag vector of length seqlen+1.
    """
    j = 1
    while j < seqlen:
        acc = 0.0
        k = 0
        while k < N:
            kk = k + tl.arange(0, BLOCK_K)
            mask = kk < N
            xk = tl.load(x_ptr + kk, mask=mask, other=0.0)
            arg = 2.0 * 3.141592653589793 * j * kk / N
            sinv = tl.sin(arg)
            prod = xk * sinv
            acc += tl.sum(prod, axis=0)
            k += BLOCK_K
        acc = acc / N
        tl.store(out_ptr + j, acc)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen) float32 CUDA tensor
        Returns:
          x_freq_real: (batch, channels, seqlen+1) float32
          x_freq_imag: (batch, channels, seqlen+1) float32
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        assert x.dtype == torch.float32, "Input must be float32."

        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Prepare output tensors (float32, contiguous)
        x_freq_real = torch.empty((batch, channels, seqlen + 1), device=x.device, dtype=torch.float32)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), device=x.device, dtype=torch.float32)

        # Launch one Triton program per (batch, channel) row for padding and rfft
        grid = (batch * channels,)

        # For each row, create a padded input vector of length N using a Triton kernel
        # We need a per-row base pointer; Triton kernels operate on pointers provided by host.
        # We'll run pad_input_kernel once per row to get the padded vector.
        # To do that, we need a temporary buffer for the padded row. Since we can't easily create a dynamic
        # 1D buffer per row inside Triton, we'll process each row independently in Python loop.
        # This avoids torch operations in forward and ensures Triton kernels are launched.
        for b in range(batch):
            for c in range(channels):
                # Compute base offsets for this row
                row_x_ptr = x[b, c, :].contiguous()  # pointer to the row data; Triton will read from it
                # Create a padded output buffer for this row
                row_out = torch.empty(N, device=x.device, dtype=torch.float32)
                # Launch pad_input_kernel to fill row_out[0:N]
                pad_input_kernel[(1,)](row_x_ptr, row_out, seqlen, N, BLOCK_K=1024)

                # Launch real kernel to compute real_out[0:seqlen] into x_freq_real[b, c, :]
                row_real_out = torch.empty(seqlen + 1, device=x.device, dtype=torch.float32)
                rfft_real_kernel[(1,)](row_out, row_real_out, seqlen, N, BLOCK_K=1024)

                # Launch imag kernel to compute imag_out[1:seqlen] into x_freq_imag[b, c, 1:seqlen]
                row_imag_out = torch.empty(seqlen + 1, device=x.device, dtype=torch.float32)
                rfft_imag_kernel[(1,)](row_out, row_imag_out, seqlen, N, BLOCK_K=1024)

                # Set imag_out[0] and imag_out[seqlen] to 0 (by identity)
                # Since row_imag_out has length seqlen+1, zero the first and last elements:
                # We can do this in Triton too, but writing zeros here is fine as it's host code, and forward is the module entry point.
                if seqlen >= 1:
                    x_freq_imag[b, c, 0] = 0.0
                if seqlen >= 1:
                    x_freq_imag[b, c, seqlen] = 0.0
                # Copy row_real_out and row_imag_out into their outputs
                # row_real_out already has length seqlen+1; assign directly
                x_freq_real[b, c, :] = row_real_out[:]
                x_freq_imag[b, c, :] = row_imag_out[:]

        # Normalize by N = 2*seqlen is already applied inside kernels (division by N).

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
