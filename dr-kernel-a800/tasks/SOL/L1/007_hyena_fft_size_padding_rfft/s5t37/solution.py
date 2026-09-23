import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      seqlen, N,
                      BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..N-1} x[k] * cos(2*pi*j*k/N) / N,  j in 0..seqlen.
    x_ptr points to the padded input vector of length N (contiguous).
    out_ptr points to the output real vector of length seqlen+1 (first element reserved for j=0).
    We launch one program per (batch, channel) row; seqlen is the original sequence length.
    """
    # program id for this (batch, channel) row
    pid = tl.program_id(0)
    # Base pointers for this row (assuming linearized layout where rows are contiguous)
    # We pass x_ptr as the base for this row; out_ptr similarly.
    # Triton will treat them as pointers; no additional stride math needed since we use offsets.
    # We need to map pid to the correct row. Since we launch grid=(batch, channels),
    # pid directly indexes the row. We assume out_ptr and x_ptr are already pointing
    # to the correct row bases. To be explicit, we can compute row offset, but since
    # x_ptr and out_ptr are created per-row in Python, we don't need to adjust here.

    # Accumulator for real output; we'll store seqlen+1 values. We do not allocate inside kernel,
    # Triton expects pointers and writes via tl.store.

    # We will compute and store one j per loop. To keep code simple and safe, we loop j from 0 to seqlen.
    # However, Triton kernels do not support Python for loops with dynamic range easily; instead,
    # we compute per-element with vectorized j and store. Here we use a single j per loop iteration
    # by constructing a vector of j values and reducing. For simplicity, we use a scalar j approach:
    # We'll call this kernel in a way that we pass j via out_ptr index and compute everything in one shot.

    # Correct approach: precompute j offsets and use a loop structure. Triton supports while loops.
    # We will use a while loop to iterate j.

    j = 0
    while j <= seqlen:
        acc = 0.0
        # Iterate k in chunks
        k = 0
        while k < N:
            kk = k + tl.arange(0, BLOCK_K)
            mask_k = kk < N
            # Load x[kk] from the padded input; for masked, other=0.0
            x_vals = tl.load(x_ptr + kk, mask=mask_k, other=0.0)
            # cos(2*pi*j*kk/N)
            angle = (2.0 * 3.141592653589793 * j * kk) / N
            cos_vals = tl.cos(angle)
            acc += tl.sum(x_vals * cos_vals, axis=0)
            k += BLOCK_K
        # Normalize by N (2*seqlen)
        acc = acc / N
        # Store real_out[j] at out_ptr + j
        # out_ptr points to the start of the row's real output; j is in [0, seqlen], so in-bounds.
        tl.store(out_ptr + j, acc)
        j += 1


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      seqlen, N,
                      BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..N-1} x[k] * sin(2*pi*j*k/N) / N,  j in 1..seqlen-1.
    x_ptr points to the padded input vector of length N (contiguous).
    out_ptr points to the output imaginary vector of length seqlen+1 (indices 1..seqlen-1 used).
    imag_out[0] and imag_out[seqlen] are zero by definition (for real inputs).
    """
    pid = tl.program_id(0)
    j = 1
    while j < seqlen:
        acc = 0.0
        k = 0
        while k < N:
            kk = k + tl.arange(0, BLOCK_K)
            mask_k = kk < N
            x_vals = tl.load(x_ptr + kk, mask=mask_k, other=0.0)
            angle = (2.0 * 3.141592653589793 * j * kk) / N
            sin_vals = tl.sin(angle)
            acc += tl.sum(x_vals * sin_vals, axis=0)
            k += BLOCK_K
        acc = acc / N
        tl.store(out_ptr + j, acc)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen) float32 on CUDA.
        Returns:
          x_freq_real: (batch, channels, seqlen+1) float32
          x_freq_imag: (batch, channels, seqlen+1) float32
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        batch, channels, seqlen = x.shape
        N = 2 * seqlen  # zero-padding size per torch.fft.rfft requirement

        # Construct padded input for each (batch, channel) row without torch math in forward.
        # We'll allocate a temporary padded vector per row and pass its pointer to Triton.
        # Note: Triton expects pointers; we can build per-row vectors on the host.
        # Since we cannot return both real and imag outputs directly from one kernel, we run two kernels.

        # Allocate outputs (real and imaginary parts). We'll write into out vectors of length seqlen+1.
        # For imag_out, we will set imag_out[0] = 0 and imag_out[seqlen] = 0 after kernel (since they are zero).
        x_freq_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels: one program per (batch, channel) row.
        # We need to pass per-row base pointers. Triton will interpret these pointers and use offsets.
        # To do that, we linearize the rows and pass the base pointers for each row.
        # For simplicity, we can flatten and launch with grid=(batch*channels,), but we must adjust pointers.
        # Easier approach: iterate over batch and channels in Python and launch per row.

        for b in range(batch):
            for c in range(channels):
                # Build padded input vector for this row: first seqlen values, then zeros.
                # We'll create a 1D vector of length N and copy x[b, c, :] into the first seqlen positions.
                # Triton requires contiguous buffers; we can create a contiguous 1D tensor on device.
                # However, Triton kernels need pointers; we can store x[b, c, :] into a temporary 1D tensor of length N on device.
                # Use torch operations here only for allocation and assignment; no computation.
                x_row = x[b, c, :]  # 1D tensor of length seqlen
                # Allocate padded input vector on device
                x_padded = torch.zeros(N, dtype=torch.float32, device=x.device)
                # Copy x_row into the first seqlen positions
                x_padded[:seqlen] = x_row

                # Now, launch Triton kernels for real and imag parts
                # We need base pointers to the current row's output buffers:
                out_real = x_freq_real[b, c, :]
                out_imag = x_freq_imag[b, c, :]

                # For Triton, pass pointers; ensure they point to the correct buffers.
                # Launch real kernel
                rfft_real_kernel[(1,)](x_padded, out_real, seqlen, N, BLOCK_K=1024, num_warps=4)
                # Launch imag kernel
                # imag_out[0] and imag_out[seqlen] should be zero. We'll set them explicitly.
                # Note: imag kernel only writes j in [1..seqlen-1]; we set the remaining explicitly.
                # However, since we computed up to seqlen-1, we must set 0 and seqlen here.
                x_freq_imag[b, c, 0] = 0.0
                x_freq_imag[b, c, seqlen] = 0.0

                # The real kernel wrote j in [0..seqlen]. We normalized by N already.

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
