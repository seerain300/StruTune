import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_N_kernel(x_row_ptr, padded_ptr, seqlen, N):
    """
    Pad the input row x_row_ptr of length 'seqlen' into padded_ptr of length 'N'.
    Assume N = 2 * seqlen. Copy x_row_ptr into padded_ptr[0:seqlen] and write zeros into padded_ptr[seqlen:N].
    """
    pid = tl.program_id(0)
    total_elems = N
    BLOCK = 1024
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    copy_mask = offsets < seqlen
    vals = tl.load(x_row_ptr + offsets, mask=copy_mask, other=0.0)
    tl.store(padded_ptr + offsets, vals, mask=mask)


@triton.jit
def compute_rfft_real_kernel(padded_ptr, out_real_ptr, seqlen, inv_N):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..2*seqlen-1} padded[k] * cos(2*pi*j*k/(2*seqlen)) * inv_N
      for j in 0..seqlen.
    out_real_ptr points to the output vector of length seqlen+1.
    inv_N = 1.0 / (2 * seqlen).
    """
    pid = tl.program_id(0)
    N = 2 * seqlen
    for j in range(0, seqlen + 1):
        acc = 0.0
        for k0 in range(0, N, 1024):
            k_offsets = k0 + tl.arange(0, 1024)
            k_mask = k_offsets < N
            xk = tl.load(padded_ptr + k_offsets, mask=k_mask, other=0.0)
            j_f = j
            angle = 2.0 * 3.141592653589793 * j_f * k_offsets / N
            cosv = tl.cos(angle)
            # Accumulate; cast to float32
            prod = xk * cosv
            prod = prod.to(tl.float32)
            acc += tl.sum(prod, axis=0)
        out_val = acc * inv_N
        tl.store(out_real_ptr + j, out_val)


@triton.jit
def compute_rfft_imag_kernel(padded_ptr, out_imag_ptr, seqlen, inv_N):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..2*seqlen-1} padded[k] * sin(2*pi*j*k/(2*seqlen)) * inv_N
      for j in 1..seqlen-1.
    out_imag_ptr points to the output vector of length seqlen+1.
    inv_N = 1.0 / (2 * seqlen).
    imag_out[0] and imag_out[seqlen] are zero.
    """
    pid = tl.program_id(0)
    N = 2 * seqlen
    for j in range(1, seqlen):
        acc = 0.0
        for k0 in range(0, N, 1024):
            k_offsets = k0 + tl.arange(0, 1024)
            k_mask = k_offsets < N
            xk = tl.load(padded_ptr + k_offsets, mask=k_mask, other=0.0)
            j_f = j
            angle = 2.0 * 3.141592653589793 * j_f * k_offsets / N
            sinv = tl.sin(angle)
            prod = xk * sinv
            prod = prod.to(tl.float32)
            acc += tl.sum(prod, axis=0)
        out_val = acc * inv_N
        tl.store(out_imag_ptr + j, out_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Input: x (batch, channels, seqlen) float32 CUDA tensor.
        Output: (batch, channels, seqlen+1) float32 tensors, real and imaginary parts.
        """
        assert x.is_cuda, "Input must be on CUDA device."
        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Allocate outputs
        x_freq_real = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels per (batch, channel) row
        for b in range(batch):
            for c in range(channels):
                # Prepare pointers
                x_row_ptr = x[b, c, :].contiguous()
                # Allocate padded buffer of length N
                padded = torch.empty(N, dtype=torch.float32, device=x.device)
                # Launch padding kernel: one program instance covering all N elements
                pad_to_N_kernel[(1,)](x_row_ptr, padded, seqlen, N, num_warps=1)
                # Launch real and imag kernels: one program instance per row
                inv_N = 1.0 / float(N)
                compute_rfft_real_kernel[(1,)](padded, x_freq_real[b, c, :], seqlen, inv_N, num_warps=1)
                compute_rfft_imag_kernel[(1,)](padded, x_freq_imag[b, c, :], seqlen, inv_N, num_warps=1)
                # Ensure imag[0] and imag[seqlen] are zero (mathematically guaranteed, but set explicitly)
                x_freq_imag[b, c, 0] = 0.0
                x_freq_imag[b, c, seqlen] = 0.0

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
