import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_one_j(x_ptr, out_ptr_j,
                     N, j, invN,
                     BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for a fixed j across rows:
      real_out[row, j] = sum_{k=0..N-1} x[row, k] * cos(2*pi*j*k/N) * invN
    x_ptr points to a matrix of shape (M, N), where M is number of rows (batch*channels).
    out_ptr_j points to a vector of length M (float32).
    """
    row_id = tl.program_id(0)
    acc = 0.0
    for k in range(0, N, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask = k_offsets < N
        x_vals = tl.load(x_ptr + row_id * N + k_offsets, mask=mask, other=0.0)
        angle = (2.0 * 3.141592653589793 * j * k_offsets) / N
        cos_term = tl.cos(angle)
        acc += tl.sum(x_vals * cos_term, axis=0)
    acc *= invN
    tl.store(out_ptr_j + row_id, acc)


@triton.jit
def rfft_imag_one_j(x_ptr, out_ptr_j,
                     N, j, invN,
                     BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for a fixed j across rows:
      imag_out[row, j] = sum_{k=0..N-1} x[row, k] * sin(2*pi*j*k/N) * invN
    x_ptr points to a matrix of shape (M, N), where M is number of rows (batch*channels).
    out_ptr_j points to a vector of length M (float32).
    """
    row_id = tl.program_id(0)
    acc = 0.0
    for k in range(0, N, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask = k_offsets < N
        x_vals = tl.load(x_ptr + row_id * N + k_offsets, mask=mask, other=0.0)
        angle = (2.0 * 3.141592653589793 * j * k_offsets) / N
        sin_term = tl.sin(angle)
        acc += tl.sum(x_vals * sin_term, axis=0)
    acc *= invN
    tl.store(out_ptr_j + row_id, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton implementation of:
          x_freq = torch.fft.rfft(x, n=2*seqlen) / (2*seqlen)
          return x_freq.real, x_freq.imag  with shape (batch, channels, seqlen+1), dtype float32
        """
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        # Ensure float32 (the original code casts to float32)
        if x.dtype != torch.float32:
            x = x.float()

        batch, channels, seqlen = x.shape
        N = 2 * seqlen
        invN = 1.0 / N

        # Flatten (batch, channels) into rows
        M = batch * channels
        x_2d = x.reshape(M, seqlen)

        # Allocate padded input per row (float32, device)
        x_padded = torch.empty((M, N), dtype=torch.float32, device=x.device)
        # Copy original row into first seqlen entries; remaining N - seqlen are zeros
        for r in range(M):
            x_row = x_2d[r]  # shape (seqlen,)
            x_padded[r, :seqlen] = x_row

        # Allocate output vectors of length M for each j bin
        out_real_list = [torch.empty(M, dtype=torch.float32, device=x.device) for j in range(0, seqlen + 1)]
        out_imag_list = [torch.empty(M, dtype=torch.float32, device=x.device) for j in range(1, seqlen)]  # imag[0] will be zero; imag[seqlen] will be zero

        # Launch real kernel: computes j=0..seqlen
        grid = (M,)
        for j in range(0, seqlen + 1):
            rfft_real_one_j[grid](
                x_padded, out_real_list[j],
                N, j, invN,
                BLOCK_K=1024,
                num_warps=4,
                num_stages=2
            )

        # Launch imag kernel: computes j=1..seqlen-1
        for j in range(1, seqlen):
            rfft_imag_one_j[grid](
                x_padded, out_imag_list[j - 1],
                N, j, invN,
                BLOCK_K=1024,
                num_warps=4,
                num_stages=2
            )

        # Construct imag full vector: imag[0] and imag[seqlen] must be zero; imag_list already excludes them
        imag_full = torch.empty(M, dtype=torch.float32, device=x.device)
        imag_full[0] = 0.0
        imag_full[-1] = 0.0
        imag_full[1:-1] = out_imag_list  # out_imag_list length = seqlen - 1

        # Reshape back to (batch, channels, seqlen+1)
        x_freq_real = out_real_list[0].view(batch, channels, seqlen + 1)
        x_freq_imag = imag_full.view(batch, channels, seqlen + 1)
        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
