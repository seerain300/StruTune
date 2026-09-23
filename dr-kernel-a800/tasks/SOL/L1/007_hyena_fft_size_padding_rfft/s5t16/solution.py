import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      seqlen, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..2*seqlen-1} x[k] * cos(2*pi*j*k/(2*seqlen)) / (2*seqlen)
      for j in 0..seqlen.
    x_ptr points to the padded input vector of length 2*seqlen for this row.
    out_ptr points to the real output vector of length seqlen+1 for this row.
    """
    # One program per (b, c) row
    row_id = tl.program_id(0)
    base_in = row_id * (2 * seqlen)
    base_out = row_id * (seqlen + 1)

    norm = 1.0 / (2.0 * seqlen)

    j = 0
    while j <= seqlen:
        acc = 0.0
        k_start = 0
        while k_start < (2 * seqlen):
            k_vec = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_vec < (2 * seqlen)

            # Load padded input x[k]
            x_val = tl.load(x_ptr + base_in + k_vec, mask=mask_k, other=0.0)

            angle = 2.0 * 3.141592653589793 * j * k_vec / (2.0 * seqlen)
            cosv = tl.cos(angle)

            acc += tl.sum(x_val * cosv, axis=0)

            k_start += BLOCK_K

        out_val = acc * norm
        tl.store(out_ptr + base_out + j, out_val)
        j += 1


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      seqlen, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..2*seqlen-1} x[k] * sin(2*pi*j*k/(2*seqlen)) / (2*seqlen)
      for j in 1..seqlen-1 (imag_out[0] = 0, imag_out[seqlen] = 0).
    x_ptr points to the padded input vector of length 2*seqlen for this row.
    out_ptr points to the imaginary output vector of length seqlen+1 for this row.
    """
    row_id = tl.program_id(0)
    base_in = row_id * (2 * seqlen)
    base_out = row_id * (seqlen + 1)

    norm = 1.0 / (2.0 * seqlen)

    j = 1
    while j < seqlen:
        acc = 0.0
        k_start = 0
        while k_start < (2 * seqlen):
            k_vec = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_vec < (2 * seqlen)

            x_val = tl.load(x_ptr + base_in + k_vec, mask=mask_k, other=0.0)

            angle = 2.0 * 3.141592653589793 * j * k_vec / (2.0 * seqlen)
            sinv = tl.sin(angle)

            acc += tl.sum(x_val * sinv, axis=0)

            k_start += BLOCK_K

        out_val = acc * norm
        tl.store(out_ptr + base_out + j, out_val)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: Input tensor of shape (batch, channels, seqlen)
        Returns:
          x_freq_real: float32 tensor of shape (batch, channels, seqlen+1)
          x_freq_imag: float32 tensor of shape (batch, channels, seqlen+1)
        """
        assert x.dim() == 3, "Input must be 3D (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        B = batch
        C = channels

        # Prepare padded input for each (b, c) row: length = 2 * seqlen, first seqlen are real, rest zeros.
        total_elems = B * C * (2 * seqlen)
        x_padded = torch.empty(total_elems, dtype=torch.float32, device=x.device)

        # Fill padded input: copy x rows into the first seqlen positions (no torch math for compute).
        # We use view/reshape and simple indexing (allowed).
        for r in range(B * C):
            row_in = x[r].reshape(-1)  # shape (seqlen,)
            base = r * (2 * seqlen)
            # Cast to float32 explicitly; view/reshape only
            x_padded[base:base + seqlen] = row_in.to(torch.float32)

        # Allocate outputs
        real_out = torch.empty(B * C * (seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty(B * C * (seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernels: one program per (b, c) row
        grid = (B * C,)
        BLOCK_K = 256  # chunk size for k-loop
        rfft_real_kernel[grid](
            x_padded, real_out,
            seqlen, BLOCK_K=BLOCK_K,
            num_warps=4
        )

        # imag_out[0] and imag_out[seqlen] are zero by definition; set explicitly
        imag_out[0] = 0.0
        imag_out[B * C * seqlen] = 0.0

        rfft_imag_kernel[grid](
            x_padded, imag_out,
            seqlen, BLOCK_K=BLOCK_K,
            num_warps=4
        )

        # Reshape back to (batch, channels, seqlen+1)
        x_freq_real = real_out.view(B, C, seqlen + 1)
        x_freq_imag = imag_out.view(B, C, seqlen + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
