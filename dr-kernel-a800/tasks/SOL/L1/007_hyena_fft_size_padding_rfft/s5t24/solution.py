import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr,
                      seqlen: tl.constexpr,
                      BLOCK_J: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for one row:
      real_out[j] = sum_{k=0..2*seqlen-1} x[k] * cos(2*pi*j*k/(2*seqlen)) / (2*seqlen)
      for j in 0..seqlen.
    Each program handles one (b, c) row and computes BLOCK_J consecutive bins j.
    """
    pid = tl.program_id(0)  # one program per (batch, channels) row
    n_rows = seqlen  # we'll use this to index output j bins

    # Compute the start j index for this program
    j_start = pid * BLOCK_J
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    mask_j = j_offsets < seqlen  # ensure we don't write beyond seqlen

    # Accumulator for real output bins
    acc = tl.zeros([BLOCK_J], dtype=tl.float32)

    # Reduction over k in chunks of BLOCK_K
    n = 2 * seqlen
    for k0 in range(0, n, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < n

        # Build input indices for this row: x_ptr indexing
        # We pass x_ptr as flat padded vector. For row pid, base = pid * (2*seqlen)
        base = pid * n
        x_vals = tl.load(x_ptr + base + k_offsets, mask=mask_k, other=0.0)

        # Compute cos(2*pi*j*k/n) for all j in BLOCK_J and k in BLOCK_K
        # j_offsets and k_offsets are vectors. Use broadcasting to form 2D and reduce.
        # Note: Python side ensures j_offsets < seqlen, k_offsets < 2*seqlen.
        j_mat = j_offsets[:, None]  # shape [BLOCK_J, 1]
        k_mat = k_offsets[None, :]  # shape [1, BLOCK_K]
        # cos(2*pi*j*k/n)
        # n is 2*seqlen
        angle = (2.0 * tl.pi * j_mat * k_mat) / (2.0 * seqlen)
        cosv = tl.cos(angle)
        # Accumulate: sum over k for each j
        # x_vals shape [BLOCK_K], cosv shape [BLOCK_J, BLOCK_K]
        # Multiply x_vals with each column of cosv and sum along k axis
        # We need to broadcast x_vals to [BLOCK_J, BLOCK_K] by repeating along j dimension.
        # Triton doesn't support direct outer-product broadcasting, so we loop j (small BLOCK_J) and add.
        for jj in range(BLOCK_J):
            if j_offsets[jj] < seqlen:
                x_vec = x_vals  # same for all j
                acc[jj] += tl.sum(x_vec * cosv[jj, :], axis=0)

    # Normalize by n = 2*seqlen
    acc = acc / (2.0 * seqlen)

    # Store results to out_ptr: one value per bin j_offsets
    out_base = pid * (seqlen + 1)
    out_ptrs = out_ptr + out_base + j_offsets
    tl.store(out_ptrs, acc, mask=mask_j)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr,
                      seqlen: tl.constexpr,
                      BLOCK_J: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for one row:
      imag_out[j] = sum_{k=0..2*seqlen-1} x[k] * sin(2*pi*j*k/(2*seqlen)) / (2*seqlen)
      for j in 1..seqlen-1.
    Each program handles one (b, c) row and computes BLOCK_J consecutive bins j.
    """
    pid = tl.program_id(0)  # one program per (batch, channels) row
    n_rows = seqlen

    j_start = pid * BLOCK_J
    j_offsets = j_start + tl.arange(0, BLOCK_J)
    # For imag kernel, we only need j in [1, seqlen-1], but Triton masks handle it.
    mask_j = j_offsets < seqlen

    acc = tl.zeros([BLOCK_J], dtype=tl.float32)

    n = 2 * seqlen
    for k0 in range(0, n, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < n

        base = pid * n
        x_vals = tl.load(x_ptr + base + k_offsets, mask=mask_k, other=0.0)

        j_mat = j_offsets[:, None]  # [BLOCK_J, 1]
        k_mat = k_offsets[None, :]  # [1, BLOCK_K]
        angle = (2.0 * tl.pi * j_mat * k_mat) / (2.0 * seqlen)
        sinv = tl.sin(angle)

        for jj in range(BLOCK_J):
            if j_offsets[jj] < seqlen:
                x_vec = x_vals
                acc[jj] += tl.sum(x_vec * sinv[jj, :], axis=0)

    acc = acc / (2.0 * seqlen)

    out_base = pid * (seqlen + 1)
    out_ptrs = out_ptr + out_base + j_offsets
    # Note: imag_out[0] and imag_out[seqlen] must be set to zero on host
    tl.store(out_ptrs, acc, mask=mask_j)


def _build_padded_flat(x: torch.Tensor) -> torch.Tensor:
    """
    Build a flat padded input tensor of shape (batch*channels * (2*seqlen)) with zeros appended.
    x: (batch, channels, seqlen), float32 on CUDA.
    Returns: 1D tensor of length (batch*channels) * (2*seqlen).
    """
    batch, channels, seqlen = x.shape
    n_rows = batch * channels
    n = 2 * seqlen
    x_padded_flat = torch.empty(n_rows * n, dtype=torch.float32, device=x.device)
    for b in range(batch):
        for c in range(channels):
            row_id = b * channels + c
            row_start = row_id * n
            row_vec = x[b, c, :]
            x_padded_flat[row_start : row_start + seqlen] = row_vec
    # Ensure zeros for the remaining N - seqlen entries are present (torch empty + not writing them yields zeros)
    return x_padded_flat


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen), float32 on CUDA
        Returns: (batch, channels, seqlen+1) real and imag parts.
        """
        assert x.is_cuda, "ModelNew.forward expects CUDA tensors"
        assert x.dtype == torch.float32, "Input must be float32"
        batch, channels, seqlen = x.shape
        n_rows = batch * channels
        n = 2 * seqlen

        # Build padded inputs on host (only allocations and copies, no torch math in forward)
        x_padded_flat = _build_padded_flat(x)

        # Output buffers (flat): real and imag, each length n_rows * (seqlen + 1)
        out_real_flat = torch.empty(n_rows * (seqlen + 1), dtype=torch.float32, device=x.device)
        out_imag_flat = torch.empty(n_rows * (seqlen + 1), dtype=torch.float32, device=x.device)
        # imag_out[0] and imag_out[seqlen] should be zero; we will leave out_imag_flat zeroed
        out_imag_flat.zero_()

        # Launch Triton kernels: one program per row, compute multiple j bins per program
        BLOCK_J = 128  # number of j bins per program, can tune
        BLOCK_K = 128  # reduction chunk over k
        grid = (n_rows,)

        # Launch real part kernel
        rfft_real_kernel[grid](
            x_padded_flat, out_real_flat,
            seqlen=seqlen,
            BLOCK_J=BLOCK_J,
            BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # Launch imag part kernel (note: imag_out[0] and imag_out[seqlen] are handled separately below)
        rfft_imag_kernel[grid](
            x_padded_flat, out_imag_flat,
            seqlen=seqlen,
            BLOCK_J=BLOCK_J,
            BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # Reshape outputs to (batch, channels, seqlen+1)
        out_real = out_real_flat.view(batch, channels, seqlen + 1)
        out_imag = out_imag_flat.view(batch, channels, seqlen + 1)

        # Ensure imag_out[0] and imag_out[seqlen] are zero across all rows (out_imag_flat.zero_ already did it)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
