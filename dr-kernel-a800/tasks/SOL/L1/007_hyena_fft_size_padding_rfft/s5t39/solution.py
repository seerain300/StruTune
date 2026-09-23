import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(x_ptr, out_ptr, total_rows, seqlen, N, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute real part of rfft for each row:
      real_out[j] = sum_{k=0..2*seqlen-1} x[k] * cos(2*pi*j*k/(2*seqlen)) / (2*seqlen)
      for j in 0..seqlen, where each row corresponds to one (batch, channel).
    x_ptr points to the padded input vector (length = 2*seqlen) for each row.
    out_ptr points to real output (length = seqlen + 1) for each row.
    """
    row_id = tl.program_id(0)
    # if row_id >= total_rows: return
    # compute (b, c) from row_id
    # Note: We assume grid is set to total_rows, so row_id < total_rows.

    # Prepare j offsets
    j_offsets = tl.arange(0, BLOCK_J)
    # We'll process j in chunks
    # We need to write out real_out[j] for j in 0..seqlen
    # We can unroll for small seqlen or process in chunks
    # Here we use a loop over j chunks
    # For each chunk, accumulate over k in chunks
    # We'll set num_warps to handle throughput; for simplicity, loop over j.

    # Because Triton does not support while with dynamic conditions cleanly in this context,
    # we implement j loop via range using constexpr chunk size, but Triton requires static ranges.
    # Therefore, we restructure: one Triton program handles one row and loops over j in Python range.
    # However, Triton kernels should be defined with static shapes. Instead, we compute j via linear index and mask.

    # Simpler approach: launch one program per row; inside, loop j and k.
    # But Triton kernels need static loops. Hence, we precompute j_offsets and mask per chunk.
    # For correctness, we keep a chunked approach:

    # We will iterate j in chunks of BLOCK_J. Triton allows loops with compile-time constants.
    # However, Triton does not support arbitrary dynamic range loops in Python; we use range with compile-time step.

    # Compute number of chunks in j dimension
    # Since BLOCK_J is constexpr, we need to encode j loop as range(start, seqlen+1, BLOCK_J).
    # Triton supports such range loops.

    # Accumulator for real_out
    acc = tl.zeros([BLOCK_J], dtype=tl.float32)

    # Loop over j in chunks
    for start_j in range(0, seqlen + 1, BLOCK_J):
        j = start_j + j_offsets
        j_mask = j < (seqlen + 1)
        # Initialize accumulator for this chunk
        acc = tl.zeros([BLOCK_J], dtype=tl.float32)
        # Loop over k in chunks
        for start_k in range(0, 2 * seqlen, BLOCK_K):
            k = start_k + tl.arange(0, BLOCK_K)
            k_mask = k < (2 * seqlen)
            # Load x[k] for current row
            # Each row is a contiguous vector of length 2*seqlen
            # row_offset = row_id * (2 * seqlen)
            x_row_ptr = x_ptr + row_id * (2 * seqlen)
            x_vals = tl.load(x_row_ptr + k, mask=k_mask, other=0.0).to(tl.float32)

            # Compute cos(2*pi*j*k/N) for current j and k chunk
            # Use float32 for stability
            # N is scalar int; cast to float
            N_f = tl.full((), 2 * seqlen, tl.float32)
            # j is vector [BLOCK_J], k is vector [BLOCK_K]
            # Create 2D for broadcast
            j_mat = j[:, None]  # shape [BLOCK_J, 1]
            k_mat = k[None, :]  # shape [1, BLOCK_K]
            cos_arg = (2.0 * 3.141592653589793 * j_mat * k_mat) / N_f
            cos_vals = tl.cos(cos_arg)  # shape [BLOCK_J, BLOCK_K]
            # Accumulate
            # x_vals is [BLOCK_K], broadcast across j: multiply cos_vals (shape [BLOCK_J, BLOCK_K]) * x_vals[:, None]
            # We need x_vals broadcast to [BLOCK_J, BLOCK_K]
            # The simplest: for each k, x_vals[k] contributes to all j
            # Loop over kk in BLOCK_K (compile-time), add to acc
            for kk in range(0, BLOCK_K):
                kkk = start_k + kk
                if kkk < (2 * seqlen):
                    # broadcast x_vals[kkk] across j
                    x_k = x_vals[kk]
                    # acc += x_k * cos_vals[:, kk]
                    # cos_vals[:, kk] is [BLOCK_J]
                    acc += x_k * cos_vals[:, kk]
        # Store results for valid j
        out_row_ptr = out_ptr + row_id * (seqlen + 1)
        # Normalize by N
        acc = acc / (2.0 * seqlen)
        # Store with mask
        tl.store(out_row_ptr + j, acc, mask=j_mask)


@triton.jit
def rfft_imag_kernel(x_ptr, out_ptr, total_rows, seqlen, N, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute imaginary part of rfft for each row for j in 1..seqlen-1:
      imag_out[j] = sum_{k=0..2*seqlen-1} x[k] * sin(2*pi*j*k/(2*seqlen)) / (2*seqlen)
    x_ptr points to the padded input vector (length = 2*seqlen) for each row.
    out_ptr points to imag output (length = seqlen + 1) for each row.
    imag_out[0] and imag_out[seqlen] are set to zero outside the kernel.
    """
    row_id = tl.program_id(0)

    acc = tl.zeros([BLOCK_J], dtype=tl.float32)

    for start_j in range(1, seqlen, BLOCK_J):
        j = start_j + j_offsets
        j_mask = j < seqlen
        acc = tl.zeros([BLOCK_J], dtype=tl.float32)
        for start_k in range(0, 2 * seqlen, BLOCK_K):
            k = start_k + tl.arange(0, BLOCK_K)
            k_mask = k < (2 * seqlen)

            x_row_ptr = x_ptr + row_id * (2 * seqlen)
            x_vals = tl.load(x_row_ptr + k, mask=k_mask, other=0.0).to(tl.float32)

            N_f = tl.full((), 2 * seqlen, tl.float32)
            j_mat = j[:, None]
            k_mat = k[None, :]
            sin_arg = (2.0 * 3.141592653589793 * j_mat * k_mat) / N_f
            sin_vals = tl.sin(sin_arg)  # [BLOCK_J, BLOCK_K]

            for kk in range(0, BLOCK_K):
                kkk = start_k + kk
                if kkk < (2 * seqlen):
                    x_k = x_vals[kk]
                    acc += x_k * sin_vals[:, kk]

        out_row_ptr = out_ptr + row_id * (seqlen + 1)
        acc = acc / (2.0 * seqlen)
        tl.store(out_row_ptr + j, acc, mask=j_mask)


def _next_power_of_two(x: int) -> int:
    # Utility to choose BLOCK sizes
    return 1 << (x - 1).bit_length()


def _pick_num_warps(size: int) -> int:
    # Simple heuristic
    if size <= 512:
        return 2
    elif size <= 2048:
        return 4
    else:
        return 8


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        x: (batch, channels, seqlen)
        returns:
          real: (batch, channels, seqlen+1), float32
          imag: (batch, channels, seqlen+1), float32
        """
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        N = 2 * seqlen
        total_rows = batch * channels

        # Prepare padded inputs per row without using torch math in forward
        # We'll allocate a 2D buffer [total_rows, N] and fill it from x.
        x_flat = x.view(total_rows, seqlen).contiguous()
        x_padded = torch.zeros((total_rows, N), dtype=torch.float32, device=x.device)
        # Fill first seqlen elements
        x_padded[:, :seqlen] = x_flat

        # Allocate outputs
        real_out = torch.empty((total_rows, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((total_rows, seqlen + 1), dtype=torch.float32, device=x.device)

        # Choose block sizes
        BLOCK_J = min(_next_power_of_two(seqlen + 1), 128)
        BLOCK_K = min(_next_power_of_two(N), 1024)
        num_warps = _pick_num_warps(max(seqlen + 1, N))

        # Launch real kernel
        grid_real = (total_rows,)
        rfft_real_kernel[grid_real](
            x_padded, real_out, total_rows, seqlen, N,
            BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=num_warps
        )

        # Launch imag kernel (compute only for j=1..seqlen-1, set j=0 and j=seqlen to zero)
        grid_imag = (total_rows,)
        rfft_imag_kernel[grid_imag](
            x_padded, imag_out, total_rows, seqlen, N,
            BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, num_warps=num_warps
        )
        # Set imag_out[0] and imag_out[seqlen] to zero (due to symmetry)
        imag_out[:, 0] = 0.0
        imag_out[:, seqlen] = 0.0

        # Reshape back to (batch, channels, seqlen+1)
        real_out = real_out.view(batch, channels, seqlen + 1)
        imag_out = imag_out.view(batch, channels, seqlen + 1)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
