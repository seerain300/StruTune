import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_ptr, padded_ptr, L: tl.int32, N: tl.int32):
    """
    Copy one row of length L to padded buffer of length N=2*L.
    x_ptr: base pointer to row data (already size L)
    padded_ptr: base pointer to output padded row (size N), we'll write first L elements
    """
    row_offset = tl.program_id(0) * L
    # Only copy valid j < L
    for j in tl.static_range(0, L):
        val = tl.load(x_ptr + row_offset + j)
        tl.store(padded_ptr + row_offset + j, val)


@triton.jit
def _compute_rfft_real_per_k(padded_ptr, out_real_ptr, N: tl.int32, L: tl.int32):
    """
    For each (row, k): accumulate sum_{j=0}^{N-1} padded[j] * cos(2*pi*k*j/N)
    into out_real[row, k]. One program per (row, k).
    """
    row = tl.program_id(0)
    k = tl.program_id(1)
    acc = 0.0
    # Loop over j from 0 to N-1
    for j in tl.static_range(0, N):
        val = tl.load(padded_ptr + row * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        cosv = tl.cos(angle)
        acc += val * cosv
    # Store partial accumulation; normalization happens in a separate kernel.
    tl.store(out_real_ptr + row * (L + 1) + k, acc)


@triton.jit
def _compute_rfft_imag_per_k(padded_ptr, out_imag_ptr, N: tl.int32, L: tl.int32):
    """
    For each (row, k): accumulate sum_{j=0}^{N-1} padded[j] * sin(2*pi*k*j/N)
    into out_imag[row, k]. One program per (row, k).
    """
    row = tl.program_id(0)
    k = tl.program_id(1)
    acc = 0.0
    # Loop over j from 0 to N-1
    for j in tl.static_range(0, N):
        val = tl.load(padded_ptr + row * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        sinv = tl.sin(angle)
        acc += val * sinv
    tl.store(out_imag_ptr + row * (L + 1) + k, acc)


@triton.jit
def _normalize_div(p_vec_ptr, out_ptr, N: tl.int32):
    """
    Elementwise division: out[i] = p_vec[i] / N for i in [0, size(P)).
    We assume P is a flat vector, size is passed as N (number of elements).
    Grid: (size(P)/N,)
    """
    offset = tl.program_id(0) * N
    for i in tl.static_range(0, N):
        val = tl.load(p_vec_ptr + offset + i)
        val = val / N
        tl.store(out_ptr + offset + i, val)


@triton.jit
def _divide_vec_scalar(in_ptr, out_ptr, N: tl.int32, divisor: tl.float32):
    """
    Elementwise division: out[i] = in[i] / divisor for i in [0, N).
    Grid: (1,)
    """
    # This is a simple 1D elementwise division over N elements
    # Triton will unroll or handle vectorized loads; divisor is a scalar.
    # Note: We launch with grid size (1,), and we iterate over N elements in the kernel.
    for i in tl.static_range(0, N):
        val = tl.load(in_ptr + i)
        val = val / divisor
        tl.store(out_ptr + i, val)


def _triton_run(x: torch.Tensor):
    """
    Triton implementation of the original run:
    - Pads L to n=2*L, computes rfft real/imag parts, normalizes by n, returns float32 tensors of shape (B, C, L+1).
    """
    B, C, L = x.shape
    N = 2 * L
    device = x.device

    # Ensure input is float32
    x_f32 = x.to(torch.float32)
    # Flatten rows: (rows, L)
    rows = B * C
    x_flat = x_f32.view(rows, L).contiguous()

    # Allocate padded buffers: (rows, N) with zeros
    padded = torch.zeros((rows, N), dtype=torch.float32, device=device)

    # Copy each row to padded[0:L]
    _copy_row_to_padded[(rows,)](x_flat, padded, L, N, num_warps=1)

    # Allocate outputs: (rows, L+1)
    out_real = torch.zeros((rows, L + 1), dtype=torch.float32, device=device)
    out_imag = torch.zeros((rows, L + 1), dtype=torch.float32, device=device)

    # Compute real parts for k in [0..L]
    grid_k = (rows, L + 1)  # one program per (row, k)
    _compute_rfft_real_per_k[grid_k](padded, out_real, N, L, num_warps=1)

    # Compute imag parts for k in [0..L]
    _compute_rfft_imag_per_k[grid_k](padded, out_imag, N, L, num_warps=1)

    # Normalize by n = 2*L. Triton elementwise division for vectors.
    # We need to divide out_real and out_imag by n. Use _divide_vec_scalar over their flattened vectors.
    # First, flatten and allocate outputs
    out_real_flat = out_real.view(-1)  # length = rows * (L+1)
    out_imag_flat = out_imag.view(-1)
    n_elems = out_real_flat.numel()

    # We launch _divide_vec_scalar with grid size (1,) and it iterates over n_elems
    _divide_vec_scalar[(1,)](out_real_flat, out_real_flat, n_elems, float(2 * L))
    _divide_vec_scalar[(1,)](out_imag_flat, out_imag_flat, n_elems, float(2 * L))

    # Reshape back to (rows, L+1)
    out_real = out_real_flat.view(rows, L + 1)
    out_imag = out_imag_flat.view(rows, L + 1)

    # Reshape to (B, C, L+1)
    out_real = out_real.view(B, C, L + 1)
    out_imag = out_imag.view(B, C, L + 1)

    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single tensor input x of shape (batch, channels, seqlen)
        x = args[0] if len(args) > 0 else None
        if x is None:
            raise ValueError("ModelNew.forward expects a tensor input.")
        # Run Triton implementation
        real, imag = _triton_run(x)
        return real, imag


def run(*args):
    return ModelNew()(*args)
