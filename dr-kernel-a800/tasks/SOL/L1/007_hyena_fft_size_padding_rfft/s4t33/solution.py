import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_ptr, padded_ptr, L: tl.int32, N: tl.int32):
    """
    Copy x[row, :] of length L into padded[row, :] at indices [0..L-1].
    padded has length N (>= L), elements [L..N-1] should be left as zeros (padded via torch.zeros in host).
    Grid: (rows,)
    """
    row = tl.program_id(0)
    # x_ptr points to [rows, L], padded_ptr points to [rows, N]
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + row * L + i)
        tl.store(padded_ptr + row * N + i, val)


@triton.jit
def _compute_rfft_real_per_k(padded_ptr, out_real_ptr, N: tl.int32):
    """
    Compute y_real[k] for a given (row, k):
      y_real[k] = sum_{j=0}^{N-1} padded[row, j] * cos(2*pi*k*j/N)
    Grid: (rows, L+1) where L is input length, N = 2*L. We launch one program per (row, k) in [0..L].
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
    # Store accumulated sum; normalization will be done in a separate division kernel
    tl.store(out_real_ptr + row * (L + 1) + k, acc)


@triton.jit
def _compute_rfft_imag_per_k(padded_ptr, out_imag_ptr, N: tl.int32):
    """
    Compute y_imag[k] for a given (row, k):
      y_imag[k] = sum_{j=0}^{N-1} padded[row, j] * sin(2*pi*k*j/N)
    Grid: (rows, L+1)
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
def _divide_by_const(in_ptr, out_ptr, divisor: tl.float32, SIZE: tl.int32):
    """
    Elementwise divide in_ptr by divisor into out_ptr for length SIZE.
    Grid: (SIZE,) not practical; better use 1D tiling. We'll launch with grid = (rows,) and iterate over (L+1) elements per row.
    """
    row = tl.program_id(0)
    total = (L + 1)
    for idx in tl.static_range(0, total):
        val = tl.load(in_ptr + row * total + idx)
        val = val / divisor
        tl.store(out_ptr + row * total + idx, val)


def _triton_run(x: torch.Tensor):
    """
    Triton implementation of run(x):
      - Pads L to n=2*L.
      - Computes real and imaginary parts of normalized rfft via Triton kernels.
      - Returns tensors of shape (B, C, L+1) for real and imaginary parts.
    """
    B, C, L = x.shape
    N = 2 * L
    device = x.device

    # Ensure float32 and contiguous rows
    x_f32 = x.to(torch.float32).contiguous()
    rows = B * C

    # Allocate padded buffer: (rows, N)
    # We will use torch.zeros for padding and Triton to copy the first L elements.
    padded = torch.zeros((rows, N), dtype=torch.float32, device=device)

    # Copy each row into padded[:, :L]
    _copy_row_to_padded[(rows,)](x_f32.view(rows, L), padded, L, N, num_warps=1)

    # Allocate outputs: (rows, L+1)
    out_real = torch.empty((rows, L + 1), dtype=torch.float32, device=device)
    out_imag = torch.empty((rows, L + 1), dtype=torch.float32, device=device)

    # Compute real and imaginary parts for k in [0..L], i.e., L+1 coefficients
    grid_k = (rows, L + 1)
    _compute_rfft_real_per_k[grid_k](padded, out_real, N, num_warps=1)
    _compute_rfft_imag_per_k[grid_k](padded, out_imag, N, num_warps=1)

    # Normalize by N = 2*L
    # Use a separate elementwise division kernel. Note: this divides each (row, k) element.
    _divide_by_const[(rows,)](out_real, out_real, float(N), (rows * (L + 1)), num_warps=1)
    _divide_by_const[(rows,)](out_imag, out_imag, float(N), (rows * (L + 1)), num_warps=1)

    # Reshape back to (B, C, L+1)
    out_real = out_real.view(B, C, L + 1).contiguous()
    out_imag = out_imag.view(B, C, L + 1).contiguous()

    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect input tensor x: (batch, channels, seqlen)
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor (batch, channels, seqlen)")
        x = args[0]
        if not isinstance(x, torch.Tensor):
            raise RuntimeError("Input to ModelNew.forward must be a torch.Tensor")
        if x.ndim != 3:
            raise RuntimeError("Input tensor must have 3 dimensions (batch, channels, seqlen)")
        return _triton_run(x)


def run(*args):
    return ModelNew()(*args)
