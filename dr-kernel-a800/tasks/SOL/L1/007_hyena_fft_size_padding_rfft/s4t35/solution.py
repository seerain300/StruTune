import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_ptr, padded_ptr, L: tl.int32, TWO_L: tl.constexpr):
    """
    Copy x[row, 0:L] into padded[row, 0:L]. padded[row, L:TWO_L] is assumed zeros.
    Grid: (rows,)
    """
    row = tl.program_id(0)
    # Each row has length L; loop statically to avoid dynamic indexing issues
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + row * L + i)
        tl.store(padded_ptr + row * TWO_L + i, val)


@triton.jit
def _compute_rfft_real_per_k(padded_ptr, out_real_ptr, TWO_L: tl.constexpr):
    """
    For each (row, k) where k = program_id(1) in [0..L], compute y_real[k]:
      y_real[k] = (1/(2*L)) * sum_{j=0}^{2*L-1} padded[row, j] * cos(2*pi*k*j/(2*L))
    Grid: (rows, L+1)
    """
    row = tl.program_id(0)
    k = tl.program_id(1)
    acc = 0.0
    # Loop over j=0..2*L-1 (constexpr)
    for j in tl.static_range(0, TWO_L):
        val = tl.load(padded_ptr + row * TWO_L + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / TWO_L
        cosv = tl.cos(angle)
        acc += val * cosv
    # Normalize by 1/(2*L)
    inv_two_l = 1.0 / float(TWO_L)
    acc *= inv_two_l
    tl.store(out_real_ptr + row * (L + 1) + k, acc)


@triton.jit
def _compute_rfft_imag_per_k(padded_ptr, out_imag_ptr, TWO_L: tl.constexpr):
    """
    For each (row, k) where k = program_id(1) in [0..L], compute y_imag[k]:
      y_imag[k] = (1/(2*L)) * sum_{j=0}^{2*L-1} padded[row, j] * sin(2*pi*k*j/(2*L))
    Grid: (rows, L+1)
    """
    row = tl.program_id(0)
    k = tl.program_id(1)
    acc = 0.0
    for j in tl.static_range(0, TWO_L):
        val = tl.load(padded_ptr + row * TWO_L + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / TWO_L
        sinv = tl.sin(angle)
        acc += val * sinv
    inv_two_l = 1.0 / float(TWO_L)
    acc *= inv_two_l
    tl.store(out_imag_ptr + row * (L + 1) + k, acc)


@triton.jit
def _normalize_div(ptr, factor: tl.float32, length: tl.int32):
    """
    Multiply each element in ptr of given length by factor.
    Grid: (1,)
    """
    for i in tl.static_range(0, length):
        val = tl.load(ptr + i)
        val = val * factor
        tl.store(ptr + i, val)


def _triton_run(x: torch.Tensor):
    """
    Triton-only implementation of the original run function:
    - Pads to n=2*L and computes real/imag parts of rfft, normalized by 2*L, using Triton kernels.
    - Returns (B, C, L+1) tensors for real and imaginary parts.
    """
    B, C, L = x.shape
    TWO_L = 2 * L

    # Ensure float32 and contiguous per row
    x_f32 = x.to(torch.float32).contiguous()  # shape (B, C, L)
    rows = B * C

    # Allocate padded buffer: (rows, TWO_L), initialize zeros
    padded = torch.zeros((rows, TWO_L), dtype=torch.float32, device=x.device)
    # Copy each row to padded
    _copy_row_to_padded[(rows,)](x_f32.view(rows, L), padded, L, TWO_L=TWO_L, num_warps=1)

    # Allocate outputs: real and imag, shape (rows, L+1)
    out_real = torch.empty((rows, L + 1), dtype=torch.float32, device=x.device)
    out_imag = torch.empty((rows, L + 1), dtype=torch.float32, device=x.device)

    # Launch rfft real/imag computation: grid = (rows, L+1)
    grid = (rows, L + 1)
    _compute_rfft_real_per_k[grid](padded, out_real, TWO_L=TWO_L, num_warps=1)
    _compute_rfft_imag_per_k[grid](padded, out_imag, TWO_L=TWO_L, num_warps=1)

    # Normalize by 2*L (i.e., multiply by inv_two_l). Launch Triton kernel to do it.
    inv_two_l = 1.0 / float(TWO_L)
    total_len = out_real.numel()
    _normalize_div[(1,)](out_real.view(-1), inv_two_l, total_len, num_warps=1)
    _normalize_div[(1,)](out_imag.view(-1), inv_two_l, total_len, num_warps=1)

    # Reshape to (B, C, L+1)
    out_real = out_real.view(B, C, L + 1)
    out_imag = out_imag.view(B, C, L + 1)
    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect single tensor input of shape (batch, channels, seqlen)
        x = args[0] if len(args) == 1 else args[0]
        return _triton_run(x)


def run(*args):
    return ModelNew()(*args)
