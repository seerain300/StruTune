import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_ptr, padded_ptr, L: tl.int32, N: tl.int32):
    """
    Copy row x[row_id, 0:L] into padded[row_id, 0:L]; padded[row_id, L:N] is assumed pre-zeroed.
    Grid: (rows,)
    """
    row_id = tl.program_id(0)
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + row_id * L + i)
        tl.store(padded_ptr + row_id * N + i, val)


@triton.jit
def _accumulate_rfft_real_per_k(padded_ptr, out_real_ptr, N: tl.int32, inv_N: tl.float32):
    """
    For each (b, c) row (program_id(0)) and output index k (program_id(1) in [0..L]):
      y_real[k] = (1/N) * sum_{j=0}^{N-1} padded[j] * cos(2*pi*k*j/N)
      Normalize by inv_N = 1/(2*L) before store. Since acc is (1/N)*sum, we scale by inv_N to get (1/(2*L))*sum.
    Grid: (rows, L+1)
    """
    row = tl.program_id(0)
    k = tl.program_id(1)
    acc = 0.0
    for j in tl.static_range(0, N):
        val = tl.load(padded_ptr + row * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        cosv = tl.cos(angle)
        acc += val * cosv
    # Normalize by 1/(2*L)
    acc = acc * inv_N
    tl.store(out_real_ptr + row * (L + 1) + k, acc)


@triton.jit
def _accumulate_rfft_imag_per_k(padded_ptr, out_imag_ptr, N: tl.int32, inv_N: tl.float32):
    """
    For each (b, c) row (program_id(0)) and output index k (program_id(1) in [0..L]):
      y_imag[k] = (1/N) * sum_{j=0}^{N-1} padded[j] * sin(2*pi*k*j/N)
      Normalize by inv_N = 1/(2*L) before store.
    Grid: (rows, L+1)
    """
    row = tl.program_id(0)
    k = tl.program_id(1)
    acc = 0.0
    for j in tl.static_range(0, N):
        val = tl.load(padded_ptr + row * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        sinv = tl.sin(angle)
        acc += val * sinv
    # Normalize by 1/(2*L)
    acc = acc * inv_N
    tl.store(out_imag_ptr + row * (L + 1) + k, acc)


def _triton_run(x: torch.Tensor):
    """
    Triton-only computation equivalent to:
        x_f32 = x.to(torch.float32)
        x_freq = torch.fft.rfft(x_f32, n=2*L) / (2*L)
        return x_freq.real, x_freq.imag
    Returns: out_real, out_imag tensors of shape (B, C, L+1), float32.
    """
    B, C, L = x.shape
    device = x.device
    # Ensure input is float32 and contiguous per row
    x_f32 = x.to(torch.float32).contiguous()
    N = 2 * L
    rows = B * C

    # Allocate padded buffer: (rows, N). Pre-zero padded.
    padded = torch.zeros((rows, N), dtype=torch.float32, device=device)

    # Copy each row into padded buffer (first L elements)
    _copy_row_to_padded[(rows,)](x_f32.view(rows, L), padded, L, N, num_warps=1)

    # Allocate outputs: (rows, L+1), initialize to zeros for atomic accumulation
    out_real = torch.zeros((rows, L + 1), dtype=torch.float32, device=device)
    out_imag = torch.zeros((rows, L + 1), dtype=torch.float32, device=device)

    # inv_N = 1/(2*L) for normalization
    inv_N = 1.0 / float(2 * L)

    # Grid over rows and k in [0..L]
    grid = (rows, L + 1)
    _accumulate_rfft_real_per_k[grid](padded, out_real, N, inv_N, num_warps=1)
    _accumulate_rfft_imag_per_k[grid](padded, out_imag, N, inv_N, num_warps=1)

    # Reshape back to (B, C, L+1)
    out_real = out_real.view(B, C, L + 1)
    out_imag = out_imag.view(B, C, L + 1)
    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We assume single input tensor as in the original Model.forward: Model.forward(self, *args) -> run(x)
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            x = args[0]
        else:
            raise RuntimeError("ModelNew expects a single input tensor (batch, channels, seqlen)")
        return _triton_run(x)


def run(*args):
    return ModelNew()(*args)
