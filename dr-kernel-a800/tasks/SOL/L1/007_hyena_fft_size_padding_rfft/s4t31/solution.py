import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    # Each program copies one (b, c) row from x_ptr of length L into out_ptr of length N.
    # x_ptr layout: row-major contiguous, row starts at row_id * L
    # out_ptr layout: row-major contiguous, row starts at row_id * N
    row_id = tl.program_id(0)
    # Copy first L elements
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + row_id * L + i)
        tl.store(out_ptr + row_id * N + i, val)
    # Zero out the tail
    for i in tl.static_range(L, N):
        tl.store(out_ptr + row_id * N + i, 0.0)


@triton.jit
def _compute_rfft_real_per_row(padded_ptr, out_real_ptr, N: tl.int32, inv_N: tl.float32):
    """
    For each (b, c) row (program_id(0)), compute y_real[k] for k = program_id(1) in [0..L]:
      y_real[k] = (1/N) * sum_{j=0}^{N-1} padded[j] * cos(2*pi*k*j/N)
    Normalize by inv_N = 1/N.
    Grid: (B*C, L+1)
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
    # Normalize by inv_N; acc already includes inv_N scaling if desired, but here we scale explicitly.
    # inv_N is passed as 1/N. Since we accumulated without inv_N, we need to multiply by inv_N.
    # We'll compute y = acc * inv_N; however acc = sum * inv_N, so simply write acc * inv_N.
    acc = acc * inv_N
    # Store at index k in the (L+1) output
    # out_real_ptr layout: contiguous per row of length L+1
    # offset = row * (L+1) + k
    tl.store(out_real_ptr + row * (L + 1) + k, acc)


@triton.jit
def _compute_rfft_imag_per_row(padded_ptr, out_imag_ptr, N: tl.int32, inv_N: tl.float32):
    """
    For each (b, c) row (program_id(0)), compute y_imag[k] for k = program_id(1) in [0..L]:
      y_imag[k] = (1/N) * sum_{j=0}^{N-1} padded[j] * sin(2*pi*k*j/N)
    Normalize by inv_N = 1/N.
    Grid: (B*C, L+1)
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
    # Normalize
    acc = acc * inv_N
    # Store at index k
    tl.store(out_imag_ptr + row * (L + 1) + k, acc)


def _triton_run(x: torch.Tensor):
    B, C, L = x.shape
    N = 2 * L
    device = x.device
    # Ensure float32
    x_f32 = x.to(torch.float32)
    # Contiguous per row (already True if x is contiguous)
    x_f32 = x_f32.contiguous()

    rows = B * C
    # Allocate padded buffers: (rows, N)
    padded = torch.empty((rows, N), dtype=torch.float32, device=device)
    # Copy each row into padded buffer
    _copy_row_to_padded[(rows,)](x_f32.view(rows, L), padded, L, N, num_warps=1)

    # Allocate outputs: real and imag, shape (rows, L+1)
    out_real = torch.empty((rows, L + 1), dtype=torch.float32, device=device)
    out_imag = torch.empty((rows, L + 1), dtype=torch.float32, device=device)

    # Scale factor
    inv_N = 1.0 / float(N)

    # Compute real and imaginary parts for k in [0..L], i.e., L+1 coefficients
    # Grid: (rows, L+1)
    grid = (rows, L + 1)
    _compute_rfft_real_per_row[grid](padded, out_real, N, inv_N, num_warps=1)
    _compute_rfft_imag_per_row[grid](padded, out_imag, N, inv_N, num_warps=1)

    # Reshape back to (B, C, L+1)
    out_real = out_real.view(B, C, L + 1)
    out_imag = out_imag.view(B, C, L + 1)
    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect exactly one input tensor x with shape (batch, channels, seqlen)
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects exactly one input tensor.")
        x = args[0]
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device for Triton kernels.")
        if x.dim() != 3:
            raise RuntimeError("Input must be a 3D tensor of shape (batch, channels, seqlen).")
        # Triton compute path
        out_real, out_imag = _triton_run(x)
        return out_real, out_imag


# Example local test (optional):
# model = ModelNew().cuda()
# x = torch.randn(8, 128, 1024, device='cuda', dtype=torch.float32)
# y_real, y_imag = model(x)
# y_complex = torch.complex(y_real, y_imag)
# y_ref = torch.fft.rfft(x.to(torch.float32), n=2048) / 2048
# torch.testing.assert_close(y_complex, y_ref)


def run(*args):
    return ModelNew()(*args)
