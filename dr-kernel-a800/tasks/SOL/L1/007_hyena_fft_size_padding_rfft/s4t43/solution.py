import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_row_ptr, out_padded_ptr, L: tl.int32, N: tl.int32):
    # Copy x_row_ptr[0:L] into out_padded_ptr[0:L].
    # Tail out_padded_ptr[L:N] is assumed to be zeroed by host before call.
    for i in tl.static_range(0, L):
        val = tl.load(x_row_ptr + i)
        tl.store(out_padded_ptr + i, val)


@triton.jit
def _accumulate_real_scalar(padded_ptr, out_real_ptr, L: tl.int32, N: tl.int32, K: tl.int32):
    # Compute out_real[K] = sum_{j=0}^{N-1} padded[j] * cos(2*pi*K*j/N)
    acc = 0.0
    # Use a simple loop to avoid vectorized loads and reductions
    for j in tl.static_range(0, N):
        v = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * (K * j) / N
        acc += v * tl.cos(angle)
    tl.store(out_real_ptr + K, acc)


@triton.jit
def _accumulate_imag_scalar(padded_ptr, out_imag_ptr, L: tl.int32, N: tl.int32, K: tl.int32):
    # Compute out_imag[K] = sum_{j=0}^{N-1} padded[j] * sin(2*pi*K*j/N)
    acc = 0.0
    for j in tl.static_range(0, N):
        v = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * (K * j) / N
        acc += v * tl.sin(angle)
    tl.store(out_imag_ptr + K, acc)


@triton.jit
def _div_elements(x_ptr, y_ptr, numel: tl.int32, inv: tl.float32):
    # y[i] = x[i] * inv (elementwise)
    for i in tl.static_range(0, numel):
        val = tl.load(x_ptr + i)
        tl.store(y_ptr + i, val * inv)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L) float32
        assert x.dtype == torch.float32, "Input must be float32"
        B, C, L = x.shape
        N = 2 * L  # padding size for rfft

        # Allocate outputs: (B, C, L+1)
        real_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Process each (b, c) slice
        for b in range(B):
            for c in range(C):
                # Prepare padded input for this slice
                padded = torch.empty(N, dtype=torch.float32, device=x.device)
                # Copy x[b, c, :] to padded[0:L], tail will be zeroed by host
                x_row = x[b, c, :]
                _copy_row_to_padded[(1,)](x_row, padded, L, N)
                # Zero the tail to satisfy n=2*L padding
                padded[L:] = 0.0

                # Output buffers for this slice
                out_real = torch.empty(L + 1, dtype=torch.float32, device=x.device)
                out_imag = torch.empty(L + 1, dtype=torch.float32, device=x.device)

                # Compute real and imaginary parts for k in [0..L]
                for k in range(L + 1):
                    _accumulate_real_scalar[(1,)](padded, out_real, L, N, k)
                    _accumulate_imag_scalar[(1,)](padded, out_imag, L, N, k)

                # Normalize by N=2*L
                inv_N = 1.0 / float(N)
                out_real_div = torch.empty_like(out_real, device=x.device)
                out_imag_div = torch.empty_like(out_imag, device=x.device)
                _div_elements[(1,)](out_real, out_real_div, L + 1, inv_N)
                _div_elements[(1,)](out_imag, out_imag_div, L + 1, inv_N)

                # Place into outputs at (b, c, :)
                real_out[b, c, :] = out_real_div
                imag_out[b, c, :] = out_imag_div

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
