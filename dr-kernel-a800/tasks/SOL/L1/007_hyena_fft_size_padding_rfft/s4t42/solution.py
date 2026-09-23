import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    """
    Copy x_row (length L) into out_ptr (length N), writing to out_ptr[pid*L : pid*L + L].
    The caller zeros the tail (out_ptr[L:N]) on the host.
    One program per (b, c) slice.
    """
    pid = tl.program_id(0)
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + i)
        tl.store(out_ptr + pid * L + i, val)


@triton.jit
def _accumulate_real(padded_ptr, out_ptr, k: tl.int32, L: tl.int32, N: tl.int32):
    """
    Compute out_ptr[k] = sum_{j=0}^{N-1} padded_ptr[j] * cos(2*pi*k*j/N).
    One scalar accumulation per k. Launch with grid=(1,).
    """
    acc = 0.0
    for j in tl.static_range(0, N):
        v = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * k * j / N
        acc += v * tl.cos(angle)
    tl.store(out_ptr + k, acc)


@triton.jit
def _accumulate_imag(padded_ptr, out_ptr, k: tl.int32, L: tl.int32, N: tl.int32):
    """
    Compute out_ptr[k] = sum_{j=0}^{N-1} padded_ptr[j] * sin(2*pi*k*j/N).
    One scalar accumulation per k. Launch with grid=(1,).
    """
    acc = 0.0
    for j in tl.static_range(0, N):
        v = tl.load(padded_ptr + j)
        angle = 2.0 * 3.141592653589793 * k * j / N
        acc += v * tl.sin(angle)
    tl.store(out_ptr + k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Input: x of shape (B, C, L) on CUDA
        Output: (x_freq_real, x_freq_imag) each of shape (B, C, L+1) float32
        """
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L  # padding for rfft to 2*L

        # Ensure float32 and on CUDA
        x_f32 = x.to(torch.float32)
        device = x_f32.device
        assert device.type == 'cuda', "Triton requires CUDA tensors"

        # Prepare outputs
        real_out = torch.zeros((B, C, L + 1), dtype=torch.float32, device=device)
        imag_out = torch.zeros((B, C, L + 1), dtype=torch.float32, device=device)

        # For each (b, c) slice, copy row to padded buffer (zeros tail on host)
        for b in range(B):
            for c in range(C):
                base = b * C + c  # not used in kernel, but kept for clarity
                padded = torch.zeros(N, dtype=torch.float32, device=device)
                _copy_row_to_padded[(1,)](x_f32[b, c, :], padded, L, N)

                # Accumulate real and imaginary parts for k in [0..L]
                out_real = torch.empty(L + 1, dtype=torch.float32, device=device)
                out_imag = torch.empty(L + 1, dtype=torch.float32, device=device)
                for k in range(L + 1):
                    _accumulate_real[(1,)](padded, out_real, k, L, N)
                    _accumulate_imag[(1,)](padded, out_imag, k, L, N)

                # Normalize by N=2*L (match torch.fft.rfft normalization)
                out_real = out_real / float(N)
                out_imag = out_imag / float(N)

                # Place into outputs at (b, c, :)
                real_out[b, c, :] = out_real
                imag_out[b, c, :] = out_imag

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
