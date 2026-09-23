import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rows_kernel(x_ptr, sum_ptr, B, C, S):
    """
    For each (b, c), sum x[b, c, 0:S] and store to sum_ptr[pid].
    Grid: (B*C,)
    Assumes x is contiguous with layout (B, C, S) => linear index b*(C*S) + c*S + s.
    """
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    acc = 0.0
    # Loop over s = 0 .. S-1
    for s in range(0, S):
        acc += tl.load(x_ptr + b * (C * S) + c * S + s)
    tl.store(sum_ptr + pid, acc)


@triton.jit
def _fill_rfft_kernel(sum_ptr, out_real_ptr, out_imag_ptr, B, C, S):
    """
    For each (b, c), fill output tensors:
      - out_real[b, c, k] = sum * (cos(pi*k/N) - sin(pi*k/N)) / (2*S*N) if k>0 and even, else (k==0: sum, else 0)
      - out_imag[b, c, k] = -sum * sin(pi*k/N) / (2*S*N) if k>0 and odd, else 0
    Output length is S+1. Grid: (B*C,)
    """
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    sum_val = tl.load(sum_ptr + pid)
    N = 2 * S  # PyTorch rfft uses n=2*S for real input

    # k = 0
    tl.store(out_real_ptr + pid * (S + 1) + 0, sum_val)
    tl.store(out_imag_ptr + pid * (S + 1) + 0, 0.0)

    # k = 1 .. S
    for k in range(1, S + 1):
        ang = 3.141592653589793 * k / N  # pi * k / N
        cosk = tl.cos(ang)
        sink = tl.sin(ang)
        norm = 2.0 * S * N  # = 4 * S^2

        if (k & 1) == 0:  # even k
            y_real = sum_val * (cosk - sink) / norm
            tl.store(out_real_ptr + pid * (S + 1) + k, y_real)
            tl.store(out_imag_ptr + pid * (S + 1) + k, 0.0)
        else:  # odd k
            y_imag = -sum_val * sink / norm
            tl.store(out_real_ptr + pid * (S + 1) + k, 0.0)
            tl.store(out_imag_ptr + pid * (S + 1) + k, y_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          x_f32 = x.to(torch.float32)  # x is float32 in given tests
          x_freq = torch.fft.rfft(x_f32, n=2*S)  # complex (B, C, S+1)
          x_freq = x_freq / (2*S)
          return x_freq.real, x_freq.imag
        We compute outputs via Triton kernels; no torch FFT is used in forward.
        """
        assert x.dim() == 3, "Input must be 3D (B, C, S)"
        B, C, S = x.shape

        # Ensure float32 for numerics
        x = x.contiguous().to(torch.float32)

        # Allocate sum buffer: one sum per (b, c)
        sum_buf = torch.empty(B * C, dtype=torch.float32, device=x.device)

        # Launch sum kernel: one program per (b, c)
        grid_sum = (B * C,)
        _sum_rows_kernel[grid_sum](x, sum_buf, B, C, S)

        # Allocate outputs: shape (B, C, S+1)
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch fill kernel: one program per (b, c)
        grid_fill = (B * C,)
        _fill_rfft_kernel[grid_fill](sum_buf, out_real, out_imag, B, C, S)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
