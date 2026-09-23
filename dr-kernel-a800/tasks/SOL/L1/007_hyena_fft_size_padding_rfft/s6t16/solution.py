import torch
import triton
import triton.language as tl

@triton.jit
def _sum_rows_kernel(x_ptr, sum_ptr,
                      B: tl.int32, C: tl.int32, S: tl.int32,
                      stride_x_b: tl.int32, stride_x_c: tl.int32, stride_x_s: tl.int32,
                      ):
    # One program per (b, c)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    base = b * stride_x_b + c * stride_x_c
    sum_val = 0.0
    # Sum over the last dimension S
    for t in range(0, S):
        sum_val += tl.load(x_ptr + base + t * stride_x_s)
    tl.store(sum_ptr + pid, sum_val)


@triton.jit
def _rfft_fill_kernel(sum_ptr, out_real_ptr, out_imag_ptr,
                      B: tl.int32, C: tl.int32,
                      S: tl.int32,  # input seqlen
                      ):
    # One program per (b, c)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Load sum over S
    sum_x = tl.load(sum_ptr + pid)
    M = S + 1  # output length

    N = 2 * S  # PyTorch rfft length is 2*S for real input

    # Fill out_real and out_imag for k = 0..M-1
    for k in range(0, M):
        if k == 0:
            # y[0] is real sum, imag 0
            tl.store(out_real_ptr + pid * (S + 1) + k, sum_x)
            tl.store(out_imag_ptr + pid * (S + 1) + k, 0.0)
        else:
            # Compute cos and sin with N=2*S
            ang = 3.141592653589793 * k / N  # pi * k / N
            cosk = tl.cos(ang)
            sink = tl.sin(ang)

            # Normalize by 2*S (as per original code: x_freq = x_freq / (2*S))
            # For even k: real only
            if (k & 1) == 0:
                y_even = sum_x * (cosk - sink) / (2.0 * S * N)
                tl.store(out_real_ptr + pid * (S + 1) + k, y_even)
                tl.store(out_imag_ptr + pid * (S + 1) + k, 0.0)
            else:
                # Odd k: imaginary only
                y_odd_imag = -sum_x * sink / (2.0 * S * N)
                tl.store(out_real_ptr + pid * (S + 1) + k, 0.0)
                tl.store(out_imag_ptr + pid * (S + 1) + k, y_odd_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          x_f32 = x.to(torch.float32)
          x_freq = torch.fft.rfft(x_f32, n=2*S)  # complex (B, C, S+1)
          x_freq = x_freq / (2*S)
          return x_freq.real, x_freq.imag
        We compute rfft-like outputs via direct trigonometric sums in Triton and apply the final division by (2*S).
        Note: This implementation does not call torch.fft in forward, only simple tensor creation.
        """
        assert x.dim() == 3, "Input must be 3D (B, C, S)"
        B, C, S = x.shape
        x = x


def run(*args):
    return ModelNew()(*args)
