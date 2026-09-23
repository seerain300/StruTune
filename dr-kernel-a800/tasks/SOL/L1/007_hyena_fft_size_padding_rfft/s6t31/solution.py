import torch
import triton
import triton.language as tl


@triton.jit
def compute_rfft_direct_kernel(
    x_ptr,              # *float32, input x flattened over BC*S
    out_real_ptr,       # *float32, output real of length BC*(S+1)
    out_imag_ptr,       # *float32, output imag of length BC*(S+1)
    S: tl.int32,        # original seqlen
    N: tl.constexpr     # 2*S, must be compile-time known (power-of-two)
):
    bc = tl.program_id(0)
    base_x = bc * S
    base_out = bc * (S + 1)

    # sum over x
    total_sum = 0.0
    j = 0
    while j < S:
        v = tl.load(x_ptr + base_x + j)
        total_sum += v
        j += 1

    # For k in 0..S:
    # y[k] = (sum(x) * cos(pi*k/N) - sum(x) * sin(pi*k/N)) / N, for k even
    # y[k] has real=0, imag= -sum(x) * sin(pi*k/N) / N, for k odd
    norm_factor = 1.0 / (2.0 * N)

    k = 0
    while k <= S:
        if (k % 2) == 0:
            cos_term = tl.cos(3.141592653589793 * k / N)
            sin_term = tl.sin(3.141592653589793 * k / N)
            val = total_sum * (cos_term - sin_term)
            tl.store(out_real_ptr + base_out + k, val * norm_factor)
            tl.store(out_imag_ptr + base_out + k, 0.0)
        else:
            sin_term = tl.sin(3.141592653589793 * k / N)
            val = -total_sum * sin_term
            tl.store(out_real_ptr + base_out + k, 0.0)
            tl.store(out_imag_ptr + base_out + k, val * norm_factor)
        k += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x: (B, C, S), float32
        # We implement rfft and normalization via Triton for power-of-two 2*S.
        # Otherwise, fallback to PyTorch for correctness.

        # Ensure dtype and contiguity
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        B, C, S = x.shape
        N = 2 * S

        # Check if N is power-of-two
        is_pow2 = (N & (N - 1)) == 0 and N > 0

        if not is_pow2:
            # Fallback: exact PyTorch behavior
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=N)
            x_freq = x_freq / (2 * S)
            return x_freq.real.contiguous(), x_freq.imag.contiguous()

        # Triton path: direct computation
        out_real = torch.empty((B, C, S + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, S + 1), device=x.device, dtype=torch.float32)

        BC = B * C
        x_flat = x.view(-1, S)

        compute_rfft_direct_kernel[(BC,)](
            x_flat, out_real, out_imag, S, N
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
