import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (constexpr for mapping pid)
    C: tl.constexpr,        # channels (constexpr for mapping pid)
    L,                      # int32, original seqlen (runtime)
    N,                      # int32, fft_size = 2*L (runtime)
    M,                      # int32, output length = L + 1 (runtime)
    invN                   # float32, 1.0 / N (runtime)
):
    # Each program handles one (b, c) slice
    pid = tl.program_id(axis=0)
    # Map program_id to (b, c)
    b = pid // C
    c = pid % C

    # Base offset into x for this (b, c) slice: x is (B, C, L) contiguous => offset = b*C*L + c*L
    base = b * C * L + c * L

    # Base output offset for this (b, c) slice: out is (B, C, M) contiguous => offset = b*C*M + c*M
    base_out = b * C * M + c * M

    # Compute j = 0..M-1
    j = 0
    while j < M:
        # Accumulators
        sum_real = 0.0
        sum_imag = 0.0

        # Iterate over t = 0..N-1 (zero-padding: for t >= L, x[t] is treated as 0)
        t = 0
        while t < N:
            # Load x[b, c, t] if t < L, else 0
            x_val = tl.load(x_ptr + base + t, mask=(t < L), other=0.0)
            # Compute angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * float(j) * float(t) / float(N)
            # Accumulate
            sum_real += x_val * tl.cos(angle)
            sum_imag += x_val * tl.sin(angle)
            t += 1

        # Normalize by 1/N
        sum_real = sum_real * invN
        sum_imag = sum_imag * invN

        # Store results to out_real and out_imag at (b, c, j)
        tl.store(out_real_ptr + base_out + j, sum_real)
        tl.store(out_imag_ptr + base_out + j, sum_imag)

        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          y = torch.fft.rfft(x, n=2*seqlen, dim=-1)  # complex
          y = y / (2*seqlen)
          return y.real, y.imag
        where x has shape (batch, channels, seqlen).
        """
        # Ensure tensor is on CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA device for Triton kernel."
        # x is (B, C, L)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Create outputs (B, C, M) as float32
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        invN = 1.0 / float(N)

        _rfft_real_imag_kernel[grid](
            x, out_real, out_imag,
            B, C, L, N, M, invN,
            num_warps=1,  # conservative launch config; avoids complexity
            num_stages=1
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
