import math
import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_kernel_with_tables(
    x_ptr,                      # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,               # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,               # *float32, output pointer to imag part (B, C, L+1)
    tables_cos_ptr,             # *const float32, pointer to cos table of length N_eff
    tables_sin_ptr,             # *const float32, pointer to sin table of length N_eff
    B: tl.constexpr,            # batch size (constexpr for index math)
    C: tl.constexpr,            # channels (constexpr for index math)
    L: tl.constexpr,            # seqlen (constexpr for index math)
    N: tl.constexpr,            # n = 2 * L (constexpr for index math)
    M: tl.constexpr,            # M = L + 1 (constexpr for index math)
    invN: tl.float32,           # normalization factor 1.0 / N
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Base offsets for input and output
    base_in = (b * C + c) * L
    base_out = (b * C + c) * M

    # For each output index j, compute sum over t in [0, N)
    for j in range(0, M):
        sum_real = 0.0
        sum_imag = 0.0

        # Direct DFT with zero-padding: x[t] = 0 for t >= L
        # N_eff is passed as N (2*L), we only loop up to N-1, and x_ptr + base_in + t is valid for t < L.
        for t in range(0, N):
            x_val = tl.load(x_ptr + base_in + t)  # x is float32

            # Lookup cos and sin for angle = (j * t) * (2*pi / N)
            # Index into tables: idx = (j * t) % N_eff, but here N_eff == N and j < M, t < N, so idx = (j * t) % N.
            # To avoid negative modulo, cast to int32 and use % with N.
            idx = (j * t) % N
            cos_val = tl.load(tables_cos_ptr + idx)
            sin_val = tl.load(tables_sin_ptr + idx)

            sum_real += x_val * cos_val
            sum_imag += x_val * sin_val

        # Normalize by N
        sum_real = sum_real * invN
        sum_imag = sum_imag * invN

        # Store to outputs at index (b, c, j)
        tl.store(out_real_ptr + base_out + j, sum_real)
        tl.store(out_imag_ptr + base_out + j, sum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Fused FFT size padding and real FFT computation for Hyena convolution.
        Computes torch.fft.rfft(x, n=2*L) along the last dimension for each (batch, channel) slice,
        returns real and imaginary parts separately (batch, channels, seqlen+1), normalized by 2*L.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernel."
        assert x.dtype == torch.float32, "Input must be float32."
        assert x.is_contiguous(), "Input must be contiguous."

        # x has shape (B, C, L)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Allocate outputs (B, C, M) as float32
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Precompute cos and sin tables on device: length N_eff = N
        # angle_k = 2*pi*k / N for k in [0, N)
        k = torch.arange(N, dtype=torch.float32, device=x.device)
        angles = (2.0 * math.pi) * (k) / float(N)
        tables_cos = torch.cos(angles)  # shape (N,)
        tables_sin = torch.sin(angles)  # shape (N,)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        invN = 1.0 / float(N)

        _rfft_real_imag_kernel_with_tables[grid](
            x, out_real, out_imag,
            tables_cos, tables_sin,
            B=B, C=C, L=L, N=N, M=M, invN=invN,
            num_warps=1,
            num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
