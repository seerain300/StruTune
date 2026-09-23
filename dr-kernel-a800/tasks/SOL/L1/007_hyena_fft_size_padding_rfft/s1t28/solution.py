import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (constexpr for indexing)
    C: tl.constexpr,        # channels (constexpr for indexing)
    L: tl.constexpr,        # seqlen (constexpr for indexing)
    N: tl.constexpr,        # n = 2 * L (constexpr for indexing)
    M: tl.constexpr,        # M = L + 1 (constexpr for indexing)
    invN: tl.float32,       # normalization factor 1.0 / N
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Base linear offsets for input and output slices
    base_in = (b * C + c) * L
    base_out = (b * C + c) * M

    # Compute DFT for each j = 0..M-1
    # For each j, sum over t = 0..N-1: x[t] * (cos(2*pi*j*t/N) +/- sin(2*pi*j*t/N))
    # We implement the direct formula; Triton provides tl.cos and tl.sin in recent versions.
    for j in range(0, M):
        sum_real = 0.0
        sum_imag = 0.0
        for t in range(0, N):
            x_val = tl.load(x_ptr + base_in + t)
            # angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * (j * t) * invN
            c_t = tl.cos(angle)
            s_t = tl.sin(angle)
            sum_real += x_val * c_t
            sum_imag += x_val * s_t

        # Normalize by N
        sum_real = sum_real * invN
        sum_imag = sum_imag * invN

        # Store to outputs
        tl.store(out_real_ptr + base_out + j, sum_real)
        tl.store(out_imag_ptr + base_out + j, sum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Fused Triton implementation of:
          x: (batch, channels, seqlen)
          y = torch.fft.rfft(x, n=2*seqlen, dim=-1) / (2*seqlen)
          return y.real, y.imag  # each of shape (batch, channels, seqlen+1), float32
        All computation is done by Triton kernels; no torch ops in forward.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        # Ensure contiguous input
        x = x.contiguous()
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Allocate outputs (B, C, M) float32
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        invN = 1.0 / float(N)

        _rfft_real_imag_kernel[grid](
            x, out_real, out_imag,
            B, C, L, N, M, invN,
            num_warps=1,
            num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
