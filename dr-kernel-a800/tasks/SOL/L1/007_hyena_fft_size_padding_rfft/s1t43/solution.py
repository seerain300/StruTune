import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_simple_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (compile-time constant for grid)
    C: tl.constexpr,        # channels (compile-time constant for grid)
    L: tl.constexpr,        # seqlen (runtime value, used to compute N and M)
    N: tl.constexpr,        # N = 2 * L (compile-time constant for loop bounds)
    M: tl.constexpr,        # M = L + 1 (compile-time constant)
):
    # program id across (B, C) slices
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    # base offset for this (b, c) slice assuming x is contiguous [B, C, L]
    base_in = (b * C + c) * L

    # precompute scaling factor
    inv_N = 1.0 / N

    # process outputs j = 0..M-1; use static_range to allow unrolling
    for j in tl.static_range(M):
        # compute sum for real and imag parts
        sum_re = 0.0
        sum_im = 0.0

        # loop over t = 0..N-1
        for t in range(N):
            # load x[t] as fp32
            x_t = tl.load(x_ptr + base_in + t)
            # angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * j * t * inv_N
            # cos and sin
            ct = tl.cos(angle)
            st = tl.sin(angle)
            # accumulate
            sum_re += x_t * ct
            sum_im += x_t * st

        # scale by 1/N to match torch.fft.rfft normalization (n=2*seqlen)
        sum_re = sum_re * inv_N
        sum_im = sum_im * inv_N

        # store to output at index j for (b, c)
        # output is contiguous [B, C, M], so offset = (b*C + c) * M + j
        base_out = (b * C + c) * M
        tl.store(out_real_ptr + base_out + j, sum_re)
        tl.store(out_imag_ptr + base_out + j, sum_im)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen), float32, contiguous
        assert x.dim() == 3, "Input must be 3D (batch, channels, seqlen)"
        B, C, L = x.shape
        # Ensure float32 and contiguous input
        x = x.contiguous()
        # Output tensors: real and imag parts of shape (B, C, L+1)
        M = L + 1
        N = 2 * L
        # Allocate outputs (fp32), contiguous
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        _rfft_real_imag_simple_kernel[grid](
            x, out_real, out_imag,
            B=B, C=C, L=L, N=N, M=M,
            num_warps=1,  # small work per program; 1 warp is sufficient
        )
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
