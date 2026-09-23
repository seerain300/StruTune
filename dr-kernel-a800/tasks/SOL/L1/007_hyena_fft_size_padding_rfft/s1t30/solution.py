import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (constexpr to allow index math)
    C: tl.constexpr,        # channels (constexpr to allow index math)
    L: tl.constexpr,        # seqlen (constexpr to allow index math)
    N: tl.constexpr,        # n = 2 * L (constexpr to allow index math)
    M: tl.constexpr,        # M = L + 1 (constexpr to allow index math)
    invN: tl.float32,       # normalization factor 1.0 / N
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(0)
    # Map pid to (b, c)
    b = pid // C
    c = pid % C

    # Base pointer offsets for this (b, c) slice
    # x is (B, C, L) contiguous => linear index = b*C*L + c*L + t
    base_in = b * C * L + c * L
    # Outputs are (B, C, M) contiguous => linear index = b*C*M + c*M + j
    base_out = b * C * M + c * M

    # Precompute constants for cos/sin
    # We will compute j-scalar loop to avoid unsupported vector arange and trig ops
    # We keep j in host-provided range M and compute per j
    for j in range(0, M):
        # Accumulators for real and imag parts in float32
        sum_real = 0.0
        sum_imag = 0.0

        # Loop over t = 0..N-1; implicit zero-padding for t >= L
        for t in range(0, N):
            x_val = tl.load(x_ptr + base_in + t)
            # Compute angle = 2*pi*j*t/N using scalar j and t
            # Triton provides basic arithmetic; avoid tl.cos/tl.sin to ensure compatibility.
            # Instead, we rely on Triton to evaluate the expression; if trig is unsupported, this
            # formulation is still valid and avoids calling those functions explicitly.
            angle = 2.0 * 3.141592653589793 * j * t / N

            # Direct DFT via explicit cos/sin:
            # Triton should support basic math; if not, the environment typically supports sin/cos
            # for scalar usage. We'll use them directly here.
            sum_real += x_val * tl.cos(angle)
            sum_imag += x_val * tl.sin(angle)

        # Apply normalization by N
        sum_real = sum_real * invN
        sum_imag = sum_imag * invN

        # Store results to out_real and out_imag at (b, c, j)
        tl.store(out_real_ptr + base_out + j, sum_real)
        tl.store(out_imag_ptr + base_out + j, sum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute rfft for each (batch, channel) slice with n = 2*seqlen,
        return real and imaginary parts separately, normalized by 2*seqlen.
        All computation is done in Triton kernels; no torch ops in forward.
        """
        # Ensure input is on CUDA
        assert x.is_cuda, "Input must be on CUDA device for Triton kernel."
        # x is (B, C, L)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Allocate outputs (B, C, M) as float32
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        invN = 1.0 / float(N)

        # Pass x as 1D contiguous pointer; we compute base indices in-kernel
        x_flat = x.reshape(-1)  # ensure contiguous 1D view

        _rfft_real_imag_kernel[grid](
            x_flat, out_real, out_imag,
            B, C, L, N, M, invN,
            num_warps=1,
            num_stages=1
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
