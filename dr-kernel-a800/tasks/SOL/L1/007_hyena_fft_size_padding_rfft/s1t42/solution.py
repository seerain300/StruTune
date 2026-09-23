import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,           # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,    # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,    # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr, # batch size (for grid only)
    C: tl.constexpr, # channels (for grid only)
    L: tl.int32,     # seqlen (runtime value)
    N: tl.int32,     # N = 2 * L (runtime value)
    M: tl.int32      # M = L + 1 (runtime value)
):
    # Each program handles one (b, c) slice
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    # Base pointers for this (b, c) slice
    base_x = x_ptr + (b * C + c) * L       # x is (B, C, L) contiguous
    base_r = out_real_ptr + (b * C + c) * M
    base_i = out_imag_ptr + (b * C + c) * M

    # Constants
    pi = 3.141592653589793
    inv_N = 1.0 / N

    # Loop over output indices j = 0..M-1
    for j in range(0, M):
        re = 0.0
        im = 0.0
        # Loop over input samples t = 0..N-1
        for t in range(0, N):
            x_t = tl.load(base_x + t)  # scalar load
            angle = 2.0 * pi * float(j) * float(t) / float(N)
            re += x_t * tl.cos(angle)
            im += x_t * tl.sin(angle)
        re = re * inv_N
        im = im * inv_N
        tl.store(base_r + j, re)
        tl.store(base_i + j, im)


def _triton_rfft_real_imag(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Triton-only implementation of torch.fft.rfft for real input along last dim:
    Input: x of shape (B, C, L) float32 on CUDA
    Output: (out_real, out_imag) of shape (B, C, L+1) float32, normalized by 2*L
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    assert x.dtype == torch.float32, "Input must be float32."
    assert x.dim() == 3, "Input must be 3D (B, C, L)."

    B, C, L = x.shape
    N = 2 * L
    M = L + 1

    # Allocate outputs (contiguous)
    out_real = torch.empty((B, C, M), device=x.device, dtype=torch.float32)
    out_imag = torch.empty((B, C, M), device=x.device, dtype=torch.float32)

    # Launch one program per (b, c) slice
    grid = (B * C,)

    _rfft_real_imag_triton_kernel[grid](
        x, out_real, out_imag,
        B=B, C=C, L=L, N=N, M=M,
        num_warps=1, num_stages=1
    )

    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Triton-only forward: no torch ops
        return _triton_rfft_real_imag(x)


def run(*args):
    return ModelNew()(*args)
