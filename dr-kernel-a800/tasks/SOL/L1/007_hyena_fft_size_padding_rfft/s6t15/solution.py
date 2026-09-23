import torch
import triton
import triton.language as tl


@triton.jit
def pad_and_zero_kernel(
    x_ptr,            # *float32, input x with shape (B*C, S)
    z_ptr,            # *float32, output z with shape (B*C, 2*S)
    S: tl.int32,      # seqlen
):
    bc = tl.program_id(0)
    base_x = bc * S
    N = 2 * S
    base_z = bc * N
    t = 0
    while t < S:
        val = tl.load(x_ptr + base_x + t)
        tl.store(z_ptr + base_z + t, val)
        t += 1
    # write zeros for t in [S, 2*S-1]
    t = S
    while t < N:
        tl.store(z_ptr + base_z + t, 0.0)
        t += 1


@triton.jit
def dft_complex_kernel(
    z_ptr,            # *float32, input real z of length 2*S
    out_real_ptr,     # *float32, output real part of length (B*C*(S+1))
    out_imag_ptr,     # *float32, output imag part of length (B*C*(S+1))
    S: tl.int32,      # seqlen
    TWO_S: tl.int32,  # 2*S
):
    bc = tl.program_id(0)
    N = TWO_S
    M = S + 1
    base_z = bc * N
    # Compute first M outputs (j=0..S)
    j = 0
    while j < M:
        sum_r = 0.0
        sum_i = 0.0
        # sum over t=0..N-1
        t = 0
        while t < N:
            zt = tl.load(z_ptr + base_z + t)
            ang = 2.0 * 3.141592653589793 * t * j / (1.0 * N)
            c = tl.cos(ang)
            s = tl.sin(ang)
            sum_r += zt * c
            sum_i += -zt * s
            t += 1
        # Normalize by 2*S (same as original: divide complex output by 2*S)
        inv_2S = 1.0 / (1.0 * TWO_S)
        tl.store(out_real_ptr + (bc * M + j), sum_r * inv_2S)
        tl.store(out_imag_ptr + (bc * M + j), sum_i * inv_2S)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, S) float32 on CUDA.
        Returns: (B, C, S+1) real and imaginary parts of rfft(x, n=2*S) normalized by 2*S.
        """
        assert x.is_cuda, "Input must be on CUDA device."
        assert x.dtype == torch.float32, "Input must be float32."
        B, C, S = x.shape
        device = x.device

        # Flatten (B, C) into one dimension for Triton programs
        BC = B * C
        x_flat = x.reshape(BC, S).contiguous()

        # Allocate z_real: (BC, 2*S) zeros + x
        N = 2 * S
        z_real = torch.empty((BC, N), dtype=torch.float32, device=device)

        # Kernel 1: pad x with zeros to length 2*S
        grid = (BC,)
        pad_and_zero_kernel[grid](x_flat, z_real, S)

        # Allocate outputs: real and imag parts of length BC*(S+1)
        M = S + 1
        out_real = torch.empty((BC, M), dtype=torch.float32, device=device)
        out_imag = torch.empty((BC, M), dtype=torch.float32, device=device)

        # Kernel 2: compute DFT via direct summation and normalize by 2*S
        dft_complex_kernel[grid](z_real, out_real, out_imag, S, N)

        # Reshape back to (B, C, S+1)
        out_real = out_real.view(B, C, M)
        out_imag = out_imag.view(B, C, M)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
