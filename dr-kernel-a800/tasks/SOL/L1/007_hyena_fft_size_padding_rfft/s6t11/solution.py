import torch
import triton
import triton.language as tl


@triton.jit
def _real_rfft_norm_kernel(x_ptr, out_real_ptr, out_imag_ptr, S: tl.int32):
    """
    For each (b, c) slice of length S in x_ptr, compute rfft over 2*S and
    write normalized real/imag parts into out_real/out_imag of shape (B, C, S+1).

    x_ptr: 1D contiguous float32 of length B*C*S
    out_real/out_imag: 1D contiguous float32 of length B*C*(S+1)
    """
    bc = tl.program_id(0)
    base = bc * S

    N = 2 * S
    inv_norm = 1.0 / (2.0 * N)  # normalize by 2*S, as in original

    # Loop over k = 0..S
    k = 0
    while k <= S:
        # Sum of x over [0, S-1]
        sum_x = 0.0
        t = 0
        while t < S:
            v = tl.load(x_ptr + base + t)
            sum_x += v
            t += 1

        ang = 3.141592653589793 * k * 2.0 / N  # 2*pi*k/N
        is_even = (k % 2 == 0)

        if is_even:
            cos_k = tl.cos(ang)
            sin_k = tl.sin(ang)
            real_k = (sum_x * cos_k - sum_x * sin_k) * (1.0 / N) * inv_norm
            imag_k = 0.0
        else:
            sin_k = tl.sin(ang)
            real_k = 0.0
            imag_k = (-sum_x * sin_k) * (1.0 / N) * inv_norm

        out_index = bc * (S + 1) + k
        tl.store(out_real_ptr + out_index, real_k)
        tl.store(out_imag_ptr + out_index, imag_k)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, S) float32 on CUDA
        Returns: (B, C, S+1) real and imaginary parts of rfft(x, n=2*S) normalized by 2*S.
        """
        assert x.is_cuda, "Input must be on CUDA device."
        assert x.dtype == torch.float32, "Input must be float32."
        B, C, S = x.shape

        # Outputs (B, C, S+1) contiguous
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Flatten input for simple 1D kernel
        x_flat = x.contiguous().view(-1)  # length B*C*S

        # Launch: one program per (b, c)
        grid = (B * C,)
        _real_rfft_norm_kernel[grid](
            x_flat, out_real, out_imag, S,
            num_warps=1, num_stages=1
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
