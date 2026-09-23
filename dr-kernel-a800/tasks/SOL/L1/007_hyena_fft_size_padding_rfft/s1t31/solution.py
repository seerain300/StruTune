import triton
import triton.language as tl


@triton.jit
def _real_rfft_sum_triton_kernel(
    x_ptr,              # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,       # *float32, output pointer to real part (B, C, L+1) contiguous
    out_imag_ptr,       # *float32, output pointer to imag part (B, C, L+1) contiguous
    L,                  # int32, seqlen
    N,                  # int32, 2 * seqlen (FFT length)
    C,                  # int32, number of channels
):
    # One program per (b, c) slice: grid is (B * C,)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # Base pointer for the (b, c) slice; within each slice, elements are contiguous in t
    x_base = x_ptr + b * (C * L) + c * L

    # Normalization factor
    invN = 1.0 / N

    # Output bases: out_real and out_imag are (B, C, L+1) contiguous
    out_base_real = out_real_ptr + b * (C * (L + 1)) + c * (L + 1)
    out_base_imag = out_imag_ptr + b * (C * (L + 1)) + c * (L + 1)

    # Accumulate re[j] and im[j] for j = 0..L
    for j in range(0, L + 1):
        re_sum = 0.0
        im_sum = 0.0
        # Sum over t = 0..2L-1
        for t in range(0, N):
            x_val = tl.load(x_base + t)  # x[b, c, t] as float32
            angle = 2.0 * 3.141592653589793 * (j * t) / N
            cosj = tl.cos(angle)
            sinj = tl.sin(angle)
            re_sum += x_val * cosj * invN
            im_sum += x_val * sinj * invN

        tl.store(out_base_real + j, re_sum)
        tl.store(out_base_imag + j, im_sum)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Input: x of shape (batch, channels, seqlen)
        - Output: (batch, channels, seqlen+1) for real and imag parts, normalized by 2*seqlen.
        """
        # Enforce contiguous layout (typical for model inputs); no torch ops after this.
        x = x.contiguous()
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Allocate outputs (B, C, M), float32
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel with 1D grid (B*C,)
        _real_rfft_sum_triton_kernel[(B * C,)](
            x, out_real, out_imag,
            L, N, C,
            num_warps=1, num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
