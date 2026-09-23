import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                # *const float32, input pointer to x (B, C, L), contiguous
    out_real_ptr,         # *float32, output pointer to real part (B, C, L+1), contiguous
    out_imag_ptr,         # *float32, output pointer to imag part (B, C, L+1), contiguous
    L,                    # int32, seqlen
    B,                    # int32, batch size (runtime arg, not used for math but passed to satisfy host context)
    C,                    # int32, channels (runtime arg)
):
    # One program per (b, c) slice; grid size must be set to B*C by host
    pid = tl.program_id(axis=0)

    # Compute b and c from pid: assume grid size == B*C
    # Note: Triton kernels don't expose B or C directly, but host must ensure correct grid. We derive using integer division and modulo.
    # We will use pid // C and pid % C. To do that safely, we must assume grid size is exactly B*C.
    b = pid // C
    c = pid % C

    # Base offsets for this (b, c) slice
    # For contiguous x of shape (B, C, L), stride_b = C*L, stride_c = L, so base offset = b*C*L + c*L
    base_x = b * C * L + c * L

    N = 2 * L
    M = L + 1
    invN = 1.0 / N

    # For each j in 0..M-1, compute re and im
    j = 0
    while j < M:
        re_sum = 0.0
        im_sum = 0.0
        t = 0
        while t < N:
            # Load x[b, c, t] (contiguous last dim)
            x_val = tl.load(x_ptr + base_x + t)
            # angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * (j * t) / N
            cosj = tl.cos(angle)
            sinj = tl.sin(angle)
            # Accumulate with normalization
            re_sum += x_val * cosj * invN
            im_sum += x_val * sinj * invN
            t += 1
        # Output indices for real/imag are contiguous along last dim
        out_base = b * C * M + c * M + j
        tl.store(out_real_ptr + out_base, re_sum)
        tl.store(out_imag_ptr + out_base, im_sum)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Input: x of shape (batch, channels, seqlen), float32, contiguous
        - Output: (batch, channels, seqlen+1) for real and imag parts, normalized by 2*seqlen.
        """
        # Ensure contiguous layout
        x = x.contiguous()
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Allocate outputs (B, C, M), float32
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel with 1D grid (B*C,)
        _rfft_real_imag_triton_kernel[(B * C,)](
            x, out_real, out_imag,
            L, B, C,
            num_warps=1, num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
