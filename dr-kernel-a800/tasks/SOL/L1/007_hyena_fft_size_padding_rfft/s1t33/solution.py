import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                # *const float32, input pointer to x with shape (B, C, L), contiguous
    out_real_ptr,         # *float32, output pointer to real part (B, C, L+1), contiguous
    out_imag_ptr,         # *float32, output pointer to imag part (B, C, L+1), contiguous
    L,                    # int32, seqlen (length per (b, c) slice)
    N,                    # int32, 2 * seqlen (FFT length)
    M,                    # int32, seqlen + 1 (output length)
    B,                    # int32, batch size (for potential indexing if needed)
    stride_x_b,           # int32, stride for batch in x
    stride_x_c,           # int32, stride for channel in x
    stride_out_b,         # int32, stride for batch in out
    stride_out_c,         # int32, stride for channel in out
):
    # One program per (batch, channel) slice
    pid = tl.program_id(axis=0)

    # Derive b, c from pid
    # Note: Triton doesn't require runtime integer division/mod, but we can pass B and C and assume pid < B*C.
    # To be safe, we compute b and c via loop counts if needed, but here we assume caller ensures pid in range.
    # We'll use pid directly as linear index and map to (b, c) via global indices.
    # However, since we have 1D grid size == B*C, we can do:
    b = pid // C
    c = pid % C

    # Compute base offsets for this (b, c) slice
    x_base = b * stride_x_b + c * stride_x_c
    out_base = b * stride_out_b + c * stride_out_c

    invN = 1.0 / N

    # Accumulators
    re_sum = 0.0
    im_sum = 0.0

    # We need to compute for each j = 0..M-1 and store; to do this, we re-initialize inside this outer loop.
    # But Triton requires static loops for unrolling. We will use a compile-time unrolled loop over j by making j a runtime variable in inner loop.
    # Since Triton can handle dynamic loops via range, we can implement this as two nested loops:
    # - outer loop over j, dynamic range [0, M)
    # - inner loop over t, dynamic range [0, N)
    # For correctness, we implement scalar accumulation per j.

    # Note: Triton doesn't allow 'for j in range(M)' directly in all versions; we emulate with while.
    j = 0
    while j < M:
        re_sum = 0.0
        im_sum = 0.0
        t = 0
        while t < N:
            # Load x[b, c, t] as float32
            x_val = tl.load(x_ptr + x_base + t)
            # Compute angle = 2*pi*j*t/N
            angle = 2.0 * 3.141592653589793 * (j * t) / N
            cosj = tl.cos(angle)
            sinj = tl.sin(angle)
            re_sum += x_val * cosj * invN
            im_sum += x_val * sinj * invN
            t += 1
        # Store results to out[b, c, j]
        tl.store(out_real_ptr + out_base + j, re_sum)
        tl.store(out_imag_ptr + out_base + j, im_sum)
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Input: x of shape (batch, channels, seqlen)
        - Output: (batch, channels, seqlen+1) for real and imag parts, normalized by 2*seqlen.
        """
        # Ensure contiguous input
        x = x.contiguous()
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Allocate outputs (B, C, M), float32, contiguous
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel with 1D grid (B*C,)
        _rfft_real_imag_triton_kernel[(B * C,)](
            x, out_real, out_imag,
            L, N, M, B,
            x.stride(0), x.stride(1),
            out_real.stride(0), out_real.stride(1),
            num_warps=1, num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
