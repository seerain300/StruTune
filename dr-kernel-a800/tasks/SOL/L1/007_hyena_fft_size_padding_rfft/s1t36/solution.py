import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.int32,            # batch size (runtime int)
    C: tl.int32,            # channels (runtime int)
    L: tl.int32,            # seqlen (runtime int)
    SCALE: tl.float32,      # 1.0 / (2 * L) normalization factor
):
    # Each program handles one (b, c) slice
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    N = 2 * L
    M = L + 1

    # Base offsets
    base_in = (b * C + c) * L
    base_out = (b * C + c) * M

    # Compute re_j and im_j for j = 0..M-1
    j = 0
    while j < M:
        sum_real = 0.0
        sum_imag = 0.0

        t = 0
        while t < N:
            x_val = tl.load(x_ptr + base_in + t)
            x_val = x_val.to(tl.float32)

            # angle = 2*pi*j*t/N = (j * t) * (2*pi / (2*L)) = (j * t) * (pi / L)
            # SCALE = 1 / (2*L)
            angle = 3.141592653589793 * j * t * SCALE

            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            sum_real += x_val * cos_term
            sum_imag += x_val * sin_term

            t += 1

        # Normalize by (2*L)
        sum_real *= SCALE
        sum_imag *= SCALE

        # Store results to out_real_ptr and out_imag_ptr at index j
        tl.store(out_real_ptr + base_out + j, sum_real)
        tl.store(out_imag_ptr + base_out + j, sum_imag)

        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-optimized implementation of:
          Input x: (B, C, L) float32 CUDA tensor.
          Output: (B, C, L+1) float32 tensors, real and imag parts of rfft(x, n=2*L) normalized by 2*L.
          All computation is performed inside Triton kernels; tensor allocations for outputs are done with torch.
        """
        assert x.is_cuda, "ModelNew expects a CUDA tensor."
        assert x.dim() == 3, "Expected input of shape (batch, channels, seqlen)."
        B, C, L = x.shape

        # Allocate outputs (float32, contiguous)
        M = L + 1
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        scale = 1.0 / float(2 * L)

        _rfft_real_imag_triton_kernel[grid](
            x, out_real, out_imag,
            B, C, L, scale,
            num_warps=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
