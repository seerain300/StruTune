import triton
import triton.language as tl


@triton.jit
def _real_rfft_compute_kernel(
    x_ptr,            # *const float32, input pointer to x with shape (B, C, L)
    out_real_ptr,     # *float32, output pointer to real part (B, C, L+1), flattened
    out_imag_ptr,     # *float32, output pointer to imag part (B, C, L+1), flattened
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    L: tl.constexpr,  # seqlen
    N: tl.constexpr,  # N = 2 * L
    M: tl.constexpr,  # M = L + 1
):
    # Each program handles one (b, c) slice
    b = tl.program_id(0)  # batch index
    c = tl.program_id(1)  # channel index

    # Input is (B, C, L) contiguous => index = ((b*C + c) * L) + t
    base_x = (b * C + c) * L

    # Precompute inv_N = 1 / N (normalization factor)
    inv_N = 1.0 / N

    # We will compute re_j and im_j for j = 0..M-1
    # Use a scalar loop over j and accumulate sums across t.
    for j in range(0, M):
        # Accumulators in fp32
        acc_re = 0.0
        acc_im = 0.0

        # Direct real DFT: sum over t
        for t in range(0, N):
            x_t = tl.load(x_ptr + base_x + t)  # x[b, c, t] as float32
            angle = 2.0 * 3.141592653589793 * j * t / N
            acc_re += x_t * tl.cos(angle)
            acc_im += x_t * tl.sin(angle)

        # Apply normalization
        acc_re = acc_re * inv_N
        acc_im = acc_im * inv_N

        # Output is (B, C, M) contiguous => index = ((b*C + c) * M) + j
        out_base = (b * C + c) * M
        out_index = out_base + j

        # Store real and imag parts
        tl.store(out_real_ptr + out_index, acc_re)
        tl.store(out_imag_ptr + out_index, acc_im)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Ensure x is contiguous and on CUDA
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()

        # Allocate outputs (B, C, M) as float32, flatten for kernel
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)

        # Launch one Triton program per (b, c) slice
        grid = (B, C)
        _real_rfft_compute_kernel[grid](
            x, out_real, out_imag,
            B, C, L, N, M,
            num_warps=1,  # simple kernel, 1 warp is sufficient
            num_stages=1,
        )

        # Reshape back to (B, C, M)
        out_real = out_real.view(B, C, M)
        out_imag = out_imag.view(B, C, M)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
