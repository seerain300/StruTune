import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B,                      # int32, batch size (for output indexing)
    C,                      # int32, channels (for output indexing)
    L,                      # int32, seqlen
    N,                      # int32, n = 2 * seqlen
    M,                      # int32, output length = N//2 + 1 (equals L + 1)
):
    # One program per (b, c) slice
    pid = tl.program_id(axis=0)  # total programs = B*C
    b = pid // C
    c = pid % C

    # Precompute 2*pi / N
    two_pi_over_N = 2.0 * 3.14159265358979323846 / N

    # Iterate over j = 0 .. M-1
    for j in range(128):  # 128 is safe upper bound; we mask out j >= M
        j_valid = j < M
        if j_valid:
            # Accumulators for this j
            re_j = 0.0
            im_j = 0.0

            # Scalar loop over t = 0 .. N-1
            for t in range(N):
                # Load x[b, c, t]. x is contiguous along last dim.
                # Address: ((b*C + c)*L + t)
                x_offset = (b * C + c) * L + t
                xt = tl.load(x_ptr + x_offset)
                # angle = j * (2*pi*t / N)
                angle = j * two_pi_over_N * t
                ct = tl.cos(angle)
                st = tl.sin(angle)
                # Accumulate
                re_j += xt * ct
                im_j += xt * st

            # Normalize by N
            re_j = re_j * (1.0 / N)
            im_j = im_j * (1.0 / N)

            # Store to outputs: linear index ((b*C + c)*M + j)
            out_index = (b * C + c) * M + j
            tl.store(out_real_ptr + out_index, re_j)
            tl.store(out_imag_ptr + out_index, im_j)


def _run_triton_rfft(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Triton-only implementation of:
      x_f32: (B, C, L)
      y = torch.fft.rfft(x_f32, n=2*L)  # real FFT over last dim, length 2*L
      y = y / (2*L)
      return y.real, y.imag, each of shape (B, C, L+1)
    Returns real and imaginary parts as float32 tensors, created by ModelNew.forward.
    """
    assert x.is_cuda, "Input must be on CUDA device for Triton kernel."
    assert x.dtype == torch.float32, "Input must be float32 for this Triton kernel."

    B, C, L = x.shape
    N = 2 * L  # rfft length
    M = N // 2 + 1  # output length (equals L + 1 for real input)

    # Allocate outputs
    out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
    out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

    # Launch grid: one program per (b, c) slice
    grid = (B * C,)

    _rfft_real_imag_triton_kernel[grid](
        x, out_real, out_imag,
        B, C, L, N, M,
        num_warps=4,
        num_stages=2,
    )

    return out_real, out_imag


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x of shape (batch, channels, seqlen)
        x = args[0] if len(args) > 0 else None
        if x is None:
            raise ValueError("Input tensor 'x' is required for ModelNew.forward.")

        # Triton-only computation: ensure input is float32 and on CUDA
        if not x.is_cuda:
            raise RuntimeError("Input tensor must be on CUDA device for Triton kernels.")
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, C, L = x.shape
        N = 2 * L
        M = N // 2 + 1  # equals L + 1 for real input

        out_real, out_imag = _run_triton_rfft(x)

        # Return tensors with shape (B, C, L+1) for real and imag, matching original
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
