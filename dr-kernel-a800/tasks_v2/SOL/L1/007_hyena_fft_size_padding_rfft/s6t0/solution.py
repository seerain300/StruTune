import torch
import triton
import triton.language as tl


@triton.jit
def pad_kernel(x_ptr, xp_ptr, B: tl.int32, C: tl.int32, S: tl.int32, BLOCK: tl.constexpr):
    """
    Construct a padded input tensor xp of shape (B, C, 2*S) from x of shape (B, C, S).
    xp[..., :S] = x
    xp[..., S:] = 0
    Assumes x_ptr points to (B, C, S) contiguous and xp_ptr points to (B, C, 2*S) contiguous.
    Indexing:
      x[b, c, k] -> x_ptr + ((b*C + c) * S + k)
      xp[b, c, j] -> xp_ptr + ((b*C + c) * (2*S) + j)
    """
    pid = tl.program_id(axis=0)  # each program handles one (b, c) slice
    b = pid // C
    c = pid % C

    base_x = (b * C + c) * S
    base_xp = (b * C + c) * (2 * S)

    # Copy non-padded part: x -> xp[:S]
    idx = tl.arange(0, BLOCK)
    mask1 = idx < S
    x_vals = tl.load(x_ptr + base_x + idx, mask=mask1, other=0.0)
    tl.store(xp_ptr + base_xp + idx, x_vals, mask=mask1)

    # Zero-pad the remaining S positions: xp[S:] = 0
    pad_idx = idx + S
    zero_vals = tl.zeros([BLOCK], dtype=tl.float32)
    tl.store(xp_ptr + base_xp + pad_idx, zero_vals, mask=idx < S)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect x of shape (batch, channels, seqlen)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        B, C, S = x.shape

        # CPU fallback: original PyTorch implementation
        if x.device.type != 'cuda':
            x_f32 = x.to(torch.float32)
            fft_size = 2 * S
            x_freq = torch.fft.rfft(x_f32, n=fft_size)
            x_freq = x_freq / (2 * S)
            x_freq_real = x_freq.real.contiguous()
            x_freq_imag = x_freq.imag.contiguous()
            return x_freq_real, x_freq_imag

        # CUDA path: use Triton for padding, then PyTorch for FFT
        x_f32 = x.to(torch.float32).contiguous()

        # Allocate padded input: (B, C, 2*S), initialized to zeros
        xp = torch.zeros((B, C, 2 * S), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: 1D grid over B*C slices
        grid = (B * C,)
        BLOCK = 1024  # process up to 1024 elements per program; mask handles smaller S
        pad_kernel[grid](x_f32, xp, B, C, S, BLOCK=BLOCK, num_warps=4)

        # Perform FFT along last dimension (length 2*S). Output is complex of length S+1.
        x_freq = torch.fft.rfft(xp, n=2 * S, dim=-1)  # (B, C, S+1, complex)
        x_freq = x_freq / (2 * S)

        # Extract real and imaginary parts
        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
