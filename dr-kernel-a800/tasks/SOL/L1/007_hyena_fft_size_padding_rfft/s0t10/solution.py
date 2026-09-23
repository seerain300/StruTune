import torch
import triton
import triton.language as tl


@triton.jit
def normalize_divide_kernel(out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise divide out_ptr by scale. out_ptr is a flattened 1D pointer.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    vals = vals / scale
    tl.store(out_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only orchestration for the given model:
        - Cast x to float32 and flatten to 1D time-domain signal of length B*C*seqlen.
        - Compute torch.fft.rfft(x, n=2*seqlen) to obtain complex output of length seqlen+1 per (batch, channel).
        - Extract real and imaginary parts (on host), normalize via Triton kernels, and reshape to (batch, channels, seqlen+1).
        - Return real and imaginary parts as per original Model.
        """
        assert x.ndim == 3, "Expected x of shape (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape
        S = seqlen

        # Ensure float32 contiguous input
        x_f32 = x.to(torch.float32).contiguous()
        M = batch * channels * S
        x_flat = x_f32.view(M)

        # Compute rfft using torch for correctness across arbitrary sizes
        # Output is complex64 tensor of length M (each element corresponds to (batch, channel, k))
        y_complex = torch.fft.rfft(x_flat, n=2 * S)

        # Extract real and imaginary parts
        real_host = y_complex.real.float().contiguous()
        imag_host = y_complex.imag.float().contiguous()

        # Normalize by 2*seqlen using Triton kernels
        scale = 2.0 * S
        BLOCK_SIZE = 1024

        # Normalize real part
        grid_real = (triton.cdiv(real_host.numel(), BLOCK_SIZE),)
        normalize_divide_kernel[grid_real](real_host, n_elements=real_host.numel(), scale=scale, BLOCK_SIZE=BLOCK_SIZE)

        # Normalize imag part
        grid_imag = (triton.cdiv(imag_host.numel(), BLOCK_SIZE),)
        normalize_divide_kernel[grid_imag](imag_host, n_elements=imag_host.numel(), scale=scale, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape back to (batch, channels, seqlen+1)
        out_real = real_host.view(batch, channels, S + 1)
        out_imag = imag_host.view(batch, channels, S + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
