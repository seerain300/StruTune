import torch
import triton
import triton.language as tl


@triton.jit
def normalize_divide_kernel(x_ptr, y_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise normalize: y[i] = x[i] / scale
    x_ptr: pointer to input tensor (real or imag)
    y_ptr: pointer to output tensor
    n_elements: total number of elements
    scale: scalar float32 (division factor = 2 * seqlen)
    BLOCK_SIZE: grid meta-parameter
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x / scale
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x of shape (batch, channels, seqlen)
        if len(args) == 0:
            raise ValueError("ModelNew.forward expects at least one input tensor.")
        x = args[0]
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor.")

        # Ensure 3D input (batch, channels, seqlen)
        if x.dim() != 3:
            raise ValueError("Input must be a 3D tensor of shape (batch, channels, seqlen).")
        batch, channels, seqlen = x.shape

        # Cast to float32 for numerical stability (as original)
        x_f32 = x.to(torch.float32)

        # Compute rFFT with implicit zero-padding to n=2*seqlen
        # Output complex length = (2*seqlen)//2 + 1 = seqlen + 1 per (batch, channel)
        x_freq = torch.fft.rfft(x_f32, n=2 * seqlen)

        # Extract real and imaginary parts and ensure contiguous memory for Triton
        x_freq_real = x_freq.real.contiguous()
        x_freq_imag = x_freq.imag.contiguous()

        # Prepare output tensors
        out_real = torch.empty_like(x_freq_real)
        out_imag = torch.empty_like(x_freq_imag)

        # Total number of elements
        n_elements = out_real.numel()
        scale = float(2 * seqlen)

        # Launch Triton normalization kernels (elementwise divide)
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        normalize_divide_kernel[grid](x_freq_real, out_real, n_elements, scale, BLOCK_SIZE=BLOCK_SIZE)
        normalize_divide_kernel[grid](x_freq_imag, out_imag, n_elements, scale, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape back to (batch, channels, seqlen + 1)
        out_real = out_real.view(batch, channels, seqlen + 1)
        out_imag = out_imag.view(batch, channels, seqlen + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
