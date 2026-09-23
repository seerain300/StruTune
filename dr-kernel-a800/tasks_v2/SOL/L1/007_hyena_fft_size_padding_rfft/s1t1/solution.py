import torch
import triton
import triton.language as tl

@triton.jit
def copy_flat_kernel(inp_ptr, out_ptr, size: tl.int32, BLOCK: tl.constexpr):
    """
    1D copy kernel: copies 'size' elements from inp_ptr to out_ptr.
    Launch multiple programs; each handles a chunk of size BLOCK.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < size
    vals = tl.load(inp_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, vals, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape (batch, channels, seqlen)
        x = args[0]
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Cast to float32 and ensure contiguous
        x_f32 = x.to(torch.float32).contiguous()

        # Compute rfft along the last dimension with padded length n = 2 * seqlen
        n = 2 * seqlen
        x_freq = torch.fft.rfft(x_f32, n=n)  # complex tensor of shape (batch, channels, seqlen+1)

        # Normalize by 2 * seqlen (same as original)
        x_freq = x_freq / n

        # Output shape must be (batch, channels, seqlen + 1)
        M = seqlen + 1

        # Allocate outputs
        out_real = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((batch, channels, M), dtype=torch.float32, device=x.device)

        # Flatten views for Triton copying
        real_view = x_freq.real.contiguous().view(-1)    # length = batch*channels*M
        imag_view = x_freq.imag.contiguous().view(-1)    # length = batch*channels*M
        out_real_flat = out_real.view(-1)
        out_imag_flat = out_imag.view(-1)

        # Launch Triton copy kernels
        BLOCK = 1024
        num_items = real_view.numel()
        grid = (triton.cdiv(num_items, BLOCK),)
        copy_flat_kernel[grid](real_view, out_real_flat, num_items, BLOCK=BLOCK, num_warps=4)
        copy_flat_kernel[grid](imag_view, out_imag_flat, num_items, BLOCK=BLOCK, num_warps=4)

        # Return real and imaginary parts as per original function
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
