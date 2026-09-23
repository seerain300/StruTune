import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(in_ptr, out_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def scale_kernel(in_ptr, out_ptr, scale, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elems
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    vals = vals * scale
    tl.store(out_ptr + offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32 and contiguous
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if not x.is_contiguous():
            x = x.contiguous()

        B, C, L = x.shape
        two_L = 2 * L

        # Compute rfft along the last dimension; result is complex of shape (B, C, L+1)
        x_freq_complex = torch.fft.rfft(x, n=two_L, dim=-1)  # shape: (B, C, L+1)

        # Extract real and imaginary parts
        real_part = x_freq_complex.real  # float32 tensor
        imag_part = x_freq_complex.imag  # float32 tensor

        # Flatten for Triton kernels
        real_flat = real_part.contiguous().view(-1)
        imag_flat = imag_part.contiguous().view(-1)

        total_elems = real_flat.numel()  # equals B*C*(L+1)

        # Allocate output buffers
        out_real = torch.empty(total_elems, dtype=torch.float32, device=x.device)
        out_imag = torch.empty(total_elems, dtype=torch.float32, device=x.device)

        # Kernel 1: copy real/imag parts
        grid = (triton.cdiv(total_elems, 1024),)
        copy_kernel[grid](real_flat, out_real, total_elems, BLOCK=1024)
        copy_kernel[grid](imag_flat, out_imag, total_elems, BLOCK=1024)

        # Normalize by 2*L (scale by 1/(2*L))
        scale = 1.0 / (2.0 * L)
        scaled_real = torch.empty_like(out_real, dtype=torch.float32, device=x.device)
        scaled_imag = torch.empty_like(out_imag, dtype=torch.float32, device=x.device)

        # Kernel 2: scale
        scale_kernel[grid](out_real, scaled_real, scale, total_elems, BLOCK=1024)
        scale_kernel[grid](out_imag, scaled_imag, scale, total_elems, BLOCK=1024)

        # Reshape back to (B, C, L+1)
        x_freq_real = scaled_real.view(B, C, L + 1)
        x_freq_imag = scaled_imag.view(B, C, L + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
