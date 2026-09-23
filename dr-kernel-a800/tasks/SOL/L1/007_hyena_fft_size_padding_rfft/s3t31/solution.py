import torch
import triton
import triton.language as tl

@triton.jit
def copy_real_kernel(src_ptr, dst_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)

@triton.jit
def copy_imag_kernel(src_ptr, dst_ptr, n_elements: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, vals, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect input shape (batch, channels, seqlen)
        assert x.ndim == 3, "Input must be 3D: (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Cast to float32 for numerical stability (matches original code)
        x_f32 = x.to(torch.float32)

        # Compute rfft along the last dimension with implicit zero-padding to 2*seqlen
        twoL = 2 * seqlen
        x_freq_complex = torch.fft.rfft(x_f32, n=twoL, dim=-1)

        # Normalize by 2*seqlen (original code divides by 2*seqlen)
        x_freq_complex = x_freq_complex / float(twoL)

        # Prepare output tensors: real and imaginary parts of shape (batch, channels, seqlen+1)
        L_out = seqlen + 1
        real_out = torch.empty((batch, channels, L_out), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L_out), dtype=torch.float32, device=x.device)

        # Total elements to copy for each part
        total_elems = batch * channels * L_out

        # Launch Triton kernels to copy real and imaginary parts
        BLOCK = 4096
        grid = (triton.cdiv(total_elems, BLOCK),)

        # Copy real parts
        copy_real_kernel[grid](x_freq_complex.real, real_out, total_elems, BLOCK=BLOCK, num_warps=4)
        # Copy imag parts
        copy_imag_kernel[grid](x_freq_complex.imag, imag_out, total_elems, BLOCK=BLOCK, num_warps=4)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
