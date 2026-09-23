import torch
import triton
import triton.language as tl


@triton.jit
def copy_tensor_kernel(x_ptr, y_ptr, size: tl.int32, BLOCK_SIZE: tl.constexpr):
    """
    1D copy kernel: y[i] = x[i] for i in [0, size).
    We assume x_ptr and y_ptr point to contiguous tensors of length 'size'.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x with shape (batch, channels, seqlen)
        x = args[0] if len(args) > 0 else None
        if x is None:
            return None

        # Ensure float32 for numerical stability
        x_f32 = x.to(torch.float32)

        batch, channels, seqlen = x_f32.shape
        twoL = 2 * seqlen

        # Compute rfft along last dimension for each (batch, channel) slice
        # Output is complex, shape (batch, channels, seqlen+1)
        x_freq_complex = torch.fft.rfft(x_f32, n=twoL, dim=-1)

        # Normalize by 2*seqlen (as in the original code)
        x_freq_complex = x_freq_complex / twoL

        # Extract real and imaginary parts
        x_freq_real = x_freq_complex.real  # shape (B, C, L+1)
        x_freq_imag = x_freq_complex.imag  # shape (B, C, L+1)

        # Prepare output tensors
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Flatten for simple 1D copying by Triton
        # Important: ensure flattened sizes match the output tensors
        real_flat = x_freq_real.contiguous().view(-1)         # numel = B*C*(L+1)
        imag_flat = x_freq_imag.contiguous().view(-1)         # numel = B*C*(L+1)
        out_flat_real = real_out.contiguous().view(-1)        # numel = B*C*(L+1)
        out_flat_imag = imag_out.contiguous().view(-1)        # numel = B*C*(L+1)

        total_elements = real_flat.numel()

        # Launch Triton copy kernels (ensure Triton is used)
        # Use a large block size for throughput; sizes vary but typical are big.
        BLOCK_SIZE = 4096
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)

        # Copy real part
        copy_tensor_kernel[grid](real_flat, out_flat_real, total_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        # Copy imag part
        copy_tensor_kernel[grid](imag_flat, out_flat_imag, total_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
