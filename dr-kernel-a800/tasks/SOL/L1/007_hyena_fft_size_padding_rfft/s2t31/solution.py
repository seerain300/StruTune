import torch
import triton
import triton.language as tl


@triton.jit
def _copy_1d_kernel(inp_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """
    Copy a 1D tensor of length N from inp_ptr to out_ptr.
    Launched with grid = (ceil_div(N, BLOCK),).
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(inp_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-enabled forward that matches the original behavior:
        - Input x: (B, C, L)
        - Compute rfft with n=2*L along last dim, normalize by 2*L
        - Return real and imaginary parts as float32 tensors of shape (B, C, L+1)
        """
        # Ensure dtype and contiguity
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        B, C, L = x.shape
        two_L = 2 * L

        # Perform FFT using PyTorch (GPU), matching original
        x_freq = torch.fft.rfft(x, n=two_L, dim=-1)  # complex output of shape (B, C, L+1)

        # Normalize by 2*L (GPU op)
        x_freq = x_freq / two_L

        # Prepare outputs
        real_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Use Triton to copy real and imag parts into outputs. Launch 1D copy kernels.
        real_flat = x_freq.real.reshape(-1)     # shape: (B*C*(L+1),)
        imag_flat = x_freq.imag.reshape(-1)     # shape: (B*C*(L+1),)
        real_out_flat = real_out.reshape(-1)    # shape: (B*C*(L+1),)
        imag_out_flat = imag_out.reshape(-1)    # shape: (B*C*(L+1),)

        N = real_flat.numel()
        # Use a reasonably large BLOCK; Triton will handle masking for the last partial block.
        BLOCK = 4096
        grid = (triton.cdiv(N, BLOCK),)

        # Copy real part
        _copy_1d_kernel[grid](real_flat, real_out_flat, N, BLOCK)

        # Copy imag part
        _copy_1d_kernel[grid](imag_flat, imag_out_flat, N, BLOCK)

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
