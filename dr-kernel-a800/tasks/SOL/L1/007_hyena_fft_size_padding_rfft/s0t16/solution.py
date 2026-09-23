import torch
import triton
import triton.language as tl


@triton.jit
def _dct2_rfft_real_kernel(x_ptr, out_ptr, L, N: tl.constexpr):
    """
    Compute real part of rfft(x, n=N) via DCT-II for a real vector x of length L,
    writing output of length N//2 + 1 into out_ptr.
    y[k] = sum_{t=0..L-1} x[t] * cos(pi*k*(t+0.5)/N) for k=1..N//2
    y[0] = 0.5 * sum_{t=0..L-1} x[t]
    """
    # One program per output index k
    pid = tl.program_id(axis=0)
    if pid == 0:
        # k = 0 special case
        sum_val = tl.zeros((), dtype=tl.float32)
        t = 0
        while t < L:
            val = tl.load(x_ptr + t)
            sum_val += val
            t += 1
        # torch.rfft scales by 1/N for real outputs; here we compute DCT-II which needs scaling by 1/N.
        sum_val *= 0.5  # rfft real output y[0] equals 0.5 * sum(x)
        tl.store(out_ptr + 0, sum_val)
    else:
        k = pid - 1  # k in [0..N//2 - 1]
        sum_val = tl.zeros((), dtype=tl.float32)
        t = 0
        while t < L:
            # cos(pi*k*(t+0.5)/N)
            theta = tl.cos(3.141592653589793 * k * (t + 0.5) / N)
            val = tl.load(x_ptr + t)
            sum_val += val * theta
            t += 1
        # Store scaled result (rfft normalization divides by N)
        sum_val = sum_val / N
        tl.store(out_ptr + k + 1, sum_val)


@triton.jit
def _divide_inplace_kernel(x_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x / scale
    tl.store(x_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of the original run(x):
        - Input: x of shape (batch, channels, seqlen)
        - Compute real part of rfft(x, n=2*seqlen) using DCT-II in Triton
        - Return real and imaginary parts (imag=0), normalized by 2*seqlen
        Output shape: (batch, channels, seqlen+1) for both real and imag
        """
        assert x.dim() == 3, f"Input must be (batch, channels, seqlen), got {x.shape}"
        batch, channels, seqlen = x.shape
        N = 2 * seqlen

        # Flatten to 1D vector for kernel processing
        x_flat = x.reshape(-1).to(torch.float32).contiguous()
        L = x_flat.numel()  # equals batch*channels*seqlen, but we only use seqlen for DCT

        # Output length: (N//2) + 1
        out_len = (N // 2) + 1

        # Allocate output for real rfft
        out_real = torch.empty(out_len, dtype=torch.float32, device=x.device)

        # Launch DCT-II kernel: one program per output index k
        grid = (out_len,)
        _dct2_rfft_real_kernel[grid](x_flat, out_real, L, N=N)

        # Normalize by N = 2*seqlen using Triton
        scale = float(N)
        BLOCK_SIZE = 1024
        grid_norm = (triton.cdiv(out_len, BLOCK_SIZE),)
        _divide_inplace_kernel[grid_norm](out_real, out_len, scale, BLOCK_SIZE=BLOCK_SIZE)

        # Reshape to (batch, channels, seqlen + 1)
        out_real = out_real.view(batch, channels, seqlen + 1)

        # Imaginary part is zero (rfft of real input has zero imaginary part)
        out_imag = torch.zeros_like(out_real)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
