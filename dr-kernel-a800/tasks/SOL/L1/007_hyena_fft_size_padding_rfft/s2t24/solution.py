import torch
import triton
import triton.language as tl

@triton.jit
def real_rfft_kernel(
    x_ptr,           # *float32, input x of shape (B, C, L) contiguous
    out_ptr,         # *float32, output real part buffer (B*C*(L+1))
    imag_ptr,        # *float32, output imaginary part buffer (B*C*(L+1))
    B: tl.constexpr, # batch size
    C: tl.constexpr, # channels
    L: tl.constexpr, # seqlen
    two_L: tl.constexpr, # 2*L
):
    # Each program handles one (b, c) slice
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Base pointer offset for this (b, c) slice in x: x is laid out as (B, C, L) contiguous
    base = (b * C + c) * L

    # 1) X[0] = sum(input) / two_L
    sum0 = 0.0
    for t in range(0, two_L):
        if t < L:
            val = tl.load(x_ptr + base + t)
        else:
            val = 0.0
        sum0 += val
    tl.store(out_ptr + (b * C + c) * (L + 1) + 0, sum0 / two_L)
    tl.store(imag_ptr + (b * C + c) * (L + 1) + 0, 0.0)

    # 2) X[1] = (sum(odd) - sum(even)) / two_L
    sum_odd = 0.0
    sum_even = 0.0
    for t in range(0, two_L):
        if t < L:
            val = tl.load(x_ptr + base + t)
        else:
            val = 0.0
        if (t % 2) == 1:
            sum_odd += val
        else:
            sum_even += val
    tl.store(out_ptr + (b * C + c) * (L + 1) + 1, (sum_odd - sum_even) / two_L)
    tl.store(imag_ptr + (b * C + c) * (L + 1) + 1, 0.0)

    # 3) X[L] = (sum(odd) + sum(even)) / two_L
    sum_odd = 0.0
    sum_even = 0.0
    for t in range(0, two_L):
        if t < L:
            val = tl.load(x_ptr + base + t)
        else:
            val = 0.0
        if (t % 2) == 1:
            sum_odd += val
        else:
            sum_even += val
    tl.store(out_ptr + (b * C + c) * (L + 1) + L, (sum_odd + sum_even) / two_L)
    tl.store(imag_ptr + (b * C + c) * (L + 1) + L, 0.0)

    # 4) X[k] for k in [2..L-1]
    for k in range(2, L):
        real_k = 0.0
        imag_k = 0.0
        for t in range(0, two_L):
            if t < L:
                val = tl.load(x_ptr + base + t)
            else:
                val = 0.0
            angle = (2.0 * 3.141592653589793 * float(k) * float(t)) / float(two_L)
            real_k += val * tl.cos(angle)
            imag_k -= val * tl.sin(angle)  # negative sign in rfft relation
        tl.store(out_ptr + (b * C + c) * (L + 1) + k, real_k / two_L)
        tl.store(imag_ptr + (b * C + c) * (L + 1) + k, imag_k / two_L)

# Launch Triton kernel(s) in ModelNew.forward
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, L) float32 tensor on CUDA
        returns: (B, C, L+1) float32 tensors for real and imaginary parts
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        # Ensure contiguous
        x = x.contiguous()
        B, C, L = x.shape
        two_L = 2 * L

        # Allocate outputs
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Launch one program per (b, c)
        grid = (B, C)
        real_rfft_kernel[grid](
            x, out_real, out_imag,
            B=B, C=C, L=L, two_L=two_L,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
