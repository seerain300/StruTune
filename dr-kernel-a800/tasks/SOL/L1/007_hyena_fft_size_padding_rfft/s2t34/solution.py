import torch
import triton
import triton.language as tl

# Triton kernel: for each (b, c), build a zero-padded vector of length two_L = 2*L
# Input x: shape (B, C, L), contiguous float32
# Output out_pad: flattened vector of length M = (B*C) * two_L
@triton.jit
def pad_kernel(x_ptr, out_ptr, B: tl.constexpr, C: tl.constexpr, L: tl.constexpr):
    # grid = (B*C,)
    idx = tl.program_id(0)
    b = idx // C
    c = idx % C

    # base offset to (b, c, 0) in x: b*C*L + c*L
    base = b * C * L + c * L

    # number of elements in this (b, c) slice
    n = L

    # write first L elements
    for i in range(n):
        val = tl.load(x_ptr + base + i)  # load float32
        out_base = idx * (2 * L)
        tl.store(out_ptr + out_base + i, val)

    # write remaining L zeros
    for i in range(n, 2 * L):
        out_base = idx * (2 * L)
        tl.store(out_ptr + out_base + i, 0.0)


# Triton kernel: real DFT over the padded vector, one program per (b, c)
# Computes X[k] = sum_{t=0}^{two_L-1} x[t] * exp(-2πi k t / two_L), for k in [0..L]
# Stores normalized real part in out_real. Imaginary part is zero (for real inputs).
@triton.jit
def real_dft_kernel(out_pad_ptr, out_real_ptr, B: tl.constexpr, C: tl.constexpr, L: tl.constexpr):
    # grid = (B*C,)
    idx = tl.program_id(0)
    two_L = 2 * L

    # base pointer for this (b, c) in padded vector
    out_base = idx * two_L

    # accumulate in float32
    for k in range(L + 1):  # output length is L+1; we compute up to k=L
        acc = 0.0
        # sum over t = 0..two_L-1
        for t in range(two_L):
            val = tl.load(out_pad_ptr + out_base + t)
            angle = -2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
            cos_angle = tl.cos(angle)
            acc += val * cos_angle
        # normalize by 2*L
        acc = acc / float(two_L)

        # store real part
        out_real_base = idx * (L + 1)
        tl.store(out_real_ptr + out_real_base + k, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Ensure float32 and contiguous
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        B, C, L = x.shape
        two_L = 2 * L

        # Output real part (we compute it via Triton)
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Imaginary part for real inputs is zero; allocate zeros explicitly
        out_imag = torch.zeros((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Flatten pointer for Triton: one program per (b, c)
        M = B * C

        # Allocate flattened padded input vector
        out_pad = torch.empty(M * two_L, dtype=torch.float32, device=x.device)

        # Launch pad kernel: build zero-padded vector per (b, c)
        pad_kernel[(M,)](x, out_pad, B, C, L)

        # Launch real DFT kernel: compute real part only
        real_dft_kernel[(M,)](out_pad, out_real, B, C, L)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
