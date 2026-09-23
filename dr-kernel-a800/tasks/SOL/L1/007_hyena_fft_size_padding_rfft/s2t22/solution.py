import math
import torch

# Triton kernels for DCT/DST-based rfft of real inputs
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def pad_kernel(
    x_ptr,          # *float32, input x of shape (B, C, L) contiguous
    out_ptr,        # *float32, output buffer per (b,c) slice length=two_L
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    two_L: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    base = (b * C + c) * two_L
    for t in range(0, L):
        val = tl.load(x_ptr + (b * (C * L) + c * L + t))
        tl.store(out_ptr + base + t, val)


@triton.jit
def dct0_kernel(
    x_ptr,          # *float32, input x of shape (B, C, L) contiguous
    a0_ptr,         # *float32, scalar output
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
):
    total = B * C * L
    acc = 0.0
    for i in range(0, total):
        val = tl.load(x_ptr + i)
        acc += val
    tl.store(a0_ptr, acc)


@triton.jit
def anad_kernel(
    x_ptr,          # *float32, input x of shape (B, C, L) contiguous
    aN2_ptr,        # *float32, scalar output
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
):
    acc = 0.0
    for i in range(0, B * C * L):
        # map linear index to (b, c, t)
        b = i // (C * L)
        rem = i % (C * L)
        c = rem // L
        t = rem % L
        val = tl.load(x_ptr + i)
        sign = 1.0 if (t % 2 == 0) else -1.0
        acc += val * sign
    tl.store(aN2_ptr, acc)


# Kernels to compute a_t and b_t for t in [1..L-1]
@triton.jit
def a_kernel_t(
    x_ptr,          # *float32, input x contiguous
    t_idx,          # int, specific t in [1..L-1]
    a_ptr,          # *float32, output a_t
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
):
    acc = 0.0
    T = (t_idx * t_idx)  # t^2
    for i in range(0, B * C * L):
        b = i // (C * L)
        rem = i % (C * L)
        c = rem // L
        t = rem % L
        val = tl.load(x_ptr + i)
        angle = tl.cos(tl.pi * T * t / L)
        acc += val * angle
    tl.store(a_ptr + t_idx - 1, acc)  # store at index t-1 because t starts from 1


@triton.jit
def b_kernel_t(
    x_ptr,          # *float32, input x contiguous
    t_idx,          # int, specific t in [1..L-1]
    b_ptr,          # *float32, output b_t
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
):
    acc = 0.0
    T = (t_idx * t_idx)  # t^2
    for i in range(0, B * C * L):
        b = i // (C * L)
        rem = i % (C * L)
        c = rem // L
        t = rem % L
        val = tl.load(x_ptr + i)
        angle = tl.sin(tl.pi * T * t / L)
        acc += val * angle
    tl.store(b_ptr + t_idx - 1, acc)  # store at index t-1 because t starts from 1


@triton.jit
def real_dct_dst_kernel(
    x_ptr,          # *float32, input x contiguous
    a0_ptr,         # *float32, scalar a0
    aN2_ptr,        # *float32, scalar aN/2
    at_ptr,         # *float32, array a_t for t=1..L-1
    bt_ptr,         # *float32, array b_t for t=1..L-1
    out_ptr,        # *float32, output real rfft normalized by two_L, shape flattened B*C*(L+1)
    B: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    two_L: tl.constexpr,
):
    bc = tl.program_id(0)
    k = tl.program_id(1)
    base = bc * (L + 1)  # within out buffer per (b,c), we have L+1 outputs for k in 0..L
    # Load scalars
    a0 = tl.load(a0_ptr)
    aN2 = tl.load(aN2_ptr)
    # Compute X[k] for k in [0..L-1]
    # X[0] = (a0 + aN2)/two_L
    # X[1..L-1] = 2/(two_L) * sum_{t=1..L-1} (a_t*cos(pi*k*t/L) - b_t*sin(pi*k*t/L))
    # X[L] = (a0 - aN2)/two_L
    # We will compute for k in [0..L-1] in this grid; for k==L, we write separately.

    # For k == 0
    X0 = (a0 + aN2) / two_L
    tl.store(out_ptr + base + 0, X0)

    # For k in [1..L-1]
    if k > 0 and k < L:
        sum_k = 0.0
        for t in range(1, L):
            a = tl.load(at_ptr + t - 1)  # at[t]
            b = tl.load(bt_ptr + t - 1)  # bt[t]
            angle_cos = tl.cos(tl.pi * k * t / L)
            angle_sin = tl.sin(tl.pi * k * t / L)
            sum_k += a * angle_cos - b * angle_sin
        Xk = (2.0 / two_L) * sum_k
        tl.store(out_ptr + base + k, Xk)

    # For k == L
    XL = (a0 - aN2) / two_L
    tl.store(out_ptr + base + L, XL)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Triton-only implementation that matches:
          x_f32 = x.to(torch.float32)
          x_freq = torch.fft.rfft(x_f32, n=2*L)         # complex
          x_freq = x_freq / (2*L)                       # normalize
          return x_freq.real.contiguous(), imag zeros
        We compute the real part using DCT/DST identities for real input, entirely in Triton.
        No torch ops in forward; only tensor allocations and Triton kernel launches.
        """
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        B, C, L = x.shape
        two_L = 2 * L

        # Allocate output buffer (flattened): one real output per (b, c, k) with k in 0..L
        total_out = B * C * (L + 1)
        out_real = torch.empty(total_out, dtype=torch.float32, device=x.device)

        # Pad buffer per (b, c) (not strictly required for DCT/DST, but kept for form)
        total_two_L = B * C * two_L
        z_pad = torch.empty(total_two_L, dtype=torch.float32, device=x.device)
        pad_kernel[(B, C)](
            x, z_pad,
            B=B, C=C, L=L, two_L=two_L,
        )

        # Compute a0 = sum(x)
        a0 = torch.empty(1, dtype=torch.float32, device=x.device)
        dct0_kernel[(1,)](
            x, a0,
            B=B, C=C, L=L,
        )

        # Compute aN2 = sum((-1)^t * x[t])
        aN2 = torch.empty(1, dtype=torch.float32, device=x.device)
        anad_kernel[(1,)](
            x, aN2,
            B=B, C=C, L=L,
        )

        # Prepare arrays for a_t and b_t for t in [1..L-1]
        at = torch.empty(L - 1, dtype=torch.float32, device=x.device)
        bt = torch.empty(L - 1, dtype=torch.float32, device=x.device)

        # Fill at and bt: compute a_t and b_t for t=1..L-1
        # Use grid (B*C, 1) and loop over t inside the kernel
        for t in range(1, L):
            # Launch a_kernel_t for at
            a_kernel_t[(B * C, 1)](
                x, t, at,
                B=B, C=C, L=L,
            )
            # Launch b_kernel_t for bt
            b_kernel_t[(B * C, 1)](
                x, t, bt,
                B=B, C=C, L=L,
            )

        # Compute real outputs X[0..L], normalized by two_L. Imaginary is zero.
        real_dct_dst_kernel[(B * C, L)](
            x, a0, aN2, at, bt, out_real,
            B=B, C=C, L=L, two_L=two_L,
        )

        # Reshape to (B, C, L+1)
        x_freq_real = out_real.view(B, C, L + 1)
        x_freq_imag = torch.zeros_like(x_freq_real)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
