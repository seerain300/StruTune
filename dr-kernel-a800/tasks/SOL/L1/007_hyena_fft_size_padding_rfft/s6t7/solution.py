import torch
import triton
import triton.language as tl


@triton.jit
def pad_and_copy_kernel(
    x_ptr,           # *float32, input x shape (B*C, S)
    zr_ptr, zim_ptr, # *float32, output z real/imag shape (B*C, 4*S), interleaved stores
    B, C, S,         # int32
    stride_x_bc: tl.int32,   # elements per (b,c): S
    stride_z_bc: tl.int32,   # elements per (b,c): 4*S
):
    bc = tl.program_id(0)
    base_x = bc * stride_x_bc
    base_z = bc * stride_z_bc

    N = 2 * S

    # First half: j=0..S-1 -> zr[j] = x[j], zim[j] = 0
    j = 0
    while j < S:
        v = tl.load(x_ptr + base_x + j)
        tl.store(zr_ptr + base_z + j, v)         # real part
        tl.store(zim_ptr + base_z + j, 0.0)      # imag part
        j += 1

    # Middle zeros: j=S..2S-1 -> zr[j]=0, zim[j]=0
    j = S
    while j < N:
        tl.store(zr_ptr + base_z + j, 0.0)
        tl.store(zim_ptr + base_z + j, 0.0)
        j += 1

    # Second half reversed: j=2S..3S-1 -> zr[j] = x[S-1-j], zim[j] = 0
    j = 0
    while j < S:
        src = S - 1 - j
        v = tl.load(x_ptr + base_x + src)
        tl.store(zr_ptr + base_z + (2 * S + j), v)
        tl.store(zim_ptr + base_z + (2 * S + j), 0.0)
        j += 1


@triton.jit
def fft_cooley_tukey_kernel(
    zr_ptr, zim_ptr,        # *float32, input/output real/imag (length 2N), interleaved storage
    TWO_N: tl.int32,        # 2N
    stride: tl.int32,       # elements per (bc): 4*S
):
    # Single program per (b,c) slice; operate in-place on zr/zim
    bc = tl.program_id(0)
    base = bc * stride

    n = TWO_N
    # Classic Cooley-Tukey: iterate stages
    k = 1
    while k < n:
        j = 0
        while j < n:
            idx1 = j
            idx2 = j + k
            # Load current values
            xr1 = tl.load(zr_ptr + base + idx1)
            xi1 = tl.load(zim_ptr + base + idx1)
            xr2 = tl.load(zr_ptr + base + idx2)
            xi2 = tl.load(zim_ptr + base + idx2)

            # Angle = -2*pi * idx1 * k / (2N)
            ang = -2.0 * 3.141592653589793 * idx1 * k / (2.0 * n)
            c = tl.cos(ang)
            s = tl.sin(ang)

            # Butterfly mix
            mixed_r = xr2 * c + xi2 * s
            mixed_i = -xr2 * s + xi2 * c

            xr1_new = xr1 + mixed_r
            xi1_new = xi1 + mixed_i
            xr2_new = xr1 - mixed_r
            xi2_new = xi1 - mixed_i

            # Store back
            tl.store(zr_ptr + base + idx1, xr1_new)
            tl.store(zim_ptr + base + idx1, xi1_new)
            tl.store(zr_ptr + base + idx2, xr2_new)
            tl.store(zim_ptr + base + idx2, xi2_new)

            j += 2 * k
        k *= 2


@triton.jit
def extract_normalize_kernel(
    yr_ptr, yim_ptr,       # *float32, real/imag parts of y after FFT, length 2N, but we read only 0..S
    out_real_ptr, out_imag_ptr,  # *float32, outputs (B*C, S+1)
    TWO_N: tl.int32,       # 2N
    stride_y: tl.int32,    # elements per (bc): 4*S
    inv_n: tl.float32,     # 1/(2*N)
    S: tl.int32,           # original S
):
    bc = tl.program_id(0)
    base_y = bc * stride_y
    out_base = bc * (S + 1)

    # We only need first S+1 entries: j=0..S
    j = 0
    while j <= S:
        yr = tl.load(yr_ptr + base_y + j)
        yi = tl.load(yim_ptr + base_y + j)
        yr = yr * inv_n
        yi = yi * inv_n
        tl.store(out_real_ptr + out_base + j, yr)
        tl.store(out_imag_ptr + out_base + j, yi)
        j += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x: (B, C, S), float32
        assert x.ndim == 3, "Input must be 3D (B, C, S)"
        B, C, S = x.shape
        device = x.device
        dtype = x.dtype

        # Prepare outputs
        out_real = torch.empty((B, C, S + 1), device=device, dtype=torch.float32)
        out_imag = torch.empty((B, C, S + 1), device=device, dtype=torch.float32)

        # We need zr/zim of length 2*S = N; in interleaved real/imag storage (4*S elements per (b,c))
        N = 2 * S
        zr = torch.empty((B * C, N), device=device, dtype=torch.float32)
        zim = torch.empty((B * C, N), device=device, dtype=torch.float32)

        # Flatten x to (B*C, S) for kernel convenience
        x_flat = x.view(B * C, S)

        # 1) Pad and copy into z: z = [x, zeros, x_rev] interleaved real/imag
        grid1 = (B * C,)
        pad_and_copy_kernel[grid1](
            x_flat, zr, zim, B, C, S,
            stride_x_bc=S, stride_z_bc=N,  # per (b,c) elements
            num_warps=1, num_stages=2
        )

        # 2) Perform Cooley-Tukey FFT in-place on zr/zim (length N), using the interleaved storage
        grid2 = (B * C,)
        fft_cooley_tukey_kernel[grid2](
            zr, zim, TWO_N=N, stride=N,
            num_warps=1, num_stages=2
        )

        # 3) Extract first S+1 entries from zr/zim (these equal rfft(x)), normalize by 2*S
        inv_n = 1.0 / (2.0 * S)
        grid3 = (B * C,)
        extract_normalize_kernel[grid3](
            zr, zim, out_real.view(B * C, S + 1), out_imag.view(B * C, S + 1),
            TWO_N=N, stride_y=N, inv_n=inv_n, S=S,
            num_warps=1, num_stages=2
        )

        # Return shaped outputs
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
