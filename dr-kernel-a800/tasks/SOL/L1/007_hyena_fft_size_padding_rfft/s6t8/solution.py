import math
import torch
import triton
import triton.language as tl


@triton.jit
def pad_z_kernel(
    x_ptr,               # *float32, input x of shape (BC, S)
    z_ptr,               # *float32, output z of shape (BC, 4*S), interleaved real/imag (imag zeros)
    BC: tl.int32,        # number of (batch, channel) slices
    S: tl.int32,         # original seqlen
    stride_x_bc: tl.int32,  # stride between (b,c) in x
    stride_z_bc: tl.int32,  # stride between (b,c) in z
):
    bc = tl.program_id(0)
    base_x = bc * stride_x_bc
    base_z = bc * stride_z_bc
    N = 2 * S
    TWO_N = 2 * N

    # First half: j = 0..N-1, zr[j] = x[j], zim[j] = 0
    j = 0
    while j < N:
        v = tl.load(x_ptr + base_x + j)
        tl.store(z_ptr + base_z + j * 2, v)        # real part at index j
        tl.store(z_ptr + base_z + j * 2 + 1, 0.0)  # imag part at index j
        j += 1

    # Middle zeros: j = N..TWO_N-1, zr = 0, zim = 0
    j = N
    while j < TWO_N:
        tl.store(z_ptr + base_z + j * 2, 0.0)      # real part at index j
        tl.store(z_ptr + base_z + j * 2 + 1, 0.0)  # imag part at index j
        j += 1

    # Second half: j = N..2*N-1, zr[j] = -x[S-1 - (j-N)], zim[j] = 0
    j = 0
    while j < N:
        src = S - 1 - j
        v = tl.load(x_ptr + base_x + src)
        v = -v
        tl.store(z_ptr + base_z + (N + j) * 2, v)  # real part at index N+j
        tl.store(z_ptr + base_z + (N + j) * 2 + 1, 0.0)  # imag part at index N+j
        j += 1


@triton.jit
def bitrev_copy_kernel(
    z_ptr,               # *float32, input z of shape (BC, 4*S), interleaved real/imag
    Z_ptr,               # *float32, output Z (bit-reversed copy) of shape (BC, 4*S), interleaved real/imag
    BC: tl.int32,
    S: tl.int32,
    stride_z_bc: tl.int32,
    stride_Z_bc: tl.int32,
    LOG2: tl.constexpr,  # log2(4*S), compile-time constant
):
    bc = tl.program_id(0)
    base_z = bc * stride_z_bc
    base_Z = bc * stride_Z_bc
    n = 4 * S

    # Bit-reversed index calculation: idx_rev = bitrev(idx, LOG2)
    idx = 0
    while idx < n:
        j = idx
        rev = 0
        t = j
        # Compute bit-reversed index for fixed LOG2
        for _ in range(LOG2):
            rev = rev * 2 + (t & 1)
            t = t >> 1
        # Copy real and imag from z to Z at bit-reversed positions
        real = tl.load(z_ptr + base_z + j * 2)
        imag = tl.load(z_ptr + base_z + j * 2 + 1)
        tl.store(Z_ptr + base_Z + rev * 2, real)
        tl.store(Z_ptr + base_Z + rev * 2 + 1, imag)
        idx += 1


@triton.jit
def fft_kernel(
    Z_ptr,               # *float32, input/output Z of shape (BC, 4*S), interleaved real/imag (in-place)
    BC: tl.int32,
    S: tl.int32,
    stride_Z_bc: tl.int32,
    LOG2: tl.constexpr,  # log2(4*S)
):
    bc = tl.program_id(0)
    base_Z = bc * stride_Z_bc
    n = 4 * S

    # Perform in-place Cooley-Tukey FFT on Z of length n
    size = 2
    while size <= n:
        half = size // 2
        j = 0
        while j < half:
            t = j + size // 2
            xr1 = tl.load(Z_ptr + base_Z + j * 2)
            xi1 = tl.load(Z_ptr + base_Z + j * 2 + 1)
            xr2 = tl.load(Z_ptr + base_Z + t * 2)
            xi2 = tl.load(Z_ptr + base_Z + t * 2 + 1)
            # angle = (2*pi*j)/(2*n)
            ang = 6.283185307179586 * j / (2.0 * n)
            c = tl.cos(ang)
            s = tl.sin(ang)
            out_r = xr1 + xr2 * c + xi2 * s
            out_i = xi1 + xr2 * s - xi2 * c
            tl.store(Z_ptr + base_Z + j * 2, out_r)
            tl.store(Z_ptr + base_Z + j * 2 + 1, out_i)
            # t partner
            tl.store(Z_ptr + base_Z + t * 2, xr1 - xr2 * c - xi2 * s)
            tl.store(Z_ptr + base_Z + t * 2 + 1, xi1 - xr2 * s + xi2 * c)
            j += 1
        size *= 2


@triton.jit
def extract_rfft_kernel(
    Z_ptr,               # *float32, after FFT, length 4*S, interleaved real/imag
    yr_ptr, yim_ptr,     # *float32, outputs real/imag of length S+1
    BC: tl.int32,
    S: tl.int32,
    stride_Z_bc: tl.int32,
    ninv: tl.float32,    # 1.0 / (2*S) for normalization
    LOG2: tl.constexpr,  # log2(4*S)
):
    bc = tl.program_id(0)
    base_Z = bc * stride_Z_bc
    M = S + 1

    # We need rfft(x) of length M. For z = [x, zeros, x], Z(2*S + k) = conj(Z(k)) for k=0..2*S.
    # Specifically, the first 2*S elements in Z correspond to conjugates of rfft bins. We extract them and conjugate.
    k = 0
    while k < M:
        # For k > 0, Z[k] = complex with real=0, imag=-imag part; For k==0, Z[0] = sum(x).
        # We extract Z[2*S - k] which is conjugate of Z[k].
        idx_src = (2 * S) - k
        real = tl.load(Z_ptr + base_Z + idx_src * 2)
        imag = tl.load(Z_ptr + base_Z + idx_src * 2 + 1)
        # Normalize by 2*S
        real = real * ninv
        imag = imag * ninv
        # Store real and imag parts
        tl.store(yr_ptr + bc * M + k, real)
        tl.store(yim_ptr + bc * M + k, imag)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          y = torch.fft.rfft(x.float(), n=2*S) / (2*S)
          return y.real, y.imag
        with x shape (B, C, S). Output: (B, C, S+1) float32 tensors.
        """
        assert x.dim() == 3, "Input must be (B, C, S)"
        B, C, S = x.shape
        device = x.device
        dtype = torch.float32

        # Flatten (B, C) into one dimension for Triton
        BC = B * C
        x_flat = x.contiguous().view(BC, S)

        # Allocate z of length 4*S (interleaved real/imag), imag part initialized to zeros
        n = 4 * S
        z = torch.empty((BC, n), device=device, dtype=dtype)
        z.real = z  # alias to access real/imag components
        z.imag = torch.zeros_like(z)

        # Allocate temporary Z (bit-reversed copy) and outputs
        Z = torch.empty_like(z)
        out_real = torch.empty((BC, S + 1), device=device, dtype=dtype)
        out_imag = torch.empty((BC, S + 1), device=device, dtype=dtype)

        # Strides (in number of elements, not bytes)
        stride_x_bc = S
        stride_z_bc = n
        stride_Z_bc = n

        # 1) Pad z = [x, zeros, -x_reversed]
        pad_z_kernel[(BC,)](
            x_flat, z, BC, S, stride_x_bc, stride_z_bc,
            num_warps=4, num_stages=2
        )

        # 2) Bit-reverse copy into Z
        LOG2 = int(math.log2(n))
        bitrev_copy_kernel[(BC,)](
            z, Z, BC, S, stride_z_bc, stride_Z_bc, LOG2,
            num_warps=4, num_stages=2
        )

        # 3) In-place FFT on Z using Cooley-Tukey
        fft_kernel[(BC,)](
            Z, BC, S, stride_Z_bc, LOG2,
            num_warps=4, num_stages=2
        )

        # 4) Extract rfft(x) of length S+1 and normalize by 2*S
        ninv = 1.0 / (2 * S)
        extract_rfft_kernel[(BC,)](
            Z, out_real, out_imag, BC, S, stride_Z_bc, ninv, LOG2,
            num_warps=4, num_stages=2
        )

        # Reshape back to (B, C, S+1)
        out_real = out_real.view(B, C, S + 1)
        out_imag = out_imag.view(B, C, S + 1)
        return out_real, out_imag


# Example usage (CUDA required for Triton):
# model = ModelNew().cuda()
# x = torch.randn(8, 16, 1024, device='cuda', dtype=torch.float32)
# y_real, y_imag = model(x)


def run(*args):
    return ModelNew()(*args)
