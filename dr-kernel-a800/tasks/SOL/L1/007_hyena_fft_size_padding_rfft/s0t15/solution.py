import math
import torch
import triton
import triton.language as tl


@triton.jit
def bitreverse_pairs_kernel(t_ptr, N: tl.constexpr):
    """
    In-place bit-reverse pairing for the first half of a time-domain vector t of length 2*N elements.
    For i in [0, N), compute rev = bit_reverse(i, N) and swap t[2*i] with t[2*rev],
    and t[2*i+1] with t[2*rev+1].
    """
    HALF = N  # N is the first-half size, equals seqlen in our case
    i = tl.program_id(axis=0)
    while i < HALF:
        # Compute bit-reverse of i in [0, HALF)
        rev = tl.zeros((), dtype=tl.int32)
        j = tl.zeros((), dtype=tl.int32)
        # 16-bit bit-reversal (covers HALF up to 65535)
        while j < 16:
            b = (i >> (15 - j)) & 1
            rev ^= b << j
            j += 1
        # Swap real and imag at i and rev
        tmp_real_i = tl.load(t_ptr + 2 * i)
        tmp_imag_i = tl.load(t_ptr + 2 * i + 1)
        tmp_real_rev = tl.load(t_ptr + 2 * rev)
        tmp_imag_rev = tl.load(t_ptr + 2 * rev + 1)
        tl.store(t_ptr + 2 * i, tmp_real_rev)
        tl.store(t_ptr + 2 * i + 1, tmp_imag_rev)
        tl.store(t_ptr + 2 * rev, tmp_real_i)
        tl.store(t_ptr + 2 * rev + 1, tmp_imag_i)
        i += 1


@triton.jit
def real_fft_stages_kernel(t_ptr, HALF: tl.constexpr, N: tl.constexpr, cos_ptr, sin_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Perform one stage of Cooley-Tukey complex FFT on interleaved t_real/t_imag of length 2*N.
    For stage with k given by cos_ptr[s], sin_ptr[s], process all pairs (i, j=i^k) in [0, HALF).
    Updates:
      u = t_real[i] + i*t_imag[i]
      v = t_real[j] + i*t_imag[j]
      alpha = u*cos + v*sin
      beta  = -u*sin + v*cos
      t_real[i] = alpha, t_imag[i] = beta
      t_real[j] = alpha, t_imag[j] = beta
    """
    i = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i < HALF
    # Load cos/sin scalars for this stage
    # cos_ptr and sin_ptr are 1D arrays; each stage uses its own indices.
    # We pass indices into cos_ptr/sin_ptr via kernel call (see forward).
    s = tl.load(cos_ptr)  # stage index into precomputed cos/sin arrays
    c = tl.load(cos_ptr + s)
    sn = tl.load(sin_ptr + s)
    # k = value of k for this stage; we compute theta = 2*pi*k/N. Since k is passed via s,
    # and cos/sin arrays are populated with cos(theta), sin(theta), we use c, sn directly.
    # Compute u and v
    u_real = tl.load(t_ptr + 2 * i, mask=mask, other=0.0)
    u_imag = tl.load(t_ptr + 2 * i + 1, mask=mask, other=0.0)
    v_real = tl.load(t_ptr + 2 * (i ^ k), mask=mask, other=0.0)  # k is derived from s via cos/sin indexing
    v_imag = tl.load(t_ptr + 2 * (i ^ k) + 1, mask=mask, other=0.0)

    # theta = 2*pi*k/N; since cos/sin are passed, we can compute alpha/beta directly.
    # Note: Triton does not have a direct pi constant; use 3.141592653589793
    pi = 3.141592653589793
    theta = 2.0 * pi * k / N  # k is implicit via cos_ptr/sin_ptr; here use c, sn
    # alpha = u*cos(theta) + v*sin(theta)
    # beta  = -u*sin(theta) + v*cos(theta)
    # Since cos(theta)=c, sin(theta)=sn, we can use them directly:
    alpha_real = u_real * c + v_real * sn
    alpha_imag = u_imag * c + v_imag * sn
    beta_real = -u_real * sn + v_real * c
    beta_imag = -u_imag * sn + v_imag * c

    # Store updated values for i and partner j=i^k
    tl.store(t_ptr + 2 * i, alpha_real, mask=mask)
    tl.store(t_ptr + 2 * i + 1, alpha_imag, mask=mask)
    j = i ^ k
    tl.store(t_ptr + 2 * j, alpha_real, mask=mask)
    tl.store(t_ptr + 2 * j + 1, alpha_imag, mask=mask)


@triton.jit
def divide_inplace_kernel(inp_ptr, out_ptr, n_elements: tl.constexpr, scale, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise division: out[i] = inp[i] / scale for i in [0, n_elements).
    Performs in-place division (out_ptr can be the same as inp_ptr).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    x = x / scale
    tl.store(out_ptr + offsets, x, mask=mask)


def _bitreverse_pairs(t_real_ptr, HALF: int):
    # Launch Triton bit-reverse pairing kernel
    grid = (triton.cdiv(HALF, 1024),)  # grid over i in [0, HALF)
    bitreverse_pairs_kernel[grid](t_real_ptr, HALF=HALF, N=HALF)


def _run_real_fft_stages(t_real_ptr, HALF: int, N: int):
    # Precompute stages and cos/sin arrays. We assume N is power-of-two for standard Cooley-Tukey.
    stages = int(math.log2(N))
    cos = []
    sin = []
    for s in range(stages):
        k = N // (2 << s)  # k = N/2, N/4, ..., 1
        theta = 2.0 * math.pi * k / N
        cos.append(math.cos(theta))
        sin.append(math.sin(theta))
    cos_tensor = torch.tensor(cos, device=t_real_ptr.device, dtype=torch.float32)
    sin_tensor = torch.tensor(sin, device=t_real_ptr.device, dtype=torch.float32)
    # Launch real FFT stages kernels
    BLOCK_SIZE = 1024
    for s in range(stages):
        grid = (triton.cdiv(HALF, BLOCK_SIZE),)
        real_fft_stages_kernel[grid](t_real_ptr, HALF=HALF, N=N, cos_ptr=cos_tensor, sin_ptr=sin_tensor, BLOCK_SIZE=BLOCK_SIZE, k=k)
        # Note: Triton accepts python integers as kernel args; pass k and s via cos_ptr/sin_ptr indexing.
        # For each stage, we re-launch the kernel with the same cos/sin arrays and k.


def _normalize_divide(out_real_ptr, out_imag_ptr, n_elements: int, scale: float):
    BLOCK_SIZE = 1024
    grid_real = (triton.cdiv(n_elements, BLOCK_SIZE),)
    grid_imag = (triton.cdiv(n_elements, BLOCK_SIZE),)
    divide_inplace_kernel[grid_real](out_real_ptr, out_real_ptr, n_elements=n_elements, scale=scale, BLOCK_SIZE=BLOCK_SIZE)
    divide_inplace_kernel[grid_imag](out_imag_ptr, out_imag_ptr, n_elements=n_elements, scale=scale, BLOCK_SIZE=BLOCK_SIZE)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect input of shape (batch, channels, seqlen)
        # Output: (batch, channels, seqlen+1) with real and imaginary parts
        batch, channels, seqlen = x.shape
        device = x.device
        N = 2 * seqlen  # padded length for rfft

        # Ensure contiguous float32
        x_f32 = x.to(torch.float32).contiguous()

        # Allocate time-domain buffers: real and imag interleaved of length 2*N
        t_real = torch.zeros(N, device=device, dtype=torch.float32)
        t_imag = torch.zeros(N, device=device, dtype=torch.float32)

        # Copy input into t_real[0:seqlen]
        t_real[:seqlen] = x_f32.view(-1)  # flatten across batch and channels implicitly via linear indexing

        # 1) Bit-reverse pairing
        HALF = seqlen  # N//2 == seqlen
        _bitreverse_pairs(t_real, HALF)

        # 2) Run Cooley-Tukey stages to compute complex FFT
        # Note: We operate on t_real/t_imag as complex interleaved signal.
        _run_real_fft_stages(t_real, HALF, N)

        # 3) Extract first HALF bins: out_real[i] = t_real[2*i], out_imag[i] = t_imag[2*i]
        out_real = t_real[0:2 * HALF:2]
        out_imag = t_imag[0:2 * HALF:2]

        # 4) Normalize by N = 2*seqlen
        scale = float(N)
        _normalize_divide(out_real, out_imag, n_elements=HALF, scale=scale)

        # 5) Reshape to (batch, channels, seqlen+1)
        out_real = out_real.view(batch, channels, seqlen + 1)
        out_imag = out_imag.view(batch, channels, seqlen + 1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
