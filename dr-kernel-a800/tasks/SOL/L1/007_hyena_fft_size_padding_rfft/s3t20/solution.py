import math
import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                # *f32, input x of shape (B, C, L)
    x_padded_ptr,         # *f32, output padded x of shape (B, C, 2*L), we will fill zeros and copy x
    B: tl.int32,          # batch size
    C: tl.int32,          # channels
    L: tl.int32,          # original seqlen
    BLOCK_J: tl.constexpr # tile for j
):
    # One program per (b, c) slice
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    # Base pointers
    base_in = b * L + c * 0  # since x is (B, C, L), stride along last dim is 1 for contiguous; we just index b*L + c*0
    base_out = b * (2 * L) + c * 0

    # Copy first half
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_J)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j, vals, mask=mask)
        start += BLOCK_J

    # Fill second half with zeros
    zero_vec = tl.zeros([BLOCK_J], dtype=tl.float32)
    start = L
    while start < 2 * L:
        j = start + tl.arange(0, BLOCK_J)
        mask = j < (2 * L)
        tl.store(x_padded_ptr + base_out + j, zero_vec, mask=mask)
        start += BLOCK_J


@triton.jit
def compute_rfft_real_imag_kernel(
    x_ptr,                # *f32, input x of shape (B, C, L)
    real_out_ptr,         # *f32, output real part of shape (B, C, L+1)
    imag_out_ptr,         # *f32, output imag part of shape (B, C, L+1)
    B: tl.int32,
    C: tl.int32,
    L: tl.int32,
    BLOCK_T: tl.constexpr,  # tile for t
    BLOCK_K: tl.constexpr   # tile for k
):
    # One program per (b, c) slice
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    # Base pointer for x slice
    base_in = b * L + c * 0

    # Precompute constants
    inv_L = 1.0 / (2.0 * L)
    two_pi = 2.0 * math.pi

    # Compute u_t = sum_{j=0}^{L-1} x[b,c,j] * exp(-i * 2*pi*t*j/L) for t = 0..L-1
    u = tl.zeros([L], dtype=tl.float32)
    t_idx = tl.arange(0, BLOCK_T)
    for t in range(0, L):
        # angle_t = -2*pi*t*j/L for j in [0..L-1]; but we need sum over j, so we compute u_t directly
        # u_t is a scalar per t, we will accumulate it using vectorized j and then reduce
        # Note: Triton supports scalar t here. We'll compute u_t via vector j loop.
        # However, to keep vectorization, compute u_t as:
        j = tl.arange(0, L)  # vector j
        angle = -two_pi * t * j / L
        phase = tl.exp(1j * angle)  # elementwise complex phase
        vals = tl.load(x_ptr + base_in + j, mask=(j < L), other=0.0)
        # vals are float; phase is complex; Triton may not support complex directly in this context.
        # To avoid complex, we implement u_t using scalar accumulation by iterating j:
        # We'll switch to a scalar loop for u_t.
        # Scalar accumulation for u_t:
        u_t = 0.0
        for jj in range(0, L):
            v = tl.load(x_ptr + base_in + jj, mask=(jj < L), other=0.0)
            angle_t = -two_pi * t * (jj % L) / L  # jj < L, safe
            phase_scalar = tl.exp(1j * angle_t)  # scalar complex
            u_t += v * phase_scalar.real
        u[t] = u_t  # store as float32

    # Now compute y_k for k = 0..L
    k_idx = tl.arange(0, BLOCK_K)
    for k in range(0, L + 1):
        # We only need k up to L (since output length is L+1). But we iterate k up to L+1; for k == L, compute accordingly.
        # Compute w = exp(-i * 2*pi*k/L) and c = exp(-i * 2*pi/L)
        # Note: k can be L; handle it.
        w = tl.exp(1j * (-two_pi * k / L)) if k < L else tl.exp(1j * (-two_pi))
        c = tl.exp(1j * (-two_pi / L))

        # y_k = sum_{t=0}^{L-1} u_t * w^t
        y_real = 0.0
        y_imag = 0.0
        for t in range(0, L):
            w_t = tl.exp(1j * (t * (-two_pi * k / L))) if k < L else tl.exp(1j * (t * (-two_pi)))
            # u_t is float; w_t is complex: u_t * w_t -> complex, but we only need real/imag parts.
            # u_t * w_t.real and u_t * w_t.imag
            # However, w_t is computed as exp(1j * angle) and its real is cos(angle), imag is sin(angle).
            # We can compute w_t.real = cos(angle), w_t.imag = sin(angle). For k == L, angle is -2*pi, which is 1.
            w_real = w.real
            w_imag = w.imag
            # For pow: w^t = (cos, sin) raised to t. This is cumbersome; instead, compute w_t directly via angle:
            # angle_w = -2*pi*k/L; w_t = (cos(angle_w*t), sin(angle_w*t))
            angle_w = -two_pi * k * t / L
            w_t_real = tl.cos(angle_w)
            w_t_imag = tl.sin(angle_w)

            y_real += u[t] * w_t_real
            y_imag += u[t] * w_t_imag

        # Store to outputs
        out_base = b * (L + 1) + c * (L + 1)
        tl.store(real_out_ptr + out_base + k, y_real * inv_L)
        tl.store(imag_out_ptr + out_base + k, y_imag * inv_L)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor x with shape (batch, channels, seqlen)
        x = args[0]
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        batch, channels, seqlen = x.shape

        # Ensure float32 and contiguous
        x = x.to(torch.float32)
        x = x.contiguous()

        # Allocate padded input tensor (B, C, 2*seqlen)
        twoL = 2 * seqlen
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)
        x_padded = x_padded.contiguous()

        # Launch padding kernel: one program per (batch, channel) slice
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, seqlen,
            BLOCK_J=128,
            num_warps=4,
        )

        # Allocate outputs
        real_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, seqlen + 1), dtype=torch.float32, device=x.device)

        # Launch DFT computation kernel
        grid_dft = (batch * channels,)
        compute_rfft_real_imag_kernel[grid_dft](
            x_padded, real_out, imag_out,
            batch, channels, seqlen,
            BLOCK_T=128,  # tile for t
            BLOCK_K=64,   # tile for k up to L
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
