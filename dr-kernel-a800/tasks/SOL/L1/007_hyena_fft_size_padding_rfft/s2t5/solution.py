import torch
import triton
import triton.language as tl


@triton.jit
def pad_kernel(
    x_ptr,             # *const float, input x of shape (B, C, L), contiguous
    padded_ptr,        # *float, output flattened padded vector length P = 2*L
    B: tl.constexpr,   # batch size
    C: tl.constexpr,   # channels
    L,                 # actual seqlen (int)
    P: tl.constexpr,   # padded length = 2*L (int)
):
    # One program per (b, c)
    pid = tl.program_id(0)
    c = pid % C
    b = pid // C
    bc = b * C + c

    # Base linear offset in padded_ptr for this (b, c) slice
    offset = bc * P

    # Load x[b, c, :] into padded_ptr[offset : offset + L]
    x_bc_ptr = x_ptr + bc * L  # pointer to start of this (b,c) slice
    for t in range(0, L):
        val = tl.load(x_bc_ptr + t)
        tl.store(padded_ptr + offset + t, val)

    # Zero-pad the remaining P - L positions
    for t in range(L, P):
        tl.store(padded_ptr + offset + t, 0.0)


@triton.jit
def real_dft_kernel(
    padded_ptr,         # *const float, padded input vector of length P = 2*L
    out_real_ptr,       # *float, output real part vector of length (B*C)*(L+1)
    out_imag_ptr,       # *float, output imag part vector of length (B*C)*(L+1)
    B: tl.constexpr,
    C: tl.constexpr,
    L,                  # actual seqlen (int)
    P: tl.constexpr,    # padded length = 2*L (int)
    inv_P,              # float32, 1.0 / P
):
    # Grid over (B, C, L+1)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    k = tl.program_id(2)  # frequency index in [0, L]

    bc = pid_b * C + pid_c
    base = bc * (L + 1)

    # Accumulator for real part
    acc = 0.0

    # Compute DFT: X[k] = sum_{t=0}^{P-1} x[t] * exp(-2πi k t / P)
    # For real input, rfft produces real outputs; we compute real part explicitly.
    for t in range(0, P):
        x_t = tl.load(padded_ptr + t)
        theta = -2.0 * 3.141592653589793 * float(k) * float(t) * inv_P
        acc += x_t * tl.cos(theta)

    # Normalize by P
    acc *= inv_P

    # Store real and imaginary parts
    tl.store(out_real_ptr + base + k, acc)
    tl.store(out_imag_ptr + base + k, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-optimized version:
        - Pads seqlen to 2*L and computes real DFT via Triton kernels.
        - Returns real and imaginary parts of shape (B, C, L+1), normalized by 2*L.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert x.dtype == torch.float32, "Input must be float32."
        x = x.contiguous()
        B, C, L = x.shape
        P = 2 * L  # padded length

        # 1) Build zero-padded vectors for all (b, c) in a single flattened buffer of length M = (B*C)*P
        M = (B * C) * P
        padded = torch.empty(M, dtype=torch.float32, device=x.device)

        # Launch pad kernel: one program per (b, c)
        grid_pad = (B * C,)
        pad_kernel[grid_pad](
            x, padded,
            B, C, L, P,
            num_warps=4,
        )

        # 2) Compute real DFT for k in [0, L], normalized by P
        out_real = torch.empty(B * C * (L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty(B * C * (L + 1), dtype=torch.float32, device=x.device)

        inv_P = 1.0 / float(P)
        grid_dft = (B, C, L + 1)
        real_dft_kernel[grid_dft](
            padded, out_real, out_imag,
            B, C, L, P, inv_P,
            num_warps=4,
        )

        # Reshape back to (B, C, L+1)
        x_freq_real = out_real.view(B, C, L + 1)
        x_freq_imag = out_imag.view(B, C, L + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
