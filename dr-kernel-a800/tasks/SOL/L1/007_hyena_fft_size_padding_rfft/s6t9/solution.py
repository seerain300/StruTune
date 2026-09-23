import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(
    x_ptr,                  # *float32, input x flattened as (BC, S)
    out_real_ptr,           # *float32, output real part flattened as (BC, S+1)
    out_imag_ptr,           # *float32, output imag part flattened as (BC, S+1)
    BC: tl.int32,           # number of (batch, channel) slices
    S: tl.int32,            # seqlen
    BLOCK_T: tl.constexpr,  # compile-time block for t loop, e.g., 2048 or 4096
):
    bc = tl.program_id(0)
    # Guard: if bc >= BC, return (in case grid > BC, though here grid=BC)
    if bc >= BC:
        return

    # Base offsets
    base_x = bc * S
    out_base = bc * (S + 1)

    # Precompute constants
    N = 2 * S
    inv_2N = 1.0 / (2.0 * S)  # normalization factor per original code

    # We will compute y[j] for j in 0..S-1
    # For each j, we accumulate contributions from t=0..2*S-1 with masks t<S (x part),
    # and then handle special cases:
    # - j == 0: sum(x), real only
    # - j even > 0: real part = (sum(x) * cos(pi*j/N) - sum(x) * sin(pi*j/N)) / (2*N), imag = 0
    # - j odd > 0: real = 0, imag = -sum(x) * sin(pi*j/N) / (2*N)

    # Helper: compute sum_x
    sum_x = 0.0
    # Accumulate sum(x) over t < S
    for t in tl.static_range(0, BLOCK_T):
        m_t = t < S
        # When m_t is False, tl.load(other=0.0) will return 0.0
        x_t = tl.load(x_ptr + base_x + t, mask=m_t, other=0.0)
        sum_x += x_t

    # Now compute outputs for j = 0..S-1
    for j in tl.static_range(0, BLOCK_T):
        m_j = j < S
        if m_j:
            # Compute cos and sin factors
            # Note: j is integer, N is integer (2*S), so ang = pi*j/N is well-defined.
            ang = 3.141592653589793 * j / N
            c = tl.cos(ang)
            s = tl.sin(ang)

            # For j == 0: real = sum_x, imag = 0
            y_real_j = sum_x * inv_2N if j == 0 else 0.0
            y_imag_j = 0.0 if j == 0 else 0.0  # initialized; we will set below

            # For j > 0:
            # even: real = (sum_x * c - sum_x * s) * inv_2N, imag = 0
            # odd:  real = 0, imag = -sum_x * s * inv_2N
            if j > 0:
                if (j % 2) == 0:
                    y_real_j = (sum_x * c - sum_x * s) * inv_2N
                else:
                    y_imag_j = -sum_x * s * inv_2N

            # Store results
            tl.store(out_real_ptr + out_base + j, y_real_j)
            tl.store(out_imag_ptr + out_base + j, y_imag_j)

    # Optional: initialize out_real/out_imag for j >= S (not used, but be safe)
    for j in tl.static_range(S, BLOCK_T):
        tl.store(out_real_ptr + out_base + j, 0.0)
        tl.store(out_imag_ptr + out_base + j, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of the original run function:
        - Input: x of shape (B, C, S), float32, CUDA
        - Output: out_real, out_imag of shape (B, C, S+1), float32
        """
        assert x.is_cuda, "ModelNew requires CUDA input tensor for Triton kernels."
        assert x.dtype == torch.float32, "Input dtype must be float32."

        B, C, S = x.shape
        BC = B * C

        # Allocate outputs
        out_real = torch.empty((B, C, S + 1), device=x.device, dtype=torch.float32)
        out_imag = torch.empty((B, C, S + 1), device=x.device, dtype=torch.float32)

        # Flatten views for Triton (contiguous)
        x_flat = x.reshape(BC, S)
        out_real_flat = out_real.reshape(BC, S + 1)
        out_imag_flat = out_imag.reshape(BC, S + 1)

        # Launch Triton kernel: one program per (b, c)
        grid = (BC,)
        # Choose BLOCK_T as a compile-time constant large enough to cover N=2*S in loops.
        # Using 4096 works for common sizes; Triton will compile specialized versions.
        rfft_real_kernel[grid](
            x_flat, out_real_flat, out_imag_flat,
            BC, S,
            BLOCK_T=4096,
            num_warps=4,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
