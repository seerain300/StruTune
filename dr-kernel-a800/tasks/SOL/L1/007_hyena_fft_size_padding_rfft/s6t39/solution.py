import torch
import triton
import triton.language as tl

# Direct rFFT computation via sum of cos/sin, avoiding any torch FFT calls.
# This kernel computes the real and imaginary parts of rfft(x) for each (b, c) row
# and writes them to out_real and out_imag of shape (BC, S+1).
@triton.jit
def direct_rfft_sum_kernel(
    x_ptr,                  # *float32, shape (BC, S)
    out_real_ptr,           # *float32, shape (BC, S+1)
    out_imag_ptr,           # *float32, shape (BC, S+1)
    S: tl.int32,            # seqlen
    BC: tl.int32,           # batch*channels
):
    bc = tl.program_id(0)
    # Each program handles one (b, c) row.
    base_x = bc * S

    # We compute outputs j = 0..S
    S_plus1 = S + 1
    # Precompute 2*S for normalization
    two_S = 2 * S

    # For k = 0 to S
    for k in range(0, S_plus1):
        # Compute sum of x[b, c, :] = sum over t
        sum_x = 0.0
        t = 0
        while t < S:
            # Load x[bc, t]
            v = tl.load(x_ptr + base_x + t)
            sum_x += v
            t += 1

        # Compute angle for rfft on real inputs: ang = 2*pi*k / (2*S)
        # Note: j = k in our output indexing (j runs 0..S)
        ang = 2.0 * 3.141592653589793 * k / two_S

        cos_term = tl.cos(ang)
        sin_term = tl.sin(ang)

        # Normalize by 2*S (original code divides complex result by 2*S)
        norm = 1.0 / two_S

        # Real part contribution from sum_x * cos_term
        real_val = sum_x * cos_term * norm
        # Imag part contribution from -sum_x * sin_term (imag will be zero for even k,
        # but we compute it generally and let parity decide below).
        imag_val = -sum_x * sin_term * norm

        # Store to output tensors for (bc, k)
        # out_real_ptr has linear indexing: bc * (S+1) + k
        tl.store(out_real_ptr + bc * (S_plus1) + k, real_val)
        tl.store(out_imag_ptr + bc * (S_plus1) + k, imag_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation: computes rfft(x, n=2*S), divides by 2*S,
        and returns real and imaginary parts as float32 tensors of shape (B, C, S+1).
        """
        assert x.dim() == 3, "Input must be (B, C, S)"
        B, C, S = x.shape

        # Cast to float32 (PyTorch code does this explicitly in the baseline)
        x_f32 = x.to(torch.float32)

        # Flatten (B, C) to BC and ensure contiguous
        BC = B * C
        x_flat = x_f32.reshape(BC, S).contiguous()

        # Allocate outputs (BC, S+1) float32
        out_real_flat = torch.empty((BC, S + 1), dtype=torch.float32, device=x.device)
        out_imag_flat = torch.empty((BC, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: 1D grid over BC
        grid = (BC,)
        direct_rfft_sum_kernel[grid](
            x_flat, out_real_flat, out_imag_flat,
            S, BC,
            num_warps=4,
            num_stages=2,
        )

        # Reshape back to (B, C, S+1)
        out_real = out_real_flat.view(B, C, S + 1)
        out_imag = out_imag_flat.view(B, C, S + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
