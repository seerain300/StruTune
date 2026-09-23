import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute rfft for real input per (b, c), return real and imag parts in out_real/out_imag.
# Input x: shape (B*C, S) where we pass x.view(-1, S). Output real/imag: shape (B*C, S+1).
if TRITON_AVAILABLE:
    @triton.jit
    def rfft_real_imag_kernel(
        x_ptr,                 # *float32, input x flattened as (B*C, S)
        out_real_ptr,          # *float32, output real part flattened as (B*C, S+1)
        out_imag_ptr,          # *float32, output imag part flattened as (B*C, S+1)
        S: tl.int32,           # seqlen
        TWO_S: tl.int32,       # 2 * S (for normalization)
    ):
        bc = tl.program_id(0)  # one program per (b,c)
        # Base pointers for this (b,c)
        x_base = bc * S
        out_base = bc * (S + 1)

        # Precompute constants
        # We'll compute cos/sin for each k in 0..S
        # Triton allows loops with runtime bounds; S is a scalar here.

        # We'll use a simple nested loop: for k in 0..S, accumulate sums over t in 0..S-1
        # Note: Triton supports scalar while loops.
        k = 0
        while k <= S:
            # Accumulators
            sum_real = tl.zeros((), dtype=tl.float32)
            sum_imag = tl.zeros((), dtype=tl.float32)

            # Sum over t = 0..S-1
            t = 0
            while t < S:
                # Load x[bc, t]
                val = tl.load(x_ptr + x_base + t)
                # Compute angle = 2*pi*k*t / (2*S) = pi*k*t / S
                angle = 3.141592653589793 * k * t / (0.5 * TWO_S)  # 2*pi/(2*S) * k*t
                c = tl.cos(angle)
                s = tl.sin(angle)
                sum_real += val * c
                sum_imag += -val * s
                t += 1

            # Normalize by 2*S
            inv_two_s = 1.0 / TWO_S
            out_real = sum_real * inv_two_s
            out_imag = sum_imag * inv_two_s

            # Store to output at index k
            tl.store(out_real_ptr + out_base + k, out_real)
            tl.store(out_imag_ptr + out_base + k, out_imag)

            k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute rfft along the last dim with n=2*S, return real and imag parts of shape (B, C, S+1).
        All computation is done in Triton kernels; no torch FFT is used.
        """
        assert x.dim() == 3, "Input must be of shape (B, C, S)"
        B, C, S = x.shape
        # Ensure float32 for numerical stability (original code casts to float32)
        x_f32 = x.to(torch.float32)
        # Flatten to (B*C, S) for Triton kernel
        x_flat = x_f32.reshape(B * C, S)

        # Allocate outputs (B*C, S+1)
        out_real = torch.empty((B * C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c)
        grid = (B * C,)
        rfft_real_imag_kernel[grid](x_flat, out_real, out_imag, S, 2 * S)

        # Reshape back to (B, C, S+1)
        out_real = out_real.reshape(B, C, S + 1)
        out_imag = out_imag.reshape(B, C, S + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
