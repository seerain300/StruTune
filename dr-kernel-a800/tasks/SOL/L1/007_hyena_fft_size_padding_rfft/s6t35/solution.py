import torch
import triton
import triton.language as tl


@triton.jit
def real_rfft_triton_kernel(
    x_ptr,                # *float32, input x flattened as (B*C*S)
    out_real_ptr,         # *float32, output real part flattened as (B*C*(S+1))
    out_imag_ptr,         # *float32, output imag part flattened as (B*C*(S+1))
    B: tl.int32,          # batch size (not used in indexing but kept for potential future use)
    C: tl.int32,          # channels
    S: tl.constexpr,      # sequence length (compile-time constant for Triton loops)
):
    bc = tl.program_id(0)
    base_x = bc * S

    # Load x as a vector [0..S-1]
    # We'll use loops below to compute sums; here we load x into registers if needed.
    # For each j in 0..S, compute y[j] for real input and store real/imag parts.
    inv_2N = 1.0 / (2.0 * S)   # overall scaling due to original division by 2*S
    inv_N = 1.0 / S

    # j = 0 term: y[0] = sum(x) / (2*S) and purely real (imag=0)
    sum_x = 0.0
    k = 0
    while k < S:
        sum_x += tl.load(x_ptr + base_x + k)
        k += 1
    y0_real = sum_x * inv_2N

    # Store y[0]
    out_bc_offset = bc * (S + 1)
    tl.store(out_real_ptr + out_bc_offset + 0, y0_real)
    tl.store(out_imag_ptr + out_bc_offset + 0, 0.0)

    # For j in 1..S
    j = 1
    while j <= S:
        # Compute sum_cos = sum_k x[k] * cos(2*pi*k*j/(2*S))
        # Compute sum_sin = sum_k x[k] * sin(2*pi*k*j/(2*S))
        sum_cos = 0.0
        sum_sin = 0.0

        k = 0
        while k < S:
            xk = tl.load(x_ptr + base_x + k)
            angle = 2.0 * 3.141592653589793 * k * j / (2.0 * S)
            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)
            sum_cos += xk * cos_term
            sum_sin += xk * sin_term
            k += 1

        if (j % 2) == 0:
            # j is even: y[j] = (sum_cos - sum_sin) / (2*N), imag = 0
            yj_real = (sum_cos - sum_sin) * inv_2N
            tl.store(out_real_ptr + out_bc_offset + j, yj_real)
            tl.store(out_imag_ptr + out_bc_offset + j, 0.0)
        else:
            # j is odd: y[j] real = 0, imag = -sum_sin / N
            yj_imag = -sum_sin * inv_N
            tl.store(out_real_ptr + out_bc_offset + j, 0.0)
            tl.store(out_imag_ptr + out_bc_offset + j, yj_imag)

        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute rfft for real input x along last dim, normalize by 2*S,
        and return real and imaginary parts as two float32 tensors of shape (B, C, S+1).
        All computation is performed via Triton kernels (no torch FFT calls).
        """
        assert x.dim() == 3, "Input must be (B, C, S)"
        B, C, S = x.shape
        # Ensure float32 as in original code
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Flatten input to 1D for simple indexing in Triton
        x_flat = x.contiguous().view(-1)  # length = B*C*S

        # Allocate outputs
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) pair
        grid = (B * C,)
        real_rfft_triton_kernel[grid](
            x_flat,
            out_real.view(-1),  # flatten output to (B*C*(S+1)) for Triton
            out_imag.view(-1),  # flatten output to (B*C*(S+1)) for Triton
            B, C, S,  # pass B, C, and make S constexpr for kernel
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
