import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_imag_analytical_kernel(
    x_ptr,          # *float32, input x linearized as (B*C, S)
    yr_ptr, yim_ptr,# *float32, outputs real/imag linearized as (B*C, S+1)
    S: tl.int32,
):
    bc = tl.program_id(0)
    base_x = bc * S

    # Accumulate sum over x[0..S-1]
    sum_x = 0.0
    j = 0
    while j < S:
        v = tl.load(x_ptr + base_x + j)
        sum_x += v
        j += 1

    # Precompute scale and N
    scale = 1.0 / (2.0 * S)
    N = 2 * S

    # Compute y_real and y_imag for k=0..S-1
    k = 0
    while k < S:
        angle = 3.141592653589793 * k / N
        c = tl.cos(angle)
        s = tl.sin(angle)

        if (k & 1) == 0:
            y_real = (sum_x * c - sum_x * s) * scale
            y_imag = 0.0
        else:
            y_real = 0.0
            y_imag = -(sum_x * s) * scale

        tl.store(yr_ptr + bc * (S + 1) + k, y_real)
        tl.store(yim_ptr + bc * (S + 1) + k, y_imag)
        k += 1

    # Nyquist component for real input when N is even: y[S] = -sum_x / N
    y_real_nyq = (-sum_x) / N
    tl.store(yr_ptr + bc * (S + 1) + S, y_real_nyq)
    tl.store(yim_ptr + bc * (S + 1) + S, 0.0)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x shape: (B, C, S), float32
        B, C, S = x.shape

        # Flatten to (B*C, S) for Triton kernel
        x_flat = x.reshape(B * C, S).contiguous()

        # Allocate outputs linearized as (B*C, S+1)
        yr_lin = torch.empty((B * C, S + 1), dtype=torch.float32, device=x.device)
        yim_lin = torch.empty((B * C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b,c)
        rfft_real_imag_analytical_kernel[(B * C,)](
            x_flat, yr_lin, yim_lin, S,
            num_warps=4, num_stages=2
        )

        # Reshape to (B, C, S+1)
        y_real = yr_lin.view(B, C, S + 1)
        y_imag = yim_lin.view(B, C, S + 1)

        return y_real, y_imag


def run(*args):
    return ModelNew()(*args)
