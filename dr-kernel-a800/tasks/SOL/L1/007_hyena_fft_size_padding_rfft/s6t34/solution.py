import torch
import triton
import triton.language as tl


@triton.jit
def real_rfft_triton_kernel(
    x_ptr,              # *float32, input x of shape (B*C, S) flattened by bc
    out_real_ptr,       # *float32, output real part of y (B*C*(S+1))
    out_imag_ptr,       # *float32, output imag part of y (B*C*(S+1))
    B: tl.int32,        # batch size
    C: tl.int32,        # channels
    S: tl.constexpr,    # seqlen (compile-time constant for loop bounds)
):
    # One program per (b, c)
    bc = tl.program_id(0)
    b = bc // C
    c = bc % C

    # Compute base offsets
    # Input x is flattened as (B*C, S): bc selects the (b,c) row
    base_x = bc * S

    # Output y is flattened as (B*C, S+1): offset is bc * (S+1)
    out_base = bc * (S + 1)

    # Prepare constants
    N = 2 * S
    inv_two_N = 1.0 / (2.0 * N)   # scale for even j in rfft
    inv_N = 1.0 / N               # scale for odd j imaginary part

    # 1) j = 0: y[0] is real, equals sum(x) / (2*N)
    # Handle empty sum when S == 0; if S == 0, 2*S = 0, which would divide by 0.
    # In typical workloads, S >= 1. We guard by summing k=0..S-1.
    sum_real = 0.0
    for k in range(0, S):
        v = tl.load(x_ptr + base_x + k)
        sum_real += v
    y0 = sum_real * inv_two_N
    tl.store(out_real_ptr + out_base + 0, y0)
    tl.store(out_imag_ptr + out_base + 0, 0.0)

    # 2) j = 1..S: compute real/imag parts
    for j in range(1, S + 1):
        # Compute real and imag contributions:
        # real_sum = sum_k x[k] * cos(2*pi*k*j/N) - sum_k x[k] * sin(2*pi*k*j/N)
        # imag_sum = -sum_k x[k] * sin(2*pi*k*j/N)
        real_sum = 0.0
        imag_sum = 0.0

        for k in range(0, S):
            v = tl.load(x_ptr + base_x + k)
            ang = 2.0 * 3.141592653589793 * (k * j) / N
            cos_t = tl.cos(ang)
            sin_t = tl.sin(ang)
            real_sum += v * cos_t
            imag_sum += v * sin_t

        if (j % 2) == 0:
            # even j: y[j] is real, imag = 0
            yj = real_sum * inv_two_N
            tl.store(out_real_ptr + out_base + j, yj)
            tl.store(out_imag_ptr + out_base + j, 0.0)
        else:
            # odd j: y[j] has real = 0, imag = -imag_sum / N
            imj = -imag_sum * inv_N
            tl.store(out_real_ptr + out_base + j, 0.0)
            tl.store(out_imag_ptr + out_base + j, imj)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute fused rfft along the last dimension (length 2*S), normalize by 2*S,
        and return real and imaginary parts as (B, C, S+1) float tensors, using Triton kernels.
        """
        assert x.ndim == 3, "Input must be 3D: (B, C, S)"
        B, C, S = x.shape

        # Ensure contiguous and float32 as in original code
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Allocate outputs (float32)
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Flatten to (B*C, S+1) for simple pointer arithmetic inside the kernel
        # We launch one program per (b, c)
        grid = (B * C,)

        # Launch Triton kernel
        real_rfft_triton_kernel[grid](
            x,  # x_ptr
            out_real,  # out_real_ptr
            out_imag,  # out_imag_ptr
            B, C,
            S,  # constexpr S
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
