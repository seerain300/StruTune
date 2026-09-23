import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(
    x_ptr,                       # *float32, input x of shape (B, C, N=2*S), contiguous
    out_real_ptr, out_imag_ptr,  # *float32, outputs of shape (B, C, S+1)
    N: tl.int32, S: tl.int32,   # N=2*S, S
    stride_x_bc: tl.int32,      # stride between bc in x (elements) = N for contiguous
    out_stride: tl.int32,       # stride between bc in outputs (elements) = S+1
    J_MAX: tl.constexpr         # maximum j to compute; we set J_MAX=S
):
    # Each program handles one (b, c)
    bc = tl.program_id(0)
    base_x = bc * stride_x_bc
    base_out = bc * out_stride

    # Precompute inv_scale = 1 / (N * 2*S) to match original normalization
    scale = 2 * S
    inv_scale = 1.0 / (N * scale)

    # Loop over j = 0..S-1
    for j in range(0, J_MAX):
        # Accumulate real and imag parts using discrete-time formula
        # y[j] = (1/(N*2*S)) * sum_t [ x[t] * cos(pi*j*t/N) + i * (-x[t] * sin(pi*j*t/N)) ]
        sum_real = 0.0
        sum_imag = 0.0
        for t in range(0, N):
            v = tl.load(x_ptr + base_x + t)
            ang = (j * t) * 3.141592653589793 / (2.0 * S)  # N = 2*S
            c = tl.cos(ang)
            s = tl.sin(ang)
            sum_real += v * c
            sum_imag += -v * s

        # Apply normalization
        real_y = sum_real * inv_scale
        imag_y = sum_imag * inv_scale

        # Store to outputs at (b, c, j)
        tl.store(out_real_ptr + base_out + j, real_y)
        tl.store(out_imag_ptr + base_out + j, imag_y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # x: (B, C, S), float32, on CUDA
        assert x.is_cuda, "ModelNew.forward expects a CUDA tensor"
        assert x.dtype == torch.float32, "ModelNew expects float32 input"
        x = x.contiguous()
        B, C, S = x.shape
        N = 2 * S

        # Allocate outputs (B, C, S+1)
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Strides: x is (B, C, N) contiguous => stride between bc = N
        stride_x_bc = N
        out_stride = S + 1  # elements per (b,c) row in output

        # Launch Triton kernel: 1 program per (b, c)
        grid = (B * C,)

        rfft_real_kernel[grid](
            x,
            out_real, out_imag,
            N, S,
            stride_x_bc,
            out_stride,
            J_MAX=S,  # compute up to j=S-1
            num_warps=1,
        )

        # The original code divides the complex rfft output by 2*S; we applied that scaling inside the kernel.
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
