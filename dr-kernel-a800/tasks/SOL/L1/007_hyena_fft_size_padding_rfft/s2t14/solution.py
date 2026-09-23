import torch
import triton
import triton.language as tl

# Triton kernel: per (b, c) slice, copy x[bc, :] (length L) into out[bc, :L]
# Assumes destination buffer 'out' is pre-zeroed on host (torch.zeros).
@triton.jit
def pad_kernel(x_ptr, out_ptr, stride_x, L):
    bc = tl.program_id(0)  # one program per (b, c)
    # Copy first L elements from x_ptr[bc, :] to out_ptr[bc, :]
    for i in range(0, L):
        tl.store(out_ptr + bc * stride_x + i, tl.load(x_ptr + bc * stride_x + i))

# Triton kernel: compute rFFT for a single (b, c) slice and all k in [0..L]
# Grid is 2D: (B*C, L+1). Each program handles one (bc, k).
# Input: in_ptr points to a (B*C, 2*L) vector (padded real input).
# Output: real_ptr and imag_ptr point to (B*C, L+1) buffers; we store normalized real and zero imag.
@triton.jit
def rfft_scalar_kernel(in_ptr, real_ptr, imag_ptr, two_L, L, bc, k):
    # Accumulate real and imaginary parts
    sum_real = 0.0
    sum_imag = 0.0
    # Loop over t from 0 to 2*L - 1
    for t in range(0, two_L):
        x_t = tl.load(in_ptr + bc * two_L + t)
        angle = 2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
        ct = tl.cos(angle)
        st = tl.sin(angle)
        sum_real += x_t * ct
        sum_imag += x_t * st
    # Normalize by 2*L
    norm = 1.0 / float(two_L)
    tl.store(real_ptr + bc * (L + 1) + k, sum_real * norm)
    # Imaginary part is zero for real inputs
    tl.store(imag_ptr + bc * (L + 1) + k, 0.0)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
          - Input: x of shape (B, C, L), float32
          - Compute rfft along last dim with n=2*L, normalize by 2*L
          - Return real and imaginary parts as (B, C, L+1), float32
        """
        # Ensure float32 and contiguous
        x = x.to(torch.float32).contiguous()
        B, C, L = x.shape
        two_L = 2 * L
        out_len = L + 1

        # Prepare padded input per (b, c): (B*C, 2*L) pre-zeroed
        x_bc = x.view(B * C, L).contiguous()  # shape (B*C, L)
        in_buf = torch.zeros((B * C, two_L), dtype=torch.float32, device=x.device)

        # Launch pad_kernel: one program per (b, c), copy x[bc, :] into in_buf[bc, :L]
        grid_pad = (B * C,)
        pad_kernel[grid_pad](x_bc, in_buf, L, L)

        # Allocate outputs: real and imag, shapes (B*C, L+1)
        real_out = torch.empty((B * C, out_len), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B * C, out_len), dtype=torch.float32, device=x.device)

        # Compute rFFT for k in [0..L]: launch 2D grid (B*C, L+1)
        rfft_scalar_kernel[(B * C, L + 1)](
            in_buf, real_out, imag_out,
            two_L, L
        )

        # Reshape to (B, C, L+1) and return
        real_out = real_out.view(B, C, out_len)
        imag_out = imag_out.view(B, C, out_len)
        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
