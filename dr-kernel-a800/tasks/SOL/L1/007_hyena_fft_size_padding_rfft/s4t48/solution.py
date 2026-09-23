import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_scalar_kernel(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    # One program per (b, c) slice: copy x[bc, :L] into out[bc, :N], where out is zeroed
    pid_bc = tl.program_id(0)
    # Scalar loop over input elements
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + pid_bc * L + i)
        tl.store(out_ptr + pid_bc * N + i, val)


@triton.jit
def _rfft_direct_sum_kernel(padded_ptr, out_real_ptr, out_imag_ptr, L: tl.int32, N: tl.int32):
    # Compute rfft coefficients for k in [0..L] via direct summation:
    # y_real[k] = (1/(2*N)) * sum_{j=0}^{N-1} padded[j] * cos(2*pi*k*j/N)
    # y_imag[k] = (1/(2*N)) * sum_{j=0}^{N-1} padded[j] * sin(2*pi*k*j/N)
    # Normalize by 2*N in-kernel.
    pid_bc = tl.program_id(0)
    # Precompute inv_2N
    inv_2N = 1.0 / (2.0 * N)

    # Loop over k from 0 to L
    for k in tl.static_range(0, L + 1):
        acc_real = 0.0
        acc_imag = 0.0
        # Sum over j in chunks of BLOCK_J (use scalar loop for safety across Triton versions)
        BLOCK_J = 128
        for j0 in tl.static_range(0, N, BLOCK_J):
            for jj in tl.static_range(0, BLOCK_J):
                j = j0 + jj
                # Safe bounds check
                if j < N:
                    val = tl.load(padded_ptr + pid_bc * N + j)
                    angle = (2.0 * 3.141592653589793) * (k * j) / N
                    acc_real += val * tl.cos(angle)
                    acc_imag += val * tl.sin(angle)
        # Normalize
        acc_real = acc_real * inv_2N
        acc_imag = acc_imag * inv_2N
        # Store results at index k
        tl.store(out_real_ptr + pid_bc * (L + 1) + k, acc_real)
        tl.store(out_imag_ptr + pid_bc * (L + 1) + k, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        assert x.ndim == 3, "Input must be 3D (B, C, L)"
        B, C, L = x.shape
        N = 2 * L  # zero-padding to 2*L as in original

        # Ensure float32
        x_f32 = x.to(torch.float32)

        # Allocate zero-padded buffer (B, C, N)
        padded = torch.zeros((B, C, N), dtype=torch.float32, device=x.device)

        # Launch Triton copy kernel: one program per (b, c) slice
        grid_copy = (B * C,)
        _copy_row_to_padded_scalar_kernel[grid_copy](x_f32.view(B * C, L), padded.view(B * C, N), L, N)

        # Allocate outputs (B, C, L+1) for real and imaginary parts
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton direct rfft sum kernel: one program per (b, c)
        grid_rfft = (B * C,)
        _rfft_direct_sum_kernel[grid_rfft](padded.view(B * C, N), out_real.view(B * C, L + 1), out_imag.view(B * C, L + 1), L, N)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
