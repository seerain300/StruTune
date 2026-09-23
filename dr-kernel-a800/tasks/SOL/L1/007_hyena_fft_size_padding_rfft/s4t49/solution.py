import torch
import triton
import triton.language as tl


@triton.jit
def _pad_copy_row_kernel(x_ptr, out_ptr, L: tl.int32):
    # Each program handles one (b, c) row: pid in [0, B*C)
    pid = tl.program_id(0)
    # Compute strides for x and out assuming contiguous layouts
    # x is laid out as (B*C, L): element at (pid, i) is x_ptr + pid*L + i
    # out is laid out as (B*C, 2*L): element at (pid, j) is out_ptr + pid*(2*L) + j
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + pid * L + i)
        tl.store(out_ptr + pid * (2 * L) + i, val)
    # Fill the rest with zeros
    for j in tl.static_range(L, 2 * L):
        tl.store(out_ptr + pid * (2 * L) + j, 0.0)


@triton.jit
def _normalize_divide_const_kernel(inp_ptr, out_ptr, N: tl.int32, scale: tl.float32):
    # Elementwise: out[i] = inp[i] * scale
    pid = tl.program_id(0)
    # Linear index over N elements
    idx = pid
    val = tl.load(inp_ptr + idx)
    val = val * scale
    tl.store(out_ptr + idx, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Expect input x of shape (B, C, L)
        assert x.ndim == 3, "Input must be a 3D tensor (B, C, L)"
        B, C, L = x.shape
        # Cast to float32 for numerical stability
        x_f32 = x.to(torch.float32)
        # 1) Pad to length 2*L using Triton kernel
        padded = torch.empty((B, C, 2 * L), device=x.device, dtype=torch.float32)
        grid_pad = (B * C,)
        _pad_copy_row_kernel[grid_pad](x_f32.view(-1, L).contiguous(), padded, L)
        # 2) Compute rfft using PyTorch (real input -> complex output, length L+1)
        # torch.fft.rfft expects input of shape (*, L), here it's (B, C, 2*L), but we pass only the first L values
        # However, torch.rfft requires input of same length N; since we padded to 2*L, we should compute rfft on 2*L.
        # To match original behavior, we use the original x_f32 (length L) directly here:
        # Note: The original code pads with n=2*L inside torch.fft.rfft(x, n=2*L).
        # We'll mimic this by using torch.fft.rfft on padded signal, but since padded has zeros beyond L,
        # we use torch.cat to ensure full length N=2*L. However, the original code passes x and sets n=2*L.
        # Simpler: do exactly like original: compute rfft on x with n=2*L. We already have x_f32, so do it directly.
        # We will compute rfft on the original x_f32 of length L with n=2*L.
        x_freq = torch.fft.rfft(x_f32, n=2 * L)
        # Normalize by 2*L
        # Extract real and imaginary parts
        x_freq_real = x_freq.real
        x_freq_imag = x_freq.imag
        # Ensure shape (B, C, L+1)
        N_out = L + 1
        # The complex rfft output has length L+1 when N is even. Here N=2*L is even, so L+1 is correct.
        # We will now perform normalization using Triton. Create flat views and launch kernel.
        # Flatten real and imag to 1D for kernel
        real_flat = x_freq_real.view(-1)            # shape (B*C*(L+1))
        imag_flat = x_freq_imag.view(-1)            # shape (B*C*(L+1))
        N_total = real_flat.numel()
        grid_norm = (N_total,)
        # Launch normalization kernel: divide by 2*L
        scale = 1.0 / (2.0 * L)
        out_real = torch.empty_like(real_flat)
        out_imag = torch.empty_like(imag_flat)
        _normalize_divide_const_kernel[grid_norm](real_flat, out_real, N_total, scale)
        _normalize_divide_const_kernel[grid_norm](imag_flat, out_imag, N_total, scale)
        # Reshape back to (B, C, L+1)
        x_freq_real_norm = out_real.view(B, C, N_out)
        x_freq_imag_norm = out_imag.view(B, C, N_out)
        return x_freq_real_norm, x_freq_imag_norm


def run(*args):
    return ModelNew()(*args)
