import torch
import triton
import triton.language as tl


@triton.jit
def _cast_to_f32(in_ptr, out_ptr, n_elements: tl.int32):
    pid = tl.program_id(0)
    start = pid * 1024
    offsets = start + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def _copy_row_to_padded(x_rows_ptr, padded_ptr, L: tl.int32, N: tl.int32):
    # Each program handles one (batch, channel) row
    row_idx = tl.program_id(0)
    # Copy x row into padded buffer
    for i in tl.static_range(0, L):
        val = tl.load(x_rows_ptr + row_idx * L + i)
        tl.store(padded_ptr + row_idx * N + i, val)
    # Zero-pad the rest
    for i in tl.static_range(L, N):
        tl.store(padded_ptr + row_idx * N + i, 0.0)


@triton.jit
def _rfft_row(row_idx, N, L, padded_ptr, out_real_ptr, out_imag_ptr):
    # Compute rfft coefficients for k in [0..L-1] and store at positions [k] in output rows
    # Each row has (L+1) outputs
    # base output offset for this row
    base = row_idx * (L + 1)
    for k in tl.static_range(0, L):
        acc_real = 0.0
        acc_imag = 0.0
        for j in tl.static_range(0, N):
            val = tl.load(padded_ptr + row_idx * N + j)
            angle = 2.0 * 3.141592653589793 * k * j / N
            acc_real += val * tl.cos(angle)
            acc_imag += val * tl.sin(angle)
        # Normalize by N (2*L)
        acc_real = acc_real / N
        acc_imag = acc_imag / N
        tl.store(out_real_ptr + base + k, acc_real)
        tl.store(out_imag_ptr + base + k, acc_imag)


@triton.jit
def _divide_by_n(vec_ptr, n: tl.float32, n_elements: tl.int32):
    pid = tl.program_id(0)
    start = pid * 1024
    offsets = start + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(vec_ptr + offsets, mask=mask, other=0.0)
    x = x / n
    tl.store(vec_ptr + offsets, x, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        assert x.dim() == 3, "Input must be of shape (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L

        # Ensure input is float32 using Triton kernel
        # If x already float32, this kernel will just copy
        x_f32 = torch.empty_like(x, dtype=torch.float32, device=x.device)
        n_elements = x.numel()
        grid_cast = (triton.cdiv(n_elements, 1024),)
        _cast_to_f32[grid_cast](x, x_f32, n_elements)

        # Reshape to (BC, L) for processing rows
        x_rows = x_f32.reshape(B * C, L)
        device = x_rows.device

        # Allocate padded buffer (BC, N)
        padded = torch.empty((B * C, N), dtype=torch.float32, device=device)

        # Copy rows into padded (zeros tail) using Triton
        grid_copy = (B * C,)
        _copy_row_to_padded[grid_copy](x_rows, padded, L, N)

        # Allocate outputs (BC, L+1) for real and imaginary parts
        out_real = torch.empty((B * C, L + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((B * C, L + 1), dtype=torch.float32, device=device)

        # Compute rfft via Triton direct accumulation
        grid_rfft = (B * C,)
        _rfft_row[grid_rfft](B * C, N, L, padded, out_real, out_imag)

        # Normalize by N = 2*L using Triton elementwise division
        n_elements_r = out_real.numel()
        grid_div = (triton.cdiv(n_elements_r, 1024),)
        _divide_by_n[grid_div](out_real, float(N), n_elements_r)
        n_elements_i = out_imag.numel()
        grid_div_i = (triton.cdiv(n_elements_i, 1024),)
        _divide_by_n[grid_div_i](out_imag, float(N), n_elements_i)

        # Reshape back to (B, C, L+1)
        x_freq_real = out_real.reshape(B, C, L + 1)
        x_freq_imag = out_imag.reshape(B, C, L + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
