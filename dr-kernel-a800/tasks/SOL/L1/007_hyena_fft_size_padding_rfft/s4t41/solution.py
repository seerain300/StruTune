import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    # One program per (b, c) row
    pid_bc = tl.program_id(0)
    # Copy x[pid_bc, :L] into out[pid_bc, :L]
    for j in tl.static_range(0, L):
        val = tl.load(x_ptr + pid_bc * L + j)
        tl.store(out_ptr + pid_bc * N + j, val)
    # Zero-fill the tail
    for j in tl.static_range(L, N):
        tl.store(out_ptr + pid_bc * N + j, 0.0)


@triton.jit
def _compute_rfft_row(out_real_ptr, out_imag_ptr, padded_ptr, L: tl.int32, N: tl.int32):
    # One program computes real and imag parts for all k in [0, L] for a single (b, c) row
    pid_bc = tl.program_id(0)  # over B*C rows
    inv_n = 0.5  # since N = 2*L in this task
    for k in tl.static_range(0, L + 1):
        acc_real = 0.0
        acc_imag = 0.0
        # Loop over j=0..N-1
        for j in tl.static_range(0, N):
            val = tl.load(padded_ptr + pid_bc * N + j)
            angle = 2.0 * 3.141592653589793 * (k * j) / N
            acc_real += val * tl.cos(angle)
            acc_imag += val * tl.sin(angle)
        # Normalize by 2*L
        acc_real = acc_real * inv_n
        acc_imag = acc_imag * inv_n
        # Store to outputs (flat indexing by (b,c) row and k)
        tl.store(out_real_ptr + pid_bc * (L + 1) + k, acc_real)
        tl.store(out_imag_ptr + pid_bc * (L + 1) + k, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # Input: (B, C, L)
        assert x.is_cuda, "Input must be on CUDA for Triton."
        assert x.dtype == torch.float32, "Input must be float32."
        B, C, L = x.shape
        N = 2 * L  # padding to n=2*L as in original code

        # Allocate padded buffer and outputs
        padded = torch.empty((B * C, N), dtype=torch.float32, device=x.device)
        out_real = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)

        # 1) Copy input rows into padded rows (zero-fill tail)
        grid_copy = (B * C,)
        _copy_row_to_padded[grid_copy](x.view(-1, L), padded, L, N)

        # 2) Compute real and imag parts per (b, c) row
        grid_bc = (B * C,)
        _compute_rfft_row[grid_bc](out_real, out_imag, padded, L, N)

        # Reshape back to (B, C, L+1)
        out_real = out_real.view(B, C, L + 1)
        out_imag = out_imag.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
