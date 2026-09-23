import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_to_padded_kernel(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    # Each program handles one (b, c) row: flattened index bc in [0, B*C)
    bc = tl.program_id(0)
    base_out = bc * N
    # Copy x[bc, :L] -> out[bc, 0:L]
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + bc * L + i)
        tl.store(out_ptr + base_out + i, val)
    # Zero-fill the rest [L, N)
    for i in tl.static_range(L, N):
        tl.store(out_ptr + base_out + i, 0.0)


@triton.jit
def _compute_rfft_k_real_kernel(padded_ptr, out_real_ptr, L: tl.int32, N: tl.int32):
    # Grid: (BC, L+1). Each program computes k for one (bc, k) and atomically adds its contribution.
    bc = tl.program_id(0)
    k = tl.program_id(1)
    # k in [0..L]
    acc = 0.0
    for j in tl.static_range(0, N):
        xj = tl.load(padded_ptr + bc * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        cosv = tl.cos(angle)
        acc += xj * cosv
    out_index = bc * (L + 1) + k
    tl.atomic_add(out_real_ptr + out_index, acc)


@triton.jit
def _compute_rfft_k_imag_kernel(padded_ptr, out_imag_ptr, L: tl.int32, N: tl.int32):
    # Grid: (BC, L+1). Each program computes k for one (bc, k) and atomically adds its contribution.
    bc = tl.program_id(0)
    k = tl.program_id(1)
    acc = 0.0
    for j in tl.static_range(0, N):
        xj = tl.load(padded_ptr + bc * N + j)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        sinv = tl.sin(angle)
        acc += xj * sinv
    out_index = bc * (L + 1) + k
    tl.atomic_add(out_imag_ptr + out_index, acc)


@triton.jit
def _normalize_divide_const(in_ptr, out_ptr, total_elems: tl.int32, divisor: tl.float32):
    # Elementwise division by a scalar 'divisor' for a 1D tensor of length total_elems.
    pid = tl.program_id(0)
    if pid < total_elems:
        val = tl.load(in_ptr + pid)
        val = val / divisor
        tl.store(out_ptr + pid, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - Compute y_real, y_imag for each (b,c) row, normalized by 2*L.
        - Return tensors of shape (B, C, L+1) for real and imaginary parts.
        """
        assert x.ndim == 3, "Input must be (batch, channels, seqlen)"
        B, C, L = x.shape
        device = x.device

        # Ensure input is contiguous
        x = x.contiguous()
        # Prepare N = 2 * L (padded length for rfft)
        N = 2 * L

        # 1) Copy row to padded buffer (length N), dtype float32
        x_f32 = x.to(torch.float32)
        BC = B * C
        # Allocate padded buffer (BC, N)
        padded = torch.empty((BC, N), dtype=torch.float32, device=device)

        # Launch copy kernel: grid (BC,)
        _copy_row_to_padded_kernel[(BC,)](
            x_f32.view(BC, L),  # x_ptr
            padded,              # out_ptr
            L, N,                # meta-parameters
        )

        # 2) Compute y_real and y_imag for k in [0..L] using Triton kernels
        out_real = torch.zeros(BC * (L + 1), dtype=torch.float32, device=device)
        out_imag = torch.zeros(BC * (L + 1), dtype=torch.float32, device=device)

        # Launch computation kernels: grid (BC, L+1)
        _compute_rfft_k_real_kernel[(BC, L + 1)](
            padded,  # padded_ptr
            out_real,  # out_real_ptr
            L, N,
        )
        _compute_rfft_k_imag_kernel[(BC, L + 1)](
            padded,  # padded_ptr
            out_imag,  # out_imag_ptr
            L, N,
        )

        # 3) Normalize by 2*N using Triton kernel
        total_elems = BC * (L + 1)
        inv_2N = 1.0 / (2.0 * N)
        out_real_norm = torch.empty_like(out_real, device=device)
        out_imag_norm = torch.empty_like(out_imag, device=device)
        _normalize_divide_const[(total_elems,)](
            out_real, out_real_norm, total_elems, inv_2N
        )
        _normalize_divide_const[(total_elems,)](
            out_imag, out_imag_norm, total_elems, inv_2N
        )

        # 4) Reshape back to (B, C, L+1)
        out_real_bc = out_real_norm.view(B, C, L + 1)
        out_imag_bc = out_imag_norm.view(B, C, L + 1)

        # Return real and imaginary parts
        return out_real_bc, out_imag_bc


def run(*args):
    return ModelNew()(*args)
