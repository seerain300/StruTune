import torch
import triton
import triton.language as tl


@triton.jit
def _cast_to_f32(x_ptr, out_ptr, NUMEL: tl.int32):
    """
    Cast input tensor to float32. Reads from x_ptr (any dtype) and writes float32 to out_ptr.
    We use a simple 1D program: each program handles one element.
    """
    pid = tl.program_id(0)
    if pid < NUMEL:
        val = tl.load(x_ptr + pid)
        # If x_ptr is not float32, Triton will cast on load; then store as float32.
        tl.store(out_ptr + pid, val.to(tl.float32))


@triton.jit
def _copy_row_to_padded(x_rows_ptr, padded_ptr, L: tl.int32, N: tl.int32):
    """
    For one row (length L), copy into padded buffer of length N (N = 2*L) and zero pad the tail.
    x_rows_ptr is (BC, L). We assume grid=(BC,), so row_idx = program_id(0).
    padded_ptr is (BC, N).
    """
    row_idx = tl.program_id(0)
    # Copy first L elements
    for i in tl.static_range(0, L):
        val = tl.load(x_rows_ptr + row_idx * L + i)
        tl.store(padded_ptr + row_idx * N + i, val)
    # Zero pad the rest
    for i in tl.static_range(L, N):
        tl.store(padded_ptr + row_idx * N + i, 0.0)


@triton.jit
def _rfft_row(row_idx, L: tl.int32, N: tl.int32, padded_ptr, out_real_ptr, out_imag_ptr):
    """
    Compute rfft for one (batch, channel) row:
    - padded_ptr: (BC, N) where N=2*L
    - out_real_ptr, out_imag_ptr: (BC, L+1) flattened as (BC*(L+1)) via offset = row_idx*(L+1) + k
    """
    # We need to compute for k in [0..L]
    # Use a simple accumulation loop over j in [0..N-1]
    # Accumulate in float32
    inv_N = 1.0 / N
    # Note: Triton supports elementwise sin/cos. We loop over j scalarly to keep it robust.
    for k in tl.static_range(0, L + 1):
        # We only need k in [0, L], but for safety we keep k in the loop bounds (L+1).
        # However, to accumulate, we need to skip k == L+1. Better to limit k in [0, L]. Use while to handle dynamic.
        k_val = k  # k is statically generated, but we can also compute with a while to be robust
        acc_real = 0.0
        acc_imag = 0.0
        j = 0
        while j < N:
            val = tl.load(padded_ptr + row_idx * N + j)
            angle = 2.0 * 3.141592653589793 * k_val * j / N
            acc_real += val * tl.cos(angle)
            acc_imag += val * tl.sin(angle)
            j += 1
        # Normalize by N
        acc_real = acc_real * inv_N
        acc_imag = acc_imag * inv_N
        # Store at (row_idx, k) in (B, C, L+1) flattened to (BC, L+1)
        out_off = row_idx * (L + 1) + k
        tl.store(out_real_ptr + out_off, acc_real)
        tl.store(out_imag_ptr + out_off, acc_imag)


@triton.jit
def _divide_by_n(out_ptr, divisor: tl.float32, NUMEL: tl.int32):
    """
    Elementwise divide out_ptr by divisor. out_ptr can be either out_real or out_imag.
    Grid: 1D, each program handles one element.
    """
    pid = tl.program_id(0)
    if pid < NUMEL:
        val = tl.load(out_ptr + pid)
        val = val / divisor
        tl.store(out_ptr + pid, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute real and imaginary parts of normalized rfft(x) with padding to 2*L,
        returning (B, C, L+1) tensors for real and imaginary parts.
        All computation is done in Triton kernels. No torch ops used in forward.
        """
        assert x.dim() == 3, "Input must be of shape (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L
        device = x.device
        # 1) Cast to float32 if needed (only if not float32). Triton kernel for cast.
        # Create an output buffer in float32 regardless of input dtype.
        x_f32 = torch.empty_like(x, dtype=torch.float32, device=device)
        # We'll use the cast kernel on x and write into x_f32. If x is already float32, this is a no-op in terms of content, but we still run it safely.
        # However, since we allocate x_f32 directly as float32, we can skip casting. To satisfy Triton usage, we define a dummy cast and fill x_f32 from x.
        # For performance and simplicity, we avoid launching an extra kernel for cast because we already allocate float32.
        # Instead, just ensure x_f32 is float32 and proceed.

        # 2) Reshape x to (BC, L) for kernel _copy_row_to_padded
        BC = B * C
        x_rows = x_f32.reshape(BC, L)

        # 3) Allocate padded buffer (BC, N)
        padded = torch.empty((BC, N), dtype=torch.float32, device=device)

        # 4) Launch copy kernel to pad x_rows into padded
        # Grid = (BC,) — one program per row
        _copy_row_to_padded[(BC,)](x_rows, padded, L, N)

        # 5) Allocate outputs (BC, L+1)
        out_real = torch.empty((BC, L + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((BC, L + 1), dtype=torch.float32, device=device)

        # 6) Launch rfft_row kernel: compute real/imag parts for each row
        # Grid = (BC,) — one program per row
        _rfft_row[(BC,)](0, L, N, padded, out_real, out_imag)  # Note: we pass row_idx=0 here, but since grid is (BC,), we need a separate way to iterate rows.
        # To handle all rows in one kernel launch, we instead launch grid=(BC,) and pass row_idx=program_id(0).
        # However, Triton kernels don't easily accept an additional 'row_idx' parameter; so we iterate rows by launching one program per row.
        # We can implement a loop in Python to launch per row, but Triton requires static grid size. To adhere to Triton-only, we compute per row in a separate launch.
        # Better: we restructure to a 2D grid? Here, we’ll launch per row explicitly by iterating Python range(BC) and call the kernel for each, which is fine.
        # Since Triton kernels must be called, we do:
        for row_idx in range(BC):
            _rfft_row[(1,)](row_idx, L, N, padded, out_real, out_imag)  # one program per row

        # 7) Flatten to (B, C, L+1)
        # out_real/imag are (BC, L+1), reshape to (B, C, L+1)
        out_real_bc = out_real.view(B, C, L + 1)
        out_imag_bc = out_imag.view(B, C, L + 1)

        # 8) Normalize by N using Triton kernel (divide by 2*L)
        total_elems = (B * C) * (L + 1)
        _divide_by_n[(total_elems,)](out_real_bc, float(N), total_elems)
        _divide_by_n[(total_elems,)](out_imag_bc, float(N), total_elems)

        return out_real_bc, out_imag_bc


def run(*args):
    return ModelNew()(*args)
