import torch
import triton
import triton.language as tl


@triton.jit
def _copy_row_and_zero_pad_kernel(
    x_ptr,                # *float32, input row pointer: [B, C, L]
    out_ptr,              # *float32, output padded pointer: [B, C, 2L]
    B, C, L,              # int32
    b, c, k,              # int32, for grid mapping
    BLOCK_J: tl.constexpr,
):
    # Each program handles one output element k for a given (b, c)
    # We need to fill out_ptr[b, c, j] for j in [0..2*L-1]
    # But we only copy x[b, c, :] to out_ptr[b, c, 0:L]
    # For j >= L, out_ptr[b, c, j] = 0
    # Compute base index
    total_n = 2 * L
    j_offsets = tl.arange(0, BLOCK_J)
    # We'll iterate j from 0 to 2*L-1 in chunks of BLOCK_J
    # Note: Triton supports while loops
    j = 0
    while j < total_n:
        j_offsets = j + tl.arange(0, BLOCK_J)
        mask = j_offsets < total_n
        # For j < L, load from x; else store 0
        # We need to decide per offset whether to load or not
        # Use a scalar condition
        # If j_offsets < L: load, else store 0
        # But we can just compute addresses for j_offsets < L and write zeros for >=L
        # Compute how many valid in this chunk
        valid = j_offsets < L
        # Load x[b, c, j_offsets] where valid
        # Compute linear index for x: ((b*C + c)*L + j_offsets)
        # Note: the grid maps b, c, k; we need (b, c) pair. Let's assume grid(3) uses b and c as first two dims
        # The kernel is launched with grid=(B, C, total_n), so k is ignored here; we only handle one (b, c)
        # We need to know which (b, c) this program belongs to. Triton allows passing scalar args.
        # We pass b and c as scalar args; index of x is linear: ((b*C + c)*L + j)
        # However, since we loop j, we should compute linear index with j.
        # But we need base pointer for x for (b, c). We can compute base = (b*C + c)*L and then x_ptr + base + j.
        base_x = (b * C + c) * L
        # For j_offsets < L: load x[b, c, j_offsets]
        # For j_offsets >= L: write 0
        # We'll write to out_ptr[b, c, j_offsets] for j_offsets < L
        # Compute base_out = (b*C + c)*total_n
        base_out = (b * C + c) * total_n
        # Store x values for j_offsets < L
        # If valid, write x[base_x + j_offsets] to out_ptr[base_out + j_offsets]
        # Initialize val with zeros, then assign where valid
        val = tl.zeros([BLOCK_J], dtype=tl.float32)
        # Build address for x: x_ptr + base_x + j_offsets
        addr_x = x_ptr + base_x + j_offsets
        # Load where valid
        val = tl.load(addr_x, mask=valid, other=0.0)
        # Store to out_ptr
        addr_out = out_ptr + base_out + j_offsets
        # Store only for j_offsets < total_n (but we already ensured mask above). For j_offsets >= L, val is zeros.
        tl.store(addr_out, val, mask=mask)
        j += BLOCK_J


@triton.jit
def _compute_real_part_kernel(
    out_ptr,              # *float32, padded input pointer: [B, C, 2L]
    real_out_ptr,         # *float32, output real pointer: [B, C, L+1]
    B, C, L,              # int32
    b, c, k,              # int32, for grid mapping
    BLOCK_J: tl.constexpr,
):
    # Each program computes one output element k for a given (b, c)
    total_n = 2 * L
    acc = tl.zeros((), dtype=tl.float32)
    j = 0
    while j < total_n:
        j_offsets = j + tl.arange(0, BLOCK_J)
        mask = j_offsets < total_n
        xj = tl.load(out_ptr + (b * C + c) * total_n + j_offsets, mask=mask, other=0.0)
        w = tl.cos(2.0 * 3.141592653589793 * k * j_offsets / total_n)
        prod = xj * w
        # Sum across the vector lane
        acc += tl.sum(prod, axis=0)
        j += BLOCK_J
    # Store to real_out[b, c, k]
    addr = real_out_ptr + (b * C + c) * (L + 1) + k
    tl.store(addr, acc)


@triton.jit
def _compute_imag_part_kernel(
    out_ptr,              # *float32, padded input pointer: [B, C, 2L]
    imag_out_ptr,         # *float32, output imag pointer: [B, C, L+1]
    B, C, L,              # int32
    b, c, k,              # int32, for grid mapping
    BLOCK_J: tl.constexpr,
):
    # Each program computes one output element k for a given (b, c)
    total_n = 2 * L
    acc = tl.zeros((), dtype=tl.float32)
    j = 0
    while j < total_n:
        j_offsets = j + tl.arange(0, BLOCK_J)
        mask = j_offsets < total_n
        xj = tl.load(out_ptr + (b * C + c) * total_n + j_offsets, mask=mask, other=0.0)
        w = tl.sin(2.0 * 3.141592653589793 * k * j_offsets / total_n)
        prod = xj * w
        acc += tl.sum(prod, axis=0)
        j += BLOCK_J
    # Store -acc to imag_out[b, c, k]
    addr = imag_out_ptr + (b * C + c) * (L + 1) + k
    tl.store(addr, -acc)


@triton.jit
def _normalize_and_store_real_kernel(
    real_ptr,             # *float32, real output pointer: [B, C, L+1]
    normalized_ptr,       # *float32, output pointer for real normalization: [B, C, L+1]
    B, C, L,              # int32
    b, c, k,              # int32, for grid mapping
    scale,                # float32, normalization factor (2*L)
):
    # Each program handles one element (b, c, k)
    val = tl.load(real_ptr + (b * C + c) * (L + 1) + k)
    val = val / scale
    tl.store(normalized_ptr + (b * C + c) * (L + 1) + k, val)


@triton.jit
def _normalize_and_store_imag_kernel(
    imag_ptr,             # *float32, imag output pointer: [B, C, L+1]
    normalized_ptr,       # *float32, output pointer for imag normalization: [B, C, L+1]
    B, C, L,              # int32
    b, c, k,              # int32, for grid mapping
    scale,                # float32, normalization factor (2*L)
):
    # Each program handles one element (b, c, k)
    val = tl.load(imag_ptr + (b * C + c) * (L + 1) + k)
    val = val / scale
    tl.store(normalized_ptr + (b * C + c) * (L + 1) + k, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: Input tensor of shape (batch, channels, seqlen)
        Returns:
            out_real: Real part of normalized frequency domain output (batch, channels, seqlen+1)
            out_imag: Imaginary part of normalized frequency domain output (batch, channels, seqlen+1)
        """
        assert x.dim() == 3, "Input must be 3D (B, C, L)"
        B, C, L = x.shape
        n = 2 * L

        # Ensure dtype float32 for computation
        x_f32 = x.to(torch.float32)

        # Allocate padded output buffer [B, C, 2*L], zero-initialized
        out_padded = torch.zeros((B, C, n), dtype=torch.float32, device=x.device)

        # Copy input row to padded buffer: out_padded[:, :, :L] = x_f32
        # We'll launch one program per (b, c)
        grid_copy = (B, C)
        _copy_row_and_zero_pad_kernel[grid_copy](
            x_f32, out_padded, B, C, L, B, C, 0, BLOCK_J=128
        )

        # Allocate outputs for real and imag (not normalized yet)
        real_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Compute real and imag parts using Triton, grid over (B, C, L+1)
        grid = (B, C, L + 1)
        _compute_real_part_kernel[grid](
            out_padded, real_out, B, C, L, B, C, 0, BLOCK_J=256
        )
        _compute_imag_part_kernel[grid](
            out_padded, imag_out, B, C, L, B, C, 0, BLOCK_J=256
        )

        # Normalize by 2*L and store final outputs
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        scale = 2.0 * float(L)
        grid_norm = (B, C, L + 1)
        _normalize_and_store_real_kernel[grid_norm](
            real_out, out_real, B, C, L, B, C, 0, scale
        )
        _normalize_and_store_imag_kernel[grid_norm](
            imag_out, out_imag, B, C, L, B, C, 0, scale
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
