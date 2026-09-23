import torch
import triton
import triton.language as tl


# Kernel 1: Zero-pad input of length L to a buffer of length n (n = 2*L)
@triton.jit
def _pad_to_n_kernel(x_ptr, padded_ptr, L, n, BLOCK: tl.constexpr):
    """
    x_ptr: *float32, length L (original seqlen)
    padded_ptr: *float32, length n (2*seqlen), output buffer to be zero-padded
    Each program writes one element of padded_ptr.
    """
    pid = tl.program_id(axis=0)
    if pid < L:
        tl.store(padded_ptr + pid, tl.load(x_ptr + pid))
    else:
        tl.store(padded_ptr + pid, 0.0)


# Kernel 2: Direct rFFT for k in [0..L] using explicit padding to n
# y[k] = sum_{j=0}^{n-1} x[j] * exp(-i * 2*pi*k*j / n)
# We compute real and imag parts separately: real += xj*cos, imag += xj*sin
# This kernel processes a block of k indices for one (batch, channel) slice; it loops over j to accumulate.
@triton.jit
def _rfft_simple_per_bc_kernel(padded_ptr, L, n, out_real_ptr, out_imag_ptr, BLOCK_K: tl.constexpr):
    """
    padded_ptr: *float32, length n, contains input x padded with zeros
    L: int, original seqlen
    n: int, 2*L
    out_real_ptr: *float32, shape (B, C, L+1), contiguous
    out_imag_ptr: *float32, shape (B, C, L+1), contiguous
    The kernel computes rFFT for a single (batch, channel) slice and writes into out_real_ptr/out_imag_ptr.
    We use program_id(1) and program_id(0) to identify (batch, channel). Program_id(1) is not used here.
    We compute for all (batch, channel) by making grid1 = B*C and relying on program_id(1) to iterate.
    """
    bc_id = tl.program_id(axis=1)  # combine batch and channel: bc_id in [0, B*C)
    # To align with earlier API where we didn't take B,C as args, we extract shape from out_real_ptr.
    # However, since we don't have B,C here, we assume out tensors are (B, C, L+1) and base indexing uses:
    # We need to know (L+1) to compute base; thus we pass a separate scalar 'S' indicating L+1 via a different approach.
    # Simplify: we'll require that the caller sets grid dims such that program_id(1) corresponds to (b,c), but we need B,C.
    # Therefore, we redefine the kernel to take B and C as arguments to compute base offset.
    pass  # Placeholder to demonstrate structure; actual kernel below will fix this.


# Correct kernel 2 with B and C as args, writing into preallocated (B, C, L+1) output
@triton.jit
def _rfft_simple_per_bc_kernel_v2(padded_ptr, B, C, L, n, out_real_ptr, out_imag_ptr, BLOCK_K: tl.constexpr):
    """
    padded_ptr: *float32, length n, contains input x padded with zeros
    B: int, batch size
    C: int, channels
    L: int, original seqlen
    n: int, 2*L
    out_real_ptr: *float32, shape (B, C, L+1), contiguous
    out_imag_ptr: *float32, shape (B, C, L+1), contiguous
    The kernel computes rFFT for a single (batch, channel) slice and writes into out_real_ptr/out_imag_ptr.
    We use program_id(axis=0) for blocks of k and program_id(axis=1) for (batch, channel).
    """
    k0 = tl.program_id(axis=0) * BLOCK_K
    ks = k0 + tl.arange(0, BLOCK_K)
    mask_k = ks <= L  # output length is L+1

    # Identify batch and channel from program_id(axis=1)
    bc_id = tl.program_id(axis=1)
    b = bc_id // C
    c = bc_id % C

    # Base offset for this (b, c) slice in a contiguous (B, C, L+1) tensor
    base = (b * C + c) * (L + 1)

    acc_real = tl.zeros([BLOCK_K], dtype=tl.float32)
    acc_imag = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Sum over j from 0 to n-1; padded_ptr has zeros for j >= L
    for j in range(0, n):
        xj = tl.load(padded_ptr + j)  # float32
        # angle = 2*pi*k*j / n
        angle = 2.0 * 3.141592653589793 * ks * j * (1.0 / n)
        cosv = tl.cos(angle)
        sinv = tl.sin(angle)
        acc_real += xj * cosv
        acc_imag += xj * sinv

    # Store results for ks up to L into out_real_ptr/out_imag_ptr at [b, c, ks]
    out_real_base = out_real_ptr + base
    out_imag_base = out_imag_ptr + base
    tl.store(out_real_base + ks, acc_real, mask=mask_k)
    tl.store(out_imag_base + ks, acc_imag, mask=mask_k)


# Kernel 3: Elementwise normalization by division with scalar
@triton.jit
def _normalize_divide_kernel(in_ptr, out_ptr, numel, scale, BLOCK: tl.constexpr):
    """
    out[i] = in[i] / scale
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = x / scale
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single input tensor of shape (batch, channels, seqlen)
        if len(args) != 1:
            raise ValueError("ModelNew.forward expects a single input tensor")
        x = args[0]
        if x.dim() != 3:
            raise ValueError("Input must be a 3D tensor of shape (batch, channels, seqlen)")

        # Cast to float32 (original code does this)
        x_f32 = x.to(torch.float32)
        batch, channels, seqlen = x_f32.shape
        device = x_f32.device

        L = seqlen
        n = 2 * L  # padding size as in original

        # 1) Zero-pad input to length n
        padded = torch.empty(n, dtype=torch.float32, device=device)
        # Launch padding kernel: one program per index
        grid_pad = (n,)
        _pad_to_n_kernel[grid_pad](x_f32, padded, L, n, BLOCK=1)

        # 2) Allocate outputs of shape (batch, channels, L+1) and compute rFFT per (batch, channel) slice
        out_real = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=device)

        # Triton grid: axis 0 over blocks of k, axis 1 over all (batch, channel) slices
        BLOCK_K = 256  # larger block to improve numerical stability; 128/256 are typical
        grid0 = triton.cdiv(L + 1, BLOCK_K)  # number of blocks in k dimension
        grid1 = batch * channels
        _rfft_simple_per_bc_kernel_v2[(grid0, grid1)](padded, batch, channels, L, n, out_real, out_imag, BLOCK_K=BLOCK_K)

        # 3) Normalize by 2*L using Triton elementwise division (per tensor)
        scale = 2.0 * L
        # Real part
        numel_r = out_real.numel()
        grid_norm = (triton.cdiv(numel_r, 1024),)
        _normalize_divide_kernel[grid_norm](out_real, out_real, numel_r, scale, BLOCK=1024)
        # Imag part
        numel_i = out_imag.numel()
        grid_norm = (triton.cdiv(numel_i, 1024),)
        _normalize_divide_kernel[grid_norm](out_imag, out_imag, numel_i, scale, BLOCK=1024)

        # Return outputs shaped (batch, channels, seqlen+1)
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
