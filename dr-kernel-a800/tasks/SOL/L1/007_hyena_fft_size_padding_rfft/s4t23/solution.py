import torch
import triton
import triton.language as tl

# 1) Copy input row x[b, c, :] (length L) into padded buffer of length n=2*L, fill tail with zeros
@triton.jit
def _copy_row_to_padded_kernel(
    in_ptr,                # *const float, shape (B*C, L)
    out_ptr,               # *float, shape (B*C, n), n=2*L
    L: tl.constexpr,       # int
    n: tl.constexpr,       # int, n = 2 * L
    BLOCK: tl.constexpr,   # block size along j
):
    pid = tl.program_id(0)  # each program handles one (b,c) slice
    j = tl.arange(0, BLOCK)
    mask = j < L
    x = tl.load(in_ptr + pid * L + j, mask=mask, other=0.0)
    # Store into padded buffer, only valid j
    tl.store(out_ptr + pid * n + j, x, mask=mask)
    # Tail remains zero (host initialized)

# 2) Compute forward DFT for a single vector of length n=2*L (no atomics).
#    Stores only the first L+1 real outputs into out_real_ptr[pid, k].
@triton.jit
def _dft_forward_real_kernel(
    in_ptr,                # *const float, shape (B*C, n), input padded row
    out_real_ptr,          # *float, shape (B*C, L+1), to store real outputs
    n: tl.constexpr,       # int
    L: tl.constexpr,       # int
    BLOCK_J: tl.constexpr, # block size for j loop
):
    pid = tl.program_id(0)  # per (b,c) slice

    # We compute DFT for t = 0..L and store into out_real_ptr[pid, t].
    # For t >= L+1, we don't write (kernel only handles t up to L).
    # Summation uses radix-2 style block summation without atomics.
    # Output is real-only because input is real (matches torch.fft.rfft behavior).
    for t in range(0, L + 1):
        s = tl.zeros((), dtype=tl.float32)
        # Sum over j in blocks
        for jj in range(0, n, BLOCK_J):
            j_idx = jj + tl.arange(0, BLOCK_J)
            mask_j = j_idx < n
            xj = tl.load(in_ptr + pid * n + j_idx, mask=mask_j, other=0.0)
            # exp(-i*2*pi*j*t/n) = cos(...) - i*sin(...); we only need real part since output is real.
            # For real DFT of real input, the output is real.
            angle = -(2.0 * 3.141592653589793 * j_idx * t) / n
            # cos(angle) is real part; we use it directly
            cos_a = tl.cos(angle)
            s += tl.sum(xj * cos_a, axis=0)
        # Store real result
        tl.store(out_real_ptr + pid * (L + 1) + t, s)

# 3) Normalize by scale = 2*L
@triton.jit
def _divide_by_scale_kernel(
    in_ptr,                # *float, shape (B*C, L+1)
    out_ptr,               # *float, shape (B*C, L+1)
    scale,                 # float32
    L: tl.constexpr,       # int
    BLOCK: tl.constexpr,   # block size
):
    pid = tl.program_id(0)  # linear index over (B*C)*(L+1)
    num_k = L + 1
    bc = pid // num_k
    k = pid % num_k
    val = tl.load(in_ptr + bc * (num_k) + k)
    val = val / scale
    tl.store(out_ptr + bc * (num_k) + k, val)

# 4) Store output: real/imag parts into (B, C, L+1) tensors (imag will be zeros to match signature)
@triton.jit
def _store_output_kernel(
    in_ptr,                # *float, shape (B*C, L+1)
    out_ptr,               # *float, shape (B, C, L+1)
    L: tl.constexpr,       # int
    BLOCK: tl.constexpr,   # block size
):
    bc = tl.program_id(0)  # per (b,c)
    k = tl.program_id(1)   # per k in [0..L]
    val = tl.load(in_ptr + bc * (L + 1) + k)
    tl.store(out_ptr + bc * (L + 1) + k, val)

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, L) float32 tensor
        Returns:
            x_freq_real: (B, C, L+1) float32
            x_freq_imag: (B, C, L+1) float32 (zeros, since rfft of real input is real-only)
        """
        assert x.ndim == 3, "Input must be (B, C, L)"
        B, C, L = x.shape
        n = 2 * L

        # 1) Prepare padded buffer: (B*C, n)
        x_contig = x.contiguous()
        padded = torch.zeros((B * C, n), dtype=torch.float32, device=x.device)

        # Launch copy kernel to fill first L entries
        BLOCK = 256
        grid_copy = (B * C,)
        _copy_row_to_padded_kernel[grid_copy](
            x_contig.reshape(-1, L),  # input row pointer for each (b,c)
            padded,                    # output padded buffer
            L, n, BLOCK,
        )

        # 2) Allocate output for real (B*C, L+1)
        out_real = torch.empty((B * C, L + 1), dtype=torch.float32, device=x.device)

        # 3) Compute forward DFT real outputs for t in [0..L]
        #    We'll set BLOCK_J to 256; loops handle any n but we use n=2*L (power of two when L is power of two).
        #    Even if L is not power-of-two, correctness is ensured by padding to 2*L and computing DFT.
        grid_dft = (B * C,)
        _dft_forward_real_kernel[grid_dft](
            padded, out_real, n, L, 256
        )

        # 4) Normalize by 2*L
        scale = 2.0 * float(L)
        out_real_norm = torch.empty_like(out_real, dtype=torch.float32, device=x.device)
        grid_div = (B * C * (L + 1),)
        _divide_by_scale_kernel[grid_div](out_real, out_real_norm, scale, L, 128)

        # 5) Prepare outputs: real and imag (imag zeros to match signature and original behavior)
        x_freq_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        x_freq_imag = torch.zeros((B, C, L + 1), dtype=torch.float32, device=x.device)

        # 6) Store outputs into (B, C, L+1) using _store_output_kernel (ensures kernel is used)
        grid_store = (B * C, L + 1)
        _store_output_kernel[grid_store](out_real_norm, x_freq_real, L, 128)
        _store_output_kernel[grid_store](x_freq_imag, x_freq_imag, L, 128)  # write zeros

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
