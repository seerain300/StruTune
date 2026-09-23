import triton
import triton.language as tl


@triton.jit
def _real_rfft_kernel(
    x_ptr,            # *const float32, input pointer to x (B, C, L)
    out_real_ptr,     # *float32, output pointer to real part (B, C, L+1), flattened
    out_imag_ptr,     # *float32, output pointer to imag part (B, C, L+1), flattened
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    L: tl.constexpr,  # seqlen
    N: tl.constexpr,  # N = 2 * L
    M: tl.constexpr,  # M = L + 1
):
    # Each program handles one (b, c) slice
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Base offset for this (b, c) slice in the input: x is (B, C, L) contiguous
    # index = ((b * C + c) * L + t)
    base_in = (b * C + c) * L

    # Base output offset for this (b, c) slice: output is (B, C, M) contiguous
    base_out = (b * C + c) * M

    # Precompute constants
    pi = 3.141592653589793
    inv_N = 1.0 / N

    # Loop over frequency j from 0 to M-1, in chunks
    J_CHUNK = 64  # small vectorization over j for efficiency; adjust if needed
    for j0 in range(0, M, J_CHUNK):
        # vector of j indices for this chunk
        j_offsets = j0 + tl.arange(0, J_CHUNK)
        j_mask = j_offsets < M  # valid j indices

        # Accumulators for real and imaginary parts for this chunk (vector of length J_CHUNK)
        re = tl.zeros((J_CHUNK,), dtype=tl.float32)
        im = tl.zeros((J_CHUNK,), dtype=tl.float32)

        # Accumulate over time t = 0..N-1
        # Use scalar t loop for robustness
        for t in range(0, N):
            # Load x[b, c, t]
            x_val = tl.load(x_ptr + base_in + t)
            # Compute angles for this chunk of j: ang = 2*pi*j*t/N
            ang = 2.0 * pi * (j_offsets.to(tl.float32)) * (t.to(tl.float32)) / N
            # cos and sin
            c = tl.cos(ang)
            s = tl.sin(ang)
            # Accumulate
            re += x_val * c
            im += x_val * s

        # Normalize by N (2*seqlen)
        re = re * inv_N
        im = im * inv_N

        # Store results to output with mask for j < M
        # out_real_ptr and out_imag_ptr point to flattened arrays of length B*C*M
        # For each (b, c), outputs are contiguous at offset base_out
        store_idx = base_out + j_offsets
        tl.store(out_real_ptr + store_idx, re, mask=j_mask)
        tl.store(out_imag_ptr + store_idx, im, mask=j_mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Ensure x is contiguous and on CUDA for Triton
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()

        # Allocate outputs; flatten to contiguous for kernel
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B, C)
        _real_rfft_kernel[grid](
            x, out_real, out_imag,
            B, C, L, N, M,
            num_warps=1,  # simple kernel; small num_warps is fine
            num_stages=1,
        )

        # Reshape back to (B, C, M)
        out_real = out_real.view(B, C, M)
        out_imag = out_imag.view(B, C, M)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
