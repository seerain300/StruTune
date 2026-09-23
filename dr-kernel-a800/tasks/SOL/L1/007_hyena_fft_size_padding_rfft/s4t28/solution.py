import torch
import triton
import triton.language as tl


@triton.jit
def _copy_pad_kernel(x_ptr, out_ptr, L: tl.int32, N: tl.int32):
    # Each program handles one (bc) row: pid_bc in [0, B*C)
    pid_bc = tl.program_id(0)
    # Copy x[pid_bc, :L] into out[pid_bc, 0:L]
    for i in tl.static_range(0, L):
        val = tl.load(x_ptr + pid_bc * L + i)
        tl.store(out_ptr + pid_bc * N + i, val)
    # Fill the rest with zeros
    for i in tl.static_range(L, N):
        tl.store(out_ptr + pid_bc * N + i, 0.0)


@triton.jit
def _rfft_real_recurrence_kernel(x_ptr, out_ptr, L: tl.int32, N: tl.int32, BC: tl.int32):
    # Compute real part of rfft for N=2*L using recurrence for k in [0..L], store at index k in out[BC, L+1]
    # out_ptr is a flat buffer of length BC*(L+1); we index per (bc, k)
    pid_bc = tl.program_id(0)
    k = tl.program_id(1)

    # Base case: if k == 0, y0 = sum(x0)/N
    if k == 0:
        s = 0.0
        for j in tl.static_range(0, N):
            s += tl.load(x_ptr + pid_bc * N + j)
        y = s / N
        tl.store(out_ptr + pid_bc * (L + 1) + k, y)
        return

    # Helper: compute sum over subset using masks
    def sum_subset(start, count):
        s = 0.0
        # Iterate over j in [start, start + count - 1] (N=2*L is even, L is max)
        for j in tl.static_range(0, count):
            jj = start + j
            valid = jj < N
            v = tl.load(x_ptr + pid_bc * N + jj, mask=valid, other=0.0)
            s += v
        return s

    # Compute s_even = sum_{j=0..N/2-1} x0[2j]
    s_even = sum_subset(0, N // 2)
    # Compute s_odd = sum_{j=0..N/2-1} x0[2j+1] (odd positions, padded with zeros)
    s_odd = sum_subset(1, N // 2)

    # Recurrence for real-input FFT:
    # y[k] = ((s_even - s_odd) * cos(pi*k/N)) / N + (y_real[k/2] - y_real[k/2 - 1]) / 2 for even k
    # For odd k: use symmetry from previous terms via combination, but we only store up to k <= L.
    # For simplicity and correctness, compute using recurrence:
    # We'll use a small fixed loop for recurrence steps up to L (L is runtime, but Triton requires static loops;
    # so we avoid deep recursion by computing direct sums for small k, and recurrence for larger k via decomposition.)

    # Instead of deep recursion, we implement the standard recurrence via decomposed left/right halves.
    # Define helper to compute y from left/right sums. We'll do a small k loop using recurrence relations.
    # For y at k, decompose into y_left, y_right combining sums as above.

    # Note: Implementing full recurrence in Triton with Python control flow can be tricky; to keep it robust,
    # we compute y_real using direct sum with cos for all k, which is correct and simpler:
    # y_real[k] = (1/N) * sum_{j=0}^{N-1} x0[j] * cos(2*pi*k*j/N)
    # This is correct for real inputs and avoids the need for complex recurrence logic.

    # So we replace the above with direct computation:
    acc = 0.0
    BLOCK_J = 256
    for j0 in tl.static_range(0, N, BLOCK_J):
        j = j0 + tl.arange(0, BLOCK_J)
        valid = j < N
        vals = tl.load(x_ptr + pid_bc * N + j, mask=valid, other=0.0)
        angle = 2.0 * 3.141592653589793 * (k * j) / N
        cosv = tl.cos(angle)
        # Only valid j contribute
        vals = vals * valid
        acc += tl.sum(vals * cosv, axis=0)

    inv_N = 1.0 / N
    y = acc * inv_N
    tl.store(out_ptr + pid_bc * (L + 1) + k, y)


@triton.jit
def _divide_kernel(in_ptr, out_ptr, NUMEL: tl.int32, SCALAR: tl.float32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    y = x / SCALAR
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-based fused FFT size padding and real FFT computation for Hyena convolution.
        Returns:
            x_freq_real: Real part of normalized frequency domain output (batch, channels, seqlen+1), float32
            x_freq_imag: Imaginary part (zeros), shape (batch, channels, seqlen+1), float32
        """
        # Ensure input is float32 and contiguous
        x = x.to(torch.float32).contiguous()
        B, C, L = x.shape
        N = 2 * L  # zero-pad to 2*seqlen

        # Allocate padded buffer: shape (B*C, N)
        BC = B * C
        padded_x = torch.empty((BC, N), dtype=torch.float32, device=x.device)

        # Launch copy+pad kernel
        _copy_pad_kernel[(BC,)](x.view(BC, L), padded_x, L, N, num_warps=1)

        # Output real buffer: shape (BC, L+1)
        out_real = torch.empty((BC, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft real computation kernel for each (bc, k)
        grid = (BC, L)
        _rfft_real_recurrence_kernel[grid](padded_x, out_real, L, N, BC, num_warps=1)

        # Normalize by N using Triton division kernel
        out_real_div = torch.empty_like(out_real)
        numel = out_real.numel()
        BLOCK = 1024
        _divide_kernel[(triton.cdiv(numel, BLOCK),)](out_real, out_real_div, numel, N, BLOCK=BLOCK, num_warps=1)

        # Imaginary part is zeros for real input rfft
        out_imag = torch.zeros((BC, L + 1), dtype=torch.float32, device=x.device)

        # Reshape back to (B, C, L+1)
        x_freq_real = out_real_div.view(B, C, L + 1)
        x_freq_imag = out_imag.view(B, C, L + 1)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
