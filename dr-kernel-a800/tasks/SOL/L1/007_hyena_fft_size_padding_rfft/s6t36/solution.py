import torch
import triton
import triton.language as tl


@triton.jit
def pad_and_copy_kernel(
    x_ptr,                 # *float32, input x flattened to [BC, S]
    zreal_ptr,             # *float32, output z real flattened to [BC, 4*S]
    S: tl.int32,
    BC: tl.int32,
    BLOCK: tl.constexpr,   # vectorization along S dimension
):
    # Each program handles one (b, c) row
    bc = tl.program_id(0)
    if bc >= BC:
        return

    # Build z = [x, 0 at index S, zeros at [S+1..2S-1], x_reversed at [2S..3S-1]]
    # z length = 4*S (real-only, imag is zero for all)
    total = 4 * S

    # First half: j=0..S-1 -> z[j] = x[j]
    # x_ptr layout: [BC, S], offset = bc*S + idx
    idx0 = tl.arange(0, BLOCK)
    mask0 = idx0 < S
    x_off = bc * S + idx0
    x_vals = tl.load(x_ptr + x_off, mask=mask0, other=0.0)
    z_off0 = bc * total + idx0
    tl.store(zreal_ptr + z_off0, x_vals)

    # Middle zeros: j=S -> z[S] = 0; j=S+1..2S-1 -> zeros
    # For simplicity and safety, write zeros for j=S+1..2S-1
    idx1 = S + tl.arange(0, BLOCK)
    mask1 = idx1 < (2 * S)
    # We don't have x for these positions; write zeros
    zero_vec = tl.zeros([BLOCK], dtype=tl.float32)
    z_off1 = bc * total + idx1
    tl.store(zreal_ptr + z_off1, zero_vec, mask=mask1)

    # Second half reversed: j=2S..3S-1 -> z[j] = x[S-1-j]
    idx2 = 2 * S + tl.arange(0, BLOCK)
    mask2 = idx2 < (3 * S)
    src_idx = S - 1 - tl.arange(0, BLOCK)
    # src_idx ranges 0..S-1; mask src < S
    src_mask = src_idx < S
    x_reversed = tl.load(x_ptr + bc * S + src_idx, mask=src_mask, other=0.0)
    z_off2 = bc * total + idx2
    tl.store(zreal_ptr + z_off2, x_reversed, mask=mask2)


@triton.jit
def bitrev_cooley_tukey_r2c_kernel(
    zreal_ptr,             # *float32, real part of input z (length 4*S)
    yr_ptr, yim_ptr,       # *float32, output real/imag parts (length 4*S)
    TWO_N: tl.int32,       # 2*N = 4*S
    STRIDE: tl.int32,      # elements per (b,c) row in outputs = 4*S
    STAGES: tl.int32,      # log2(TWO_N) stages for FFT
):
    # Single program per (b,c) row: we use flattened pointers here, so index by pid=0
    # Implement in-place Cooley-Tukey FFT on zreal_ptr -> write yr_ptr, yim_ptr.
    # We assume total length = TWO_N = 4*S (passed as argument).
    # Bit-reverse addressing: For each k in 0..TWO_N-1, compute bk = bitrev(k, STAGES),
    # load from zreal at bk, update at k. Since we need two arrays for real/imag outputs,
    # we will write to yr_ptr and yim_ptr accordingly. We only read zreal_ptr (real-only).
    # Note: This kernel assumes zreal_ptr contains interleaved real/imag for z, but here
    # z is purely real. We'll implement sin/cos-based mixing per stage to produce complex outputs.
    # However, Triton doesn't support complex; we will compute real and imag contributions
    # via sin/cos mixing and store into yr_ptr and yim_ptr.

    # Since Triton JIT requires static loops, we unroll stages manually using a compile-time list.
    # However, STAGES is runtime; Triton can handle while loops. We implement full Cooley-Tukey
    # using while loops and vectorized operations. For simplicity, we implement one full pass
    # using classical nested while structure (TWO_N iterations and half-step updates). This is
    # a canonical implementation; despite dynamic loops, Triton supports them.

    # We will implement the standard butterfly steps: for size = 2,4,8,... up to TWO_N
    # For each size, process all pairs (j, j+half) for half = size//2 down to 1.
    size = 2
    while size <= TWO_N:
        half = size // 2
        while half >= 1:
            j = 0
            while j < TWO_N:
                idx1 = j
                idx2 = j + half
                # Load real components
                a = tl.load(zreal_ptr + idx1)  # real part at idx1
                b = tl.load(zreal_ptr + idx2)  # real part at idx2
                # Compute angle for each pair using k = idx1, and half steps
                # Using angle = -2*pi * idx1 * half / (2*N) = -2*pi * idx1 * half / (4*S)
                two_pi = 6.283185307179586
                ang = two_pi * idx1 * half / TWO_N
                c = tl.cos(ang)
                s = tl.sin(ang)
                # Mixed real and imag contributions
                mixed_r = b * c + 0.0  # since b is real; imag part of b is 0
                mixed_i = -b * s
                # Update outputs: yr[idx1] = a + mixed_r; yim[idx1] = mixed_i
                # Note: The classical bit-reversed addressing would read from bk indices,
                # but here we can't access z at bk directly. So we compute using current j.
                # This is a simplification: we compute for each j and write to yr/yim at j.
                # For full bit-reversed FFT, we need to compute per bit-reversed index.
                # Implementing full bit-reversed in Triton is complex. Hence we switch to
                # direct mapping using known rfft properties instead of bit-reverse here.

                # Store results
                tl.store(yr_ptr + idx1, a + mixed_r)
                tl.store(yim_ptr + idx1, mixed_i)
                j += size
            half = half // 2
        size = size * 2


@triton.jit
def map_to_rfft_outputs_kernel(
    yr_ptr, yim_ptr,       # *float32, real/imag parts of FFT(z), length 4*S
    out_real_ptr, out_imag_ptr,  # *float32, final outputs for x, length S+1
    S: tl.int32,
    TWO_N: tl.int32,       # 4*S
):
    # For real input x, FFT(z) where z = [x, 0 at S, zeros at [S+1..2S-1], x_reversed at [2S..3S-1]]
    # The rfft output y for x has length S+1. We can reconstruct y using known relations.
    # For k = 0..S:
    #   If k is even: y[k] = (yr[k] + yr[2S + (S - k)]) / (2*2S)
    #   If k is odd:  y[k] has real = yim[k], imag = -yim[2S + (S - k)], scaled by 1/(2*2S)
    # But simpler and exact: compute real and imag parts directly:
    #   y[k].real = 0 for odd k, and for even k: (yr[k] + yr[2S + (S - k)]) / (4*S)
    #   y[k].imag = yim[k] for odd k; for even k: 0
    # However, a more accurate mapping from real rfft is:
    #   y[k] = sum_{t=0..2S-1} x[t] * (cos(2π k t / (2S)) - i sin(2π k t / (2S))) / (2S)
    # Since we don't have cos/sin here, we use the identity that rfft(x) can be obtained from
    # the DFT of z. The standard mapping is:
    #   y[k] = (FFT(z)[k] + complex conjugate of FFT(z)[k_even]) / (2*2S), for even k,
    #   y[k] = imaginary part mapping using yim. Instead, we implement exact direct formula:
    # We will compute using Triton vectorized approach per k.

    # We implement a simple vectorized mapping:
    # For k = 0..S, compute real and imag via known relations.
    # To keep it simple, we loop k in Triton using while. This avoids complex operations.
    k = 0
    while k <= S:
        # For even k: real = (yr[k] + yr[2S + (S - k)]) / (4*S); imag = 0
        # For odd k: real = 0; imag = yim[k] / (4*S)
        # Note: The exact mapping uses DFT of z. Since z is real, we can compute real rfft by:
        # y[k].real = 0 if odd; else (yr[k] + yr[2S + (S - k)]) / (4*S)
        # y[k].imag = yim[k] if odd; else 0
        # But to ensure correctness against PyTorch, we implement the direct summation formula
        # using Triton vectorized operations per k. However, since Triton doesn't support complex,
        # we compute only real or imag part using available data. Given complexity, we use
        # a hybrid approach: we compute real and imag via direct DFT formula using trig functions
        # which Triton provides. To avoid torch, we use a direct sum formula with precomputed
        # sin/cos, but that reintroduces torch. Hence, for correctness, we use the exact
        # identity mapping from FFT(z): For even k, y.real = (yr[k] + yr[2S + (S - k)]) / (4*S).
        # For odd k, y.imag = yim[k] / (4*S). The original code divides by 2*S, so we use 4*S here.

        # Even/odd check
        is_even = (k % 2 == 0)
        scale = 1.0 / (4.0 * S)  # since overall scaling in original is 1/(2*2S)

        if is_even:
            # Real part
            yr_k = tl.load(yr_ptr + k)
            # index idx2 = 2S + (S - k)
            idx2 = 2 * S + (S - k)
            yr_idx2 = tl.load(yr_ptr + idx2)
            out_real_k = (yr_k + yr_idx2) * scale
            out_imag_k = 0.0
        else:
            # Imaginary part
            yim_k = tl.load(yim_ptr + k)
            out_real_k = 0.0
            out_imag_k = yim_k * scale

        # Store at (b,c) row and position k
        # We assume out tensors are flattened as [BC, S+1]. For simplicity, we use bc=0 here.
        # But since we have per-(b,c) outputs, we need to know bc. Triton kernels usually handle
        # flattened outputs. Here, we assume out pointers are for a specific (b,c); however,
        # we don't have bc in this kernel. We'll pass outputs from a higher-level launcher
        # with per-(b,c) pointers. For clarity, we return. In ModelNew.forward, we call this
        # kernel per (b,c) using separate out buffers.

        # Since Triton kernel cannot return, we store directly to provided out buffers.
        # The caller must pass out_real_ptr and out_imag_ptr for the specific (b,c).

        # We cannot store here without bc; thus we define this as helper per-(b,c). Instead,
        # we'll implement a wrapper in Python that launches this per (b,c).

        k += 1


# The above bitrev_cooley_tukey_r2c_kernel is complex; given previous failures, we switch to
# a direct rfft via sin/cos summation in Triton, which is simpler to implement correctly.
# We'll use this in the forward path to avoid torch FFT.

@triton.jit
def direct_rfft_sum_kernel(
    x_ptr,                  # *float32, input x flattened to [BC, S]
    out_real_ptr,           # *float32, output real part flattened to [BC, S+1]
    out_imag_ptr,           # *float32, output imaginary part flattened to [BC, S+1]
    S: tl.int32,            # seqlen
    BC: tl.int32,           # total (b,c) rows
    BLOCK: tl.constexpr,    # vectorization block along S
):
    bc = tl.program_id(0)
    if bc >= BC:
        return

    N = 2 * S
    # We compute rfft(x) of length S+1 via direct summation:
    # y[k] = sum_{t=0..N-1} x[t] * (cos(2π k t / N) - i sin(2π k t / N)) / N
    # Original code divides by 2*S, so final scale is 1/(2*N).
    k = 0
    while k <= S:
        idx = tl.arange(0, BLOCK)
        mask = idx < S
        x_off = bc * S + idx
        x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)  # float32
        # Compute trig terms for current k
        two_pi = 6.283185307179586
        ang = two_pi * k * idx / N
        cosk = tl.cos(ang)
        sink = tl.sin(ang)
        # Compute sum over idx: real_sum = sum(x * cos), imag_sum = -sum(x * sin)
        real_sum = tl.sum(x_vals * cosk, axis=0)
        imag_sum = tl.sum(-x_vals * sink, axis=0)
        scale = 1.0 / (2.0 * N)
        real_sum = real_sum * scale
        imag_sum = imag_sum * scale
        out_real_off = bc * (S + 1) + k
        out_imag_off = bc * (S + 1) + k
        tl.store(out_real_ptr + out_real_off, real_sum)
        tl.store(out_imag_ptr + out_imag_off, imag_sum)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Input x: (B, C, S), float32
        x = args[0] if len(args) == 1 else args[0]
        assert x.dtype == torch.float32, "Input must be float32"
        B, C, S = x.shape
        BC = B * C

        # Prepare outputs: (B, C, S+1)
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Flatten for Triton
        x_flat = x.reshape(BC, S)
        out_real_flat = out_real.reshape(BC, S + 1)
        out_imag_flat = out_imag.reshape(BC, S + 1)

        # Launch direct rfft summation Triton kernel: 1D grid over BC
        grid = (BC,)
        direct_rfft_sum_kernel[grid](
            x_flat, out_real_flat, out_imag_flat,
            S, BC,
            BLOCK=1024,
            num_warps=4,
            num_stages=2,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
