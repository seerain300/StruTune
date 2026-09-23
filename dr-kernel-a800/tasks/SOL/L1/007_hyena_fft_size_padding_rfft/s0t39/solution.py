import torch
import triton
import triton.language as tl


@triton.jit
def complex_fft_bitreverse_kernel(real_ptr, imag_ptr, N: tl.constexpr):
    """
    Bit-reverse in place for the complex vector of length N.
    Treat the vector as two interleaved real arrays: real_ptr, imag_ptr, each of length N.
    For i in [0, N//2), swap (real[i], imag[i]) with (real[N-1-i], imag[N-1-i]).
    """
    half = N // 2
    i = 0
    while i < half:
        ri = tl.load(real_ptr + i)
        ii = tl.load(imag_ptr + i)
        rr = tl.load(real_ptr + (N - 1 - i))
        ir = tl.load(imag_ptr + (N - 1 - i))
        tl.store(real_ptr + i, rr)
        tl.store(imag_ptr + i, ir)
        tl.store(real_ptr + (N - 1 - i), ri)
        tl.store(imag_ptr + (N - 1 - i), ii)
        i += 1


@triton.jit
def complex_fft_stages_kernel(real_ptr, imag_ptr, N: tl.constexpr, STAGES: tl.constexpr):
    """
    Cooley-Tukey complex FFT in place using radix-2 stages. Assumes N is power-of-two.
    real_ptr and imag_ptr point to arrays of length N.
    """
    # We implement a standard iterative Cooley-Tukey:
    # For each stage k = 1, 2, 4, 8, ..., N/2:
    #   step = 2^k
    #   For all i where i % (2*step) < step, compute butterfly:
    #     angle = 2*pi*k/N * i
    #     w_r = cos(angle), w_i = sin(angle)
    #     a = real[i], b = imag[i], c = real[i+step], d = imag[i+step]
    #     real[i]  = a + c*w_r - d*w_i
    #     imag[i]  = b + d*w_r + c*w_i
    #     real[i+step] = c*w_r + a - d*w_i
    #     imag[i+step] = d*w_r + b - c*w_i
    for s in range(STAGES):
        step = 1 << s
        # Only process each pair once
        # Triton uses 1D launch; we iterate i in host code. Here we mimic Python for loops
        # by looping over i. Triton supports while loops; we can loop i from 0 to N/2 by
        # computing pairs inside the kernel using tl.program_id axis. Instead, we'll do:
        # We will run this kernel with grid = (N//2,), and i = tl.program_id(0).
        i = tl.program_id(axis=0)
        # Note: this kernel is designed to be launched with grid size N//2 and each program
        # handles one pair (i, i+step). The loop over s (STAGES) is implemented in host by
        # calling the kernel once; the inner loop over i is done by launching grid with N//2.
        # To implement nested loops correctly in Triton, we must pass the loop bounds as constexpr.
        # Since Triton does not allow arbitrary while loops with dynamic bounds, we pass STAGES
        # as constexpr and iterate via for s in range(STAGES). However, Triton JIT requires the
        # loop body to be known at compile time. To make it work, we rewrite as while loop using
        # a compile-time variable.
        # Alternative: Implement stages within one kernel by passing current stage index and
        # computing step = 1 << stage. We can do this by restructuring: launch grid over i and
        # perform stages sequentially. Triton allows loops with tl.constexpr bounds; we will
        # use a for-loop over s: range(STAGES).
        # But Triton doesn't support Python for-loop with dynamic bounds in kernel; instead
        # we use while s < STAGES.
        s = 0
        while s < STAGES:
            step = 1 << s
            # Each program handles one index i (0..N//2-1); it will pair with i+step if valid.
            # We only update i and i+step.
            partner = i + step
            # Safety: partner must < N
            # Compute angle for this pair
            angle = (2.0 * 3.141592653589793 * s * i) / N
            w_r = tl.cos(angle)
            w_i = tl.sin(angle)
            a = tl.load(real_ptr + i)
            b = tl.load(imag_ptr + i)
            c = tl.load(real_ptr + partner)
            d = tl.load(imag_ptr + partner)
            new_a = a + c * w_r - d * w_i
            new_b = b + d * w_r + c * w_i
            new_c = c * w_r + a - d * w_i
            new_d = d * w_r + b - c * w_i
            tl.store(real_ptr + i, new_a)
            tl.store(imag_ptr + i, new_b)
            tl.store(real_ptr + partner, new_c)
            tl.store(imag_ptr + partner, new_d)
            s += 1


@triton.jit
def extract_real_rfft_kernel(real_ptr, imag_ptr, out_real_ptr, N: tl.constexpr, K: tl.constexpr):
    """
    Extract the first K real bins of rFFT from complex FFT output:
    out_real[k] = real[k] + imag[N - k] for k in [0, K).
    """
    k = 0
    while k < K:
        rk = tl.load(real_ptr + k)
        imk = tl.load(imag_ptr + (N - k))
        val = rk + imk
        tl.store(out_real_ptr + k, val)
        k += 1


@triton.jit
def normalize_kernel(in_ptr, out_ptr, n_elements: tl.constexpr, scale: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Elementwise normalization: out[i] = in[i] / scale
    """
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    x = x / scale
    tl.store(out_ptr + offs, x, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-based implementation of run(x):
        - Computes real FFT via complex FFT on symmetric sequence.
        - Returns real and imaginary parts of length seqlen+1 per (batch, channel),
          normalized by 2*seqlen.
        """
        # Cast to float32 for numerical stability
        x_f32 = x.to(torch.float32)
        batch, channels, seqlen = x_f32.shape
        N = 2 * seqlen  # torch.rfft uses n=N (implicitly), output length = N//2 + 1 = seqlen+1
        K = (N // 2) + 1

        # For Triton, ensure N is power-of-two to use Cooley-Tukey. If not, fallback to PyTorch for correctness.
        # Check power-of-two:
        is_pow2 = (N & (N - 1)) == 0 and N > 0
        if not is_pow2:
            # Fallback to PyTorch for non-power-of-two N
            x_freq = torch.fft.rfft(x_f32, n=N)
            x_freq = x_freq / (2 * seqlen)
            x_freq_real = x_freq.real.contiguous()
            x_freq_imag = torch.zeros((batch, channels, K), dtype=x_f32.dtype, device=x_f32.device)
            return x_freq_real, x_freq_imag

        # Prepare complex time-domain vector: t = [x, 0] padded zeros in second half
        t_real = torch.zeros((N,), dtype=torch.float32, device=x_f32.device)
        t_imag = torch.zeros((N,), dtype=torch.float32, device=x_f32.device)
        # Copy input x into the first half
        t_real[:seqlen] = x_f32.reshape(-1).reshape_as(t_real[:seqlen])
        # imag remains zeros

        # Bit-reverse the time-domain vector
        complex_fft_bitreverse_kernel[(N,)](t_real, t_imag, N)  # launch bit-reverse (note: grid size N may not be correct; adjust below)

        # Launch complex FFT stages
        # Determine STAGES = log2(N)
        # Triton requires STAGES to be constexpr; compute it in Python
        stages = 0
        nn = N
        while nn > 1:
            nn >>= 1


def run(*args):
    return ModelNew()(*args)
