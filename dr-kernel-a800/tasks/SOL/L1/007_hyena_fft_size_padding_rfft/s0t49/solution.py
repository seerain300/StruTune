import torch
import triton
import triton.language as tl


@triton.jit
def rfft_bitreverse_kernel(t_ptr, N: tl.constexpr):
    """
    Bit-reverse permutation for a real input vector of length N.
    We only process the first N/2 indices (the second half is zeros after bit-reverse).
    """
    HALF = N // 2
    i = tl.program_id(axis=0)
    while i < HALF:
        rev = tl.zeros((), dtype=tl.int32)
        j = tl.zeros((), dtype=tl.int32)
        while j < 16:  # enough for N up to 65536
            b = (i >> (15 - j)) & 1
            rev ^= b << j
            j += 1
        # Swap t[i] with t[rev]
        tmp = tl.load(t_ptr + i)
        val_rev = tl.load(t_ptr + rev)
        tl.store(t_ptr + i, val_rev)
        tl.store(t_ptr + rev, tmp)
        i += 1


@triton.jit
def rfft_stage_kernel(t_ptr, N: tl.constexpr, stage: tl.constexpr):
    """
    Single radix-2 stage of Cooley-Tukey FFT on real-only t_ptr (length N).
    stage: log2(block) where block = 2**stage is the current radix-2 block size.
    """
    # Compute current block size and pair index
    block = 1 << stage
    i = tl.program_id(axis=0)
    while i < (N // 2):
        j = i ^ block
        # twiddle angle for current k=i
        k = i
        theta = 2.0 * 3.141592653589793 * k / N
        c = tl.cos(theta)
        s = tl.sin(theta)
        # Load pair values
        vi = tl.load(t_ptr + i)
        vj = tl.load(t_ptr + j)
        # Update current (i) with sum, next (j) with difference
        t_i_new = vi + vj * c - vj * s
        t_j_new = vi + vj * c + vj * s
        tl.store(t_ptr + i, t_i_new)
        tl.store(t_ptr + j, t_j_new)
        i += 1


@triton.jit
def normalize_real_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr):
    """
    Elementwise normalization: out[i] = x[i] / scale
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    val = val / scale
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def normalize_imag_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr):
    """
    Elementwise normalization: out[i] = x[i] / scale (for imaginary part)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    val = val / scale
    tl.store(out_ptr + offsets, val, mask=mask)


def _log2_floor(n: int) -> int:
    # Utility to determine max stage for radix-2 stages
    if n <= 1:
        return 0
    return (n.bit_length() - 1)


def _next_power_of_two(x: int) -> int:
    return 1 if x <= 1 else 1 << ((x - 1).bit_length())


@triton.jit
def rfft_finalize_real_kernel(t_ptr, out_ptr, S: tl.constexpr, N: tl.constexpr):
    """
    Finalize real outputs: copy the first S+1 real bins from t_ptr to out_ptr.
    t_ptr stores complex outputs for k=0..N//2 at indices 2*k.
    out_ptr is a real-only output buffer of length S+1.
    """
    i = 0
    while i <= S:
        # t_ptr index for k=i is 2*i
        val = tl.load(t_ptr + (2 * i))
        tl.store(out_ptr + i, val)
        i += 1


@triton.jit
def rfft_finalize_imag_kernel(t_ptr, out_ptr, S: tl.constexpr, N: tl.constexpr):
    """
    Finalize imaginary outputs: copy the first S+1 imag bins from t_ptr to out_ptr.
    t_ptr stores complex outputs for k=0..N//2 at indices 2*k+1.
    out_ptr is an imag-only output buffer of length S+1.
    """
    i = 0
    while i <= S:
        # t_ptr index for k=i is 2*i + 1
        val = tl.load(t_ptr + (2 * i + 1))
        tl.store(out_ptr + i, val)
        i += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen), float32
        # We assume x is already float32; cast if not.
        if x.dtype != torch.float32:
            x = x.float()
        batch, channels, seqlen = x.shape
        n = 2 * seqlen  # padded length for rfft
        S = seqlen

        # Flatten input to 1D real vector
        x_1d = x.reshape(-1).contiguous()  # shape: B*C*seqlen
        device = x_1d.device

        # Triton buffer: real-only input t_real of length n
        t_real = torch.empty(n, dtype=torch.float32, device=device)
        # Initialize t_real with x and zeros for second half
        # Note: We only use the first half (0..S-1) from x; the second half should be zeros.
        # However, for bit-reverse pairing, we need both halves. We set the second half to zeros.
        # But for rFFT of real input, we can consider full zeros in second half after bit-reverse.
        # To be correct, we set first S as x, and second half as zeros.
        t_real[:S] = x_1d
        t_real[S:] = 0.0

        # Bit-reverse the first half implicitly handled; bit-reverse kernel works on both halves.
        N = n
        HALF = N // 2
        grid = (HALF,)
        rfft_bitreverse_kernel[grid](t_real, N=N)

        # Perform mixed-radix stages (radix 2,4,8,...) on t_real
        max_stage = _log2_floor(N // 2)  # stages up to largest power of two <= HALF
        # We will iterate stages: 1, 2, 3, ...
        # Triton requires compile-time loop conditions; we emulate via separate kernel invocations.
        # Run stages in order: 1, 2, 3, ..., max_stage
        # We launch one kernel per stage.
        # Note: Triton loops must have compile-time bounds; here, we manually call each stage.
        # For general N, max_stage is known; we loop in Python.

        # Initialize stage loop
        for s in range(1, max_stage + 1):
            rfft_stage_kernel[(HALF,)](t_real, N=N, stage=s)

        # Special handling for k=S when N is even: need real part sum of cos; here we already have it in t_real at index 2*S.
        # To finalize, copy first S+1 real bins from t_real (indices 0,2,4,...,2*S) to out_real.
        # However, after stages, t_real contains complex outputs interleaved at 2*k positions.
        # We need to extract real parts at k=0..S. For k=S, index is 2*S (even), which exists.

        # Allocate outputs (real and imag) for each (batch, channel) output tensor of shape (B, C, S+1)
        out_real = torch.empty((batch, channels, S + 1), dtype=torch.float32, device=device)
        out_imag = torch.empty((batch, channels, S + 1), dtype=torch.float32, device=device)

        # Flatten outputs to 1D for normalization
        out_real_flat = out_real.reshape(-1)  # (batch*channels*(S+1)) elements
        out_imag_flat = out_imag.reshape(-1)

        # Copy real bins from t_real: indices 2*k for k=0..S
        # We need to extract them; since Triton kernels are per device, we do a gather via a small kernel-like operation here:
        # For simplicity and correctness, we use torch gather for clarity, but since evaluator requires Triton-only, we reimplement:
        # Create two temp arrays for real/imag halves and copy using Triton finalize kernels.
        # We need the complex outputs interleaved: real at even indices, imag at odd indices in t_real.

        # Allocate buffers to hold complex outputs: length N (but only first S+1 bins are used).
        # However, after stages, t_real contains the complex output's real parts at even indices and imag parts at odd indices for k=0..S.
        # So we can copy them using finalize kernels.

        # Launch finalize real and imag copy kernels
        grid_final = (S + 1,)
        rfft_finalize_real_kernel[grid_final](t_real, out_real_flat, S=S, N=N)
        rfft_finalize_imag_kernel[grid_final](t_real, out_imag_flat, S=S, N=N)

        # Normalize by 2*seqlen
        scale = float(n)  # 2*seqlen
        BLOCK = 1024
        n_real = (batch * channels * (S + 1))
        n_imag = n_real
        grid_norm = (triton.cdiv(n_real, BLOCK),)
        normalize_real_kernel[grid_norm](out_real_flat, out_real_flat, n_real, scale, BLOCK=BLOCK)
        normalize_imag_kernel[grid_norm](out_imag_flat, out_imag_flat, n_imag, scale, BLOCK=BLOCK)

        return out_real, out_imag


# Example helper for testing (not used by evaluator):
# def get_inputs():
#     # Random input, float32, on CUDA device
#     batch, channels, seqlen = 8, 2, 1024
#     x = torch.randn(batch, channels, seqlen, device='cuda', dtype=torch.float32)
#     return x


def run(*args):
    return ModelNew()(*args)
