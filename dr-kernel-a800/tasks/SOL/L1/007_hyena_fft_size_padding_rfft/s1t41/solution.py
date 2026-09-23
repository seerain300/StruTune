import triton
import triton.language as tl


@triton.jit
def _rfft_real_imag_triton_kernel(
    x_ptr,                  # *const float32, input pointer to x (B, C, L) contiguous
    out_real_ptr,           # *float32, output pointer to real part (B, C, L+1)
    out_imag_ptr,           # *float32, output pointer to imag part (B, C, L+1)
    B: tl.constexpr,        # batch size (for grid only)
    C: tl.constexpr,        # channels (for grid only)
    L: tl.constexpr,        # seqlen (runtime value, used to compute N and M)
    N: tl.constexpr,        # N = 2 * L (runtime value)
    M: tl.constexpr,        # M = L + 1 (runtime value)
):
    # Each program handles one (b, c) slice
    pid = tl.program_id(axis=0)  # 0 .. (B*C - 1)
    c = pid % C
    b = pid // C

    # Base offset for this (b, c) slice in input x (contiguous across L)
    # x is laid out as (B, C, L) contiguous => offset = (b*C + c) * L
    base = (b * C + c) * L

    # Precompute invN = 1 / N for scaling
    invN = 1.0 / N

    # We'll compute for j = 0..M-1 and store to output at (b, c, j)
    # Output is laid out as (B, C, M) contiguous => offset_out = (b*C + c)*M + j
    # Loop over j; vectorize j within the kernel to avoid 2D indexing pitfalls
    # Keep j as scalar to avoid Triton tensor indexing issues
    # We use a single kernel instance to compute all j by looping; Triton supports Python-level loops in kernels.

    # Loop j from 0 to M-1
    j = 0
    while j < M:
        # Accumulate real and imaginary parts
        sum_re = 0.0
        sum_im = 0.0

        # Loop over t = 0..N-1
        t = 0
        while t < N:
            # Load x[t] (as float32); x_ptr points to base of this slice
            x_val = tl.load(x_ptr + base + t)
            # Compute cos and sin for current j and t
            # tl.cos/tl.sin require float arguments; ensure float
            angle = (2.0 * 3.141592653589793) * float(j) * float(t) / float(N)
            sum_re += x_val * tl.cos(angle)
            sum_im += x_val * tl.sin(angle)
            t += 1

        # Normalize by N and store
        re_val = sum_re * invN
        im_val = sum_im * invN

        # Store to out_real and out_imag at (b, c, j)
        out_base = (b * C + c) * M
        tl.store(out_real_ptr + out_base + j, re_val)
        tl.store(out_imag_ptr + out_base + j, im_val)

        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
        - Cast to float32 (implicitly, as Triton will treat input as float32; no torch ops allowed).
        - Compute rfft(x, n=2*seqlen) along the last dim for each (batch, channel) slice, then normalize by 2*seqlen.
        - Return real and imaginary parts separately, each of shape (batch, channels, seqlen+1).
        """
        assert x.dim() == 3, "Input must be of shape (batch, channels, seqlen)"
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Ensure input is contiguous and use float32 (Triton expects float32 pointers; we'll pass x as float32 view)
        # Important: No torch operations here; just ensure contiguity. We can call .contiguous() but it's a no-op if already contiguous.
        x_contig = x.contiguous()

        # Allocate outputs
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        _rfft_real_imag_triton_kernel[grid](
            x_contig, out_real, out_imag,
            B=B, C=C, L=L, N=N, M=M,
            num_warps=1,  # conservative launch; correctness first
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
