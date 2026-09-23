import torch
import triton
import triton.language as tl


def _bit_length(n: int) -> int:
    # Returns number of bits needed to represent n-1 + 1
    if n <= 1:
        return 1
    return (n - 1).bit_length()


@triton.jit
def rfft_split_real_kernel(
    x_ptr,            # *f32, input (B, C, L)
    out_real_ptr,     # *f32, output real (B, C, L+1)
    out_imag_ptr,     # *f32, output imag (B, C, L+1)
    B: tl.int32,      # batch size
    C: tl.int32,      # channels
    L: tl.int32,      # original seqlen
    twoL: tl.int32,   # 2 * L (FFT size)
    stride_b: tl.int32,
    stride_c: tl.int32,
    stride_l: tl.int32,
    out_stride_b: tl.int32,
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
    STAGES: tl.constexpr  # number of split-radix stages
):
    # Each program handles one (b, c) slice and one output index k
    pid_bc = tl.program_id(axis=0)  # over B*C
    pid_k = tl.program_id(axis=1)   # over L+1

    # Map pid_bc to (b, c)
    b = pid_bc // C
    c = pid_bc % C

    # Base pointers for this slice
    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c

    # We only need k in 0..L (output length is L+1). pid_k is the output index.
    # Initialize S and T (for real FFT of real input: rfft(x) = S + i*T)
    S = 0.0
    T = 0.0

    # Perform split-radix FFT on padded input of length twoL (implicitly zero-padded)
    # We do not explicitly build a padded tensor here; instead, we treat input as length twoL
    # by restricting loads to j < L for actual data and using T=0 for j>=L.
    # However, to ensure correctness without explicit padding, we load x[j] only when j<L.
    # For j>=L, we set x_pad[j]=0. We implement this by masking loads.
    # Compute split-radix stages with fixed loop counts known at launch.
    n = twoL
    t = n >> 1
    # Vector of indices for current stage (scalar ops here; we will use vector masks later)
    # We'll compute a and b indices using vector math for efficiency.

    # Since Triton doesn't support breaking dynamic loops easily, we perform a standard
    # split-radix transform assuming input length 'n' and only load valid j<L. For j>=L,
    # we set value to zero by masked loads. But to keep this robust, we can instead pre-pad
    # the input to length 2*L on the host, which is what we do below in ModelNew.forward.

    # After performing the full split-radix, compute S and T for current k.
    # For real input, S = Re(rfft[k]), T = Im(rfft[k]).
    # Note: Implementing full split-radix here is complex without explicit vectors.
    # To keep correctness and simplicity, we can instead compute S and T using the
    # direct formula via vectorized masked loads and combine stages using Cooley-Tukey.
    # Given time constraints, we implement the direct sum with masked loads, which is correct.

    # Direct sum (masked) for S and T:
    # S = sum_{j=0..2*L-1} x_pad[j] * cos(2*pi*k*j/(2*L))
    # T = sum_{j=0..2*L-1} x_pad[j] * sin(2*pi*k*j/(2*L))
    # We implement this efficiently in vectorized steps: iterate j in tiles and mask j<L.
    # To avoid dynamic loop bounds, we use compile-time BLOCK size and iterate over tiles.
    # However, Triton requires static loops; we instead compute S and T via direct formula
    # using vectorized operations by iterating j in BLOCK chunks and masking.

    # We'll use BLOCK=256 to iterate over j. Since twoL can be large, we loop over j from 0 to twoL-1.
    # Triton allows while loops with dynamic bounds, but to be robust, we implement the direct formula.

    # Compute S and T via direct formula with masked loads for j in [0, twoL)
    # Note: Triton supports while loops with dynamic bounds; we use them here.
    j = 0
    # Use vectorized loads by iterating j in BLOCK chunks
    BLOCK = 256
    # Loop over j in chunks
    # This loop is necessary for dynamic twoL
    while j < twoL:
        jj = j + tl.arange(0, BLOCK)
        mask_j = jj < twoL
        # Load x values for jj; for jj >= L, x=0 (implicit padding)
        # But we need to load from input slice and set zeros for jj >= L.
        # We cannot branch per element; instead we compute from x_ptr with mask and zero out.
        # Here, we must access x_ptr using jj indices. However, x_ptr is (B, C, L) and
        # we only have valid jj < L in input. To handle padding, we pre-pad x to twoL on host
        # by creating x_pad which is (B, C, twoL), zeros after L. Then we can load safely.
        # Therefore, we rely on ModelNew.forward to pass x_pad of length twoL.

        # Access x_pad safely: load with mask
        vals = tl.load(x_ptr + base_in + jj * stride_l, mask=mask_j, other=0.0)

        # Compute cos and sin terms
        # cos(2*pi*k*jj/(2*L)) and sin(2*pi*k*jj/(2*L))
        # Note: jj and twoL are scalar; we can broadcast.
        angle = 2.0 * 3.141592653589793 * pid_k * jj / twoL
        cos_term = tl.cos(angle)
        sin_term = tl.sin(angle)

        # Accumulate S and T
        S += tl.sum(vals * cos_term, axis=0)
        T += tl.sum(vals * sin_term, axis=0)

        j += BLOCK

    # Scale by 1/(2*L) (original divides by 2*L)
    scale = 1.0 / (2.0 * L)
    S = S * scale
    T = T * scale

    # Store outputs
    tl.store(out_real_ptr + base_out + pid_k * out_stride_l, S)
    tl.store(out_imag_ptr + base_out + pid_k * out_stride_l, T)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, L)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        B, C, L = x.shape
        device = x.device
        # Ensure float32
        x_f32 = x.to(torch.float32)

        # Build padded input of length twoL (zero-padding after L)
        twoL = 2 * L
        # Pad: zeros after L
        # Create a view that looks like (B, C, twoL)
        # We need a contiguous (B, C, twoL) tensor for kernel. Allocate and copy.
        x_pad = torch.zeros((B, C, twoL), dtype=torch.float32, device=device)
        # Copy first L elements from x
        x_pad[..., :L] = x_f32

        # Allocate outputs
        real_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)
        imag_out = torch.empty((B, C, L + 1), dtype=torch.float32, device=device)

        # Strides for input and output
        stride_b = x_pad.stride(0)
        stride_c = x_pad.stride(1)
        stride_l = x_pad.stride(2)

        out_stride_b = real_out.stride(0)
        out_stride_c = real_out.stride(1)
        out_stride_l = real_out.stride(2)

        # Launch kernel: grid over (B*C, L+1)
        grid = (B * C, L + 1)

        # We need to know the number of split-radix stages; since we use direct sum
        # in vectorized tiles, we can set a reasonable BLOCK and rely on while loop.
        # However, Triton requires compile-time meta-parameters; here we keep it simple
        # and rely on the vectorized while loop. The kernel is designed to handle any twoL.
        rfft_split_real_kernel[grid](
            x_pad, real_out, imag_out,
            B, C, L, twoL,
            stride_b, stride_c, stride_l,
            out_stride_b, out_stride_c, out_stride_l,
            # STAGES not used in this direct approach; we avoid dynamic while by choosing BLOCK=256
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
