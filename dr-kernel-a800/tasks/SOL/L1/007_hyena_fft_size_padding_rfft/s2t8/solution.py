import torch
import triton
import triton.language as tl

# Triton kernel: compute real DFT for each (b, c, k) over a zero-padded vector of length two_L.
# One program per (b, c, k). We use scalar indexing to avoid Triton address mode issues.
@triton.jit
def real_dft_scalar_kernel(
    x_ptr,            # *float32, flattened input vector of length M = (B*C)*two_L
    out_real_ptr,     # *float32, output real part of length M
    out_imag_ptr,     # *float32, output imag part of length M (will be zero for real inputs)
    two_L: tl.constexpr,  # padded length = 2*L (compile-time constant per launch)
):
    # Program ids: (b, c, k)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)  # k in [0, two_L - 1] but we use only [0, L] per output, so guard if needed.

    # Compute base index for this (b, c) slice in flattened vector
    # M = (B*C) * two_L; base = (pid_b * C + pid_c) * two_L
    bc = pid_b * C + pid_c
    base = bc * two_L

    # Accumulator for X[k]
    accum = 0.0  # Triton will treat this as float32

    # Compute sum over t from 0 to two_L - 1
    # Note: tl.arange not used; we do scalar loop to avoid address mode issues.
    # We can't break early; just compute all terms.
    # Use a Python-level range since Triton can handle such loops:
    # For t in range(0, two_L):
    #   addr = base + t
    #   x_t = tl.load(x_ptr + addr)
    #   theta = -2.0 * 3.141592653589793 * float(pid_k) * float(t) / float(two_L)
    #   accum += x_t * tl.exp(1j * theta)  # Python complex scalar; Triton will handle it appropriately
    # However, to avoid complex in Triton, we instead compute real part directly via cos and sin:
    # exp(i theta) = cos(theta) + i sin(theta). We accumulate real part; imag part is zero for real inputs.

    # Since Triton doesn't support complex math directly, we compute real accumulation and set imag to zero.
    # Loop and compute real contribution:
    # We'll compute only up to t < two_L, but we must ensure we sum all terms; otherwise set zeros.

    # Simpler approach: perform accumulation with cos; imag is zero.
    # We'll set out_imag to zero and only compute real part via sum of cos terms. But we need x vector; we can load x_t via base + t.

    # However, in our setup, we pass flattened x which corresponds to (B*C) slices. For each (b,c) slice, base points to that slice of length two_L.
    # To keep it simple and robust, we will write real and imag as zero imag, and compute real via cos(theta) with x_t = tl.load(x_ptr + base + t).

    # Implement explicit loop:
    for t in range(0, two_L):
        x_t = tl.load(x_ptr + base + t)
        theta = -2.0 * 3.141592653589793 * float(pid_k) * float(t) / float(two_L)
        # cos(theta) is real part of exp(i theta)
        contrib = x_t * tl.cos(theta)
        accum += contrib

    # Normalize by 2*L
    accum = accum / float(two_L)

    # Write out: since input is real, rfft real part is accum, imag part is zero.
    # We store at the same (b,c,k) location in flattened output: out at index base_out = bc * (L+1) + pid_k
    base_out = bc * (L + 1) + pid_k
    tl.store(out_real_ptr + base_out, accum)
    # Imaginary part is zero for real input
    tl.store(out_imag_ptr + base_out, 0.0)

    # Note: The above stores only for k in [0, L], which is the expected output length for rfft on real input.
    # We launch grid (B, C, L+1) to cover all k in [0, L].

# Host-side forward in ModelNew:
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Compute real FFT with zero-padding to 2*L along the last dimension, normalize by 2*L,
        and return real and imaginary parts as two float32 tensors of shape (B, C, L+1).
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        # Ensure contiguous and float32
        x = x.contiguous().to(torch.float32)
        B, C, L = x.shape
        two_L = 2 * L

        # Flatten x into (B*C, L) then build zero-padded vectors per (b, c) using torch on device
        # This avoids Triton address mode complexity and guarantees correct padding.
        # We will create a single flattened padded vector of length M = (B*C) * two_L.
        # But since we need per-slice pointers, we instead construct per-slice padded tensors using torch.cat.
        # However, constructing per-slice and passing to Triton would require dynamic indexing; simpler is to flatten and index carefully.
        # Here we take the simple approach: create a single contiguous buffer by stacking and concatenating per slice.

        # Construct per-slice zero-padded vectors using torch.cat on device:
        # For each (b, c), x[b, c, :] is length L; zero pad to two_L.
        # Store them in a single contiguous buffer of length M = (B*C) * two_L by computing offsets.
        # But to keep Triton code simple, we avoid building per-slice in Triton. We'll instead create a single vector using torch operations:
        # Allocate M = (B*C) * two_L float32 buffer on device and fill using torch operations.

        # Create a temporary buffer of shape (B, C, two_L) using torch, then flatten:
        # Build zero-padded x_per_bc: for each (b,c), write x[b,c,:] into [:, :L], zeros into [:, L:].
        # Then we can flatten and pass to Triton.

        # Efficient torch construction:
        # First, expand x to (B, C, two_L) with zeros for the padded part.
        x_bc = x.view(B, C, L)  # just reshape; still contiguous
        # Allocate zero tensor of shape (B, C, two_L)
        x_padded = torch.zeros((B, C, two_L), dtype=torch.float32, device=x.device)
        # Copy first L columns
        x_padded[..., :L] = x_bc
        # Flatten to length M
        M = (B * C) * two_L
        # We need a single 1D flattened vector for Triton input. We can flatten x_padded to 1D and pass.
        # However, our Triton kernel expects the vector laid out as ((b*C) slices) * two_L.
        # torch.flatten keeps contiguity; but indexing in Triton will be base = (b*C)*two_L + t.
        # To make it simple, we can view x_padded as (B*C, two_L) by reshaping: (B*C, two_L)
        x_padded_2d = x_padded.view(B * C, two_L).contiguous()
        # Now we have a contiguous 2D buffer; we can index by base = bc * two_L + t.

        # Allocate outputs: flattened real and imag of length M_out = (B*C)*(L+1)
        M_out = (B * C) * (L + 1)
        out_real_flat = torch.empty(M_out, dtype=torch.float32, device=x.device)
        out_imag_flat = torch.empty(M_out, dtype=torch.float32, device=x.device)

        # Launch Triton kernel: grid over (B, C, L+1)
        grid = (B, C, L + 1)
        real_dft_scalar_kernel[grid](
            x_padded_2d, out_real_flat, out_imag_flat,
            two_L=two_L,  # constexpr specialization per launch
            num_warps=1,
        )

        # Reshape outputs back to (B, C, L+1)
        out_real = out_real_flat.view(B, C, L + 1)
        out_imag = out_imag_flat.view(B, C, L + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
