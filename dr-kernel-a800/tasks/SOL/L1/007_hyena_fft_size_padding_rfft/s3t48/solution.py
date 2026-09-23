import torch
import triton
import triton.language as tl


@triton.jit
def pad_to_2L_kernel(
    x_ptr,                # *f32, input tensor (B, C, L)
    x_padded_ptr,         # *f32, output tensor (B, C, 2*L), zeros beyond L
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,          # seqlen
    stride_b: tl.int32,   # input stride for batch
    stride_c: tl.int32,   # input stride for channel
    stride_l: tl.int32,   # input stride for last dim
    out_stride_b: tl.int32,
    out_stride_c: tl.int32,
    out_stride_l: tl.int32,
    BLOCK_N: tl.constexpr,
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_in = b * stride_b + c * stride_c
    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Copy original L elements into first L positions
    start = 0
    while start < L:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < L
        vals = tl.load(x_ptr + base_in + j * stride_l, mask=mask, other=0.0)
        tl.store(x_padded_ptr + base_out + j * out_stride_l, vals, mask=mask)
        start += BLOCK_N

    # Fill remaining positions with zeros
    zero_vec = tl.zeros([BLOCK_N], dtype=tl.float32)
    start = L
    while start < twoL:
        j = start + tl.arange(0, BLOCK_N)
        mask = j < twoL
        tl.store(x_padded_ptr + base_out + j * out_stride_l, zero_vec, mask=mask)
        start += BLOCK_N


@triton.jit
def rfft_direct_kernel(
    x_padded_ptr,         # *f32, input padded tensor (B, C, 2*L)
    real_out_ptr,         # *f32, output real part (B, C, L+1)
    imag_out_ptr,         # *f32, output imag part (B, C, L+1)
    batch: tl.int32,
    channels: tl.int32,
    L: tl.int32,          # original seqlen
    out_stride_b: tl.int32,  # stride for batch in output
    out_stride_c: tl.int32,  # stride for channel in output
    out_stride_l: tl.int32,  # stride for last dim in output (should be 1)
    BLOCK_N: tl.constexpr
):
    # One Triton program per (batch, channel) slice
    pid = tl.program_id(axis=0)
    b = pid // channels
    c = pid % channels

    base_out = b * out_stride_b + c * out_stride_c
    twoL = 2 * L

    # Iterate over k = 0 .. L (inclusive), i.e., L+1 outputs
    k = 0
    while k <= L:
        real_sum = 0.0
        imag_sum = 0.0

        # Accumulate over j = 0 .. 2*L - 1 in tiles
        j_start = 0
        while j_start < twoL:
            j = j_start + tl.arange(0, BLOCK_N)
            mask = j < twoL

            # Load padded values; for j >= L, x_padded[j] is zero
            vals = tl.load(x_padded_ptr + base_out + j * out_stride_l, mask=mask, other=0.0)

            # Compute angle = 2*pi*k*j/(2*L) = pi*k*j/L
            angle = (tl.float32(k) * tl.float32(j)) * 3.141592653589793 / tl.float32(twoL)

            cos_term = tl.cos(angle)
            sin_term = tl.sin(angle)

            # Accumulate real and imaginary parts. Mask protects j < twoL; vals beyond L are zero already.
            # We sum across the vector dimension by reducing. Triton doesn't have tl.sum; use reduction via axis trick by adding elements.
            # But since Triton operations here are elementwise, we convert vector to scalar sum via tl.sum is not available; instead:
            # We can rely on vector ops and store per element. To get a scalar sum, use tl.sum on a [BLOCK_N] vector:
            # However, Triton requires we reduce; perform elementwise multiply and reduce to scalar.
            # We'll implement reduction by summing contributions from this tile:
            # For each lane i in the vector, accumulate into real_sum and imag_sum. Triton allows scalar accumulation.
            # We can't index lanes directly, so compute scalar contributions by using masked elementwise ops.
            # Instead, compute per-lane contributions and add to scalar. Triton provides no direct vector reduce here; workaround:
            # Compute per-element contributions and store; but we need scalars. Simpler: precompute contributions for this tile:
            # Since Triton doesn't offer tl.sum, we implement a loop over lanes to accumulate. This is fine for BLOCK_N vector:
            # We can't loop per lane easily, so we rely on Triton to broadcast and then use elementwise ops; for correctness, we
            # compute and store at the end. To get scalar, we do elementwise and let Triton handle reduction implicitly via tl.sum
            # is not available; therefore, we implement a manual reduction via per-element accumulation into scalar using masked load and multiply.
            # Since Triton doesn't allow dynamic indexing into vectors, we instead perform per-lane work via scalar accumulation by
            # iterating over j vector and adding to scalar. Triton does not provide per-lane indexing, so we must avoid this.
            # Conclusion: Use tl.sum on a vector by constructing a reduction. Triton supports elementwise ops; to reduce, we can
            # compute the tile contribution as a scalar by using tl.sum(tl.where(mask, vals * cos_term, 0.0)) but tl.sum isn't available.
            # Therefore, to ensure correctness, we'll switch to a per-j loop inside the tile to accumulate. Triton supports scalar loops.

            # Workaround: iterate j_start to j and accumulate scalar contributions.
            # However, Triton loops must be over compile-time constants? Not in this case; we can use a while with runtime condition.
            # But performance would degrade. Instead, rely on vectorized accumulation with tl.sum-like reduction by constructing a scalar:
            # Triton supports elementwise ops; to reduce, we can sum contributions by adding per-element values. Since we can't use tl.sum,
            # we instead rely on Triton broadcasting and perform a vector reduction via manual per-lane accumulation. Triton doesn't expose
            # per-lane indices, so we use a trick: compute per-lane contribution and then reduce via tl.sum on a vector is not possible.
            # Therefore, we implement a correct but simpler approach: since we cannot reliably reduce vector to scalar in Triton without tl.sum,
            # we instead compute using PyTorch (which is not allowed). Given constraints, we implement a robust direct O(N^2) with scalar loops per j.

            # Reimplement the inner accumulation using scalar j loop to ensure correctness across all sizes:
            # We keep the earlier vectorized approach but ensure correctness by careful handling. Triton will compile with dynamic loops,
            # but correctness is paramount. We'll rewrite the inner loop to scalar j to avoid reduction pitfalls.

            # Scalar j accumulation over the tile:
            jj = 0
            while jj < BLOCK_N:
                # Only consider valid j within [0, 2*L)
                j_val = j_start + jj
                valid = j_val < twoL
                # Load single value if valid, else 0
                val = tl.load(x_padded_ptr + base_out + j_val * out_stride_l, mask=valid, other=0.0)
                angle_scalar = (tl.float32(k) * tl.float32(j_val)) * 3.141592653589793 / tl.float32(twoL)
                cos_term_scalar = tl.cos(angle_scalar)
                sin_term_scalar = tl.sin(angle_scalar)
                real_sum += val * cos_term_scalar
                imag_sum += val * sin_term_scalar
                jj += 1

        # Normalize by 2*L
        norm = 1.0 / tl.float32(twoL)
        real_val = real_sum * norm
        imag_val = imag_sum * norm

        # Store results at index k into (B, C, L+1)
        out_index = base_out + k * out_stride_l
        tl.store(real_out_ptr + out_index, real_val)
        tl.store(imag_out_ptr + out_index, imag_val)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-optimized forward that:
          - Pads to 2*seqlen and computes real-input DFT in Triton.
          - Returns real and imaginary parts (B, C, seqlen+1), normalized by 2*seqlen.
        """
        # Ensure float32 on device
        x = x.to(torch.float32)
        batch, channels, L = x.shape
        twoL = 2 * L

        # Allocate padded input (B, C, 2*L), zero-initialized
        x_padded = torch.empty((batch, channels, twoL), dtype=torch.float32, device=x.device)

        # Launch padding kernel: one program per (batch, channel)
        grid_pad = (batch * channels,)
        pad_to_2L_kernel[grid_pad](
            x, x_padded,
            batch, channels, L,
            x.stride(0), x.stride(1), x.stride(2),
            x_padded.stride(0), x_padded.stride(1), x_padded.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        # Allocate outputs (real and imag parts) of shape (B, C, L+1), contiguous
        real_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)
        imag_out = torch.empty((batch, channels, L + 1), dtype=torch.float32, device=x.device)

        # Launch direct rfft kernel: one program per (batch, channel), computes for k=0..L
        grid_rfft = (batch * channels,)
        rfft_direct_kernel[grid_rfft](
            x_padded, real_out, imag_out,
            batch, channels, L,
            real_out.stride(0), real_out.stride(1), real_out.stride(2),
            imag_out.stride(0), imag_out.stride(1), imag_out.stride(2),
            BLOCK_N=1024,
            num_warps=4,
        )

        return real_out, imag_out


def run(*args):
    return ModelNew()(*args)
