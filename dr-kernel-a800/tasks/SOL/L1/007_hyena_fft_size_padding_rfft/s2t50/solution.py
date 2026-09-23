import torch
import triton
import triton.language as tl


@triton.jit
def pad_and_run_kernel(
    x_ptr,            # *float32, input x flattened over (B, C, L)
    out_real_ptr,     # *float32, output real flattened over (B, C, L+1)
    out_imag_ptr,     # *float32, output imag flattened over (B, C, L+1)
    L: tl.constexpr,      # int32, seqlen
    two_L: tl.constexpr,  # int32, 2 * seqlen
    B: tl.constexpr,      # int32, batch size (passed for index math)
    C: tl.constexpr,      # int32, channels
    b: tl.constexpr,      # int32, current batch index for this program
):
    # We process all channels for this batch index in this single program.
    # Base offset for this batch in the flattened input is b * (C * L).
    # However, to simplify addressing, we recompute offsets per channel.
    # Note: We'll loop over channels and compute DFT for each (b, c) slice.

    # We'll build the zero-padded input vector inside the kernel: length = two_L.
    # Allocate temporary padded vector (we cannot use torch.cat here; use Triton tl.zeros).
    # Create a temporary buffer of length two_L for this (b, c) pair. We'll recompute it for each c.

    # We'll compute DFT for each channel c in [0..C).
    for c_idx in range(0, C):
        # Compute base input offset for this (b, c_idx) slice in the flattened x: start at offset = (b * C + c_idx) * L
        # But since we flattened x to a single vector of length B*C*L, we can directly index using c_idx.
        # To reconstruct pointer, we need to compute the offset. For simplicity, we pass x as [B*C*L] and compute base = b * (C * L) + c_idx * L.
        # However, because x_ptr is provided flattened as B*C*L, we need to know how to index. A simpler approach is to pass x as [B, C, L] contiguous and treat x_ptr as a 3D pointer. Triton kernels prefer 1D pointers; thus we flatten x to 1D and compute offsets as (b * C + c) * L + t.
        # Here, we use the flattened x: offset = (b * C + c_idx) * L.
        base_in = (b * C + c_idx) * L

        # Allocate temporary padded vector of length two_L (float32 zeros)
        # Triton supports creating vectors with tl.zeros(shape, dtype). We'll use a 1D vector.
        two_L_vec = tl.full((two_L,), 0.0, tl.float32)

        # Copy first L elements from x to the padded vector
        for t in range(0, L):
            x_t = tl.load(x_ptr + base_in + t)
            two_L_vec[t] = x_t

        # Now compute DFT for k in [0..L]
        for k in range(0, L + 1):
            acc_real = 0.0
            acc_imag = 0.0
            for t in range(0, two_L):
                x_t = two_L_vec[t]
                angle = -2.0 * 3.141592653589793 * float(k) * float(t) / float(two_L)
                cos_part = tl.cos(angle)
                sin_part = tl.sin(angle)
                acc_real += x_t * cos_part
                acc_imag -= x_t * sin_part  # imaginary contribution

            # Normalize by 2*L
            norm = 1.0 / float(two_L)
            acc_real = acc_real * norm
            acc_imag = acc_imag * norm

            # Compute output offset for (b, c_idx, k). Output is flattened over (B, C, L+1).
            # Flatten index = (b * C + c_idx) * (L + 1) + k
            out_base = (b * C + c_idx) * (L + 1) + k
            tl.store(out_real_ptr + out_base, acc_real)
            tl.store(out_imag_ptr + out_base, acc_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-optimized version of run(x):
        - Pads each (b, c) slice to 2*L and computes rfft along last dim (length L+1).
        - Returns real and imaginary parts, both float32 tensors of shape (B, C, L+1), normalized by 2*L.
        """
        # Ensure input is contiguous and float32
        x = x.contiguous().to(torch.float32)

        B, C, L = x.shape
        two_L = 2 * L

        # Flatten x to 1D: total elements = B * C * L
        x_flat = x.view(-1)  # 1D view over (B*C*L)

        # Allocate outputs: flatten over (B, C, L+1)
        out_real = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, L + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per batch element; inside program, handle all channels
        grid = (B,)
        pad_and_run_kernel[grid](
            x_flat, out_real.view(-1), out_imag.view(-1),
            L=L, two_L=two_L, B=B, C=C,
            num_warps=4,
        )

        # The outputs are already normalized inside the kernel; no further scaling needed.
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
