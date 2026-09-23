import torch
import triton
import triton.language as tl


@triton.jit
def sum_all_kernel(x_ptr, sum_ptr, S: tl.constexpr):
    """
    Compute per-(b,c) sum of a vector of length S.
    Grid: 1 program per (b,c)
    x_ptr points to a [B*C, S] flattened layout; we index it as sum_ptr[bc] loads one scalar at a time.
    But to keep it simple and safe, we pass x as (B, C, S) and index bc-specific slice manually in host.
    """
    # One program per (b,c)
    bc = tl.program_id(0)
    # We'll iterate over S elements to accumulate the sum.
    total = 0.0
    for i in tl.static_range(0, S):
        # Load x[bc, i] (assuming x was made contiguous as (B,C,S) and we pass bc-labeled slice via stride).
        # However, simpler approach: host will pass a tensor of shape (B*C, S), and each program reads its own slice.
        # The pointer arithmetic below assumes x_ptr points to flattened (B*C, S) and bc selects that row.
        # We need to load x_ptr[bc, i] correctly; Triton supports loading from pointer + offset.
        # To do that robustly, we set up the correct per-program base pointer. In practice, host will pass
        # x as (B, C, S) and we compute pointer as base = bc * S + i.
        # Since we cannot access x here, we instead rely on host to call this kernel on a tensor of shape (B*C, S)
        # where each row corresponds to a (b,c). Then we can do:
        base = bc * S
        val = tl.load(x_ptr + base + i)
        total += val
    # Write the sum for this (b,c)
    tl.store(sum_ptr + bc, total)


@triton.jit
def rfft_real_imag_kernel(
    out_real_ptr, out_imag_ptr,
    sum_ptr,  # per-(b,c) scalar sum
    S: tl.constexpr,
    inv_twoN,  # scalar = 1.0 / (2 * (2*S))
    stride_out_bc,  # elements between (b,c) rows for output
):
    """
    For each (b,c), compute rfft real/imag outputs of length S+1 directly using real-input formulas,
    scaled by inv_twoN = 1 / (2 * (2*S)).
    Grid: 1 program per (b,c)
    """
    bc = tl.program_id(0)
    # Load sum_all for this (b,c)
    sum_all = tl.load(sum_ptr + bc)

    # Prepare indices j = 0..S for outputs (since length is S+1, but we'll handle j=S in code below).
    # We'll build vectors for j in chunks and use tl.where to select even/odd behavior.
    # Output stride between rows is stride_out_bc (elements), because out_real/out_imag are shape (B,C,S+1) contiguous.

    # We will compute outputs vectorized for j in [0, S], then handle j=S separately.
    # Create j vector for 0..S-1
    j_vec = tl.arange(0, S)
    k = j_vec  # k = j for the general formula

    # Compute angle in radians: angle = pi * k / (2*S)
    N = 2 * S
    angle = tl.pi * k / N

    cos_term = tl.cos(angle)
    sin_term = tl.sin(angle)

    # For general real-input rfft:
    # real = sum_all * (cos - sin)
    # imag = -sum_all * sin
    # But this is only valid for k in 1..S. For k=0, real = sum_all / N, imag = 0.
    # We'll first compute general for k=1..S-1 and then handle k=0 and k=S separately.
    # However, to avoid dynamic loops, we'll compute for j=1..S-1 and then set j=0 and j=S separately.

    # Compute for j=1..S-1
    # Note: We'll do this as a separate vectorized operation, but we need scalar for j=0 and j=S.
    # We'll construct outputs by initializing zeros and then assigning j=0 and j=1..S-1 and j=S.

    # Initialize outputs with zeros
    # We can't directly "memset" in Triton, so we set them one by one (no dynamic loop, only vectorized for j=1..S-1).
    # We need to store to out_real_ptr and out_imag_ptr for j=0..S. Triton allows indexing vector ops; we'll do:
    # For j=0..S-1:
    # real[j] = 0, imag[j] = 0
    # Then overwrite real[0], imag[0] and imag[S] appropriately.

    # We'll use a loop to set these; Triton supports static_range, and S is constexpr here.

    # Store j=0: real = sum_all / N, imag = 0
    real0 = sum_all / N
    imag0 = 0.0
    tl.store(out_real_ptr + bc * stride_out_bc + 0, real0 * inv_twoN)
    tl.store(out_imag_ptr + bc * stride_out_bc + 0, imag0 * inv_twoN)

    # Store j=1..S-1: general formula
    # For even k: imag=0; for odd k: imag = -sum_all * sin(angle) * inv_twoN; real = 0 if k>0 odd, else general formula doesn't apply here since we set even/odd below.
    # To avoid dynamic loops, we compute per j explicitly:
    # We'll loop j from 1 to S-1 using static_range
    for j in tl.static_range(1, S):
        angle_j = tl.pi * j / N
        cos_j = tl.cos(angle_j)
        sin_j = tl.sin(angle_j)
        # Even/odd detection
        is_even = (j % 2) == 0
        # real contribution: sum_all * (cos - sin)
        real_part = sum_all * (cos_j - sin_j)
        imag_part = -sum_all * sin_j
        # Apply inv_twoN scaling
        real_j = real_part * inv_twoN
        imag_j = imag_part * inv_twoN
        # Even k: real_j stored, imag_j = 0; Odd k: imag_j stored, real_j = 0
        # But general formula applies only for k>=1; rfft real outputs are real, imag outputs are 0 except at odd k.
        # Since we are computing both, we set:
        if is_even:
            tl.store(out_real_ptr + bc * stride_out_bc + j, real_j)
            tl.store(out_imag_ptr + bc * stride_out_bc + j, 0.0)
        else:
            tl.store(out_real_ptr + bc * stride_out_bc + j, 0.0)
            tl.store(out_imag_ptr + bc * stride_out_bc + j, imag_j)

    # j=S: real = 0, imag = -sum_all * sin(pi * S / (2*S)) / (2 * (2*S))
    # sin(pi * S / 2S) = sin(pi / 2) = 1
    angle_S = tl.pi * S / N
    sin_S = tl.sin(angle_S)  # equals 1.0
    imag_S = -sum_all * sin_S * inv_twoN  # = -sum_all / (2*(2*S))
    tl.store(out_real_ptr + bc * stride_out_bc + S, 0.0)
    tl.store(out_imag_ptr + bc * stride_out_bc + S, imag_S)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, S) float tensor
        Returns: (out_real, out_imag) each (B, C, S+1) float tensors.
        """
        B, C, S = x.shape

        # Ensure float32 and contiguous layout
        x = x.to(torch.float32)
        x = x.contiguous()

        # Compute per-(b,c) sum using Triton reduction: shape (B*C,)
        x_flat = x.reshape(B * C, S).contiguous()
        sum_all = torch.empty(B * C, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per (b,c)
        sum_all_kernel[(B * C,)](
            x_flat,  # pointer to flattened tensor
            sum_all,
            S,       # constexpr S for static_range
        )

        # Allocate outputs
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch output kernel: one program per (b,c)
        inv_twoN = 1.0 / (2.0 * (2.0 * S))
        stride_out_bc = (S + 1)  # since out tensors are contiguous in (b,c) rows, stride between rows is S+1

        rfft_real_imag_kernel[(B * C,)](
            out_real, out_imag,
            sum_all,
            S,
            inv_twoN,
            stride_out_bc,
            num_warps=1,
            num_stages=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
