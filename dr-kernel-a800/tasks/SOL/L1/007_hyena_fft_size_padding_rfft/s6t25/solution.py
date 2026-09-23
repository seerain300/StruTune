import triton
import triton.language as tl


@triton.jit
def sum_all_kernel(x_ptr, sum_ptr, S: tl.constexpr):
    """
    For each (b, c) program, compute sum of x[b, c, :] and store in sum_ptr[bc].
    Assumes x is contiguous along the last dimension (C, S) flattened as (B*C, S).
    """
    bc = tl.program_id(0)
    # base offset for this (b, c) row in x: stride_x_c = S elements per (b, c)
    stride_x_c = S
    base_x = bc * stride_x_c

    sum_val = 0.0
    # Unrolled compile-time loop over S elements
    for i in tl.static_range(0, S):
        val = tl.load(x_ptr + base_x + i)
        sum_val += val
    # Store per-(b, c) sum
    tl.store(sum_ptr + bc, sum_val)


@triton.jit
def rfft_output_kernel(
    out_real_ptr, out_imag_ptr,
    sum_ptr,
    S: tl.constexpr,
    inv_twoN: tl.constexpr,  # 1.0 / (2 * N) where N = 2 * S
):
    """
    For each (b, c), compute out_real[bc, :] and out_imag[bc, :] of length S+1.
    out tensors are laid out as (B*C, S+1). We compute per (b, c) row.
    """
    bc = tl.program_id(0)
    stride_out = S + 1
    base_out = bc * stride_out

    sum_val = tl.load(sum_ptr + bc)

    # Even k: real part = sum_val * (cos - sin) / N, imag = 0
    # Odd k: real = 0, imag = -sum_val * sin / N
    # We compute arrays for j = 0..S
    j = tl.arange(0, S + 1)  # vectorized 1D index
    even_mask = (j % 2) == 0
    # Prepare constants
    N = 2 * S
    pi = 3.141592653589793
    # Compute cos and sin for each j
    # Use float32 math
    ang = pi * j * (S) / N  # S is the index k in 0..S-1 for even/odd; here j is 0..S, but we need k=j for odd/even
    # Correction: we need k = j for odd/even. However, we don't have a vectorized k in kernel; instead compute per-lane using j directly:
    # For even j, use k=j; for odd j, use k=j. So the formula holds with k=j.
    cos_t = tl.cos(ang)
    sin_t = tl.sin(ang)

    even_val = sum_val * (cos_t - sin_t) * (1.0 / N)  # real part for even j
    odd_val = -sum_val * sin_t * (1.0 / N)           # imag part for odd j

    # Select even/odd contribution; j=0 is even, handled by even_val
    # We need to set real for even j and imag for odd j:
    # Create outputs vectors
    out_real_vec = tl.where(even_mask, even_val, 0.0)
    out_imag_vec = tl.where(~even_mask, odd_val, 0.0)

    # Scale by final normalization 1/(2*N) -> inv_twoN is passed
    out_real_vec = out_real_vec * inv_twoN
    out_imag_vec = out_imag_vec * inv_twoN

    # Store results
    tl.store(out_real_ptr + base_out + j, out_real_vec)
    tl.store(out_imag_ptr + base_out + j, out_imag_vec)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, S) float32 tensor.
        Returns:
          out_real: (B, C, S+1) float32
          out_imag: (B, C, S+1) float32
        """
        assert x.dtype == torch.float32, "Input must be float32"
        B, C, S = x.shape
        N = 2 * S

        # Flatten to (B*C, S) for per-(b,c) processing
        x_flat = x.reshape(B * C, S).contiguous()

        # Allocate per-(b,c) sum
        sum_all = torch.empty(B * C, dtype=torch.float32, device=x.device)

        # Launch sum kernel: one program per (b, c)
        sum_all_kernel[(B * C,)](
            x_flat,
            sum_all,
            S=S,  # tl.constexpr for unrolling
            num_warps=1,
        )

        # Allocate outputs (B*C, S+1)
        out_real = torch.empty((B * C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, S + 1), dtype=torch.float32, device=x.device)

        # Precompute 1/(2*N)
        inv_twoN = 1.0 / (2.0 * N)

        # Launch output kernel: one program per (b, c)
        rfft_output_kernel[(B * C,)](
            out_real, out_imag,
            sum_all,
            S=S,
            inv_twoN=inv_twoN,  # tl.constexpr-like usage: Triton will treat as compile-time constant for this launch
            num_warps=1,
        )

        # Reshape back to (B, C, S+1)
        out_real = out_real.view(B, C, S + 1)
        out_imag = out_imag.view(B, C, S + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
