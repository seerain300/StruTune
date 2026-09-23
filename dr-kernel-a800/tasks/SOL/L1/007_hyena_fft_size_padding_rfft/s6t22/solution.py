import torch
import triton
import triton.language as tl


@triton.jit
def sum_all_kernel(x_ptr, sum_ptr, S, stride_bc):
    """
    Triton kernel: compute sum over last dim S for each (b, c) row.
    x_ptr points to a flattened (B*C, S) tensor.
    sum_ptr is output of shape (B*C,).
    """
    pid = tl.program_id(0)  # program id over B*C
    acc = 0.0
    # Scalar loop over S: Triton supports Python 'for' with dynamic bounds
    for idx in range(0, S):
        val = tl.load(x_ptr + pid * stride_bc + idx)
        acc += val
    tl.store(sum_ptr + pid, acc)


@triton.jit
def rfft_real_imag_kernel(out_real_ptr, out_imag_ptr, sum_ptr, S, inv_twoN, stride_out_bc, BLOCK_S: tl.constexpr):
    """
    Triton kernel: for each (b, c), compute rfft-like real/imag parts of length S+1 using formulas,
    apply final normalization inv_twoN = 1 / (2 * (2*S)).
    out_real_ptr/out_imag_ptr: outputs shaped (B*C, S+1). stride_out_bc = (S+1) for contiguous.
    sum_ptr: per-(b, c) sum of x, shape (B*C,).
    """
    pid = tl.program_id(0)  # program id over B*C
    sum_all = tl.load(sum_ptr + pid)  # scalar

    # We will compute for j = 0..S-1 using a single vectorized chunk of size BLOCK_S = S+1
    j_vec = tl.arange(0, BLOCK_S)
    mask_j = j_vec < S

    N = 2 * S
    pi = 3.141592653589793

    # Handle j==0 explicitly: k=0, real = sum_all / (2*S), imag = 0
    real0 = sum_all * 0.5 / S
    imag0 = 0.0

    # General j>0: ang = pi * j / N, real = sum_all * (cos(ang) - sin(ang)) * inv_twoN,
    # imag = -sum_all * sin(ang) * inv_twoN
    ang = pi * j_vec / N
    cos_term = tl.cos(ang)
    sin_term = tl.sin(ang)

    real_general = sum_all * (cos_term - sin_term) * inv_twoN
    imag_general = -sum_all * sin_term * inv_twoN

    # Override general j==0 with real0, imag0
    is_j0 = j_vec == 0
    real_vec = tl.where(is_j0, real0, real_general)
    imag_vec = tl.where(is_j0, imag0, imag_general)

    # Store results to outputs at indices j_vec
    out_row_base = out_real_ptr + pid * stride_out_bc
    tl.store(out_row_base + j_vec, real_vec, mask=mask_j)
    out_row_base_imag = out_imag_ptr + pid * stride_out_bc
    tl.store(out_row_base_imag + j_vec, imag_vec, mask=mask_j)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation:
        - No torch operations (not even .sum, torch.fft).
        - Compute rfft-like outputs for real input x of shape (B, C, S),
          return real and imaginary parts of shape (B, C, S+1), normalized by 2*S.
        """
        # Ensure float32 and contiguous
        x = x.to(torch.float32).contiguous()
        B, C, S = x.shape

        # Flatten to (B*C, S) for row-wise reduction
        x_flat = x.reshape(B * C, S)

        # Allocate sum buffer (B*C,)
        sum_all = torch.empty(B * C, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per (b, c) row
        sum_all_kernel[(B * C,)](
            x_flat,
            sum_all,
            S,
            x_flat.stride(0),  # stride between rows in elements (S for contiguous)
        )

        # Prepare outputs: (B, C, S+1)
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Final normalization factor 1 / (2 * (2*S)) for the original code's division by 2*S
        inv_twoN = 1.0 / (2.0 * (2 * S))

        # Stride between (b, c) rows in output (for contiguous layout): S+1
        stride_out_bc = (S + 1)

        # Launch output kernel: one program per (b, c)
        rfft_real_imag_kernel[(B * C,)](
            out_real,
            out_imag,
            sum_all,          # per-(b,c) sum of x (computed in Triton)
            S,
            inv_twoN,
            stride_out_bc,    # stride between rows in output (S+1 for contiguous)
            BLOCK_S=S + 1,    # vectorized chunk covers full S+1 outputs
            num_warps=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
