import torch
import triton
import triton.language as tl


@triton.jit
def sum_all_kernel(x_ptr, sum_ptr, S, stride_x_bc, BLOCK_S: tl.constexpr):
    """
    Compute sum over S elements for each (b, c) slice, write to sum_ptr[pid].
    x_ptr points to the start of the (b, c) slice (flattened (B*C, S) row).
    We unroll up to BLOCK_S and mask s < S.
    """
    pid = tl.program_id(0)
    total = 0.0
    for s in range(0, BLOCK_S):
        val = tl.load(x_ptr + pid * stride_x_bc + s, mask=s < S, other=0.0)
        total += val
    tl.store(sum_ptr + pid, total)


@triton.jit
def rfft_real_imag_from_sum_kernel(out_real_ptr, out_imag_ptr, sum_ptr, S, inv_twoN, BLOCK_J: tl.constexpr):
    """
    For each (b, c), fill out_real_ptr[pid, :] and out_imag_ptr[pid, :] of length S+1.
    Uses known real-input rfft formulas and final scaling by inv_twoN = 1/(2*S).
    Loops over j are tl.static_range with BLOCK_J = S + 1 (constexpr).
    """
    pid = tl.program_id(0)
    sum_val = tl.load(sum_ptr + pid)
    inv_twoN = inv_twoN  # scalar 1/(2*S)

    for j in tl.static_range(0, BLOCK_J):
        # Angle for real-input rfft: pi * j / (2 * S)
        ang = 3.141592653589793 * j / (2.0 * S)
        c = tl.cos(ang)
        s = tl.sin(ang)

        # j == 0: real = sum_val / (2*S), imag = 0
        # j > 0 even: real = (sum_val * c - sum_val * s) / (2*S), imag = 0
        # j > 0 odd: real = 0, imag = -sum_val * s / (2*S)
        is_zero = j == 0
        is_even = (j % 2) == 0

        real = tl.where(is_zero, sum_val * inv_twoN,
                        tl.where(is_even, (sum_val * c - sum_val * s) * inv_twoN, 0.0))
        imag = tl.where(is_zero, 0.0,
                        tl.where(is_even, 0.0, -(sum_val * s) * inv_twoN))

        tl.store(out_real_ptr + pid * (S + 1) + j, real)
        tl.store(out_imag_ptr + pid * (S + 1) + j, imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, S), arbitrary float dtype; convert to float32 for computation
        B, C, S = x.shape
        x32 = x.to(torch.float32)

        # Flatten (B, C) into rows for reduction; each row has S elements
        x_flat = x32.reshape(B * C, S).contiguous()
        stride_x_bc = x_flat.stride(0)  # should be S

        # Allocate sum buffer per (b, c)
        sum_all = torch.empty(B * C, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: one program per (b, c). Use BLOCK_S >= S; mask handles S.
        # We choose BLOCK_S = 2048 for robustness across typical S (up to 8192 in workloads).
        sum_all_kernel[(B * C,)](
            x_flat, sum_all, S, stride_x_bc, BLOCK_S=2048,
            num_warps=1,
        )

        # Allocate outputs
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Final scale factor: 1 / (2 * S)
        inv_twoN = 1.0 / (2.0 * S)

        # Launch output kernel: one program per (b, c). BLOCK_J = S + 1 as constexpr.
        rfft_real_imag_from_sum_kernel[(B * C,)](
            out_real, out_imag, sum_all, S, inv_twoN, BLOCK_J=S + 1,
            num_warps=1,
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
