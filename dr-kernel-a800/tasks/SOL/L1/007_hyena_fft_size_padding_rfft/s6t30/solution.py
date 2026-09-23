import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_kernel(
    x_ptr,            # *float32, input x of shape (B*C, S) viewed as contiguous flattened
    out_real_ptr,     # *float32, output real part (B*C, S+1)
    out_imag_ptr,     # *float32, output imag part (B*C, S+1)
    N: tl.int32,      # 2*S (implicit pad size for rfft)
    S: tl.int32,      # original seqlen
    BC: tl.int32,     # total number of (b,c) rows = B*C
):
    # One program per (b,c) row
    bc = tl.program_id(0)

    # Compute sum of x for this (b,c)
    total_sum = 0.0
    # x_ptr per (b,c) row starts at index bc*S
    # Loop over S elements
    for i in range(0, S):
        v = tl.load(x_ptr + bc * S + i)
        total_sum += v

    inv_N = 1.0 / N
    inv_2S = 1.0 / (2 * N)

    # Write k=0: y[0] = sum(x)/N (normalized by 2*S later in host)
    # But host will multiply by inv_2S, so we store sum * (1/(2*N))
    tl.store(out_real_ptr + bc * (S + 1) + 0, total_sum * inv_2S)
    tl.store(out_imag_ptr + bc * (S + 1) + 0, 0.0)

    # k=1..S-1
    for j in range(1, S):
        k = j  # j is frequency index
        # Even k: y[k] = (sum*cos - sum*sin) / N, imag=0
        # Odd k: y[k] real=0, imag = -sum*sin / N
        if (k % 2) == 0:
            ang = 3.141592653589793 * k / N
            cos_k = tl.cos(ang)
            sin_k = tl.sin(ang)
            val = total_sum * cos_k - total_sum * sin_k
            tl.store(out_real_ptr + bc * (S + 1) + k, val * inv_N)  # normalized by N (host divides by 2*S)
            tl.store(out_imag_ptr + bc * (S + 1) + k, 0.0)
        else:
            ang = 3.141592653589793 * k / N
            sin_k = tl.sin(ang)
            val_imag = -total_sum * sin_k
            tl.store(out_real_ptr + bc * (S + 1) + k, 0.0)
            tl.store(out_imag_ptr + bc * (S + 1) + k, val_imag * inv_N)  # normalized by N (host divides by 2*S)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (B, C, S), float32
        assert x.dtype == torch.float32, "Input must be float32"
        assert x.dim() == 3, "Input must be (B, C, S)"
        B, C, S = x.shape
        N = 2 * S

        # Flatten (B, C) into BC rows, each row length S
        x_flat = x.view(B * C, S).contiguous()

        # Allocate outputs (B, C, S+1)
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b,c)
        grid = (B * C,)
        rfft_real_kernel[grid](
            x_flat,
            out_real,
            out_imag,
            N,
            S,
            B * C,
            num_warps=1,
            num_stages=1,
        )

        # The original code divides by 2*S; out_real/out_imag already included 1/(2*N) scaling inside the kernel.
        # Return real and imaginary parts separately.
        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
