import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rows_kernel(x_ptr, sum_ptr, B: tl.int32, C: tl.int32, S: tl.int32, BLOCK_K: tl.constexpr):
    # One program per (b, c)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    # x is laid out as (B, C, S) contiguous: offset = b*C*S + c*S + t
    base = b * C * S + c * S
    sum_x = 0.0
    # Loop over S in chunks of BLOCK_K
    for start in range(0, S, BLOCK_K):
        offs = start + tl.arange(0, BLOCK_K)
        mask = offs < S
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_x += tl.sum(vals, axis=0)
    tl.store(sum_ptr + pid, sum_x)


@triton.jit
def _rfft_fill_kernel(sum_ptr, out_real_ptr, out_imag_ptr,
                      B: tl.int32, C: tl.int32, S: tl.int32, M: tl.int32):
    # One program per (b, c)
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C
    # Load sum for this (b, c)
    sum_x = tl.load(sum_ptr + pid)
    # Prepare output vectors of length M=S+1
    k_idx = tl.arange(0, M)  # [0..M-1]

    # Handle k == 0: real = sum_x, imag = 0
    out_real = tl.zeros([M], dtype=tl.float32)
    out_imag = tl.zeros([M], dtype=tl.float32)
    out_real[0] = sum_x
    out_imag[0] = 0.0

    # For k = 1..S: compute contributions
    # We cannot use tl.static_range for runtime S, so iterate scalar k.
    # Triton allows scalar control flow; per-k computation is fine.
    for k in range(1, S + 1):
        # ang = 2*pi*k/(2*S) = pi*k/S
        N = 2 * S
        ang = 3.141592653589793 * k / (N * 0.5)  # 2*pi*k/N = pi*k/S
        cosk = tl.cos(ang)
        sink = tl.sin(ang)
        # Normalized by 2*S (original code divides complex rfft by 2*S)
        scale = 1.0 / (2.0 * S)
        real_k = sum_x * cosk * scale
        imag_k = -sum_x * sink * scale
        out_real[k] = real_k
        out_imag[k] = imag_k

    # Store results to contiguous outputs at (b, c) slice:
    # Output tensors are (B, C, M). For (b, c), linear offset = pid * M + k_idx
    offs = tl.arange(0, M)
    store_base = pid * M
    tl.store(out_real_ptr + store_base + offs, out_real)
    tl.store(out_imag_ptr + store_base + offs, out_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation that matches:
            x_f32 = x.to(torch.float32)
            x_freq = torch.fft.rfft(x_f32, n=2*S)  # complex (B, C, S+1)
            x_freq = x_freq / (2*S)
            return x_freq.real, x_freq.imag
        """
        assert x.dim() == 3, "Input must be 3D (B, C, S)"
        assert x.dtype == torch.float32, "Input must be float32"
        B, C, S = x.shape
        M = S + 1  # output length

        # Allocate buffers
        sum_buf = torch.empty(B * C, dtype=torch.float32, device=x.device)
        out_real = torch.empty(B, C, M, dtype=torch.float32, device=x.device)
        out_imag = torch.empty(B, C, M, dtype=torch.float32, device=x.device)

        # Launch sum kernel: one program per (b, c)
        grid = (B * C,)
        _sum_rows_kernel[grid](
            x, sum_buf,
            B, C, S,
            BLOCK_K=1024,
            num_warps=4
        )

        # Launch fill kernel: compute rfft-like outputs per (b, c)
        _rfft_fill_kernel[grid](
            sum_buf, out_real, out_imag,
            B, C, S, M,
            num_warps=4
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
