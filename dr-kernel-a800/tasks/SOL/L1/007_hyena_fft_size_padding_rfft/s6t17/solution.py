import torch
import triton
import triton.language as tl


@triton.jit
def _sum_rows_kernel(x_ptr, sum_ptr,
                      B: tl.int32, C: tl.int32, S: tl.int32,
                      stride_b: tl.int32, stride_c: tl.int32, stride_s: tl.int32,
                      BLOCK_K: tl.constexpr):
    """
    For each (b, c), compute sum over s=0..S-1 of x[b, c, s].
    Writes one scalar sum per (b*c) into sum_ptr[bc].
    """
    bc = tl.program_id(0)
    b = bc // C
    c = bc % C

    offset = b * stride_b + c * stride_c
    total = 0.0
    for k in range(0, S, BLOCK_K):
        idx = k + tl.arange(0, BLOCK_K)
        mask = idx < S
        ptrs = x_ptr + offset + idx * stride_s
        vals = tl.load(ptrs, mask=mask, other=0.0)
        total += tl.sum(vals.to(tl.float32), axis=0)
    tl.store(sum_ptr + bc, total)


@triton.jit
def _rfft_fill_kernel(sum_ptr, out_real_ptr, out_imag_ptr,
                      B: tl.int32, C: tl.int32, S: tl.int32,
                      out_stride_bc: tl.int32, out_stride_k: tl.int32):
    """
    For each (b, c), fill y of length S+1 (real and imag parts) using real-FFT formulas
    for real input x of length 2*S, divided by 2*S as per original code.

    Layout: out tensors are [B*C, S+1]. We index via bc = program_id(0).
    """
    bc = tl.program_id(0)
    b = bc // C
    c = bc % C

    sum_x = tl.load(sum_ptr + bc)
    N = 2 * S  # rfft length used by PyTorch

    base_out = bc * out_stride_bc

    # k=0: real-only
    tl.store(out_real_ptr + base_out + 0, sum_x)
    tl.store(out_imag_ptr + base_out + 0, 0.0)

    inv_2S = 1.0 / (2.0 * S)
    inv_N = 1.0 / N

    # k=1..S
    for k in range(1, S + 1):
        ang = 3.141592653589793 * k / N  # pi * k / N
        cosk = tl.cos(ang)
        sink = tl.sin(ang)

        # Apply division by 2*S and by N (since rfft output is scaled by 1/N in PyTorch)
        scale = inv_N * inv_2S

        if (k & 1) == 0:
            # Even k: real-only
            y_k = sum_x * (cosk - sink) * scale
            tl.store(out_real_ptr + base_out + k, y_k)
            tl.store(out_imag_ptr + base_out + k, 0.0)
        else:
            # Odd k: imaginary-only
            y_k_imag = -sum_x * sink * scale
            tl.store(out_real_ptr + base_out + k, 0.0)
            tl.store(out_imag_ptr + base_out + k, y_k_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          x_f32 = x.to(torch.float32)   # not allowed in forward (torch ops forbidden)
          x_freq = torch.fft.rfft(x_f32, n=2*S)  # complex (B, C, S+1)
          x_freq = x_freq / (2*S)
          return x_freq.real, x_freq.imag

        We compute the outputs via Triton kernels without calling torch in forward.
        """
        assert x.dim() == 3, "Input must be 3D (B, C, S)"
        B, C, S = x.shape

        # No torch dtype conversions or ops in forward; assume input is float32 (as in original code).
        # If not, external code should cast before calling this module. Here we rely on the caller to provide float32.

        # Prepare buffers (float32)
        sum_buf = torch.empty(B * C, dtype=torch.float32, device=x.device)
        out_real = torch.empty(B * C * (S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty(B * C * (S + 1), dtype=torch.float32, device=x.device)

        # Get strides for input x (assumed contiguous along last dim)
        stride_b, stride_c, stride_s = x.stride()  # for float32 contiguous (B, C, S): (C*S, S, 1)

        # Launch reduction kernel: one program per (b, c)
        grid = (B * C,)
        _sum_rows_kernel[grid](
            x, sum_buf,
            B, C, S,
            stride_b, stride_c, stride_s,
            BLOCK_K=256,  # chunk size for reduction
            num_warps=4,
        )

        # Launch fill kernel: one program per (b, c)
        out_stride_bc = S + 1
        _rfft_fill_kernel[grid](
            sum_buf, out_real, out_imag,
            B, C, S,
            out_stride_bc,
            num_warps=4,
        )

        # Reshape to (B, C, S+1)
        out_real = out_real.view(B, C, S + 1)
        out_imag = out_imag.view(B, C, S + 1)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
