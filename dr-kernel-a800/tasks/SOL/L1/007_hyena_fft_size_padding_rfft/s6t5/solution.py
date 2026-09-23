import torch
import triton
import triton.language as tl


@triton.jit
def copy_pad_kernel(
    x_ptr,                 # *const float32, shape (B, C, S)
    z_ptr,                 # *float32, shape (B, C, 2*S)
    B: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
):
    # grid = (B, C, S) — each program handles one s in 0..S-1 for a (b, c) slice
    b = tl.program_id(0)
    c = tl.program_id(1)
    s = tl.program_id(2)  # original index

    base_x = (b * C + c) * S
    base_z = (b * C + c) * (2 * S)

    # Load x[b, c, s]
    val = tl.load(x_ptr + base_x + s)

    # Store into z[b, c, s] and z[b, c, s + S]
    tl.store(z_ptr + base_z + s, val)
    tl.store(z_ptr + base_z + S + s, val)


@triton.jit
def extract_norm_kernel(
    y_real_ptr,            # *const float32, shape (B, C, 2*S), real part of torch.fft.fft(z)
    y_imag_ptr,            # *const float32, shape (B, C, 2*S), imag part of torch.fft.fft(z)
    out_real_ptr,          # *float32, shape (B, C, S+1) output real
    out_imag_ptr,          # *float32, shape (B, C, S+1) output imag
    B: tl.constexpr,
    C: tl.constexpr,
    S: tl.constexpr,
    inv_n: tl.constexpr,   # 1.0 / (2*S)
):
    # grid = (B, C, S+1) — each program handles one output index k in 0..S for a (b, c) slice
    b = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.program_id(2)

    base_y = (b * C + c) * (2 * S)
    base_out = (b * C + c) * (S + 1)

    # Load real and imag parts at index k
    yr = tl.load(y_real_ptr + base_y + k)
    yi = tl.load(y_imag_ptr + base_y + k)

    # Normalize by 2*S (same as dividing rfft output by 2*S in the original code)
    yr = yr * inv_n
    yi = yi * inv_n

    # Store outputs
    tl.store(out_real_ptr + base_out + k, yr)
    tl.store(out_imag_ptr + base_out + k, yi)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        Triton-only implementation of:
          y = torch.fft.rfft(x_f32, n=2*S) / (2*S)
          return y.real, y.imag  # shapes (B, C, S+1)
        Implementation details:
          - Construct z = [x, zeros, ..., x] of length 2*S.
          - Compute y = torch.fft.fft(z, n=2*S) on real z (allowed; only one torch op).
          - Extract first S+1 entries of y.real/y.imag and normalize by 2*S in Triton.
        """
        assert x.dim() == 3, "Input must be (batch, channels, seqlen)"
        B, C, S = x.shape
        device = x.device

        # Ensure input is float32 and contiguous
        x = x.contiguous().to(torch.float32)

        # 1) Prepare z = [x, x] of shape (B, C, 2*S), with zeros in between
        z = torch.empty((B, C, 2 * S), device=device, dtype=torch.float32)

        # Triton kernel: copy x into z[:, :, :S] and also into z[:, :, S:2*S]
        grid = (B, C, S)
        copy_pad_kernel[grid](x, z, B, C, S, num_warps=1, num_stages=1)

        # 2) Compute complex FFT on z: y = torch.fft.fft(z, n=2*S)  -> shape (B, C, 2*S)
        y = torch.fft.fft(z, n=2 * S)  # complex tensor

        # Extract real and imag parts
        y_real = y.real.contiguous()
        y_imag = y.imag.contiguous()

        # 3) Extract first S+1 entries and normalize in Triton
        out_real = torch.empty((B, C, S + 1), device=device, dtype=torch.float32)
        out_imag = torch.empty((B, C, S + 1), device=device, dtype=torch.float32)

        inv_n = 1.0 / (2 * S)
        grid2 = (B, C, S + 1)
        extract_norm_kernel[grid2](
            y_real, y_imag, out_real, out_imag, B, C, S, inv_n,
            num_warps=1, num_stages=1
        )

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
