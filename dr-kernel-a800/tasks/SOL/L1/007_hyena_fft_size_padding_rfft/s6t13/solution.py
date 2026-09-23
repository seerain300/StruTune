import torch
import triton
import triton.language as tl


@triton.jit
def rfft_real_imag_direct_kernel(
    x_ptr,            # *float32, input x of length S per (b,c)
    out_real_ptr,     # *float32, output real part of length N=S+1 per (b,c)
    out_imag_ptr,     # *float32, output imag part of length N=S+1 per (b,c)
    S: tl.int32,      # input length
    inv_2S: tl.float32,  # 1.0 / (2 * S), normalization factor
):
    # Each program handles one (b,c) slice.
    bc = tl.program_id(0)
    base_x = bc * S

    # We need output length N = S + 1
    N = S + 1

    # Loop over k from 0 to S (inclusive). We will compute real and imag parts.
    # For k=0:
    #   real = sum(x) * inv_2S, imag = 0
    # For k>0:
    #   even k: real = (sum(x) * cos(2*pi*k/(2*S)) - sum(x) * sin(2*pi*k/(2*S))) * inv_2S, imag = 0
    #   odd k: real = 0, imag = -sum(x) * sin(2*pi*k/(2*S)) * inv_2S
    k = 0
    total_sum = 0.0
    t = 0
    while t < S:
        v = tl.load(x_ptr + base_x + t)
        total_sum += v
        t += 1

    # k = 0: real part = total_sum * inv_2S, imag = 0
    tl.store(out_real_ptr + bc * N + 0, total_sum * inv_2S)
    tl.store(out_imag_ptr + bc * N + 0, 0.0)

    # For k=1..S
    k = 1
    while k <= S:
        N2 = 2 * S  # n for rfft is 2*S
        ang = 2.0 * 3.141592653589793 * k / N2  # 2*pi*k/(2*S)
        c = tl.cos(ang)
        s = tl.sin(ang)

        is_even = (k & 1) == 0
        # Compute output index idx = k (since N = S+1 and k in [0, S], idx in [0, S] == [0, N-2], but we compute separately)
        # For even k: real contribution is total_sum * (c - s) * inv_2S
        # For odd k: imag contribution is -total_sum * s * inv_2S
        if is_even:
            real_val = (total_sum * (c - s)) * inv_2S
            tl.store(out_real_ptr + bc * N + k, real_val)
            tl.store(out_imag_ptr + bc * N + k, 0.0)
        else:
            imag_val = (-total_sum * s) * inv_2S
            tl.store(out_real_ptr + bc * N + k, 0.0)
            tl.store(out_imag_ptr + bc * N + k, imag_val)

        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        """
        x: (B, C, S) float32 tensor on CUDA device.
        Returns:
          - x_freq_real: (B, C, S+1) float32, real part of rfft(x, n=2*S) normalized by 2*S.
          - x_freq_imag: (B, C, S+1) float32, imag part of rfft(x, n=2*S) normalized by 2*S.
        """
        assert x.is_cuda, "Input must be on CUDA."
        assert x.dtype == torch.float32, "Input must be float32."
        B, C, S = x.shape
        N = S + 1  # output length

        # Allocate outputs
        out_real = torch.empty((B * C, N), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B * C, N), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b,c)
        grid = (B * C,)
        inv_2S = 1.0 / (2 * S)

        rfft_real_imag_direct_kernel[grid](
            x, out_real, out_imag, S, inv_2S,
            num_warps=1, num_stages=1,
        )

        # Reshape to (B, C, N)
        x_freq_real = out_real.view(B, C, N)
        x_freq_imag = out_imag.view(B, C, N)

        return x_freq_real, x_freq_imag


def run(*args):
    return ModelNew()(*args)
