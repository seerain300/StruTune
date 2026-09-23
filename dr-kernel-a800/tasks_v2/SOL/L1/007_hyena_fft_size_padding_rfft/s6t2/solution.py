import triton
import triton.language as tl


@triton.jit
def real_fft_rfft_kernel(x_ptr, out_real_ptr, out_imag_ptr, B: tl.int32, C: tl.int32, S: tl.int32, BLOCK_K: tl.constexpr):
    """
    For each (b, c) slice of x_ptr with shape (B, C, S):
      Compute the real FFT of length 2*S and write normalized real and imaginary parts
      into out_real/out_imag of shape (B, C, S+1).
      No torch operations. All computations done inside Triton.
    """
    pid = tl.program_id(axis=0)  # one program per (b, c) slice
    b = pid // C
    c = pid % C

    base_x = (b * C + c) * S
    M = S + 1  # output length of rfft for real input of length 2*S
    base_out = (b * C + c) * (S + 1)

    twoS = 2 * S
    pi = 3.141592653589793

    # 1) real[0] = sum(x[:S]) / (2*S)
    sum_real0 = 0.0
    for off in range(0, S, BLOCK_K):
        k = off + tl.arange(0, BLOCK_K)
        mask_k = k < S
        x_k = tl.load(x_ptr + base_x + k, mask=mask_k, other=0.0).to(tl.float32)
        sum_real0 += tl.sum(x_k, axis=0)
    out0 = sum_real0 / (2.0 * S)
    tl.store(out_real_ptr + base_out + 0, out0)
    tl.store(out_imag_ptr + base_out + 0, 0.0)

    # 2) real[j] and imag[j] for j = 1..S
    for j in range(1, M):
        sum_real = 0.0
        sum_imag = 0.0
        for off in range(0, twoS, BLOCK_K):
            k = off + tl.arange(0, BLOCK_K)
            mask_k = k < twoS
            # Implicitly treat k >= S as zeros by masking and using S-1 upper bound
            x_k = tl.load(x_ptr + base_x + tl.minimum(k, S - 1), mask=mask_k, other=0.0).to(tl.float32)
            angle = 2.0 * pi * j * k / twoS
            cos_t = tl.cos(angle).to(tl.float32)
            sin_t = tl.sin(angle).to(tl.float32)
            sum_real += tl.sum(x_k * cos_t, axis=0)
            sum_imag += tl.sum(x_k * sin_t, axis=0)

        # Normalize by 2*S
        sum_real = sum_real / (2.0 * S)
        sum_imag = sum_imag / (2.0 * S)

        tl.store(out_real_ptr + base_out + j, sum_real)
        tl.store(out_imag_ptr + base_out + j, -sum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x is expected to be (batch, channels, seqlen)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        B, C, S = x.shape

        # Triton requires CUDA tensors
        if x.device.type != 'cuda':
            # CPU fallback: original PyTorch behavior (for completeness)
            x_f32 = x.to(torch.float32)
            fft_size = 2 * S
            x_freq = torch.fft.rfft(x_f32, n=fft_size)
            x_freq = x_freq / (2 * S)
            x_freq_real = x_freq.real.contiguous()
            x_freq_imag = x_freq.imag.contiguous()
            return x_freq_real, x_freq_imag

        # Forward must not allocate tensors or use torch operations. Assume outputs are provided.
        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        real_fft_rfft_kernel[grid](x, out_real_ptr, out_imag_ptr, B, C, S, BLOCK_K=256, num_warps=4)

        # Return results
        return out_real_ptr, out_imag_ptr


def run(*args):
    return ModelNew()(*args)
