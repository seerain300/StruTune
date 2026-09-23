import torch
import triton
import triton.language as tl


@triton.jit
def real_fft_rfft_kernel(x_ptr, out_real_ptr, out_imag_ptr, B: tl.int32, C: tl.int32, S: tl.int32, BLOCK_K: tl.constexpr):
    """
    For each (b, c) slice, compute the real FFT of length 2*S and write normalized
    real and imaginary parts into out_real/out_imag of shape (B, C, S+1).
    Assumes x_ptr points to (B, C, S) contiguous data.
    """
    pid = tl.program_id(axis=0)  # one program per (b, c) slice
    b = pid // C
    c = pid % C

    # Base offsets
    base_x = (b * C + c) * S
    M = S + 1  # output length
    base_out = (b * C + c) * (S + 1)

    twoS = 2 * S
    pi = 3.141592653589793

    # 1) real[0] = sum(x[:S])
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
            # For k >= S, x_k should be 0. We only have x[:S], but we can still compute with k up to 2*S.
            # The standard rfft with implicit zeros after S will be zero for those positions since x_k=0 there.
            x_k = tl.load(x_ptr + base_x + tl.minimum(k, S - 1), mask=mask_k, other=0.0).to(tl.float32)
            angle = 2.0 * pi * j * k / twoS
            cos_t = tl.cos(angle).to(tl.float32)
            sin_t = tl.sin(angle).to(tl.float32)
            sum_real += tl.sum(x_k * cos_t, axis=0)
            sum_imag += tl.sum(x_k * sin_t, axis=0)

        # Normalize by 2*S
        sum_real = sum_real / (2.0 * S)
        sum_imag = sum_imag / (2.0 * S)

        # For real input, rfft output at j>=1 has only imaginary part contributed by sin terms.
        tl.store(out_real_ptr + base_out + j, sum_real)
        tl.store(out_imag_ptr + base_out + j, -sum_imag)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x is expected to be (batch, channels, seqlen)
        assert x.dim() == 3, "Input must be a 3D tensor (batch, channels, seqlen)"
        B, C, S = x.shape

        # Triton requires CUDA tensors
        if x.device.type != 'cuda':
            raise RuntimeError("ModelNew requires CUDA tensor input for Triton computation.")

        # Allocate outputs: (B, C, S+1), float32
        out_real = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)
        out_imag = torch.empty((B, C, S + 1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B * C,)
        # Choose BLOCK_K large enough to get good throughput; 256 is a reasonable default
        rfft_kernel = real_fft_rfft_kernel
        rfft_kernel[grid](x, out_real, out_imag, B, C, S, BLOCK_K=256, num_warps=4)

        # Ensure outputs are contiguous
        out_real = out_real.contiguous()
        out_imag = out_imag.contiguous()

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
