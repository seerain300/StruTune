# solution=GPT-5.6-Sol_007_hyena_fft_size_padding_rfft_triton_optimized_r10 score=1.8581027399826482 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _direct_rfft_kernel(
    x_ptr,
    real_ptr,
    imag_ptr,
    seqlen: tl.constexpr,
    freq_len: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row_block = tl.program_id(0)
    freq_block = tl.program_id(1)

    row0 = row_block * 4
    row1 = row0 + 1
    row2 = row0 + 2
    row3 = row0 + 3
    k = freq_block * BLOCK_K + tl.arange(0, BLOCK_K)

    acc_real0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_imag0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_real1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_imag1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_real2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_imag2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_real3 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_imag3 = tl.zeros((BLOCK_K,), dtype=tl.float32)

    scale = 1.0 / (2.0 * seqlen)
    angle_scale = -3.141592653589793 / seqlen

    for start in tl.static_range(0, seqlen, BLOCK_T):
        t = start + tl.arange(0, BLOCK_T)
        mask_t = t < seqlen

        values0 = tl.load(
            x_ptr + row0 * seqlen + t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32) * scale
        values1 = tl.load(
            x_ptr + row1 * seqlen + t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32) * scale
        values2 = tl.load(
            x_ptr + row2 * seqlen + t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32) * scale
        values3 = tl.load(
            x_ptr + row3 * seqlen + t,
            mask=mask_t,
            other=0.0,
        ).to(tl.float32) * scale

        angles = (
            k[:, None].to(tl.float32)
            * t[None, :].to(tl.float32)
            * angle_scale
        )
        cosines = tl.cos(angles)
        sines = tl.sin(angles)

        acc_real0 += tl.sum(values0[None, :] * cosines, axis=1)
        acc_imag0 += tl.sum(values0[None, :] * sines, axis=1)
        acc_real1 += tl.sum(values1[None, :] * cosines, axis=1)
        acc_imag1 += tl.sum(values1[None, :] * sines, axis=1)
        acc_real2 += tl.sum(values2[None, :] * cosines, axis=1)
        acc_imag2 += tl.sum(values2[None, :] * sines, axis=1)
        acc_real3 += tl.sum(values3[None, :] * cosines, axis=1)
        acc_imag3 += tl.sum(values3[None, :] * sines, axis=1)

    mask_k = k < freq_len
    offsets0 = row0 * freq_len + k
    offsets1 = row1 * freq_len + k
    offsets2 = row2 * freq_len + k
    offsets3 = row3 * freq_len + k

    tl.store(real_ptr + offsets0, acc_real0, mask=mask_k)
    tl.store(imag_ptr + offsets0, acc_imag0, mask=mask_k)
    tl.store(real_ptr + offsets1, acc_real1, mask=mask_k)
    tl.store(imag_ptr + offsets1, acc_imag1, mask=mask_k)
    tl.store(real_ptr + offsets2, acc_real2, mask=mask_k)
    tl.store(imag_ptr + offsets2, acc_imag2, mask=mask_k)
    tl.store(real_ptr + offsets3, acc_real3, mask=mask_k)
    tl.store(imag_ptr + offsets3, acc_imag3, mask=mask_k)


@torch.no_grad()
def run(x: torch.Tensor):
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen
    freq_len = seqlen + 1

    if x.is_cuda and seqlen <= 131:
        real = torch.empty(
            (batch, channels, freq_len),
            device=x.device,
            dtype=torch.float32,
        )
        imag = torch.empty_like(real)

        rows = batch * channels
        block_k = 16
        grid = (triton.cdiv(rows, 4), triton.cdiv(freq_len, block_k))
        _direct_rfft_kernel[grid](
            x,
            real,
            imag,
            seqlen=seqlen,
            freq_len=freq_len,
            BLOCK_K=block_k,
            BLOCK_T=32,
            num_warps=4,
        )
        return real, imag

    x_freq = torch.fft.rfft(
        x.to(torch.float32),
        n=fft_size,
        norm="forward",
    )
    return x_freq.real, x_freq.imag