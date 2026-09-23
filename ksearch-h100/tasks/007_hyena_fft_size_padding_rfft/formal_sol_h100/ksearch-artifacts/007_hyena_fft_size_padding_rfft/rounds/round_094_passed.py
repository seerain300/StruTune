# solution=GPT-5.6-Sol_007_hyena_fft_size_padding_rfft_triton_optimized_r4 score=1.7968312557060109 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _direct_rfft_kernel(
    x_ptr,
    output_ptr,
    seqlen: tl.constexpr,
    freq_len: tl.constexpr,
    output_plane_size: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    freq_block = tl.program_id(1)

    k = freq_block * BLOCK_K + tl.arange(0, BLOCK_K)
    acc_real = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc_imag = tl.zeros((BLOCK_K,), dtype=tl.float32)

    scale = 1.0 / (2.0 * seqlen)
    angle_scale = -3.141592653589793 / seqlen

    for start in tl.static_range(0, seqlen, BLOCK_T):
        t = start + tl.arange(0, BLOCK_T)
        values = tl.load(
            x_ptr + row * seqlen + t,
            mask=t < seqlen,
            other=0.0,
        ).to(tl.float32)

        angles = (
            k[:, None].to(tl.float32)
            * t[None, :].to(tl.float32)
            * angle_scale
        )
        values = values[None, :] * scale
        acc_real += tl.sum(values * tl.cos(angles), axis=1)
        acc_imag += tl.sum(values * tl.sin(angles), axis=1)

    offsets = row * freq_len + k
    mask = k < freq_len
    tl.store(output_ptr + offsets, acc_real, mask=mask)
    tl.store(output_ptr + output_plane_size + offsets, acc_imag, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor):
    batch, channels, seqlen = x.shape
    fft_size = 2 * seqlen
    freq_len = seqlen + 1

    if x.is_cuda and seqlen <= 131:
        output = torch.empty(
            (2, batch, channels, freq_len),
            device=x.device,
            dtype=torch.float32,
        )
        real = output[0]
        imag = output[1]

        rows = batch * channels
        block_k = 32
        grid = (rows, triton.cdiv(freq_len, block_k))
        _direct_rfft_kernel[grid](
            x,
            output,
            seqlen=seqlen,
            freq_len=freq_len,
            output_plane_size=rows * freq_len,
            BLOCK_K=block_k,
            BLOCK_T=32,
            num_warps=4,
        )
        return real, imag

    x_freq = torch.fft.rfft(
        x,
        n=fft_size,
        norm="forward",
    )
    return x_freq.real, x_freq.imag