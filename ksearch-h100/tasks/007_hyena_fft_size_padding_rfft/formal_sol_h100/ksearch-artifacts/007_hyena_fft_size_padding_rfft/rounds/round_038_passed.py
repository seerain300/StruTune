# solution=GPT-5.6-Sol_007_hyena_fft_size_padding_rfft_triton_optimized_r38 score=2.5956008211085977 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _short_rfft_kernel(
    x,
    out,
    seqlen: tl.constexpr,
    fft_size: tl.constexpr,
    freq_len: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)

    k = block * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_k = k < freq_len

    angle = k.to(tl.float32) * (-2.0 * 3.141592653589793 / fft_size)
    step_real = tl.cos(angle)
    step_imag = tl.sin(angle)

    endpoint = (k == 0) | (k == seqlen)
    step_real = tl.where(k == 0, 1.0, step_real)
    step_real = tl.where(k == seqlen, -1.0, step_real)
    step_imag = tl.where(endpoint, 0.0, step_imag)

    step2_real = step_real * step_real - step_imag * step_imag
    step2_imag = 2.0 * step_real * step_imag

    row_offset = row * seqlen
    value0 = tl.load(x + row_offset).to(tl.float32)
    real = tl.full((BLOCK_K,), value0, dtype=tl.float32)
    imag = tl.zeros((BLOCK_K,), dtype=tl.float32)
    phase_real = step_real
    phase_imag = step_imag

    for n in range(1, seqlen - 1, 2):
        value_a = tl.load(x + row_offset + n).to(tl.float32)
        value_b = tl.load(x + row_offset + n + 1).to(tl.float32)

        coeff_real = value_a + value_b * step_real
        coeff_imag = value_b * step_imag

        real += phase_real * coeff_real - phase_imag * coeff_imag
        imag += phase_real * coeff_imag + phase_imag * coeff_real

        next_real = phase_real * step2_real - phase_imag * step2_imag
        next_imag = phase_real * step2_imag + phase_imag * step2_real
        phase_real = next_real
        phase_imag = next_imag

    if seqlen % 2 == 0:
        value = tl.load(x + row_offset + seqlen - 1).to(tl.float32)
        real += value * phase_real
        imag += value * phase_imag

    imag = tl.where(endpoint, 0.0, imag)

    offsets = row * freq_len + k
    scale = 1.0 / fft_size
    tl.store(out + 2 * offsets, real * scale, mask=valid_k)
    tl.store(out + 2 * offsets + 1, imag * scale, mask=valid_k)


@torch.no_grad()
def run(x: torch.Tensor):
    seqlen = x.shape[2]

    use_short_rfft = seqlen <= 160 or seqlen == 211
    if x.is_cuda and x.is_contiguous() and use_short_rfft:
        batch, channels, _ = x.shape
        freq_len = seqlen + 1
        packed = torch.empty(
            (batch, channels, freq_len, 2),
            device=x.device,
            dtype=torch.float32,
        )

        if seqlen == 211:
            block_k = 256
            num_warps = 8
            maxnreg = 48
        else:
            block_k = 64
            num_warps = 1
            maxnreg = None

        rows = batch * channels
        grid = (rows, triton.cdiv(freq_len, block_k))
        if maxnreg is not None:
            _short_rfft_kernel[grid](
                x,
                packed,
                seqlen=seqlen,
                fft_size=2 * seqlen,
                freq_len=freq_len,
                BLOCK_K=block_k,
                num_warps=num_warps,
                maxnreg=maxnreg,
            )
        else:
            _short_rfft_kernel[grid](
                x,
                packed,
                seqlen=seqlen,
                fft_size=2 * seqlen,
                freq_len=freq_len,
                BLOCK_K=block_k,
                num_warps=num_warps,
            )
        return packed[..., 0], packed[..., 1]

    fft_size = 2 * seqlen
    spectrum = torch.fft.rfft(x, n=fft_size, norm="forward")
    packed = torch.view_as_real(spectrum)
    return packed[..., 0], packed[..., 1]