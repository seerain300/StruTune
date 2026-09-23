# solution=GPT-5.6-Sol_007_hyena_fft_size_padding_rfft_triton_optimized_r22 score=2.414491052553279 passed=True
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
    STORE_NYQUIST: tl.constexpr,
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
    step3_real = step2_real * step_real - step2_imag * step_imag
    step3_imag = step2_real * step_imag + step2_imag * step_real
    step4_real = step2_real * step2_real - step2_imag * step2_imag
    step4_imag = 2.0 * step2_real * step2_imag

    row_offset = row * seqlen
    value0 = tl.load(x + row_offset).to(tl.float32)
    real = tl.full((BLOCK_K,), value0, dtype=tl.float32)
    imag = tl.zeros((BLOCK_K,), dtype=tl.float32)
    phase_real = step_real
    phase_imag = step_imag

    if STORE_NYQUIST:
        nyquist = value0

    if seqlen == 211:
        step5_real = step4_real * step_real - step4_imag * step_imag
        step5_imag = step4_real * step_imag + step4_imag * step_real
        step6_real = step4_real * step2_real - step4_imag * step2_imag
        step6_imag = step4_real * step2_imag + step4_imag * step2_real
        step7_real = step4_real * step3_real - step4_imag * step3_imag
        step7_imag = step4_real * step3_imag + step4_imag * step3_real
        step8_real = step4_real * step4_real - step4_imag * step4_imag
        step8_imag = 2.0 * step4_real * step4_imag

        for n in range(1, 209, 8):
            value_a = tl.load(x + row_offset + n).to(tl.float32)
            value_b = tl.load(x + row_offset + n + 1).to(tl.float32)
            value_c = tl.load(x + row_offset + n + 2).to(tl.float32)
            value_d = tl.load(x + row_offset + n + 3).to(tl.float32)
            value_e = tl.load(x + row_offset + n + 4).to(tl.float32)
            value_f = tl.load(x + row_offset + n + 5).to(tl.float32)
            value_g = tl.load(x + row_offset + n + 6).to(tl.float32)
            value_h = tl.load(x + row_offset + n + 7).to(tl.float32)

            polynomial_real = (
                value_a
                + value_b * step_real
                + value_c * step2_real
                + value_d * step3_real
                + value_e * step4_real
                + value_f * step5_real
                + value_g * step6_real
                + value_h * step7_real
            )
            polynomial_imag = (
                value_b * step_imag
                + value_c * step2_imag
                + value_d * step3_imag
                + value_e * step4_imag
                + value_f * step5_imag
                + value_g * step6_imag
                + value_h * step7_imag
            )

            real += phase_real * polynomial_real - phase_imag * polynomial_imag
            imag += phase_real * polynomial_imag + phase_imag * polynomial_real

            next_real = phase_real * step8_real - phase_imag * step8_imag
            next_imag = phase_real * step8_imag + phase_imag * step8_real
            phase_real = next_real
            phase_imag = next_imag

        value_a = tl.load(x + row_offset + 209).to(tl.float32)
        value_b = tl.load(x + row_offset + 210).to(tl.float32)
        polynomial_real = value_a + value_b * step_real
        polynomial_imag = value_b * step_imag

        real += phase_real * polynomial_real - phase_imag * polynomial_imag
        imag += phase_real * polynomial_imag + phase_imag * polynomial_real
    else:
        main_values: tl.constexpr = ((seqlen - 1) // 4) * 4
        remainder_start: tl.constexpr = 1 + main_values

        for n in range(1, remainder_start, 4):
            value_a = tl.load(x + row_offset + n).to(tl.float32)
            value_b = tl.load(x + row_offset + n + 1).to(tl.float32)
            value_c = tl.load(x + row_offset + n + 2).to(tl.float32)
            value_d = tl.load(x + row_offset + n + 3).to(tl.float32)

            polynomial_real = (
                value_a
                + value_b * step_real
                + value_c * step2_real
                + value_d * step3_real
            )
            polynomial_imag = (
                value_b * step_imag
                + value_c * step2_imag
                + value_d * step3_imag
            )

            real += phase_real * polynomial_real - phase_imag * polynomial_imag
            imag += phase_real * polynomial_imag + phase_imag * polynomial_real

            next_real = phase_real * step4_real - phase_imag * step4_imag
            next_imag = phase_real * step4_imag + phase_imag * step4_real
            phase_real = next_real
            phase_imag = next_imag

            if STORE_NYQUIST:
                nyquist += value_b - value_a + value_d - value_c

        for n in range(remainder_start, seqlen):
            value = tl.load(x + row_offset + n).to(tl.float32)
            real += value * phase_real
            imag += value * phase_imag

            if STORE_NYQUIST:
                if n % 2 == 0:
                    nyquist += value
                else:
                    nyquist -= value

            next_real = phase_real * step_real - phase_imag * step_imag
            next_imag = phase_real * step_imag + phase_imag * step_real
            phase_real = next_real
            phase_imag = next_imag

    imag = tl.where(endpoint, 0.0, imag)

    offsets = row * freq_len + k
    scale = 1.0 / fft_size

    if STORE_NYQUIST:
        tl.store(out + 2 * offsets, real * scale)
        tl.store(out + 2 * offsets + 1, imag * scale)
    else:
        tl.store(out + 2 * offsets, real * scale, mask=valid_k)
        tl.store(out + 2 * offsets + 1, imag * scale, mask=valid_k)

    if STORE_NYQUIST:
        nyquist_offset = row * freq_len + seqlen
        tl.store(out + 2 * nyquist_offset, nyquist * scale)
        tl.store(out + 2 * nyquist_offset + 1, 0.0)


@torch.no_grad()
def run(x: torch.Tensor):
    seqlen = x.shape[2]

    use_short_rfft = seqlen <= 160 or seqlen == 211 or seqlen == 256
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
            grid_freq_len = freq_len
            num_warps = 8
            maxnreg = 64
            store_nyquist = False
        elif seqlen == 256:
            block_k = 256
            grid_freq_len = seqlen
            num_warps = 8
            maxnreg = 56
            store_nyquist = True
        else:
            block_k = 64
            grid_freq_len = freq_len
            num_warps = 1
            maxnreg = None
            store_nyquist = False

        rows = batch * channels
        grid = (rows, triton.cdiv(grid_freq_len, block_k))

        if maxnreg is not None:
            _short_rfft_kernel[grid](
                x,
                packed,
                seqlen=seqlen,
                fft_size=2 * seqlen,
                freq_len=freq_len,
                BLOCK_K=block_k,
                STORE_NYQUIST=store_nyquist,
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
                STORE_NYQUIST=store_nyquist,
                num_warps=num_warps,
            )

        return packed[..., 0], packed[..., 1]

    fft_size = 2 * seqlen
    spectrum = torch.fft.rfft(x, n=fft_size, norm="forward")
    packed = torch.view_as_real(spectrum)
    return packed[..., 0], packed[..., 1]