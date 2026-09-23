# task: 007_hyena_fft_size_padding_rfft
# bench: SOL-L1 | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=1.375x
# feedback best (5-workload sample during search): 1.309x
# torch fallback audit: A·核心靠库 (rfft×2)
# tokens: 1,207,483

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _direct_rfft_kernel(
    x_ptr,
    real_ptr,
    imag_ptr,
    stride_xb,
    stride_xc,
    stride_xt,
    SEQLEN: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FREQ_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    freq_len = SEQLEN + 1
    num_freq_groups = (freq_len + FREQ_GROUP - 1) // FREQ_GROUP

    row = pid // num_freq_groups
    group_idx = pid - row * num_freq_groups

    batch_idx = row // CHANNELS
    channel_idx = row - batch_idx * CHANNELS

    time = tl.arange(0, BLOCK_SIZE)
    time_mask = time < SEQLEN

    x_offsets = (
        batch_idx * stride_xb
        + channel_idx * stride_xc
        + time * stride_xt
    )
    values = tl.load(
        x_ptr + x_offsets,
        mask=time_mask,
        other=0.0,
    ).to(tl.float32)

    frequencies = group_idx * FREQ_GROUP + tl.arange(0, FREQ_GROUP)
    frequency_mask = frequencies < freq_len

    step_angles = frequencies.to(tl.float32) * (-math.pi / SEQLEN)
    step_cosine = tl.cos(step_angles)
    step_sine = tl.sin(step_angles)

    endpoint = (frequencies == 0) | (frequencies == SEQLEN)
    step_cosine = tl.where(
        frequencies == 0,
        1.0,
        tl.where(frequencies == SEQLEN, -1.0, step_cosine),
    )
    step_sine = tl.where(endpoint, 0.0, step_sine)

    cosine = tl.full((FREQ_GROUP,), 1.0, tl.float32)
    sine = tl.zeros((FREQ_GROUP,), tl.float32)
    real_values = tl.zeros((FREQ_GROUP,), tl.float32)
    imag_values = tl.zeros((FREQ_GROUP,), tl.float32)

    for t in tl.static_range(0, SEQLEN):
        value = values[t]
        real_values += value * cosine
        imag_values += value * sine

        next_cosine = cosine * step_cosine - sine * step_sine
        next_sine = sine * step_cosine + cosine * step_sine
        cosine = next_cosine
        sine = next_sine

    scale = 1.0 / (2 * SEQLEN)
    real_values *= scale
    imag_values *= scale
    imag_values = tl.where(endpoint, 0.0, imag_values)

    output_offsets = row * freq_len + frequencies
    tl.store(
        real_ptr + output_offsets,
        real_values,
        mask=frequency_mask,
    )
    tl.store(
        imag_ptr + output_offsets,
        imag_values,
        mask=frequency_mask,
    )


@triton.jit
def _split_scale_kernel(
    freq_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    components = tl.arange(0, 2)
    pair_offsets = 2 * offsets[:, None] + components[None, :]
    values = tl.load(
        freq_ptr + pair_offsets,
        mask=mask[:, None],
        other=0.0,
    )
    real, imag = tl.split(values)

    tl.store(output_ptr + offsets, real * SCALE, mask=mask)
    tl.store(output_ptr + N_ELEMENTS + offsets, imag * SCALE, mask=mask)


@torch.no_grad()
def run(x: torch.Tensor):
    batch_size, channels, seqlen = x.shape
    fft_size = 2 * seqlen
    freq_len = seqlen + 1
    n_elements = batch_size * channels * freq_len

    if seqlen <= 32 and channels == 256 and x.is_cuda:
        output = torch.empty(
            (2, batch_size, channels, freq_len),
            device=x.device,
            dtype=torch.float32,
        )
        x_freq_real = output[0]
        x_freq_imag = output[1]

        block_size = triton.next_power_of_2(seqlen)
        freq_group = 4
        num_freq_groups = triton.cdiv(freq_len, freq_group)
        grid = (batch_size * channels * num_freq_groups,)

        _direct_rfft_kernel[grid](
            x,
            x_freq_real,
            x_freq_imag,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            SEQLEN=seqlen,
            CHANNELS=channels,
            BLOCK_SIZE=block_size,
            FREQ_GROUP=freq_group,
            num_warps=1,
        )
        return x_freq_real, x_freq_imag

    if x.is_cuda:
        x_freq = torch.fft.rfft(x, n=fft_size)
        freq_storage = torch.view_as_real(x_freq)

        output = torch.empty(
            (2, batch_size, channels, freq_len),
            device=x.device,
            dtype=torch.float32,
        )

        if n_elements >= 1_000_000:
            block_size = 1024
            num_warps = 8
        else:
            block_size = 512
            num_warps = 4

        _split_scale_kernel[(triton.cdiv(n_elements, block_size),)](
            freq_storage,
            output,
            N_ELEMENTS=n_elements,
            SCALE=1.0 / fft_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output[0], output[1]

    x_freq = torch.fft.rfft(x, n=fft_size) / fft_size
    return x_freq.real.contiguous(), x_freq.imag.contiguous()