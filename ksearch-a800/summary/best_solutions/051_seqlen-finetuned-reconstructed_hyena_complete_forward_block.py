# task: 051_seqlen-finetuned-reconstructed_hyena_complete_forward_block
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=2.351x
# feedback best (5-workload sample during search): 2.282x
# torch fallback audit: C·待消融 (addmm×2,rfft/irfft)
# tokens: 3,024,273

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    base = row * 256 + offsets

    x = tl.load(x_ptr + base).to(tl.float32)
    mean = tl.sum(x, axis=0) * (1.0 / 256.0)
    centered = x - mean
    var = tl.sum(centered * centered, axis=0) * (1.0 / 256.0)
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    bias = tl.load(bias_ptr + offsets).to(tl.float32)
    tl.store(out_ptr + base, centered * inv_std * weight + bias)


@triton.jit
def _residual_layer_norm_kernel(
    hidden_ptr,
    update_ptr,
    weight_ptr,
    bias_ptr,
    output_bias_ptr,
    residual_bias_ptr,
    normed_ptr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    base = row * 256 + offsets

    hidden = tl.load(hidden_ptr + base).to(tl.float32)
    update = tl.load(update_ptr + base).to(tl.float32)
    residual = hidden + update

    mean = tl.sum(residual, axis=0) * (1.0 / 256.0)
    centered = residual - mean
    var = tl.sum(centered * centered, axis=0) * (1.0 / 256.0)
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    bias = tl.load(bias_ptr + offsets).to(tl.float32)
    output_bias = tl.load(output_bias_ptr + offsets).to(tl.float32)

    normed = centered * inv_std * weight + bias
    tl.store(residual_bias_ptr + base, residual + output_bias)
    tl.store(normed_ptr + base, normed)


@triton.jit
def _short_conv_gate_kernel(
    u_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    seq_len,
    batch_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    time_block = tl.program_id(0)
    channel_block = tl.program_id(1)
    batch = tl.program_id(2)

    times = time_block * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)

    t = times[:, None]
    c = channels[None, :]
    valid_t = t < seq_len

    input_batch = batch * seq_len * 768
    output_batch = batch * 256 * seq_len
    output_offsets = output_batch + c * seq_len + t

    channel0 = c
    base0 = input_batch + t * 768 + channel0
    u00 = tl.load(u_ptr + base0, mask=valid_t, other=0.0).to(tl.float32)
    u01 = tl.load(
        u_ptr + base0 - 768,
        mask=valid_t & (t >= 1),
        other=0.0,
    ).to(tl.float32)
    u02 = tl.load(
        u_ptr + base0 - 1536,
        mask=valid_t & (t >= 2),
        other=0.0,
    ).to(tl.float32)

    w00 = tl.load(weight_ptr + channel0 * 3).to(tl.float32)
    w01 = tl.load(weight_ptr + channel0 * 3 + 1).to(tl.float32)
    w02 = tl.load(weight_ptr + channel0 * 3 + 2).to(tl.float32)
    b0 = tl.load(bias_ptr + channel0).to(tl.float32)

    x0 = u02 * w00 + u01 * w01
    x0 = x0 + u00 * w02 + b0
    tl.store(out_ptr + output_offsets, x0, mask=valid_t)

    channel1 = c + 256
    base1 = input_batch + t * 768 + channel1
    u10 = tl.load(u_ptr + base1, mask=valid_t, other=0.0).to(tl.float32)
    u11 = tl.load(
        u_ptr + base1 - 768,
        mask=valid_t & (t >= 1),
        other=0.0,
    ).to(tl.float32)
    u12 = tl.load(
        u_ptr + base1 - 1536,
        mask=valid_t & (t >= 2),
        other=0.0,
    ).to(tl.float32)

    w10 = tl.load(weight_ptr + channel1 * 3).to(tl.float32)
    w11 = tl.load(weight_ptr + channel1 * 3 + 1).to(tl.float32)
    w12 = tl.load(weight_ptr + channel1 * 3 + 2).to(tl.float32)
    b1 = tl.load(bias_ptr + channel1).to(tl.float32)

    x1 = u12 * w10 + u11 * w11
    x1 = x1 + u10 * w12 + b1

    channel2 = c + 512
    base2 = input_batch + t * 768 + channel2
    u20 = tl.load(u_ptr + base2, mask=valid_t, other=0.0).to(tl.float32)
    u21 = tl.load(
        u_ptr + base2 - 768,
        mask=valid_t & (t >= 1),
        other=0.0,
    ).to(tl.float32)
    u22 = tl.load(
        u_ptr + base2 - 1536,
        mask=valid_t & (t >= 2),
        other=0.0,
    ).to(tl.float32)

    w20 = tl.load(weight_ptr + channel2 * 3).to(tl.float32)
    w21 = tl.load(weight_ptr + channel2 * 3 + 1).to(tl.float32)
    w22 = tl.load(weight_ptr + channel2 * 3 + 2).to(tl.float32)
    b2 = tl.load(bias_ptr + channel2).to(tl.float32)

    v = u22 * w20 + u21 * w21
    v = v + u20 * w22 + b2

    gated_offset = batch_size * 256 * seq_len + output_offsets
    tl.store(out_ptr + gated_offset, v * x1, mask=valid_t)


@triton.jit
def _filter_hidden_kernel(
    linear1_weight_ptr,
    linear1_bias_ptr,
    frequency_ptr,
    linear2_weight_ptr,
    linear2_bias_ptr,
    linear3_weight_ptr,
    linear3_bias_ptr,
    out_ptr,
    seq_len,
    BLOCK_T: tl.constexpr,
):
    block = tl.program_id(0)

    times = block * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = tl.arange(0, 64)
    t_idx = times[:, None]
    c_idx = channels[None, :]
    valid_t = t_idx < seq_len

    time_f = t_idx.to(tl.float32)
    denom = tl.maximum(seq_len - 1, 1).to(tl.float32)
    normalized_t = time_f / denom
    angle = time_f * (6.283185307179586 / seq_len)

    z0 = normalized_t
    z1 = tl.cos(-angle * 1.0e-4)
    z2 = tl.cos(-angle)
    z3 = tl.sin(-angle * 1.0e-4)
    z4 = tl.sin(-angle)

    weight_base = c_idx * 5
    w0 = tl.load(linear1_weight_ptr + weight_base).to(tl.float32)
    w1 = tl.load(linear1_weight_ptr + weight_base + 1).to(tl.float32)
    w2 = tl.load(linear1_weight_ptr + weight_base + 2).to(tl.float32)
    w3 = tl.load(linear1_weight_ptr + weight_base + 3).to(tl.float32)
    w4 = tl.load(linear1_weight_ptr + weight_base + 4).to(tl.float32)
    b1 = tl.load(linear1_bias_ptr + c_idx).to(tl.float32)
    freq = tl.load(frequency_ptr + c_idx).to(tl.float32)

    hidden = z0 * w0 + z1 * w1
    hidden = hidden + z2 * w2 + z3 * w3
    hidden = tl.sin(freq * (hidden + z4 * w4 + b1))

    k = tl.arange(0, 64)[:, None]
    n = tl.arange(0, 64)[None, :]

    linear2_weight = tl.load(
        linear2_weight_ptr + n * 64 + k
    ).to(tl.float32)
    hidden = tl.dot(hidden, linear2_weight)
    b2 = tl.load(linear2_bias_ptr + n).to(tl.float32)
    freq2 = tl.load(frequency_ptr + n).to(tl.float32)
    hidden = tl.sin(freq2 * (hidden + b2))

    linear3_weight = tl.load(
        linear3_weight_ptr + n * 64 + k
    ).to(tl.float32)
    hidden = tl.dot(hidden, linear3_weight)
    b3 = tl.load(linear3_bias_ptr + n).to(tl.float32)
    hidden = tl.sin(freq2 * (hidden + b3))

    offsets = t_idx * 64 + c_idx
    tl.store(out_ptr + offsets, hidden, mask=valid_t)


@triton.jit
def _filter_project_modulate_kernel(
    hidden_ptr,
    final_weight_ptr,
    delta_ptr,
    bias_ptr,
    out_ptr,
    seq_len,
    shift,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    time_block = tl.program_id(0)
    channel_block = tl.program_id(1)

    times = time_block * BLOCK_T + tl.arange(0, BLOCK_T)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.arange(0, 64)

    t = times[:, None]
    c = channels[None, :]
    k_row = k_offsets[None, :]
    k_col = k_offsets[:, None]

    hidden = tl.load(
        hidden_ptr + t * 64 + k_row,
        mask=t < seq_len,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        final_weight_ptr + c * 64 + k_col
    ).to(tl.float32)

    projected = tl.dot(hidden, weight)

    delta = tl.load(delta_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    denom = tl.maximum(seq_len - 1, 1).to(tl.float32)
    normalized_t = t.to(tl.float32) / denom
    decay = tl.exp(-normalized_t * tl.abs(delta))
    value = projected * (decay + shift) + bias

    tl.store(out_ptr + t * 256 + c, value, mask=t < seq_len)


@triton.jit
def _post_fft_projection_kernel(
    convolved_ptr,
    gated_ptr,
    x0_ptr,
    filter_bias_ptr,
    projection_weight_ptr,
    projection_bias_ptr,
    out_ptr,
    conv_stride_b,
    conv_stride_c,
    seq_len,
    BLOCK_T: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    time_block = tl.program_id(0)
    output_block = tl.program_id(1)
    batch = tl.program_id(2)

    times = time_block * BLOCK_T + tl.arange(0, BLOCK_T)
    outputs = output_block * BLOCK_O + tl.arange(0, BLOCK_O)
    accumulator = tl.zeros((BLOCK_T, BLOCK_O), dtype=tl.float32)

    t = times[:, None]
    valid_t = t < seq_len

    for k_start in range(0, 256, BLOCK_K):
        channels = k_start + tl.arange(0, BLOCK_K)
        c = channels[None, :]

        channel_major = batch * 256 * seq_len + c * seq_len + t
        convolved_offsets = batch * conv_stride_b + c * conv_stride_c + t

        convolved = tl.load(
            convolved_ptr + convolved_offsets,
            mask=valid_t,
            other=0.0,
        ).to(tl.float32)
        gated = tl.load(
            gated_ptr + channel_major,
            mask=valid_t,
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            x0_ptr + channel_major,
            mask=valid_t,
            other=0.0,
        ).to(tl.float32)
        filter_bias = tl.load(filter_bias_ptr + c).to(tl.float32)

        y = (convolved + gated * filter_bias) * x0

        k = channels[:, None]
        o = outputs[None, :]
        projection_weight = tl.load(
            projection_weight_ptr + o * 256 + k
        ).to(tl.float32)

        accumulator += tl.dot(y, projection_weight)

    projection_bias = tl.load(
        projection_bias_ptr + outputs
    ).to(tl.float32)
    accumulator += projection_bias[None, :]

    output_offsets = (
        batch * seq_len * 256
        + times[:, None] * 256
        + outputs[None, :]
    )
    tl.store(out_ptr + output_offsets, accumulator, mask=valid_t)


@triton.jit
def _gelu_fc2_residual_kernel(
    hidden_ptr,
    weight_ptr,
    residual_ptr,
    out_ptr,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)

    row_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offsets = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_rows = row_offsets[:, None] < rows
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, 1024, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        hidden = tl.load(
            hidden_ptr
            + row_offsets[:, None] * 1024
            + k_offsets[None, :],
            mask=valid_rows,
            other=0.0,
        ).to(tl.float32)

        hidden_cubed = hidden * hidden * hidden
        gelu_arg = 0.7978845608028654 * (
            hidden + 0.044715 * hidden_cubed
        )
        tanh_value = 2.0 * tl.sigmoid(2.0 * gelu_arg) - 1.0
        activated = 0.5 * hidden * (1.0 + tanh_value)

        weight = tl.load(
            weight_ptr
            + output_offsets[None, :] * 1024
            + k_offsets[:, None]
        ).to(tl.float32)

        accumulator += tl.dot(activated, weight)

    residual = tl.load(
        residual_ptr
        + row_offsets[:, None] * 256
        + output_offsets[None, :],
        mask=valid_rows,
        other=0.0,
    ).to(tl.float32)

    tl.store(
        out_ptr
        + row_offsets[:, None] * 256
        + output_offsets[None, :],
        accumulator + residual,
        mask=valid_rows,
    )


def _layer_norm_prefix(x, weight, bias, eps):
    rows = x.numel() // 256
    out = torch.empty_like(x)
    _layer_norm_kernel[(rows,)](
        x,
        weight,
        bias,
        out,
        eps,
        BLOCK=256,
        num_warps=8,
    )
    return out


def _short_convolution_and_gate(u, weight, bias):
    batch_size, seq_len, _ = u.shape
    out = torch.empty(
        (2, batch_size, 256, seq_len),
        dtype=torch.float32,
        device=u.device,
    )
    _short_conv_gate_kernel[
        (triton.cdiv(seq_len, 32), 8, batch_size)
    ](
        u,
        weight,
        bias,
        out,
        seq_len,
        batch_size=batch_size,
        BLOCK_T=32,
        BLOCK_C=32,
        num_warps=4,
    )
    return out


def _filter_hidden(
    linear1_weight,
    linear1_bias,
    frequency,
    linear2_weight,
    linear2_bias,
    linear3_weight,
    linear3_bias,
    seq_len,
    device,
):
    out = torch.empty(
        (1, seq_len, 64),
        dtype=torch.float32,
        device=device,
    )
    _filter_hidden_kernel[(triton.cdiv(seq_len, 16),)](
        linear1_weight,
        linear1_bias,
        frequency,
        linear2_weight,
        linear2_bias,
        linear3_weight,
        linear3_bias,
        out,
        seq_len,
        BLOCK_T=16,
        num_warps=4,
    )
    return out


def _filter_project_modulate(hidden, weight, deltas, bias, shift):
    seq_len = hidden.shape[1]
    out = torch.empty(
        (256, seq_len),
        dtype=torch.float32,
        device=hidden.device,
    )
    _filter_project_modulate_kernel[
        (triton.cdiv(seq_len, 32), 4)
    ](
        hidden,
        weight,
        deltas,
        bias,
        out,
        seq_len,
        shift,
        BLOCK_T=32,
        BLOCK_C=64,
        num_warps=8,
    )
    return out


def _post_fft_projection(
    convolved,
    gated,
    x0,
    filter_bias,
    projection_weight,
    projection_bias,
):
    batch_size, _, seq_len = convolved.shape
    out = torch.empty(
        (batch_size, seq_len, 256),
        dtype=torch.float32,
        device=convolved.device,
    )
    _post_fft_projection_kernel[
        (triton.cdiv(seq_len, 64), 4, batch_size)
    ](
        convolved,
        gated,
        x0,
        filter_bias,
        projection_weight,
        projection_bias,
        out,
        convolved.stride(0),
        convolved.stride(1),
        seq_len,
        BLOCK_T=64,
        BLOCK_O=64,
        BLOCK_K=32,
        num_warps=4,
    )
    return out


def _gelu_fc2_residual(hidden, weight, residual):
    rows = hidden.shape[0]
    out = torch.empty(
        (rows, 256),
        dtype=torch.float32,
        device=hidden.device,
    )
    _gelu_fc2_residual_kernel[
        (triton.cdiv(rows, 64), 4)
    ](
        hidden,
        weight,
        residual,
        out,
        rows,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
    )
    return out


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    short_conv_weight: torch.Tensor,
    short_conv_bias: torch.Tensor,
    filter_linear1_weight: torch.Tensor,
    filter_linear1_bias: torch.Tensor,
    sin_freq: torch.Tensor,
    filter_linear2_weight: torch.Tensor,
    filter_linear2_bias: torch.Tensor,
    filter_linear3_weight: torch.Tensor,
    filter_linear3_bias: torch.Tensor,
    filter_linear_final_weight: torch.Tensor,
    filter_bias: torch.Tensor,
    exp_mod_deltas: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    mlp_fc1_weight: torch.Tensor,
    mlp_fc1_bias: torch.Tensor,
    mlp_fc2_weight: torch.Tensor,
    mlp_fc2_bias: torch.Tensor,
    layer_norm_eps: float,
    exp_mod_shift: float,
):
    batch_size, seq_len, _ = hidden_states.shape
    l_filter = min(seq_len, 32768)
    device = hidden_states.device

    hidden_prefix = hidden_states[:, :l_filter, :]
    normed1 = _layer_norm_prefix(
        hidden_prefix,
        norm1_weight,
        norm1_bias,
        layer_norm_eps,
    )

    normed1_tokens = normed1.reshape(batch_size * l_filter, 256)
    projected_tokens = torch.addmm(
        in_proj_bias,
        normed1_tokens,
        in_proj_weight.t(),
    )
    projected = projected_tokens.reshape(batch_size, l_filter, 768)

    short_conv = _short_convolution_and_gate(
        projected,
        short_conv_weight,
        short_conv_bias,
    )
    x0 = short_conv[0]
    gated = short_conv[1]

    filter_hidden = _filter_hidden(
        filter_linear1_weight,
        filter_linear1_bias,
        sin_freq,
        filter_linear2_weight,
        filter_linear2_bias,
        filter_linear3_weight,
        filter_linear3_bias,
        l_filter,
        device,
    )
    k = _filter_project_modulate(
        filter_hidden,
        filter_linear_final_weight,
        exp_mod_deltas,
        filter_bias,
        exp_mod_shift,
    )

    fft_size = 2 * l_filter
    k_f = torch.fft.rfft(
        k,
        n=fft_size,
        dim=-1,
        norm="forward",
    )
    v_f = torch.fft.rfft(gated, n=fft_size, dim=-1)
    v_f.mul_(k_f)
    convolved = torch.fft.irfft(
        v_f,
        n=fft_size,
        dim=-1,
        norm="forward",
    )[..., :l_filter]

    hyena_prefix = _post_fft_projection(
        convolved,
        gated,
        x0,
        filter_bias,
        out_proj_weight,
        out_proj_bias,
    )

    if l_filter == seq_len:
        hyena_out = hyena_prefix
    else:
        hyena_out = (
            out_proj_bias.reshape(1, 1, 256)
            .expand(batch_size, seq_len, 256)
            .clone()
        )
        hyena_out[:, :l_filter, :].copy_(hyena_prefix)

    residual_bias = hyena_out
    if l_filter == seq_len:
        normed2 = normed1
    else:
        normed2 = torch.empty_like(hidden_states)

    rows = batch_size * seq_len
    _residual_layer_norm_kernel[(rows,)](
        hidden_states,
        hyena_out,
        norm2_weight,
        norm2_bias,
        mlp_fc2_bias,
        residual_bias,
        normed2,
        layer_norm_eps,
        BLOCK=256,
        num_warps=8,
    )

    normed2_tokens = normed2.reshape(rows, 256)
    mlp_hidden = torch.addmm(
        mlp_fc1_bias,
        normed2_tokens,
        mlp_fc1_weight.t(),
    )

    output = _gelu_fc2_residual(
        mlp_hidden,
        mlp_fc2_weight,
        residual_bias.reshape(rows, 256),
    )
    return output.reshape(batch_size, seq_len, 256)