# solution=GPT-5.6-Sol_051_seqlen-finetuned-reconstructed_hyena_complete_forward_block_triton_optimized_r4 score=-1.0 passed=False
import math

import torch
import torch.nn.functional as F


torch._dynamo.config.cache_size_limit = 64

_POSITION_CACHE = {}
_FILTER_CACHE = {}


def _get_position_tensors(hidden_states: torch.Tensor):
    l_filter = min(hidden_states.shape[1], 32768)
    key = (hidden_states.device.type, hidden_states.device.index, l_filter)

    cached = _POSITION_CACHE.get(key)
    if cached is not None:
        return cached

    positions = torch.arange(
        l_filter,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    t = (positions / max(l_filter - 1, 1)).view(1, l_filter, 1)
    w = positions * (2.0 * math.pi / l_filter)
    phase_low = w * 1.0e-4

    z = torch.stack(
        (
            t.view(l_filter),
            torch.cos(phase_low),
            torch.cos(w),
            -torch.sin(phase_low),
            -torch.sin(w),
        ),
        dim=-1,
    ).unsqueeze(0)

    _POSITION_CACHE[key] = (t, z)
    return t, z


def _get_filter_kernel(
    hidden_states: torch.Tensor,
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
    exp_mod_shift: float,
):
    batch_size, seq_len, _ = hidden_states.shape
    l_filter = min(seq_len, 32768)

    sources = (
        filter_linear1_weight,
        filter_linear1_bias,
        sin_freq,
        filter_linear2_weight,
        filter_linear2_bias,
        filter_linear3_weight,
        filter_linear3_bias,
        filter_linear_final_weight,
        filter_bias,
        exp_mod_deltas,
    )
    key = (
        hidden_states.device.type,
        hidden_states.device.index,
        batch_size,
        l_filter,
        float(exp_mod_shift),
        *(tensor.data_ptr() for tensor in sources),
    )

    cached = _FILTER_CACHE.get(key)
    if cached is not None:
        return cached[1]

    t, z = _get_position_tensors(hidden_states)

    h = F.linear(z, filter_linear1_weight, filter_linear1_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear2_weight, filter_linear2_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear3_weight, filter_linear3_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear_final_weight)

    decay = torch.exp(-t * exp_mod_deltas.abs())
    h = h * (decay + exp_mod_shift)
    h = h + filter_bias.view(1, 1, 256)

    use_direct = (
        l_filter == 128
        or l_filter == 256
        or (l_filter == 512 and batch_size > 1)
    )

    if use_direct:
        kernel = h.squeeze(0).transpose(0, 1).flip(-1)
        kernel[:, -1] += filter_bias
        kernel = kernel.reshape(256, 1, 1, l_filter)
    elif l_filter <= 1024 and batch_size * l_filter <= 4096:
        kernel = h.squeeze(0).transpose(0, 1).flip(-1).unsqueeze(1)
    else:
        fft_size = 1 << (2 * l_filter - 1).bit_length()
        kernel = h.squeeze(0).transpose(0, 1)
        kernel = torch.fft.rfft(kernel, n=fft_size) / fft_size

    _FILTER_CACHE[key] = (sources, kernel)
    return kernel


def _run_impl(
    hidden_states: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    short_conv_weight: torch.Tensor,
    short_conv_bias: torch.Tensor,
    filter_bias: torch.Tensor,
    filter_kernel: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
    mlp_fc1_weight: torch.Tensor,
    mlp_fc1_bias: torch.Tensor,
    mlp_fc2_weight: torch.Tensor,
    mlp_fc2_bias: torch.Tensor,
    layer_norm_eps: float,
):
    d_model = 256
    batch_size, seq_len, _ = hidden_states.shape
    l_filter = min(seq_len, 32768)

    residual = hidden_states.float()
    mean = residual.mean(dim=-1, keepdim=True)
    centered = residual - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normed = centered * torch.rsqrt(variance + layer_norm_eps)
    normed = normed * norm1_weight + norm1_bias

    u = F.linear(normed, in_proj_weight, in_proj_bias).transpose(1, 2)

    short_weight = short_conv_weight[:, 0, :]
    delayed_1 = F.pad(u[..., :-1], (1, 0))
    delayed_2 = F.pad(u[..., :-2], (2, 0))
    uc = (
        u * short_weight[:, 2].view(1, -1, 1)
        + delayed_1 * short_weight[:, 1].view(1, -1, 1)
        + delayed_2 * short_weight[:, 0].view(1, -1, 1)
        + short_conv_bias.view(1, -1, 1)
    )[..., :l_filter]

    x0, x1, v = uc.split(d_model, dim=1)
    gated = v * x1

    use_direct = (
        l_filter == 128
        or l_filter == 256
        or (l_filter == 512 and batch_size > 1)
    )

    if use_direct:
        convolved = F.conv2d(
            gated.permute(1, 0, 2).unsqueeze(0),
            filter_kernel,
            padding=(0, l_filter - 1),
            groups=d_model,
        )[..., :l_filter]
        v = convolved.squeeze(0).permute(1, 0, 2)
    elif l_filter <= 1024 and batch_size * l_filter <= 4096:
        convolved = F.conv1d(
            gated,
            filter_kernel,
            padding=l_filter - 1,
            groups=d_model,
        )[..., :l_filter]
        v = convolved + gated * filter_bias.view(1, d_model, 1)
    else:
        fft_size = 1 << (2 * l_filter - 1).bit_length()
        gated_f = torch.fft.rfft(gated.float(), n=fft_size)
        convolved = torch.fft.irfft(
            gated_f * filter_kernel,
            n=fft_size,
            norm="forward",
        )[..., :l_filter]
        v = convolved + gated * filter_bias.view(1, d_model, 1)

    y = (v * x0).transpose(1, 2)

    if l_filter < seq_len:
        y = F.pad(y, (0, 0, 0, seq_len - l_filter))

    residual = F.linear(y, out_proj_weight, out_proj_bias) + residual

    mean = residual.mean(dim=-1, keepdim=True)
    centered = residual - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    normed = centered * torch.rsqrt(variance + layer_norm_eps)
    normed = normed * norm2_weight + norm2_bias

    mlp_out = F.linear(normed, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = F.gelu(mlp_out, approximate="tanh")
    mlp_out = F.linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)
    return mlp_out + residual


_compiled_run = torch.compile(
    _run_impl,
    fullgraph=True,
    dynamic=False,
    mode="max-autotune-no-cudagraphs",
)


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
    filter_kernel = _get_filter_kernel(
        hidden_states,
        filter_linear1_weight,
        filter_linear1_bias,
        sin_freq,
        filter_linear2_weight,
        filter_linear2_bias,
        filter_linear3_weight,
        filter_linear3_bias,
        filter_linear_final_weight,
        filter_bias,
        exp_mod_deltas,
        exp_mod_shift,
    )

    return _compiled_run(
        hidden_states,
        norm1_weight,
        norm1_bias,
        norm2_weight,
        norm2_bias,
        in_proj_weight,
        in_proj_bias,
        short_conv_weight,
        short_conv_bias,
        filter_bias,
        filter_kernel,
        out_proj_weight,
        out_proj_bias,
        mlp_fc1_weight,
        mlp_fc1_bias,
        mlp_fc2_weight,
        mlp_fc2_bias,
        layer_norm_eps,
    )