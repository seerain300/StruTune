# solution=GPT-5.6-Sol_051_seqlen-finetuned-reconstructed_hyena_complete_forward_block_triton_optimized_r11 score=2.665994825011985 passed=True
import math

import torch
import torch.nn.functional as F


torch._dynamo.config.cache_size_limit = 64


def _direct_causal_core(
    v: torch.Tensor,
    x1: torch.Tensor,
    h: torch.Tensor,
    filter_bias: torch.Tensor,
    l_filter: int,
):
    gated = v * x1
    kernel = h.squeeze(0).transpose(0, 1).flip(-1).unsqueeze(1)
    convolved = F.conv1d(
        gated,
        kernel,
        padding=l_filter - 1,
        groups=256,
    )[..., :l_filter]
    return convolved + gated * filter_bias.view(1, 256, 1)


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

    positions = torch.arange(
        l_filter,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    t = (positions / max(l_filter - 1, 1)).view(1, l_filter, 1)
    w = (positions * (2.0 * math.pi / l_filter)).view(1, l_filter, 1)
    frequencies = torch.tensor(
        [1.0e-4, 1.0],
        device=hidden_states.device,
        dtype=torch.float32,
    ).view(1, 1, 2)
    phases = -w * frequencies
    z = torch.cat((t, torch.cos(phases), torch.sin(phases)), dim=-1)

    h = F.linear(z, filter_linear1_weight, filter_linear1_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear2_weight, filter_linear2_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear3_weight, filter_linear3_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear_final_weight)

    decay = torch.exp(-t * exp_mod_deltas.abs())
    h = h * (decay + exp_mod_shift)
    h = h + filter_bias.view(1, 1, d_model)

    if l_filter in (128, 256, 512):
        v = _direct_causal_core(v, x1, h, filter_bias, l_filter)
    elif l_filter <= 1024 and batch_size * l_filter <= 4096:
        gated = v * x1
        kernel = h.squeeze(0).transpose(0, 1).flip(-1).unsqueeze(1)
        convolved = F.conv1d(
            gated,
            kernel,
            padding=l_filter - 1,
            groups=d_model,
        )[..., :l_filter]
        v = convolved + gated * filter_bias.view(1, d_model, 1)
    else:
        fft_size = 2 * l_filter
        kernel = h.transpose(0, 1).reshape(1, d_model, l_filter)
        kernel_f = torch.fft.rfft(kernel[0], n=fft_size) / fft_size
        gated = v * x1
        gated_f = torch.fft.rfft(gated.float(), n=fft_size)
        convolved = torch.fft.irfft(
            gated_f * kernel_f,
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


_compiled_run = torch.compile(_run_impl, fullgraph=True, dynamic=False)


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
        out_proj_weight,
        out_proj_bias,
        mlp_fc1_weight,
        mlp_fc1_bias,
        mlp_fc2_weight,
        mlp_fc2_bias,
        layer_norm_eps,
        exp_mod_shift,
    )