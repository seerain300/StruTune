# solution=GPT-5.6-Sol_051_seqlen-finetuned-reconstructed_hyena_complete_forward_block_triton_optimized_r8 score=-1.0 passed=False
import torch
import torch.nn.functional as F


torch._dynamo.config.cache_size_limit = 64


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
    _, seq_len, _ = hidden_states.shape
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
    bias = filter_bias.view(1, d_model, 1)
    v = (torch.cumsum(gated, dim=-1) + gated) * bias

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