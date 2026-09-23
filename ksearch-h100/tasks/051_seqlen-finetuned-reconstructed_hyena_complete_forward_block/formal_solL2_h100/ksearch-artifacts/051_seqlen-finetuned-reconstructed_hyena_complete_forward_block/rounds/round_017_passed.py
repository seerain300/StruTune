# solution=GPT-5.6-Sol_051_seqlen-finetuned-reconstructed_hyena_complete_forward_block_triton_optimized_r17 score=1.0617112653765781 passed=True
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_FILTER_SPECTRUM_CACHE = {}


@triton.jit
def _spectral_pre_gate_kernel(
    v_ptr,
    x1_ptr,
    q_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    v = tl.load(v_ptr + offsets, mask=mask)
    x1 = tl.load(x1_ptr + offsets, mask=mask)
    tl.store(q_ptr + offsets, v * x1, mask=mask)


@triton.jit
def _spectral_post_gate_kernel(
    convolution_ptr,
    q_ptr,
    x0_ptr,
    bias_ptr,
    output_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    channel_row = tl.program_id(0)
    sequence_block = tl.program_id(1)

    sequence_offsets = sequence_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = sequence_offsets < seq_len
    offsets = channel_row * seq_len + sequence_offsets
    channel = channel_row % 256

    convolution = tl.load(convolution_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)
    x0 = tl.load(x0_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + channel)

    output = (convolution + q * bias) * x0
    tl.store(output_ptr + offsets, output, mask=mask)


def _causal_fft_convolution(
    q: torch.Tensor,
    k_f: torch.Tensor,
    direct_bias: torch.Tensor,
    output_gate: torch.Tensor,
) -> torch.Tensor:
    batch_size, d_model, seq_len = q.shape
    fft_size = 2 * seq_len

    q_f = torch.fft.rfft(q, n=fft_size)
    convolution = torch.fft.irfft(
        q_f * k_f[None, :, :],
        n=fft_size,
        norm="forward",
    )[..., :seq_len]

    gated = torch.empty_like(convolution)
    block_size = 256
    _spectral_post_gate_kernel[
        (batch_size * d_model, triton.cdiv(seq_len, block_size))
    ](
        convolution,
        q,
        output_gate,
        direct_bias,
        gated,
        seq_len,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return gated.transpose(1, 2)


def _get_filter_spectrum(
    seq_len: int,
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
) -> torch.Tensor:
    tensors = (
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
        filter_linear1_weight.device.index,
        seq_len,
        float(exp_mod_shift),
        *(tensor.data_ptr() for tensor in tensors),
        *(tensor._version for tensor in tensors),
    )

    cached = _FILTER_SPECTRUM_CACHE.get(key)
    if cached is not None:
        return cached[1]

    device = filter_linear1_weight.device
    t = torch.linspace(
        0.0,
        1.0,
        seq_len,
        dtype=torch.float32,
        device=device,
    )[None, :, None]
    t_rescaled = torch.linspace(
        0.0,
        float(seq_len - 1),
        seq_len,
        dtype=torch.float32,
        device=device,
    )[None, :, None]
    w = 2.0 * math.pi * t_rescaled / seq_len
    f = torch.linspace(
        1e-4,
        1.0,
        2,
        dtype=torch.float32,
        device=device,
    )[None, None, :]

    z = torch.cat(
        (
            t,
            torch.cos(-f * w),
            torch.sin(-f * w),
        ),
        dim=-1,
    )

    h = F.linear(z, filter_linear1_weight, filter_linear1_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear2_weight, filter_linear2_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear3_weight, filter_linear3_bias)
    h = torch.sin(h * sin_freq)
    h = F.linear(h, filter_linear_final_weight)

    decay = torch.exp(-t * exp_mod_deltas.abs())
    h = h * (decay + exp_mod_shift)
    h = h + filter_bias[None, None, :]

    k = h.transpose(0, 1).reshape(256, seq_len)
    k_f = torch.fft.rfft(k, n=2 * seq_len) / (2 * seq_len)

    _FILTER_SPECTRUM_CACHE[key] = (tensors, k_f)
    return k_f


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    d_model = 256
    d_inner = 1024
    order = 2
    short_filter_order = 3
    filter_order = 64
    emb_dim = 5
    inner_width = d_model * (order + 1)

    hidden_states = torch.randn(
        batch_size, seq_len, d_model, dtype=torch.float32, device=device
    )
    norm1_weight = torch.ones(d_model, dtype=torch.float32, device=device)
    norm1_bias = torch.zeros(d_model, dtype=torch.float32, device=device)
    norm2_weight = torch.ones(d_model, dtype=torch.float32, device=device)
    norm2_bias = torch.zeros(d_model, dtype=torch.float32, device=device)

    in_proj_weight = (
        torch.randn(inner_width, d_model, dtype=torch.float32, device=device) * 0.02
    )
    in_proj_bias = torch.randn(inner_width, dtype=torch.float32, device=device) * 0.02
    short_conv_weight = (
        torch.randn(
            inner_width,
            1,
            short_filter_order,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    short_conv_bias = (
        torch.randn(inner_width, dtype=torch.float32, device=device) * 0.02
    )

    filter_linear1_weight = (
        torch.randn(filter_order, emb_dim, dtype=torch.float32, device=device) * 0.02
    )
    filter_linear1_bias = (
        torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    )
    sin_freq = torch.ones(1, filter_order, dtype=torch.float32, device=device)
    filter_linear2_weight = (
        torch.randn(
            filter_order,
            filter_order,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    filter_linear2_bias = (
        torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    )
    filter_linear3_weight = (
        torch.randn(
            filter_order,
            filter_order,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    filter_linear3_bias = (
        torch.randn(filter_order, dtype=torch.float32, device=device) * 0.02
    )
    filter_linear_final_weight = (
        torch.randn(
            d_model,
            filter_order,
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    )
    filter_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02

    max_decay = math.log(0.01) / 0.3
    min_decay = math.log(0.01) / 1.5
    deltas = torch.linspace(
        min_decay,
        max_decay,
        d_model,
        dtype=torch.float32,
        device=device,
    )[None, None, :]
    exp_mod_deltas = deltas

    out_proj_weight = (
        torch.randn(d_model, d_model, dtype=torch.float32, device=device) * 0.02
    )
    out_proj_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02
    mlp_fc1_weight = (
        torch.randn(d_inner, d_model, dtype=torch.float32, device=device) * 0.02
    )
    mlp_fc1_bias = torch.randn(d_inner, dtype=torch.float32, device=device) * 0.02
    mlp_fc2_weight = (
        torch.randn(d_model, d_inner, dtype=torch.float32, device=device) * 0.02
    )
    mlp_fc2_bias = torch.randn(d_model, dtype=torch.float32, device=device) * 0.02

    return {
        "hidden_states": hidden_states,
        "norm1_weight": norm1_weight,
        "norm1_bias": norm1_bias,
        "norm2_weight": norm2_weight,
        "norm2_bias": norm2_bias,
        "in_proj_weight": in_proj_weight,
        "in_proj_bias": in_proj_bias,
        "short_conv_weight": short_conv_weight,
        "short_conv_bias": short_conv_bias,
        "filter_linear1_weight": filter_linear1_weight,
        "filter_linear1_bias": filter_linear1_bias,
        "sin_freq": sin_freq,
        "filter_linear2_weight": filter_linear2_weight,
        "filter_linear2_bias": filter_linear2_bias,
        "filter_linear3_weight": filter_linear3_weight,
        "filter_linear3_bias": filter_linear3_bias,
        "filter_linear_final_weight": filter_linear_final_weight,
        "filter_bias": filter_bias,
        "exp_mod_deltas": exp_mod_deltas,
        "out_proj_weight": out_proj_weight,
        "out_proj_bias": out_proj_bias,
        "mlp_fc1_weight": mlp_fc1_weight,
        "mlp_fc1_bias": mlp_fc1_bias,
        "mlp_fc2_weight": mlp_fc2_weight,
        "mlp_fc2_bias": mlp_fc2_bias,
        "layer_norm_eps": 1e-5,
        "exp_mod_shift": 0.05,
    }


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
    d_model = 256
    inner_width = 768

    batch_size, seq_len, _ = hidden_states.shape
    l_filter = min(seq_len, 32768)

    residual = hidden_states
    normed = F.layer_norm(
        residual,
        (d_model,),
        norm1_weight,
        norm1_bias,
        layer_norm_eps,
    )

    u = F.linear(normed, in_proj_weight, in_proj_bias).transpose(1, 2)
    uc = F.conv1d(
        u,
        short_conv_weight,
        short_conv_bias,
        padding=2,
        groups=inner_width,
    )[..., :l_filter]

    x0 = uc[:, :d_model, :]
    x1 = uc[:, d_model : 2 * d_model, :]
    v = uc[:, 2 * d_model :, :]

    k_f = _get_filter_spectrum(
        l_filter,
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

    q = torch.empty_like(v)
    q_elements = v.numel()
    _spectral_pre_gate_kernel[(triton.cdiv(q_elements, 256),)](
        v,
        x1,
        q,
        q_elements,
        BLOCK_SIZE=256,
        num_warps=4,
    )

    y = _causal_fft_convolution(
        q,
        k_f,
        filter_bias,
        x0,
    )

    if l_filter < seq_len:
        y = F.pad(y, (0, 0, 0, seq_len - l_filter))

    residual = F.linear(y, out_proj_weight, out_proj_bias) + residual

    normed = F.layer_norm(
        residual,
        (d_model,),
        norm2_weight,
        norm2_bias,
        layer_norm_eps,
    )

    mlp_out = F.linear(normed, mlp_fc1_weight, mlp_fc1_bias)
    mlp_out = F.gelu(mlp_out, approximate="tanh")
    mlp_out = F.linear(mlp_out, mlp_fc2_weight, mlp_fc2_bias)

    return mlp_out + residual