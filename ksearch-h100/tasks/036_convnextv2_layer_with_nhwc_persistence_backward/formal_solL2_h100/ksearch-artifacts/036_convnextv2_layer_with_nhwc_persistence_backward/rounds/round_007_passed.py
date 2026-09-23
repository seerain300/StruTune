# solution=GPT-5.6-Sol_036_convnextv2_layer_with_nhwc_persistence_backward_triton_optimized_r7 score=22.38887738239358 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_vector_parameter_reductions_kernel(
    grad_x_projected,
    grad_x_grn,
    x_grn_scaled,
    grad_x_expanded,
    grad_x_ln,
    x_normalized,
    grad_x_dwconv,
    grad_pwconv2_bias,
    grad_grn_weight,
    grad_grn_bias,
    grad_pwconv1_bias,
    grad_layernorm_weight,
    grad_layernorm_bias,
    grad_dwconv_bias,
    height,
    width,
    gp_s0,
    gp_s1,
    gp_s2,
    gp_s3,
    gg_s0,
    gg_s1,
    gg_s2,
    gg_s3,
    gs_s0,
    gs_s1,
    gs_s2,
    gs_s3,
    ge_s0,
    ge_s1,
    ge_s2,
    ge_s3,
    gl_s0,
    gl_s1,
    gl_s2,
    gl_s3,
    xn_s0,
    xn_s1,
    xn_s2,
    xn_s3,
    gd_s0,
    gd_s1,
    gd_s2,
    gd_s3,
    n_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    channel_pid = tl.program_id(0)
    tile_pid = tl.program_id(1)

    token = tile_pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = token < n_tokens

    spatial_size = height * width
    batch = token // spatial_size
    spatial = token - batch * spatial_size
    row = spatial // width
    col = spatial - row * width

    if channel_pid < 512:
        channel = channel_pid

        grn_offset = (
            batch * gg_s0
            + row * gg_s1
            + col * gg_s2
            + channel * gg_s3
        )
        scaled_offset = (
            batch * gs_s0
            + row * gs_s1
            + col * gs_s2
            + channel * gs_s3
        )
        expanded_offset = (
            batch * ge_s0
            + row * ge_s1
            + col * ge_s2
            + channel * ge_s3
        )

        grn_grad = tl.load(
            grad_x_grn + grn_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scaled = tl.load(
            x_grn_scaled + scaled_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        expanded_grad = tl.load(
            grad_x_expanded + expanded_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        grn_weight_partial = tl.sum(grn_grad * scaled, axis=0)
        grn_bias_partial = tl.sum(grn_grad, axis=0)
        pwconv1_bias_partial = tl.sum(expanded_grad, axis=0)

        tl.atomic_add(
            grad_grn_weight + channel,
            grn_weight_partial,
        )
        tl.atomic_add(
            grad_grn_bias + channel,
            grn_bias_partial,
        )
        tl.atomic_add(
            grad_pwconv1_bias + channel,
            pwconv1_bias_partial,
        )
    else:
        channel = channel_pid - 512

        projected_offset = (
            batch * gp_s0
            + row * gp_s1
            + col * gp_s2
            + channel * gp_s3
        )
        ln_offset = (
            batch * gl_s0
            + row * gl_s1
            + col * gl_s2
            + channel * gl_s3
        )
        normalized_offset = (
            batch * xn_s0
            + row * xn_s1
            + col * xn_s2
            + channel * xn_s3
        )
        dwconv_offset = (
            batch * gd_s0
            + channel * gd_s1
            + row * gd_s2
            + col * gd_s3
        )

        projected_grad = tl.load(
            grad_x_projected + projected_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        ln_grad = tl.load(
            grad_x_ln + ln_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normalized = tl.load(
            x_normalized + normalized_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dwconv_grad = tl.load(
            grad_x_dwconv + dwconv_offset,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        pwconv2_bias_partial = tl.sum(projected_grad, axis=0)
        layernorm_weight_partial = tl.sum(
            ln_grad * normalized,
            axis=0,
        )
        layernorm_bias_partial = tl.sum(ln_grad, axis=0)
        dwconv_bias_partial = tl.sum(dwconv_grad, axis=0)

        tl.atomic_add(
            grad_pwconv2_bias + channel,
            pwconv2_bias_partial,
        )
        tl.atomic_add(
            grad_layernorm_weight + channel,
            layernorm_weight_partial,
        )
        tl.atomic_add(
            grad_layernorm_bias + channel,
            layernorm_bias_partial,
        )
        tl.atomic_add(
            grad_dwconv_bias + channel,
            dwconv_bias_partial,
        )


def _batched_weight_gradient(
    grad: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor:
    batch = grad.shape[0]
    grad_batched = grad.reshape(batch, -1, grad.shape[-1])
    activation_batched = activation.reshape(
        batch,
        -1,
        activation.shape[-1],
    )
    partials = torch.bmm(
        grad_batched.transpose(1, 2),
        activation_batched,
    )
    return partials.sum(dim=0)


def _parameter_graph_impl(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    x_dwconv: torch.Tensor,
    x_nhwc: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    x_normalized: torch.Tensor,
    x_ln: torch.Tensor,
    x_expanded: torch.Tensor,
    x_gelu: torch.Tensor,
    global_features: torch.Tensor,
    gf_mean: torch.Tensor,
    norm_features: torch.Tensor,
    x_grn_scaled: torch.Tensor,
    x_grn: torch.Tensor,
    dwconv_weight: torch.Tensor,
    layernorm_weight: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    grn_weight: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    drop_mask: torch.Tensor,
    drop_path_prob: float,
    eps: float,
):
    B = grad_output.shape[0]
    C = grad_output.shape[1]
    H = grad_output.shape[2]
    W = grad_output.shape[3]
    C4 = x_expanded.shape[-1]

    grad_residual = grad_output
    if drop_path_prob > 0.0:
        grad_x_nchw = (
            grad_output * drop_mask / (1.0 - drop_path_prob)
        )
    else:
        grad_x_nchw = grad_output

    grad_x_projected = grad_x_nchw.permute(0, 2, 3, 1)

    grad_x_grn = torch.matmul(
        grad_x_projected,
        pwconv2_weight,
    )
    grad_pwconv2_weight = _batched_weight_gradient(
        grad_x_projected,
        x_grn,
    )

    grad_x_grn_scaled = grad_x_grn * grn_weight
    grad_x_gelu = (
        grad_x_grn
        + grad_x_grn_scaled * norm_features
    )
    grad_norm_features = (
        grad_x_grn_scaled * x_gelu
    ).sum(dim=(1, 2), keepdim=True)

    gf_denom = gf_mean + eps
    grad_global_features = grad_norm_features / gf_denom
    grad_gf_mean = (
        -grad_norm_features
        * global_features
        / (gf_denom * gf_denom)
    )
    grad_global_features = (
        grad_global_features + grad_gf_mean / C4
    )
    grad_x_gelu = (
        grad_x_gelu
        + x_gelu
        * grad_global_features
        / (global_features + eps)
    )

    sqrt_2_over_pi = 0.7978845608028654
    cdf_coeff = 0.044715
    x2 = x_expanded * x_expanded
    inner = sqrt_2_over_pi * (
        x_expanded
        + cdf_coeff * x_expanded * x2
    )
    tanh_inner = torch.tanh(inner)
    cdf_approx = 0.5 * (1.0 + tanh_inner)
    pdf_approx = (
        0.5
        * (1.0 - tanh_inner * tanh_inner)
        * sqrt_2_over_pi
        * (1.0 + 3.0 * cdf_coeff * x2)
    )
    grad_x_expanded = grad_x_gelu * (
        cdf_approx + x_expanded * pdf_approx
    )

    grad_x_ln = torch.matmul(
        grad_x_expanded,
        pwconv1_weight,
    )
    grad_pwconv1_weight = _batched_weight_gradient(
        grad_x_expanded,
        x_ln,
    )

    grad_x_normalized = grad_x_ln * layernorm_weight

    centered = x_nhwc - mean
    std = torch.sqrt(var + eps)
    inv_std = 1.0 / std

    grad_x_nhwc = grad_x_normalized * inv_std
    grad_var = -(
        grad_x_normalized * centered
    ).sum(dim=-1, keepdim=True) / (
        2.0 * (var + eps) * std
    )
    grad_mean = -(
        grad_x_normalized * inv_std
    ).sum(dim=-1, keepdim=True)
    grad_mean = grad_mean + grad_var * (
        -2.0 * centered.sum(dim=-1, keepdim=True) / C
    )
    grad_x_nhwc = (
        grad_x_nhwc
        + grad_var * (2.0 * centered / C)
        + grad_mean / C
    )
    grad_x_dwconv = grad_x_nhwc.permute(0, 3, 1, 2)

    grad_pwconv2_bias = torch.zeros(
        (C,),
        device=grad_output.device,
        dtype=torch.float32,
    )
    grad_grn_weight_flat = torch.zeros(
        (C4,),
        device=grad_output.device,
        dtype=torch.float32,
    )
    grad_grn_bias_flat = torch.zeros(
        (C4,),
        device=grad_output.device,
        dtype=torch.float32,
    )
    grad_pwconv1_bias = torch.zeros(
        (C4,),
        device=grad_output.device,
        dtype=torch.float32,
    )
    grad_layernorm_weight = torch.zeros(
        (C,),
        device=grad_output.device,
        dtype=torch.float32,
    )
    grad_layernorm_bias = torch.zeros(
        (C,),
        device=grad_output.device,
        dtype=torch.float32,
    )
    grad_dwconv_bias = torch.zeros(
        (C,),
        device=grad_output.device,
        dtype=torch.float32,
    )

    block_size = 1024
    n_tokens = B * H * W
    grid = (
        C4 + C,
        triton.cdiv(n_tokens, block_size),
    )

    _fused_vector_parameter_reductions_kernel[grid](
        grad_x_projected,
        grad_x_grn,
        x_grn_scaled,
        grad_x_expanded,
        grad_x_ln,
        x_normalized,
        grad_x_dwconv,
        grad_pwconv2_bias,
        grad_grn_weight_flat,
        grad_grn_bias_flat,
        grad_pwconv1_bias,
        grad_layernorm_weight,
        grad_layernorm_bias,
        grad_dwconv_bias,
        H,
        W,
        grad_x_projected.stride(0),
        grad_x_projected.stride(1),
        grad_x_projected.stride(2),
        grad_x_projected.stride(3),
        grad_x_grn.stride(0),
        grad_x_grn.stride(1),
        grad_x_grn.stride(2),
        grad_x_grn.stride(3),
        x_grn_scaled.stride(0),
        x_grn_scaled.stride(1),
        x_grn_scaled.stride(2),
        x_grn_scaled.stride(3),
        grad_x_expanded.stride(0),
        grad_x_expanded.stride(1),
        grad_x_expanded.stride(2),
        grad_x_expanded.stride(3),
        grad_x_ln.stride(0),
        grad_x_ln.stride(1),
        grad_x_ln.stride(2),
        grad_x_ln.stride(3),
        x_normalized.stride(0),
        x_normalized.stride(1),
        x_normalized.stride(2),
        x_normalized.stride(3),
        grad_x_dwconv.stride(0),
        grad_x_dwconv.stride(1),
        grad_x_dwconv.stride(2),
        grad_x_dwconv.stride(3),
        n_tokens,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    grad_grn_weight = grad_grn_weight_flat.reshape(
        1,
        1,
        1,
        C4,
    )
    grad_grn_bias = grad_grn_bias_flat.reshape(
        1,
        1,
        1,
        C4,
    )

    grad_x_conv, grad_dwconv_weight, _ = (
        torch.ops.aten.convolution_backward.default(
            grad_x_dwconv,
            residual,
            dwconv_weight,
            [C],
            [1, 1],
            [3, 3],
            [1, 1],
            False,
            [0, 0],
            C,
            [True, True, False],
        )
    )

    grad_x = grad_x_conv + grad_residual

    return (
        grad_x,
        grad_dwconv_weight,
        grad_dwconv_bias,
        grad_layernorm_weight,
        grad_layernorm_bias,
        grad_pwconv1_weight,
        grad_pwconv1_bias,
        grad_grn_weight,
        grad_grn_bias,
        grad_pwconv2_weight,
        grad_pwconv2_bias,
    )


_parameter_graph = torch.compile(
    _parameter_graph_impl,
    fullgraph=True,
    dynamic=True,
)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    x_dwconv: torch.Tensor,
    x_nhwc: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    x_normalized: torch.Tensor,
    x_ln: torch.Tensor,
    x_expanded: torch.Tensor,
    x_gelu: torch.Tensor,
    global_features: torch.Tensor,
    gf_mean: torch.Tensor,
    norm_features: torch.Tensor,
    x_grn_scaled: torch.Tensor,
    x_grn: torch.Tensor,
    dwconv_weight: torch.Tensor,
    layernorm_weight: torch.Tensor,
    pwconv1_weight: torch.Tensor,
    grn_weight: torch.Tensor,
    pwconv2_weight: torch.Tensor,
    drop_mask: torch.Tensor,
    drop_path_prob: float,
    eps: float,
):
    return _parameter_graph(
        grad_output,
        residual,
        x_dwconv,
        x_nhwc,
        mean,
        var,
        x_normalized,
        x_ln,
        x_expanded,
        x_gelu,
        global_features,
        gf_mean,
        norm_features,
        x_grn_scaled,
        x_grn,
        dwconv_weight,
        layernorm_weight,
        pwconv1_weight,
        grn_weight,
        pwconv2_weight,
        drop_mask,
        drop_path_prob,
        eps,
    )