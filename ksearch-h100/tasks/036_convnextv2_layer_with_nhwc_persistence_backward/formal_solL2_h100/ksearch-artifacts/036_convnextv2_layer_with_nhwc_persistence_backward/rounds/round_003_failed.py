# solution=GPT-5.6-Sol_036_convnextv2_layer_with_nhwc_persistence_backward_triton_optimized_r3 score=-1.0 passed=False
import torch


def _sum_spatial_then_batch(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=(1, 2)).sum(dim=0)


def _batched_weight_gradient(
    grad: torch.Tensor,
    activation: torch.Tensor,
) -> torch.Tensor:
    batch = grad.shape[0]
    grad_batched = grad.reshape(batch, -1, grad.shape[-1])
    activation_batched = activation.reshape(batch, -1, activation.shape[-1])
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
    C = grad_output.shape[1]
    C4 = x_expanded.shape[-1]

    grad_residual = grad_output
    if drop_path_prob > 0.0:
        grad_x_nchw = grad_output * drop_mask / (1.0 - drop_path_prob)
    else:
        grad_x_nchw = grad_output

    grad_x_projected = grad_x_nchw.permute(0, 2, 3, 1)

    grad_x_grn = torch.matmul(grad_x_projected, pwconv2_weight)
    grad_pwconv2_weight = _batched_weight_gradient(
        grad_x_projected,
        x_grn,
    )
    grad_pwconv2_bias = _sum_spatial_then_batch(grad_x_projected)

    grad_x_grn_scaled = grad_x_grn * grn_weight
    grad_grn_weight = _sum_spatial_then_batch(
        grad_x_grn * x_grn_scaled
    ).reshape(1, 1, 1, C4)
    grad_grn_bias = _sum_spatial_then_batch(
        grad_x_grn
    ).reshape(1, 1, 1, C4)

    grad_x_gelu = grad_x_grn + grad_x_grn_scaled * norm_features
    grad_norm_features = (
        grad_x_grn_scaled * x_gelu
    ).sum(dim=(1, 2), keepdim=True)

    gf_denom = gf_mean + eps
    grad_global_features = (
        grad_norm_features / gf_denom
        - grad_norm_features
        * global_features
        / (gf_denom * gf_denom * C4)
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
        x_expanded + cdf_coeff * x_expanded * x2
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

    grad_x_ln = torch.matmul(grad_x_expanded, pwconv1_weight)
    grad_pwconv1_weight = _batched_weight_gradient(
        grad_x_expanded,
        x_ln,
    )
    grad_pwconv1_bias = _sum_spatial_then_batch(grad_x_expanded)

    grad_x_normalized = grad_x_ln * layernorm_weight
    grad_layernorm_weight = _sum_spatial_then_batch(
        grad_x_ln * x_normalized
    )
    grad_layernorm_bias = _sum_spatial_then_batch(grad_x_ln)

    inv_std = torch.rsqrt(var + eps)
    grad_sum = grad_x_normalized.sum(dim=-1, keepdim=True)
    grad_norm_sum = (
        grad_x_normalized * x_normalized
    ).sum(dim=-1, keepdim=True)
    grad_x_nhwc = (
        grad_x_normalized
        - grad_sum / C
        - x_normalized * (grad_norm_sum / C)
    ) * inv_std
    grad_x_dwconv = grad_x_nhwc.permute(0, 3, 1, 2)

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
    grad_dwconv_bias = grad_x_dwconv.sum(dim=(2, 3)).sum(dim=0)

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
    dynamic=False,
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