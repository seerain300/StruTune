# solution=GPT-5.6-Sol_036_convnextv2_layer_with_nhwc_persistence_backward_triton_optimized_r2 score=-1.0 passed=False
import torch


@torch.compile(
    fullgraph=True,
    dynamic=False,
    mode="max-autotune",
)
def _run_impl(
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
    B, C, H, W = grad_output.shape
    C4 = x_expanded.shape[-1]
    spatial_size = B * H * W

    if drop_path_prob > 0.0:
        grad_branch_nchw = (
            grad_output * drop_mask * (1.0 / (1.0 - drop_path_prob))
        )
    else:
        grad_branch_nchw = grad_output

    grad_projected = (
        grad_branch_nchw.permute(0, 2, 3, 1)
        .contiguous()
        .view(spatial_size, C)
    )
    x_grn_2d = x_grn.reshape(spatial_size, C4)

    grad_x_grn_2d = torch.mm(grad_projected, pwconv2_weight)
    grad_pwconv2_weight = torch.mm(grad_projected.t(), x_grn_2d)
    grad_pwconv2_bias = grad_projected.sum(dim=0)

    grad_x_grn = grad_x_grn_2d.view(B, H, W, C4)
    grad_x_grn_scaled = grad_x_grn * grn_weight

    grad_grn_weight = (grad_x_grn * x_grn_scaled).sum(
        dim=(0, 1, 2), keepdim=True
    )
    grad_grn_bias = grad_x_grn.sum(dim=(0, 1, 2), keepdim=True)

    grad_norm_features = (grad_x_grn_scaled * x_gelu).sum(
        dim=(1, 2), keepdim=True
    )

    gf_denominator = gf_mean + eps
    grad_global_features = (
        grad_norm_features / gf_denominator
        - grad_norm_features
        * global_features
        / (gf_denominator * gf_denominator * C4)
    )

    grad_x_gelu = (
        grad_x_grn
        + grad_x_grn_scaled * norm_features
        + x_gelu
        * grad_global_features
        / (global_features + eps)
    )

    sqrt_2_over_pi = 0.7978845608028654
    cdf_coeff = 0.044715
    expanded_sq = x_expanded * x_expanded
    inner = sqrt_2_over_pi * (
        x_expanded + cdf_coeff * x_expanded * expanded_sq
    )
    tanh_inner = torch.tanh(inner)
    gelu_grad = (
        0.5 * (1.0 + tanh_inner)
        + x_expanded
        * 0.5
        * (1.0 - tanh_inner * tanh_inner)
        * sqrt_2_over_pi
        * (1.0 + 3.0 * cdf_coeff * expanded_sq)
    )
    grad_x_expanded = grad_x_gelu * gelu_grad

    grad_x_expanded_2d = grad_x_expanded.view(spatial_size, C4)
    x_ln_2d = x_ln.reshape(spatial_size, C)

    grad_x_ln_2d = torch.mm(grad_x_expanded_2d, pwconv1_weight)
    grad_pwconv1_weight = torch.mm(
        grad_x_expanded_2d.t(), x_ln_2d
    )
    grad_pwconv1_bias = grad_x_expanded_2d.sum(dim=0)

    grad_x_ln = grad_x_ln_2d.view(B, H, W, C)
    grad_layernorm_weight = (
        grad_x_ln * x_normalized
    ).sum(dim=(0, 1, 2))
    grad_layernorm_bias = grad_x_ln_2d.sum(dim=0)

    grad_x_normalized = grad_x_ln * layernorm_weight
    centered = x_nhwc - mean
    inv_std = torch.rsqrt(var + eps)

    grad_var = -0.5 * (
        grad_x_normalized * centered
    ).sum(dim=-1, keepdim=True) * inv_std * inv_std * inv_std

    grad_mean = -(
        grad_x_normalized * inv_std
    ).sum(dim=-1, keepdim=True)
    grad_mean = grad_mean - (
        2.0
        * grad_var
        * centered.sum(dim=-1, keepdim=True)
        / C
    )

    grad_x_nhwc = (
        grad_x_normalized * inv_std
        + grad_var * (2.0 * centered / C)
        + grad_mean / C
    )
    grad_x_dwconv = grad_x_nhwc.permute(0, 3, 1, 2).contiguous()

    (
        grad_x_conv,
        grad_dwconv_weight,
        grad_dwconv_bias,
    ) = torch.ops.aten.convolution_backward(
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
        [True, True, True],
    )

    grad_x = grad_x_conv + grad_output

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
    return _run_impl(
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