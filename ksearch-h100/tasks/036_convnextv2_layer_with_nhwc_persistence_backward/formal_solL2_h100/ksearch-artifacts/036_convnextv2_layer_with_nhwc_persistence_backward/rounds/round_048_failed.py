# solution=GPT-5.6-Sol_036_convnextv2_layer_with_nhwc_persistence_backward_triton_optimized_r48 score=-1.0 passed=False
I’m keeping the established fused graph and targeting the small-workload reduction overhead. The LayerNorm kernel will use direct parameter stores when a launch has a single program, while retaining atomic accumulation for larger shapes.import torch
import triton
import triton.language as tl


torch._dynamo.config.cache_size_limit = 32


@triton.jit
def _layernorm_backward_nchw_kernel(
    grad_x_ln_ptr,
    x_normalized_ptr,
    var_ptr,
    layernorm_weight_ptr,
    grad_x_dwconv_ptr,
    grad_parameter_ptr,
    eps: tl.constexpr,
    ROWS: tl.constexpr,
    HW: tl.constexpr,
    W: tl.constexpr,
    xnorm_stride_b: tl.constexpr,
    xnorm_stride_h: tl.constexpr,
    xnorm_stride_w: tl.constexpr,
    xnorm_stride_c: tl.constexpr,
    var_stride_b: tl.constexpr,
    var_stride_h: tl.constexpr,
    var_stride_w: tl.constexpr,
    C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GROUP_BLOCKS: tl.constexpr,
    SINGLE_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    channels = tl.arange(0, C)
    layernorm_weight = tl.load(layernorm_weight_ptr + channels)

    grad_weight_acc = tl.zeros((C,), dtype=tl.float32)
    grad_bias_acc = tl.zeros((C,), dtype=tl.float32)
    grad_dwconv_bias_acc = tl.zeros((C,), dtype=tl.float32)

    for group_idx in tl.static_range(0, GROUP_BLOCKS):
        block_idx = pid * GROUP_BLOCKS + group_idx
        rows = block_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < ROWS

        hw = rows % HW
        batch = rows // HW
        h = hw // W
        w = hw - h * W

        grad_x_ln = tl.load(
            grad_x_ln_ptr + channels[:, None] + rows[None, :] * C,
            mask=row_mask[None, :],
            other=0.0,
        )

        xnorm_base = (
            batch * xnorm_stride_b
            + h * xnorm_stride_h
            + w * xnorm_stride_w
        )
        x_normalized = tl.load(
            x_normalized_ptr
            + channels[:, None] * xnorm_stride_c
            + xnorm_base[None, :],
            mask=row_mask[None, :],
            other=0.0,
        )

        grad_weight_acc += tl.sum(
            grad_x_ln * x_normalized,
            axis=1,
        )
        grad_bias_acc += tl.sum(grad_x_ln, axis=1)

        grad_x_normalized = grad_x_ln * layernorm_weight[:, None]
        grad_sum = tl.sum(grad_x_normalized, axis=0)
        grad_norm_sum = tl.sum(
            grad_x_normalized * x_normalized,
            axis=0,
        )

        var_offset = (
            batch * var_stride_b
            + h * var_stride_h
            + w * var_stride_w
        )
        var_value = tl.load(
            var_ptr + var_offset,
            mask=row_mask,
            other=0.0,
        )
        inv_std = tl.rsqrt(var_value + eps)
        inv_c = 1.0 / C

        grad_x_nhwc = inv_std[None, :] * (
            grad_x_normalized
            - grad_sum[None, :] * inv_c
            - x_normalized * grad_norm_sum[None, :] * inv_c
        )

        grad_dwconv_bias_acc += tl.sum(grad_x_nhwc, axis=1)

        output_offsets = (
            batch[None, :] * C * HW
            + channels[:, None] * HW
            + hw[None, :]
        )
        tl.store(
            grad_x_dwconv_ptr + output_offsets,
            grad_x_nhwc,
            mask=row_mask[None, :],
        )

    if SINGLE_PROGRAM:
        tl.store(
            grad_parameter_ptr + channels,
            grad_weight_acc,
        )
        tl.store(
            grad_parameter_ptr + C + channels,
            grad_bias_acc,
        )
        tl.store(
            grad_parameter_ptr + 2 * C + channels,
            grad_dwconv_bias_acc,
        )
    else:
        tl.atomic_add(
            grad_parameter_ptr + channels,
            grad_weight_acc,
        )
        tl.atomic_add(
            grad_parameter_ptr + C + channels,
            grad_bias_acc,
        )
        tl.atomic_add(
            grad_parameter_ptr + 2 * C + channels,
            grad_dwconv_bias_acc,
        )


_layernorm_lib = torch.library.Library(
    "convnext_fused",
    "DEF",
)
_layernorm_lib.define(
    "layernorm_backward_nchw("
    "Tensor grad_x_ln, "
    "Tensor x_normalized, "
    "Tensor var, "
    "Tensor layernorm_weight, "
    "float eps, "
    "int B, "
    "int H, "
    "int W"
    ") -> (Tensor, Tensor)"
)


def _layernorm_backward_nchw_impl(
    grad_x_ln,
    x_normalized,
    var,
    layernorm_weight,
    eps,
    B,
    H,
    W,
):
    C = grad_x_ln.shape[-1]
    rows = B * H * W

    output = torch.empty(
        (B, C, H, W),
        device=grad_x_ln.device,
        dtype=grad_x_ln.dtype,
    )

    (
        xnorm_stride_b,
        xnorm_stride_h,
        xnorm_stride_w,
        xnorm_stride_c,
    ) = x_normalized.stride()
    var_stride_b, var_stride_h, var_stride_w, _ = var.stride()

    if rows <= 1024:
        block_rows = 8
        group_blocks = 1
        num_warps = 2
    elif rows < 4096:
        block_rows = 8
        group_blocks = 1
        num_warps = 4
    elif rows < 32768:
        block_rows = 16
        group_blocks = 2
        num_warps = 4
    else:
        block_rows = 32
        group_blocks = 4
        num_warps = 8

    num_blocks = triton.cdiv(rows, block_rows)
    num_programs = triton.cdiv(num_blocks, group_blocks)

    parameter_grads = torch.zeros(
        (3, C),
        device=grad_x_ln.device,
        dtype=grad_x_ln.dtype,
    )

    _layernorm_backward_nchw_kernel[(num_programs,)](
        grad_x_ln,
        x_normalized,
        var,
        layernorm_weight,
        output,
        parameter_grads,
        eps,
        rows,
        H * W,
        W,
        xnorm_stride_b,
        xnorm_stride_h,
        xnorm_stride_w,
        xnorm_stride_c,
        var_stride_b,
        var_stride_h,
        var_stride_w,
        C=C,
        BLOCK_ROWS=block_rows,
        GROUP_BLOCKS=group_blocks,
        SINGLE_PROGRAM=num_programs == 1,
        num_warps=num_warps,
    )

    return output, parameter_grads


_layernorm_cuda_lib = torch.library.Library(
    "convnext_fused",
    "IMPL",
    "CUDA",
)
_layernorm_cuda_lib.impl(
    "layernorm_backward_nchw",
    _layernorm_backward_nchw_impl,
)


@torch.library.register_fake(
    "convnext_fused::layernorm_backward_nchw"
)
def _layernorm_backward_nchw_fake(
    grad_x_ln,
    x_normalized,
    var,
    layernorm_weight,
    eps,
    B,
    H,
    W,
):
    C = grad_x_ln.shape[-1]
    return (
        grad_x_ln.new_empty((B, C, H, W)),
        grad_x_ln.new_empty((3, C)),
    )


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
            grad_output
            * drop_mask
            * (1.0 / (1.0 - drop_path_prob))
        )
    else:
        grad_branch_nchw = grad_output

    grad_projected = (
        grad_branch_nchw.permute(0, 2, 3, 1)
        .contiguous()
        .view(spatial_size, C)
    )
    x_grn_2d = x_grn.reshape(spatial_size, C4)

    grad_x_grn_2d = torch.mm(
        grad_projected,
        pwconv2_weight,
    )
    grad_pwconv2_weight = torch.mm(
        grad_projected.t(),
        x_grn_2d,
    )
    grad_pwconv2_bias = grad_projected.sum(dim=0)

    grad_x_grn = grad_x_grn_2d.view(B, H, W, C4)

    grad_grn_weight = (
        grad_x_grn * x_grn_scaled
    ).sum(dim=(0, 1, 2), keepdim=True)
    grad_grn_bias = grad_x_grn.sum(
        dim=(0, 1, 2),
        keepdim=True,
    )

    grad_norm_features = (
        grad_x_grn * grn_weight * x_gelu
    ).sum(dim=(1, 2), keepdim=True)

    gf_denominator = gf_mean + eps
    grad_global_features = (
        grad_norm_features
        * (1.0 - norm_features / C4)
        / gf_denominator
    )

    sqrt_2_over_pi = 0.7978845608028654
    cdf_coeff = 0.044715
    expanded_sq = x_expanded * x_expanded
    expanded_nonzero = x_expanded != 0.0

    cdf_approx = torch.where(
        expanded_nonzero,
        x_gelu
        / torch.where(
            expanded_nonzero,
            x_expanded,
            1.0,
        ),
        0.5,
    )

    pdf_approx = (
        2.0
        * cdf_approx
        * (1.0 - cdf_approx)
        * sqrt_2_over_pi
        * (1.0 + 3.0 * cdf_coeff * expanded_sq)
    )

    grad_x_expanded = (
        grad_x_grn
        * (1.0 + grn_weight * norm_features)
        + x_gelu
        * grad_global_features
        / (global_features + eps)
    ) * (
        cdf_approx + x_expanded * pdf_approx
    )

    grad_x_expanded_2d = grad_x_expanded.view(
        spatial_size,
        C4,
    )
    x_ln_2d = x_ln.reshape(spatial_size, C)

    grad_x_ln_2d = torch.mm(
        grad_x_expanded_2d,
        pwconv1_weight,
    )
    grad_pwconv1_weight = torch.mm(
        grad_x_expanded_2d.t(),
        x_ln_2d,
    )
    grad_pwconv1_bias = grad_x_expanded_2d.sum(dim=0)

    grad_x_ln = grad_x_ln_2d.view(B, H, W, C)

    (
        grad_x_dwconv,
        grad_parameter_values,
    ) = torch.ops.convnext_fused.layernorm_backward_nchw(
        grad_x_ln,
        x_normalized,
        var,
        layernorm_weight,
        eps,
        B,
        H,
        W,
    )

    grad_layernorm_weight = grad_parameter_values[0]
    grad_layernorm_bias = grad_parameter_values[1]
    grad_dwconv_bias = grad_parameter_values[2]

    (
        grad_x_conv,
        grad_dwconv_weight,
        _,
    ) = torch.ops.aten.convolution_backward(
        grad_x_dwconv,
        residual,
        dwconv_weight,
        None,
        [1, 1],
        [3, 3],
        [1, 1],
        False,
        [0, 0],
        C,
        [True, True, False],
    )

    grad_x_conv.add_(grad_output)
    grad_x = grad_x_conv

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