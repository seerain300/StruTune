# task: 036_convnextv2_layer_with_nhwc_persistence_backward
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=14/14 geomean=364.242x
# feedback best (5-workload sample during search): 614.033x
# torch fallback audit: B·自研为主 (mm×4)
# tokens: 3,422,715

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_projected_grad_kernel(
    grad_output_ptr, drop_mask_ptr, grad_projected_ptr, grad_bias_ptr,
    M: tl.constexpr, HW: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    grad_stride_b, grad_stride_c, grad_stride_h, grad_stride_w,
    mask_stride_b, inv_keep_prob,
    USE_DROP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
    SPLITS: tl.constexpr, SINGLE_WRITER_BIAS: tl.constexpr,
    PARTIAL_BIAS: tl.constexpr,
):
    pid_split = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    acc_bias = tl.zeros((BLOCK_C,), tl.float32)

    for start in range(0, M, BLOCK_M * SPLITS):
        offs_m = start + pid_split * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask = mask_m[:, None] & mask_c[None, :]

        b = offs_m // HW
        spatial = offs_m - b * HW
        h = spatial // W
        w = spatial - h * W

        offsets = (
            b[:, None] * grad_stride_b
            + offs_c[None, :] * grad_stride_c
            + h[:, None] * grad_stride_h
            + w[:, None] * grad_stride_w
        )
        grad = tl.load(
            grad_output_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        if USE_DROP:
            drop = tl.load(
                drop_mask_ptr + b * mask_stride_b,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            grad *= drop[:, None] * inv_keep_prob

        tl.store(
            grad_projected_ptr + offs_m[:, None] * C + offs_c[None, :],
            grad,
            mask=mask,
        )
        acc_bias += tl.sum(grad, axis=0)

    if SINGLE_WRITER_BIAS:
        tl.store(grad_bias_ptr + offs_c, acc_bias, mask=mask_c)
    elif PARTIAL_BIAS:
        tl.store(
            grad_bias_ptr + pid_split * C + offs_c,
            acc_bias,
            mask=mask_c,
        )
    else:
        tl.atomic_add(grad_bias_ptr + offs_c, acc_bias, mask=mask_c)


@triton.jit
def _reduce_partials_kernel(
    partials_ptr, output_ptr,
    C: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    acc = tl.zeros((BLOCK_C,), tl.float32)

    for start in range(0, SPLITS, BLOCK_SPLITS):
        offs_s = start + tl.arange(0, BLOCK_SPLITS)
        mask_s = offs_s < SPLITS
        values = tl.load(
            partials_ptr + offs_s[:, None] * C + offs_c[None, :],
            mask=mask_s[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(values, axis=0)

    tl.store(output_ptr + offs_c, acc, mask=mask_c)


@triton.jit
def _grn_reduction_kernel(
    grad_x_grn_ptr, x_gelu_ptr, norm_features_ptr, grn_weight_ptr,
    reductions_ptr, param_grads_ptr,
    HW: tl.constexpr, K: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    ONE_BATCH: tl.constexpr,
):
    b = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    grn_weight = tl.load(
        grn_weight_ptr + offs_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    norm_features = tl.load(
        norm_features_ptr + b * K + offs_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)

    acc_grad_x = tl.zeros((BLOCK_K,), tl.float32)
    acc_bias = tl.zeros((BLOCK_K,), tl.float32)

    for start in range(0, HW, BLOCK_S):
        offs_s = start + tl.arange(0, BLOCK_S)
        mask_s = offs_s < HW
        mask = mask_s[:, None] & mask_k[None, :]
        offsets = (b * HW + offs_s[:, None]) * K + offs_k[None, :]

        grad = tl.load(
            grad_x_grn_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        x_gelu = tl.load(
            x_gelu_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        acc_grad_x += tl.sum(grad * x_gelu, axis=0)
        acc_bias += tl.sum(grad, axis=0)

    tl.store(
        reductions_ptr + b * K + offs_k,
        acc_grad_x * grn_weight,
        mask=mask_k,
    )

    grad_weight = acc_grad_x * norm_features
    if ONE_BATCH:
        tl.store(param_grads_ptr + offs_k, grad_weight, mask=mask_k)
        tl.store(param_grads_ptr + K + offs_k, acc_bias, mask=mask_k)
    else:
        tl.atomic_add(param_grads_ptr + offs_k, grad_weight, mask=mask_k)
        tl.atomic_add(param_grads_ptr + K + offs_k, acc_bias, mask=mask_k)


@triton.jit
def _grn_gelu_backward_kernel(
    grad_x_grn_ptr, x_expanded_ptr, x_gelu_ptr,
    global_features_ptr, gf_mean_ptr, grn_weight_ptr, reductions_ptr,
    grad_x_expanded_ptr, grad_bias_ptr,
    HW: tl.constexpr, K: tl.constexpr, eps,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLITS: tl.constexpr, SINGLE_WRITER_BIAS: tl.constexpr,
    PARTIAL_BIAS: tl.constexpr,
):
    b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_split = tl.program_id(2)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    global_features = tl.load(
        global_features_ptr + b * K + offs_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)
    gf_mean = tl.load(gf_mean_ptr + b).to(tl.float32)
    grn_weight = tl.load(
        grn_weight_ptr + offs_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    grad_norm_features = tl.load(
        reductions_ptr + b * K + offs_k,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)

    inv_denom = 1.0 / (gf_mean + eps)
    norm_features = global_features * inv_denom
    grad_global_features = (
        grad_norm_features * inv_denom
        - grad_norm_features * global_features
        * inv_denom * inv_denom / K
    )
    inv_global = 1.0 / (global_features + eps)
    acc_bias = tl.zeros((BLOCK_K,), tl.float32)

    for start in range(0, HW, BLOCK_S * SPLITS):
        offs_s = start + pid_split * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < HW
        mask = mask_s[:, None] & mask_k[None, :]
        offsets = (b * HW + offs_s[:, None]) * K + offs_k[None, :]

        grad = tl.load(
            grad_x_grn_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        x_expanded = tl.load(
            x_expanded_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        x_gelu = tl.load(
            x_gelu_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        nonzero = tl.abs(x_expanded) > 1.0e-7
        safe_x = tl.where(nonzero, x_expanded, 1.0)
        cdf_approx = tl.where(nonzero, x_gelu / safe_x, 0.5)
        tanh_inner = 2.0 * cdf_approx - 1.0

        grad_x_gelu = (
            grad
            + grad * grn_weight[None, :] * norm_features[None, :]
            + x_gelu * grad_global_features[None, :] * inv_global[None, :]
        )

        x_sq = x_expanded * x_expanded
        pdf_approx = (
            0.5
            * (1.0 - tanh_inner * tanh_inner)
            * 0.7978845608028654
            * (1.0 + 0.134145 * x_sq)
        )
        gelu_grad = cdf_approx + x_expanded * pdf_approx
        grad_x_expanded = grad_x_gelu * gelu_grad

        tl.store(
            grad_x_expanded_ptr + offsets,
            grad_x_expanded,
            mask=mask,
        )
        acc_bias += tl.sum(grad_x_expanded, axis=0)

    if SINGLE_WRITER_BIAS:
        tl.store(grad_bias_ptr + offs_k, acc_bias, mask=mask_k)
    elif PARTIAL_BIAS:
        partial_id = b * SPLITS + pid_split
        tl.store(
            grad_bias_ptr + partial_id * K + offs_k,
            acc_bias,
            mask=mask_k,
        )
    else:
        tl.atomic_add(grad_bias_ptr + offs_k, acc_bias, mask=mask_k)


@triton.jit
def _persistent_grn_gelu_backward_kernel(
    grad_x_grn_ptr, x_expanded_ptr, x_gelu_ptr,
    global_features_ptr, gf_mean_ptr, grn_weight_ptr,
    grad_x_expanded_ptr, all_grads_ptr,
    B: tl.constexpr, HW: tl.constexpr, K: tl.constexpr, eps,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_s = offs_s < HW
    mask_k = offs_k < K
    mask = mask_s[:, None] & mask_k[None, :]

    grn_weight = tl.load(
        grn_weight_ptr + offs_k, mask=mask_k, other=0.0
    ).to(tl.float32)

    total_grn_weight = tl.zeros((BLOCK_K,), tl.float32)
    total_grn_bias = tl.zeros((BLOCK_K,), tl.float32)
    total_pwconv1_bias = tl.zeros((BLOCK_K,), tl.float32)

    for b in range(0, B):
        offsets = (b * HW + offs_s[:, None]) * K + offs_k[None, :]

        grad = tl.load(
            grad_x_grn_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        x_expanded = tl.load(
            x_expanded_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        x_gelu = tl.load(
            x_gelu_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        global_features = tl.load(
            global_features_ptr + b * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        gf_mean = tl.load(gf_mean_ptr + b).to(tl.float32)

        acc_grad_x = tl.sum(grad * x_gelu, axis=0)
        acc_grn_bias = tl.sum(grad, axis=0)

        inv_denom = 1.0 / (gf_mean + eps)
        norm_features = global_features * inv_denom
        grad_norm_features = acc_grad_x * grn_weight
        grad_global_features = (
            grad_norm_features * inv_denom
            - grad_norm_features * global_features
            * inv_denom * inv_denom / K
        )
        inv_global = 1.0 / (global_features + eps)

        grad_x_gelu = (
            grad
            + grad * grn_weight[None, :] * norm_features[None, :]
            + x_gelu * grad_global_features[None, :] * inv_global[None, :]
        )

        nonzero = tl.abs(x_expanded) > 1.0e-7
        safe_x = tl.where(nonzero, x_expanded, 1.0)
        cdf_approx = tl.where(nonzero, x_gelu / safe_x, 0.5)
        tanh_inner = 2.0 * cdf_approx - 1.0
        x_sq = x_expanded * x_expanded
        pdf_approx = (
            0.5
            * (1.0 - tanh_inner * tanh_inner)
            * 0.7978845608028654
            * (1.0 + 0.134145 * x_sq)
        )
        grad_x_expanded = grad_x_gelu * (
            cdf_approx + x_expanded * pdf_approx
        )

        tl.store(
            grad_x_expanded_ptr + offsets,
            grad_x_expanded,
            mask=mask,
        )

        total_grn_weight += acc_grad_x * norm_features
        total_grn_bias += acc_grn_bias
        total_pwconv1_bias += tl.sum(grad_x_expanded, axis=0)

    tl.store(all_grads_ptr + offs_k, total_grn_weight, mask=mask_k)
    tl.store(all_grads_ptr + K + offs_k, total_grn_bias, mask=mask_k)
    tl.store(
        all_grads_ptr + 2 * K + offs_k,
        total_pwconv1_bias,
        mask=mask_k,
    )


@triton.jit
def _chunked_persistent_grn_gelu_backward_kernel(
    grad_x_grn_ptr, x_expanded_ptr, x_gelu_ptr,
    global_features_ptr, gf_mean_ptr, grn_weight_ptr,
    grad_x_expanded_ptr, all_grads_ptr,
    B: tl.constexpr, HW: tl.constexpr, K: tl.constexpr, eps,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    grn_weight = tl.load(
        grn_weight_ptr + offs_k, mask=mask_k, other=0.0
    ).to(tl.float32)

    total_grn_weight = tl.zeros((BLOCK_K,), tl.float32)
    total_grn_bias = tl.zeros((BLOCK_K,), tl.float32)
    total_pwconv1_bias = tl.zeros((BLOCK_K,), tl.float32)

    for b in range(0, B):
        acc_grad_x = tl.zeros((BLOCK_K,), tl.float32)
        acc_grn_bias = tl.zeros((BLOCK_K,), tl.float32)

        for start in range(0, HW, BLOCK_S):
            offs_s = start + tl.arange(0, BLOCK_S)
            mask_s = offs_s < HW
            mask = mask_s[:, None] & mask_k[None, :]
            offsets = (
                (b * HW + offs_s[:, None]) * K + offs_k[None, :]
            )

            grad = tl.load(
                grad_x_grn_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            x_gelu = tl.load(
                x_gelu_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)

            acc_grad_x += tl.sum(grad * x_gelu, axis=0)
            acc_grn_bias += tl.sum(grad, axis=0)

        global_features = tl.load(
            global_features_ptr + b * K + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        gf_mean = tl.load(gf_mean_ptr + b).to(tl.float32)

        inv_denom = 1.0 / (gf_mean + eps)
        norm_features = global_features * inv_denom
        grad_norm_features = acc_grad_x * grn_weight
        grad_global_features = (
            grad_norm_features * inv_denom
            - grad_norm_features * global_features
            * inv_denom * inv_denom / K
        )
        inv_global = 1.0 / (global_features + eps)
        acc_pwconv1_bias = tl.zeros((BLOCK_K,), tl.float32)

        for start in range(0, HW, BLOCK_S):
            offs_s = start + tl.arange(0, BLOCK_S)
            mask_s = offs_s < HW
            mask = mask_s[:, None] & mask_k[None, :]
            offsets = (
                (b * HW + offs_s[:, None]) * K + offs_k[None, :]
            )

            grad = tl.load(
                grad_x_grn_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            x_expanded = tl.load(
                x_expanded_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            x_gelu = tl.load(
                x_gelu_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)

            grad_x_gelu = (
                grad
                + grad * grn_weight[None, :] * norm_features[None, :]
                + x_gelu * grad_global_features[None, :] * inv_global[None, :]
            )

            nonzero = tl.abs(x_expanded) > 1.0e-7
            safe_x = tl.where(nonzero, x_expanded, 1.0)
            cdf_approx = tl.where(nonzero, x_gelu / safe_x, 0.5)
            tanh_inner = 2.0 * cdf_approx - 1.0
            x_sq = x_expanded * x_expanded
            pdf_approx = (
                0.5
                * (1.0 - tanh_inner * tanh_inner)
                * 0.7978845608028654
                * (1.0 + 0.134145 * x_sq)
            )
            grad_x_expanded = grad_x_gelu * (
                cdf_approx + x_expanded * pdf_approx
            )

            tl.store(
                grad_x_expanded_ptr + offsets,
                grad_x_expanded,
                mask=mask,
            )
            acc_pwconv1_bias += tl.sum(grad_x_expanded, axis=0)

        total_grn_weight += acc_grad_x * norm_features
        total_grn_bias += acc_grn_bias
        total_pwconv1_bias += acc_pwconv1_bias

    tl.store(all_grads_ptr + offs_k, total_grn_weight, mask=mask_k)
    tl.store(all_grads_ptr + K + offs_k, total_grn_bias, mask=mask_k)
    tl.store(
        all_grads_ptr + 2 * K + offs_k,
        total_pwconv1_bias,
        mask=mask_k,
    )


@triton.jit
def _layernorm_input_backward_kernel(
    grad_x_ln_ptr, x_normalized_ptr, var_ptr, layernorm_weight_ptr,
    grad_x_dwconv_ptr,
    M: tl.constexpr, HW: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    grad_stride_b, grad_stride_h, grad_stride_w, grad_stride_c,
    norm_stride_b, norm_stride_h, norm_stride_w, norm_stride_c,
    var_stride_b, var_stride_h, var_stride_w,
    eps, BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask_row = row < M
    mask = mask_row & mask_c

    b = row // HW
    spatial = row - b * HW
    h = spatial // W
    w = spatial - h * W

    grad_offsets = (
        b * grad_stride_b
        + h * grad_stride_h
        + w * grad_stride_w
        + offs_c * grad_stride_c
    )
    norm_offsets = (
        b * norm_stride_b
        + h * norm_stride_h
        + w * norm_stride_w
        + offs_c * norm_stride_c
    )

    grad = tl.load(
        grad_x_ln_ptr + grad_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    normalized = tl.load(
        x_normalized_ptr + norm_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    gamma = tl.load(
        layernorm_weight_ptr + offs_c, mask=mask_c, other=0.0
    ).to(tl.float32)
    variance = tl.load(
        var_ptr
        + b * var_stride_b
        + h * var_stride_h
        + w * var_stride_w,
        mask=mask_row,
        other=0.0,
    ).to(tl.float32)

    grad_normalized = grad * gamma
    sum_grad = tl.sum(grad_normalized, axis=0)
    sum_grad_norm = tl.sum(grad_normalized * normalized, axis=0)
    inv_std = tl.rsqrt(variance + eps)

    grad_x = (
        grad_normalized
        - sum_grad / C
        - normalized * (sum_grad_norm / C)
    ) * inv_std

    tl.store(
        grad_x_dwconv_ptr + b * C * HW + offs_c * HW + spatial,
        grad_x,
        mask=mask,
    )


@triton.jit
def _layernorm_param_backward_tiled_kernel(
    grad_x_ln_ptr, x_normalized_ptr, grad_params_ptr,
    M: tl.constexpr, HW: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    grad_stride_b, grad_stride_h, grad_stride_w, grad_stride_c,
    norm_stride_b, norm_stride_h, norm_stride_w, norm_stride_c,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr, SPLITS: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_split = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    acc_weight = tl.zeros((BLOCK_C,), tl.float32)
    acc_bias = tl.zeros((BLOCK_C,), tl.float32)

    for start in range(0, M, BLOCK_M * SPLITS):
        offs_m = start + pid_split * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask = mask_m[:, None] & mask_c[None, :]

        b = offs_m // HW
        spatial = offs_m - b * HW
        h = spatial // W
        w = spatial - h * W

        grad_offsets = (
            b[:, None] * grad_stride_b
            + h[:, None] * grad_stride_h
            + w[:, None] * grad_stride_w
            + offs_c[None, :] * grad_stride_c
        )
        norm_offsets = (
            b[:, None] * norm_stride_b
            + h[:, None] * norm_stride_h
            + w[:, None] * norm_stride_w
            + offs_c[None, :] * norm_stride_c
        )

        grad = tl.load(
            grad_x_ln_ptr + grad_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        normalized = tl.load(
            x_normalized_ptr + norm_offsets, mask=mask, other=0.0
        ).to(tl.float32)

        acc_weight += tl.sum(grad * normalized, axis=0)
        acc_bias += tl.sum(grad, axis=0)

    partial_base = pid_split * (2 * C)
    tl.store(
        grad_params_ptr + partial_base + offs_c,
        acc_weight,
        mask=mask_c,
    )
    tl.store(
        grad_params_ptr + partial_base + C + offs_c,
        acc_bias,
        mask=mask_c,
    )


@triton.jit
def _layernorm_input_param_backward_fused_kernel(
    grad_x_ln_ptr, x_normalized_ptr, var_ptr, layernorm_weight_ptr,
    grad_x_dwconv_ptr, grad_params_ptr,
    M: tl.constexpr, HW: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    grad_stride_b, grad_stride_h, grad_stride_w, grad_stride_c,
    norm_stride_b, norm_stride_h, norm_stride_w, norm_stride_c,
    var_stride_b, var_stride_h, var_stride_w,
    eps, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
    SPLITS: tl.constexpr, SINGLE_WRITER: tl.constexpr,
    PARTIAL_PARAMS: tl.constexpr,
):
    pid_split = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    gamma = tl.load(
        layernorm_weight_ptr + offs_c, mask=mask_c, other=0.0
    ).to(tl.float32)

    acc_weight = tl.zeros((BLOCK_C,), tl.float32)
    acc_bias = tl.zeros((BLOCK_C,), tl.float32)

    for start in range(0, M, BLOCK_M * SPLITS):
        offs_m = start + pid_split * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        mask = mask_m[:, None] & mask_c[None, :]

        b = offs_m // HW
        spatial = offs_m - b * HW
        h = spatial // W
        w = spatial - h * W

        grad_offsets = (
            b[:, None] * grad_stride_b
            + h[:, None] * grad_stride_h
            + w[:, None] * grad_stride_w
            + offs_c[None, :] * grad_stride_c
        )
        norm_offsets = (
            b[:, None] * norm_stride_b
            + h[:, None] * norm_stride_h
            + w[:, None] * norm_stride_w
            + offs_c[None, :] * norm_stride_c
        )

        grad = tl.load(
            grad_x_ln_ptr + grad_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        normalized = tl.load(
            x_normalized_ptr + norm_offsets, mask=mask, other=0.0
        ).to(tl.float32)

        acc_weight += tl.sum(grad * normalized, axis=0)
        acc_bias += tl.sum(grad, axis=0)

        grad_normalized = grad * gamma[None, :]
        sum_grad = tl.sum(grad_normalized, axis=1)
        sum_grad_norm = tl.sum(grad_normalized * normalized, axis=1)

        variance = tl.load(
            var_ptr
            + b * var_stride_b
            + h * var_stride_h
            + w * var_stride_w,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        inv_std = tl.rsqrt(variance + eps)

        grad_x = (
            grad_normalized
            - sum_grad[:, None] / C
            - normalized * (sum_grad_norm[:, None] / C)
        ) * inv_std[:, None]

        output_offsets = (
            b[:, None] * C * HW
            + offs_c[None, :] * HW
            + spatial[:, None]
        )
        tl.store(
            grad_x_dwconv_ptr + output_offsets,
            grad_x,
            mask=mask,
        )

    if SINGLE_WRITER:
        tl.store(grad_params_ptr + offs_c, acc_weight, mask=mask_c)
        tl.store(grad_params_ptr + C + offs_c, acc_bias, mask=mask_c)
    elif PARTIAL_PARAMS:
        partial_base = pid_split * (2 * C)
        tl.store(
            grad_params_ptr + partial_base + offs_c,
            acc_weight,
            mask=mask_c,
        )
        tl.store(
            grad_params_ptr + partial_base + C + offs_c,
            acc_bias,
            mask=mask_c,
        )
    else:
        tl.atomic_add(grad_params_ptr + offs_c, acc_weight, mask=mask_c)
        tl.atomic_add(
            grad_params_ptr + C + offs_c, acc_bias, mask=mask_c
        )


def _pwconv2_backward(
    grad_output, x_grn, pwconv2_weight, drop_mask, drop_path_prob,
):
    B, C, H, W = grad_output.shape
    C4 = x_grn.shape[-1]
    M = B * H * W

    grad_projected = torch.empty(
        (M, C), device=grad_output.device, dtype=torch.float32
    )

    block_m = 64
    block_c = 32
    if M < 2048:
        split_cap = 32
    elif M < 16384:
        split_cap = 64
    else:
        split_cap = 128

    splits = min(split_cap, triton.cdiv(M, block_m))
    single_writer = splits == 1
    use_partials = splits >= 64

    grad_bias = torch.empty(
        (C,), device=grad_output.device, dtype=torch.float32
    )
    if use_partials:
        bias_acc = torch.empty(
            (splits, C), device=grad_output.device, dtype=torch.float32
        )
    elif single_writer:
        bias_acc = grad_bias
    else:
        grad_bias.zero_()
        bias_acc = grad_bias

    use_drop = drop_path_prob > 0.0
    inv_keep_prob = (
        1.0 / (1.0 - drop_path_prob) if use_drop else 1.0
    )

    _prepare_projected_grad_kernel[
        (splits, triton.cdiv(C, block_c))
    ](
        grad_output,
        drop_mask,
        grad_projected,
        bias_acc,
        M=M,
        HW=H * W,
        W=W,
        C=C,
        grad_stride_b=grad_output.stride(0),
        grad_stride_c=grad_output.stride(1),
        grad_stride_h=grad_output.stride(2),
        grad_stride_w=grad_output.stride(3),
        mask_stride_b=drop_mask.stride(0),
        inv_keep_prob=inv_keep_prob,
        USE_DROP=use_drop,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        SPLITS=splits,
        SINGLE_WRITER_BIAS=single_writer,
        PARTIAL_BIAS=use_partials,
        num_warps=4,
    )

    if use_partials:
        _reduce_partials_kernel[(triton.cdiv(C, 32),)](
            bias_acc,
            grad_bias,
            C=C,
            SPLITS=splits,
            BLOCK_SPLITS=16,
            BLOCK_C=32,
            num_warps=4,
        )

    grad_x_grn = torch.mm(
        grad_projected, pwconv2_weight
    ).reshape(B, H, W, C4)
    grad_weight = torch.mm(
        grad_projected.t(), x_grn.reshape(M, C4)
    )
    return grad_x_grn, grad_weight, grad_bias


def _layernorm_dispatch_geometry(M, HW):
    use_split = M >= 32768 or (HW >= 16384 and M >= 16384)

    if use_split:
        if M < 32768:
            return True, 16, 16, 16, 4
        if M < 131072:
            return True, 32, 16, 16, 4
        return True, 32, 16, 32, 4

    if M < 128:
        return False, 1, 128, M, 4
    if M < 512:
        return False, 2, 128, min(128, triton.cdiv(M, 2)), 4
    if M < 1024:
        return False, 4, 128, min(128, triton.cdiv(M, 4)), 8
    if M < 8192:
        return False, 8, 128, min(128, triton.cdiv(M, 8)), 8
    if HW >= 4096:
        return False, 8, 128, min(256, triton.cdiv(M, 8)), 8
    return False, 4, 128, min(256, triton.cdiv(M, 4)), 8


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
    B, C, H, W = grad_output.shape
    HW = H * W
    M = B * HW
    C4 = x_grn.shape[-1]

    grad_x_grn, grad_pwconv2_weight, grad_pwconv2_bias = (
        _pwconv2_backward(
            grad_output,
            x_grn,
            pwconv2_weight,
            drop_mask,
            drop_path_prob,
        )
    )

    grad_x_expanded = torch.empty(
        (B, H, W, C4),
        device=grad_output.device,
        dtype=torch.float32,
    )

    if HW <= 256:
        grn_all_grads = torch.empty(
            (3 * C4,),
            device=grad_output.device,
            dtype=torch.float32,
        )

        persistent_block_s = triton.next_power_of_2(HW)
        persistent_block_k = 16 if persistent_block_s <= 16 else 8

        _persistent_grn_gelu_backward_kernel[
            (triton.cdiv(C4, persistent_block_k),)
        ](
            grad_x_grn,
            x_expanded,
            x_gelu,
            global_features,
            gf_mean,
            grn_weight,
            grad_x_expanded,
            grn_all_grads,
            B=B,
            HW=HW,
            K=C4,
            eps=eps,
            BLOCK_S=persistent_block_s,
            BLOCK_K=persistent_block_k,
            num_warps=4,
        )

        grn_param_grads = grn_all_grads[:2 * C4].reshape(2, C4)
        grad_pwconv1_bias = grn_all_grads[2 * C4:]

    elif HW <= 1024 and B <= 4:
        grn_all_grads = torch.empty(
            (3 * C4,),
            device=grad_output.device,
            dtype=torch.float32,
        )

        chunk_block_s = 32 if HW <= 512 else 64
        chunk_block_k = 8

        _chunked_persistent_grn_gelu_backward_kernel[
            (triton.cdiv(C4, chunk_block_k),)
        ](
            grad_x_grn,
            x_expanded,
            x_gelu,
            global_features,
            gf_mean,
            grn_weight,
            grad_x_expanded,
            grn_all_grads,
            B=B,
            HW=HW,
            K=C4,
            eps=eps,
            BLOCK_S=chunk_block_s,
            BLOCK_K=chunk_block_k,
            num_warps=4,
        )

        grn_param_grads = grn_all_grads[:2 * C4].reshape(2, C4)
        grad_pwconv1_bias = grn_all_grads[2 * C4:]

    else:
        grn_reductions = torch.empty(
            (B, C4),
            device=grad_output.device,
            dtype=torch.float32,
        )
        if B == 1:
            grn_param_grads = torch.empty(
                (2, C4),
                device=grad_output.device,
                dtype=torch.float32,
            )
        else:
            grn_param_grads = torch.zeros(
                (2, C4),
                device=grad_output.device,
                dtype=torch.float32,
            )

        if B <= 2:
            grn_block_k, grn_block_s = 8, 64
        elif B <= 4:
            grn_block_k, grn_block_s = 16, 32
        else:
            grn_block_k, grn_block_s = 32, 16

        _grn_reduction_kernel[
            (B, triton.cdiv(C4, grn_block_k))
        ](
            grad_x_grn,
            x_gelu,
            norm_features,
            grn_weight,
            grn_reductions,
            grn_param_grads,
            HW=HW,
            K=C4,
            BLOCK_S=grn_block_s,
            BLOCK_K=grn_block_k,
            ONE_BATCH=B == 1,
            num_warps=4,
        )

        if B <= 4:
            gelu_block_s, gelu_block_k = 16, 16
        else:
            gelu_block_s, gelu_block_k = 8, 32

        if HW <= 16:
            spatial_splits = 1
        elif HW <= 64:
            spatial_splits = 2
        elif HW <= 256:
            spatial_splits = 4
        else:
            spatial_splits = 8

        if B >= 8 and HW >= 1024:
            writer_target = 32
        else:
            writer_target = 8
        gelu_splits = min(
            spatial_splits, max(1, writer_target // B)
        )

        bias_writers = B * gelu_splits
        single_writer_bias = bias_writers == 1
        use_bias_partials = M >= 8192 and bias_writers >= 16

        grad_pwconv1_bias = torch.empty(
            (C4,),
            device=grad_output.device,
            dtype=torch.float32,
        )
        if use_bias_partials:
            pwconv1_bias_acc = torch.empty(
                (bias_writers, C4),
                device=grad_output.device,
                dtype=torch.float32,
            )
        elif single_writer_bias:
            pwconv1_bias_acc = grad_pwconv1_bias
        else:
            grad_pwconv1_bias.zero_()
            pwconv1_bias_acc = grad_pwconv1_bias

        _grn_gelu_backward_kernel[
            (B, triton.cdiv(C4, gelu_block_k), gelu_splits)
        ](
            grad_x_grn,
            x_expanded,
            x_gelu,
            global_features,
            gf_mean,
            grn_weight,
            grn_reductions,
            grad_x_expanded,
            pwconv1_bias_acc,
            HW=HW,
            K=C4,
            eps=eps,
            BLOCK_S=gelu_block_s,
            BLOCK_K=gelu_block_k,
            SPLITS=gelu_splits,
            SINGLE_WRITER_BIAS=single_writer_bias,
            PARTIAL_BIAS=use_bias_partials,
            num_warps=8,
        )

        if use_bias_partials:
            _reduce_partials_kernel[(triton.cdiv(C4, 32),)](
                pwconv1_bias_acc,
                grad_pwconv1_bias,
                C=C4,
                SPLITS=bias_writers,
                BLOCK_SPLITS=16,
                BLOCK_C=32,
                num_warps=4,
            )

    grad_grn_weight = grn_param_grads[0].reshape(1, 1, 1, C4)
    grad_grn_bias = grn_param_grads[1].reshape(1, 1, 1, C4)

    grad_x_expanded_flat = grad_x_expanded.reshape(M, C4)
    grad_x_ln = torch.mm(
        grad_x_expanded_flat, pwconv1_weight
    ).reshape(B, H, W, C)
    grad_pwconv1_weight = torch.mm(
        grad_x_expanded_flat.t(), x_ln.reshape(M, C)
    )

    grad_x_dwconv = torch.empty(
        (B, C, H, W),
        device=grad_output.device,
        dtype=torch.float32,
    )

    (
        use_split_layernorm,
        ln_block_m,
        ln_block_c,
        ln_splits,
        ln_num_warps,
    ) = _layernorm_dispatch_geometry(M, HW)

    if use_split_layernorm:
        layernorm_param_grads = torch.empty(
            (2, C),
            device=grad_output.device,
            dtype=torch.float32,
        )
        layernorm_param_acc = torch.empty(
            (ln_splits, 2 * C),
            device=grad_output.device,
            dtype=torch.float32,
        )

        _layernorm_param_backward_tiled_kernel[
            (triton.cdiv(C, ln_block_c), ln_splits)
        ](
            grad_x_ln,
            x_normalized,
            layernorm_param_acc,
            M=M,
            HW=HW,
            W=W,
            C=C,
            grad_stride_b=grad_x_ln.stride(0),
            grad_stride_h=grad_x_ln.stride(1),
            grad_stride_w=grad_x_ln.stride(2),
            grad_stride_c=grad_x_ln.stride(3),
            norm_stride_b=x_normalized.stride(0),
            norm_stride_h=x_normalized.stride(1),
            norm_stride_w=x_normalized.stride(2),
            norm_stride_c=x_normalized.stride(3),
            BLOCK_M=ln_block_m,
            BLOCK_C=ln_block_c,
            SPLITS=ln_splits,
            num_warps=ln_num_warps,
        )

        _reduce_partials_kernel[(triton.cdiv(2 * C, 32),)](
            layernorm_param_acc,
            layernorm_param_grads,
            C=2 * C,
            SPLITS=ln_splits,
            BLOCK_SPLITS=16,
            BLOCK_C=32,
            num_warps=4,
        )

        _layernorm_input_backward_kernel[(M,)](
            grad_x_ln,
            x_normalized,
            var,
            layernorm_weight,
            grad_x_dwconv,
            M=M,
            HW=HW,
            W=W,
            C=C,
            grad_stride_b=grad_x_ln.stride(0),
            grad_stride_h=grad_x_ln.stride(1),
            grad_stride_w=grad_x_ln.stride(2),
            grad_stride_c=grad_x_ln.stride(3),
            norm_stride_b=x_normalized.stride(0),
            norm_stride_h=x_normalized.stride(1),
            norm_stride_w=x_normalized.stride(2),
            norm_stride_c=x_normalized.stride(3),
            var_stride_b=var.stride(0),
            var_stride_h=var.stride(1),
            var_stride_w=var.stride(2),
            eps=eps,
            BLOCK_C=128,
            num_warps=4,
        )
    else:
        single_writer_ln = ln_splits == 1
        use_ln_partials = M >= 8192 and ln_splits >= 128

        layernorm_param_grads = torch.empty(
            (2, C),
            device=grad_output.device,
            dtype=torch.float32,
        )

        if use_ln_partials:
            layernorm_param_acc = torch.empty(
                (ln_splits, 2 * C),
                device=grad_output.device,
                dtype=torch.float32,
            )
        elif single_writer_ln:
            layernorm_param_acc = layernorm_param_grads
        else:
            layernorm_param_grads.zero_()
            layernorm_param_acc = layernorm_param_grads

        _layernorm_input_param_backward_fused_kernel[
            (ln_splits,)
        ](
            grad_x_ln,
            x_normalized,
            var,
            layernorm_weight,
            grad_x_dwconv,
            layernorm_param_acc,
            M=M,
            HW=HW,
            W=W,
            C=C,
            grad_stride_b=grad_x_ln.stride(0),
            grad_stride_h=grad_x_ln.stride(1),
            grad_stride_w=grad_x_ln.stride(2),
            grad_stride_c=grad_x_ln.stride(3),
            norm_stride_b=x_normalized.stride(0),
            norm_stride_h=x_normalized.stride(1),
            norm_stride_w=x_normalized.stride(2),
            norm_stride_c=x_normalized.stride(3),
            var_stride_b=var.stride(0),
            var_stride_h=var.stride(1),
            var_stride_w=var.stride(2),
            eps=eps,
            BLOCK_M=ln_block_m,
            BLOCK_C=ln_block_c,
            SPLITS=ln_splits,
            SINGLE_WRITER=single_writer_ln,
            PARTIAL_PARAMS=use_ln_partials,
            num_warps=ln_num_warps,
        )

        if use_ln_partials:
            _reduce_partials_kernel[(triton.cdiv(2 * C, 32),)](
                layernorm_param_acc,
                layernorm_param_grads,
                C=2 * C,
                SPLITS=ln_splits,
                BLOCK_SPLITS=16,
                BLOCK_C=32,
                num_warps=4,
            )

    grad_layernorm_weight = layernorm_param_grads[0]
    grad_layernorm_bias = layernorm_param_grads[1]

    grad_x, grad_dwconv_weight, grad_dwconv_bias = (
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
            [True, True, True],
        )
    )
    grad_x.add_(grad_output)

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