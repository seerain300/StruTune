import torch
import triton
import triton.language as tl


# 1) Depthwise conv2d (groups=C): y[b, c, oh, ow] = sum_{kh,kw} x[b, c, oh+kh-pad, ow+kw-pad] * w[c, 0, kh, kw]
@triton.jit
def depthwise_conv2d_groupsC_per_output_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    pad_h, pad_w,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_off = b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            w_off = c * stride_wC + kh * stride_wKH + kw * stride_wKW
            x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
            w_val = tl.load(w_ptr + w_off)
            acc += x_val * w_val

    # store result (cast to original dtype of x)
    y_off = b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_ptr + y_off, acc)


# 2) NHWC permute: out[b, h, w, c] = in[b, c, h, w]
@triton.jit
def permute_nchw_to_nhwc_kernel(
    in_ptr, out_ptr,
    B, C, H, W,
    stride_inB, stride_inC, stride_inH, stride_inW,
    stride_outB, stride_outH, stride_outW, stride_outC,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    c = tl.program_id(3)

    in_off = b * stride_inB + c * stride_inC + h * stride_inH + w * stride_inW
    out_off = b * stride_outB + h * stride_outH + w * stride_outW + c * stride_outC

    val = tl.load(in_ptr + in_off)
    tl.store(out_ptr + out_off, val)


# 3) LayerNorm across last dim (C) per (B, H, W):
# Compute per (b, h, w): mean = sum_c x / C, var = sum_c (x-mean)^2 / C
# Normalize and scale by layernorm_weight: out = (x - mean) / sqrt(var + eps) * layernorm_weight[c]
@triton.jit
def layernorm_lastdim_kernel(
    x_ptr, weight_ptr, out_ptr,
    B, C, H, W,
    stride_xB, stride_xH, stride_xW, stride_xC,
    stride_outB, stride_outH, stride_outW, stride_outC,
    eps,  # float
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # compute mean and var across C
    sum_val = 0.0
    for c in range(C):
        x_off = b * stride_xB + h * stride_xH + w * stride_xW + c * stride_xC
        x_val = tl.load(x_ptr + x_off)
        sum_val += x_val
    mean = sum_val / C

    sum_sq = 0.0
    for c in range(C):
        x_off = b * stride_xB + h * stride_xH + w * stride_xW + c * stride_xC
        x_val = tl.load(x_ptr + x_off)
        diff = x_val - mean
        sum_sq += diff * diff
    var = sum_sq / C
    inv_std = 1.0 / tl.sqrt(var + eps)

    for c in range(C):
        x_off = b * stride_xB + h * stride_xH + w * stride_xW + c * stride_xC
        x_val = tl.load(x_ptr + x_off)
        w_off = c * stride_outC  # weight has shape (C,)
        w_val = tl.load(weight_ptr + w_off)
        out_off = b * stride_outB + h * stride_outH + w * stride_outW + c * stride_outC
        out_val = (x_val - mean) * inv_std * w_val
        tl.store(out_ptr + out_off, out_val)


# 4) GEMV: out[b, h, w, c_out] = sum_{c_in} x_ln[b, h, w, c_in] * weight[c_out, c_in]
# x_ln is NHWC: shape (B, H, W, C), weight is (C_out, C_in) with C_in=C
@triton.jit
def gemv_nhwc_weightt_kernel(
    x_ptr, weight_ptr, out_ptr,
    B, C_in, H, W, C_out,
    stride_xB, stride_xH, stride_xW, stride_xC,
    stride_weightCout, stride_weightCin,
    stride_outB, stride_outH, stride_outW, stride_outC,
):
    # Grid: (B, H, W, C_out)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    c_out = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)
    # reduction over C_in
    for c_in in range(C_in):
        x_off = b * stride_xB + h * stride_xH + w * stride_xW + c_in * stride_xC
        x_val = tl.load(x_ptr + x_off)
        w_off = c_out * stride_weightCout + c_in * stride_weightCin
        w_val = tl.load(weight_ptr + w_off)
        acc += x_val * w_val

    out_off = b * stride_outB + h * stride_outH + w * stride_outW + c_out * stride_outC
    tl.store(out_ptr + out_off, acc)


# 5) GELU (tanh approximation): y = x * (0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))))
@triton.jit
def gelu_tanh_kernel(
    x_ptr, y_ptr,
    N,  # number of elements
    stride_x, stride_y,
):
    idx = tl.program_id(0)
    x_off = idx * stride_x
    y_off = idx * stride_y
    x_val = tl.load(x_ptr + x_off)
    sqrt_2_over_pi = 0.7978845608028654
    c = 0.044715
    inner = sqrt_2_over_pi * (x_val + c * x_val * x_val * x_val)
    tanh_inner = tl.tanh(inner)
    y_val = 0.5 * x_val * (1.0 + tanh_inner)
    tl.store(y_ptr + y_off, y_val)


# 6) GRN:
# global_features: per (b, c) L2 over spatial dims -> shape (B, C), gf_mean: per (B,) mean over C
# norm_features: gf_mean / (gf_mean + eps), x_grn_scaled: x_gelu * norm_features
# x_grn: grn_weight * x_grn_scaled + x_gelu (broadcast over spatial)
# Implement in Triton:
# - kernel to compute global_features[b, c] = sqrt(sum over h,w of x_gelu[b,h,w,c]^2)
# - kernel to compute gf_mean[b] = (1/C) * sum_c global_features[b,c]
# - kernel to compute norm_features[b, c] = gf_mean[b] / (gf_mean[b] + eps)
# - kernel to compute x_grn_scaled[b, h, w, c] = x_gelu[b, h, w, c] * norm_features[b, c]
# - kernel to compute final x_grn[b, h, w, c] = grn_weight[c] * x_grn_scaled + x_gelu[b, h, w, c]
@triton.jit
def grn_compute_global_features_kernel(
    x_ptr, gf_ptr,
    B, C, H, W,
    stride_xB, stride_xH, stride_xW, stride_xC,
    stride_gfB, stride_gfC,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    sum_sq = 0.0
    for h in range(H):
        for w in range(W):
            x_off = b * stride_xB + h * stride_xH + w * stride_xW + c * stride_xC
            x_val = tl.load(x_ptr + x_off)
            sum_sq += x_val * x_val
    gf_val = tl.sqrt(sum_sq)
    gf_off = b * stride_gfB + c * stride_gfC
    tl.store(gf_ptr + gf_off, gf_val)


@triton.jit
def grn_compute_mean_kernel(
    gf_ptr, mean_ptr,
    B, C,
    stride_gfB, stride_gfC,
):
    b = tl.program_id(0)
    sum_val = 0.0
    for c in range(C):
        gf_off = b * stride_gfB + c * stride_gfC
        gf_val = tl.load(gf_ptr + gf_off)
        sum_val += gf_val
    mean_val = sum_val / C
    tl.store(mean_ptr + b, mean_val)


@triton.jit
def grn_compute_norm_features_kernel(
    mean_ptr, norm_ptr,
    B, eps,
    stride_mean, stride_norm,
):
    b = tl.program_id(0)
    mean_val = tl.load(mean_ptr + b)
    denom = mean_val + eps
    norm_val = mean_val / denom
    tl.store(norm_ptr + b, norm_val)


@triton.jit
def grn_scale_and_final_kernel(
    x_ptr, grn_weight_ptr, scaled_ptr, final_ptr,
    B, C, H, W,
    stride_xB, stride_xH, stride_xW, stride_xC,
    stride_scaledB, stride_scaledH, stride_scaledW, stride_scaledC,
    stride_finalB, stride_finalH, stride_finalW, stride_finalC,
    stride_norm,  # single scalar base
):
    # This kernel is invoked for each (b, h, w, c). We pass norm via norm_ptr[b], and grn_weight via grn_weight_ptr[c].
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    c = tl.program_id(3)

    # Load norm and weight
    norm_val = tl.load(norm_ptr + b)
    w_off = c
    w_val = tl.load(grn_weight_ptr + w_off)  # grn_weight has shape (C,)

    x_off = b * stride_xB + h * stride_xH + w * stride_xW + c * stride_xC
    x_val = tl.load(x_ptr + x_off)
    scaled = x_val * norm_val  # since norm_features is 1 for each c? Not correct. We need to scale per c.
    # Correction: norm_features is per (b, c). Here norm_ptr[b] corresponds to mean of all features for that b.
    # The reference code uses norm_features = gf_mean / (gf_mean + eps), so norm_val is per b.
    # We need to apply per-c norm. We should instead compute norm per c in the previous kernel:
    # But in this kernel, we need per-c norm. We'll recompute: norm_per_c = gf_mean[b] / (gf_mean[b] + eps) for that c?
    # No, norm_features is the same for all c under the given code. So scaled = x_val * norm_val is correct.
    out = scaled * w_val + x_val

    scaled_off = b * stride_scaledB + h * stride_scaledH + w * stride_scaledW + c * stride_scaledC
    final_off = b * stride_finalB + h * stride_finalH + w * stride_finalW + c * stride_finalC
    tl.store(scaled_ptr + scaled_off, scaled)
    tl.store(final_ptr + final_off, out)


# Now, ModelNew.forward will invoke these Triton kernels. We will avoid torch operations for math.
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
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
        # Ensure CUDA and Triton usage; no torch math ops in forward
        device = residual.device
        dtype = residual.dtype

        # 1) Depthwise conv2d (groups=C) using Triton
        residual = residual.contiguous()
        dwconv_weight = dwconv_weight.contiguous()
        B, C, H, W = residual.shape
        H_out = H + 2 * 3
        W_out = W + 2 * 3
        y = torch.empty((B, C, H_out, W_out), device=device, dtype=dtype)

        grid_conv = (B, C, H_out, W_out)
        depthwise_conv2d_groupsC_per_output_kernel[grid_conv](
            residual, dwconv_weight, y,
            B, C, H, W, H_out, W_out,
            3, 3,
            residual.stride(0), residual.stride(1), residual.stride(2), residual.stride(3),
            dwconv_weight.stride(0), dwconv_weight.stride(1), dwconv_weight.stride(2), dwconv_weight.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            num_warps=4, num_stages=2,
        )

        # 2) NHWC permute using Triton: x_nhwc = y.permute(0,2,3,1)
        x_nhwc = torch.empty((B, H_out, W_out, C), device=device, dtype=dtype)
        grid_perm = (B, H_out, W_out, C)
        permute_nchw_to_nhwc_kernel[grid_perm](
            y, x_nhwc,
            B, C, H_out, W_out,
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            num_warps=4, num_stages=2,
        )

        # 3) LayerNorm across last dim (C) per (B, H_out, W_out) using Triton
        # Input: x_nhwc; Output: x_ln
        x_ln = torch.empty_like(x_nhwc, dtype=dtype, device=device)
        grid_ln = (B, H_out, W_out)
        layernorm_lastdim_kernel[grid_ln](
            x_nhwc, layernorm_weight, x_ln,
            B, C, H_out, W_out,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            x_ln.stride(0), x_ln.stride(1), x_ln.stride(2), x_ln.stride(3),
            eps,
            num_warps=4, num_stages=2,
        )

        # 4) GEMV: x_expanded = x_ln @ pwconv1_weight.T using Triton
        # x_ln: (B, H_out, W_out, C), weight: (C_out=4*C, C_in=C)
        x_expanded = torch.empty((B, H_out, W_out, C * 4), device=device, dtype=dtype)
        grid_gemv = (B, H_out, W_out, C * 4)
        gemv_nhwc_weightt_kernel[grid_gemv](
            x_ln, pwconv1_weight, x_expanded,
            B, C, H_out, W_out, C * 4,
            x_ln.stride(0), x_ln.stride(1), x_ln.stride(2), x_ln.stride(3),
            pwconv1_weight.stride(0), pwconv1_weight.stride(1),
            x_expanded.stride(0), x_expanded.stride(1), x_expanded.stride(2), x_expanded.stride(3),
            num_warps=4, num_stages=2,
        )

        # 5) GELU (tanh approximation) on x_expanded using Triton
        x_gelu = torch.empty_like(x_expanded, device=device, dtype=dtype)
        N = x_expanded.numel()
        grid_gelu = (N,)
        gelu_tanh_kernel[grid_gelu](
            x_expanded, x_gelu,
            N,
            x_expanded.stride(0), x_gelu.stride(0),
            num_warps=4, num_stages=2,
        )

        # 6) GRN:
        # Compute global_features per (b, c): L2 over spatial dims
        global_features = torch.empty((B, C), device=device, dtype=dtype)
        grid_gf = (B, C)
        grn_compute_global_features_kernel[grid_gf](
            x_gelu, global_features,
            B, C, H_out, W_out,
            x_gelu.stride(0), x_gelu.stride(1), x_gelu.stride(2), x_gelu.stride(3),
            global_features.stride(0), global_features.stride(1),
            num_warps=4, num_stages=2,
        )
        # Compute gf_mean per b
        gf_mean = torch.empty((B,), device=device, dtype=dtype)
        grid_mean = (B,)
        grn_compute_mean_kernel[grid_mean](
            global_features, gf_mean,
            B, C,
            global_features.stride(0), gf_mean.stride(0),
            num_warps=4, num_stages=2,
        )
        # Compute norm_features per (b, c) = gf_mean[b] / (gf_mean[b] + eps)
        norm_features = torch.empty((B, C), device=device, dtype=dtype)
        grid_norm = (B,)
        grn_compute_norm_features_kernel[grid_norm](
            gf_mean, norm_features,
            B, eps,
            gf_mean.stride(0), norm_features.stride(0),
            num_warps=4, num_stages=2,
        )
        # Compute x_grn_scaled and final x_grn
        x_grn_scaled = torch.empty_like(x_gelu, device=device, dtype=dtype)
        x_grn = torch.empty_like(x_gelu, device=device, dtype=dtype)
        grid_scale_final = (B, H_out, W_out, C)
        grn_scale_and_final_kernel[grid_scale_final](
            x_gelu, grn_weight, x_grn_scaled, x_grn,
            B, C, H_out, W_out,
            x_gelu.stride(0), x_gelu.stride(1), x_gelu.stride(2), x_gelu.stride(3),
            x_grn_scaled.stride(0), x_grn_scaled.stride(1), x_grn_scaled.stride(2), x_grn_scaled.stride(3),
            x_grn.stride(0), x_grn.stride(1), x_grn.stride(2), x_grn.stride(3),
            norm_features.stride(0),
            num_warps=4, num_stages=2,
        )

        # Return the main output x_dwconv and related tensors to satisfy signature (forward uses Triton for math)
        return (
            y,  # x_dwconv
            x_nhwc,
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


def run(*args):
    return ModelNew()(*args)
