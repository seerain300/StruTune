import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_depthwise_kernel(
    residual_ptr,        # *f32, [B, C, H, W]
    weight_ptr,          # *f32, [C, 1, 7, 7]
    out_ptr,             # *f32, [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_wblk = tl.program_id(2)

    # decode b, c
    b = pid_bc // C
    c = pid_bc % C
    h_out = pid_h

    # output spatial vector
    w_start = pid_wblk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W_out

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # loop over 7x7 kernel
    for kh in range(7):
        for kw in range(7):
            weight_idx = c * 49 + kh * 7 + kw  # weight is [C, 1, 7, 7] flattened per channel
            w_val = tl.load(weight_ptr + weight_idx)
            h_in = h_out + kh - PAD_H
            w_in = w_offsets - PAD_W
            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_w
            base = b * C * H * W + c * H * W + h_in * W + w_in
            val = tl.load(residual_ptr + base, mask=in_bounds, other=0.0)
            acc += val * w_val

    # store result
    out_base = b * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_offsets
    tl.store(out_ptr + out_base, acc, mask=mask_w)


@triton.jit
def layernorm_reduce_mean_var_kernel(
    x_ptr,               # *f32, NHWC layout: [B, H, W, C]
    mean_ptr,            # *f32, [B, H, W]
    var_ptr,             # *f32, [B, H, W]
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
):
    # grid over (b, h, w)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # reduce over channels
    for c in range(C):
        base = pid_b * H * W * C + pid_h * W * C + pid_w * C + c
        val = tl.load(x_ptr + base)
        sum_val += val
        sum_sq += val * val

    mean = sum_val / C
    var = sum_sq / C - mean * mean

    mean_store = pid_b * H * W + pid_h * W + pid_w
    var_store = pid_b * H * W + pid_h * W + pid_w
    tl.store(mean_ptr + mean_store, mean)
    tl.store(var_ptr + var_store, var)


@triton.jit
def rsqrt_inplace_kernel(
    var_ptr,             # *f32, [B, H, W]
    eps,                 # f32
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    idx = pid_b * H * W + pid_h * W + pid_w
    var_val = tl.load(var_ptr + idx)
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    tl.store(var_ptr + idx, inv_std)


@triton.jit
def linear_matmul_kernel(
    a_ptr,               # *f32, [B, C, H, W] (input features)
    w_ptr,               # *f32, [K, C] (weights), K = output_channels
    out_ptr,             # *f32, [B, K, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
):
    # grid over (B*K, H, W) — each program computes one output channel per (b,h,w)
    pid_bk = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    b = pid_bk // K
    oc = pid_bk % K

    acc = tl.zeros((), dtype=tl.float32)
    # reduce over input channels
    for ic in range(C):
        in_val = tl.load(a_ptr + b * C * H * W + ic * H * W + pid_h * W + pid_w)
        w_val = tl.load(w_ptr + oc * C + ic)
        acc += in_val * w_val

    out_base = b * K * H * W + oc * H * W + pid_h * W + pid_w
    tl.store(out_ptr + out_base, acc)


@triton.jit
def gelu_tanh_kernel(
    x_ptr,               # *f32, [N] flattened
    out_ptr,             # *f32, [N] flattened
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < N:
        x = tl.load(x_ptr + pid)
        # GELU tanh approximation
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        c = 0.044715
        inner = sqrt_2_over_pi * (x + c * x * x * x)
        tanh_inner = tl.tanh(inner)
        y = 0.5 * x * (1.0 + tanh_inner)
        tl.store(out_ptr + pid, y)


@triton.jit
def norm_mean_scale_kernel(
    x_ptr,               # *f32, [B, C, H, W] (input features)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    factor_ptr,          # *f32, [B, C]
    eps,                 # f32
):
    # grid over (B, C)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    sum_sq = tl.zeros((), dtype=tl.float32)
    # reduce over H and W for channel c
    for h in range(H):
        for w in range(W):
            base = pid_b * C * H * W + pid_c * H * W + h * W + w
            val = tl.load(x_ptr + base)
            sum_sq += val * val

    global_feat = tl.sqrt(sum_sq)

    # compute mean across channels per batch for L2 norms
    mean_c = tl.zeros((), dtype=tl.float32)
    for c2 in range(C):
        sum_c = tl.zeros((), dtype=tl.float32)
        for h in range(H):
            for w in range(W):
                base = pid_b * C * H * W + c2 * H * W + h * W + w
                val = tl.load(x_ptr + base)
                sum_c += val * val
        mean_c += tl.sqrt(sum_c)
    mean_c = mean_c / C

    scale = global_feat / (mean_c + eps)
    fact_store = pid_b * C + pid_c
    tl.store(factor_ptr + fact_store, scale)


@triton.jit
def conv_transpose2d_groups_kernel(
    inp_ptr,             # *f32, [B, C, H, W] (input to transposed conv)
    weight_ptr,          # *f32, [C, 1, 7, 7] (weights, per-channel)
    out_ptr,             # *f32, [B, C, H, W] (output)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,  # For transposed conv, H_out = H + 6 and W_out = W + 6
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_wblk = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C
    h_out = pid_h

    # output spatial vector
    w_start = pid_wblk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W_out

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # transposed conv: output at (h_out, w) accumulates input at positions (h_out - kh, w - kw)
    for kh in range(7):
        for kw in range(7):
            h_in = h_out - kh  # no padding in transposed conv for this demo
            w_in = w_offsets - kw
            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_w
            base_in = b * C * H * W + c * H * W + h_in * W + w_in
            val_in = tl.load(inp_ptr + base_in, mask=in_bounds, other=0.0)
            # load weight scalar per output channel c
            weight_idx = c * 49 + kh * 7 + kw
            w_val = tl.load(weight_ptr + weight_idx)
            acc += val_in * w_val

    out_base = b * C * H_out * W_out + c * H_out * W_out + h_out * W_out + w_offsets
    tl.store(out_ptr + out_base, acc, mask=mask_w)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Triton: no torch operations here; kernels only.

    def forward(self, grad_output: torch.Tensor,
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
                eps: float):
        """
        Triton-only forward: launches all required kernels. No torch math in host.
        Inputs are assumed to be tensors provided by the evaluator.
        """
        # 1) Depthwise conv: conv2d_depthwise_kernel
        B, C, H, W = residual.shape
        H_out = H + 6  # 7-1 - padding 3 on both sides
        W_out = W + 6
        x_dwconv_out = torch.empty((B, C, H, W), device=residual.device, dtype=residual.dtype)
        grid_conv = (B * C, H_out, triton.cdiv(W_out, 32))
        conv2d_depthwise_kernel[grid_conv](
            residual, dwconv_weight, x_dwconv_out,
            B, C, H, W, H_out, W_out, 3, 3, 32
        )

        # 2) LayerNorm reduction: layernorm_reduce_mean_var_kernel (NHWC)
        x_nhwc_flat = x_nhwc.reshape(-1, x_nhwc.shape[-1]).contiguous()  # [B*H*W, C]
        mean_out = torch.empty((B, x_nhwc.shape[1], x_nhwc.shape[2]), device=residual.device, dtype=residual.dtype)
        var_out = torch.empty((B, x_nhwc.shape[1], x_nhwc.shape[2]), device=residual.device, dtype=residual.dtype)
        grid_layernorm = (B, x_nhwc.shape[1], x_nhwc.shape[2])
        layernorm_reduce_mean_var_kernel[grid_layernorm](
            x_nhwc_flat, mean_out, var_out,
            B, x_nhwc.shape[1], x_nhwc.shape[2], x_nhwc.shape[-1]
        )

        # 3) rsqrt(var + eps): rsqrt_inplace_kernel
        inv_std = torch.empty_like(var_out)
        rsqrt_inplace_kernel[grid_layernorm](
            var_out, eps,
            B, x_nhwc.shape[1], x_nhwc.shape[2]
        )

        # 4) Linear projection: linear_matmul_kernel (x_ln @ pwconv1_weight.T)
        B2, C2, H2, W2 = x_ln.shape
        K = pwconv1_weight.shape[0]
        x_expanded_out = torch.empty((B2, K, H2, W2), device=residual.device, dtype=residual.dtype)
        grid_linear = (B2 * K, H2, W2)
        linear_matmul_kernel[grid_linear](
            x_ln, pwconv1_weight, x_expanded_out,
            B2, C2, H2, W2, K
        )

        # 5) GELU (tanh approx): gelu_tanh_kernel
        N = x_expanded_out.numel()
        x_gelu_out = torch.empty_like(x_expanded_out)
        grid_gelu = (N,)
        gelu_tanh_kernel[grid_gelu](
            x_expanded_out.reshape(-1), x_gelu_out.reshape(-1), N
        )

        # 6) Grouped Refined Norm (GRN): norm_mean_scale_kernel
        Bg


def run(*args):
    return ModelNew()(*args)
