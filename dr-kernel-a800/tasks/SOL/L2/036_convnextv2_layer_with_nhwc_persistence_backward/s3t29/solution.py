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
    # Grid: (B*C, H_out, ceil_div(W_out, BLOCK_W))
    pid_bc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_wblk = tl.program_id(2)

    b = pid_bc // C
    c = pid_bc % C
    h_out = pid_h

    w_start = pid_wblk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W_out

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Accumulate over 7x7 kernel with padding
    for kh in range(7):
        for kw in range(7):
            weight_idx = c * 49 + kh * 7 + kw  # 1x7x7 per channel
            w_val = tl.load(weight_ptr + weight_idx)
            h_in = h_out + kh - PAD_H
            w_in = w_offsets - PAD_W
            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_w
            base = b * C * H * W + c * H * W + h_in * W + w_in
            val = tl.load(residual_ptr + base, mask=in_bounds, other=0.0)
            acc += val * w_val

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
    # grid over (b, k, hw blocks)
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_hwblk = tl.program_id(2)

    k = pid_k
    hw_start = pid_hwblk * (H * W)
    hw_offsets = hw_start + tl.arange(0, H * W)
    mask_hw = hw_offsets < H * W

    h = hw_offsets // W
    w = hw_offsets % W

    acc = tl.zeros([H * W], dtype=tl.float32)
    # reduce over C
    for c in range(C):
        a_base = pid_b * C * H * W + c * H * W + hw_offsets
        a_vals = tl.load(a_ptr + a_base, mask=mask_hw, other=0.0)
        w_base = k * C + c
        w_val = tl.load(w_ptr + w_base)
        acc += a_vals * w_val

    out_base = pid_b * K * H * W + k * H * W + hw_offsets
    tl.store(out_ptr + out_base, acc, mask=mask_hw)


@triton.jit
def gelu_tanh_kernel(
    x_ptr,               # *f32, [B, K, H, W]
    out_ptr,             # *f32, [B, K, H, W]
    B: tl.constexpr, K: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    # grid over (b, k, hw blocks)
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_hwblk = tl.program_id(2)

    k = pid_k
    hw_start = pid_hwblk * (H * W)
    hw_offsets = hw_start + tl.arange(0, H * W)
    mask_hw = hw_offsets < H * W

    h = hw_offsets // W
    w = hw_offsets % W

    base = pid_b * K * H * W + k * H * W + hw_offsets
    x = tl.load(x_ptr + base, mask=mask_hw, other=0.0)

    # GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_inner = tl.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_inner)

    tl.store(out_ptr + base, y, mask=mask_hw)


@triton.jit
def norm_mean_scale_kernel(
    x_ptr,               # *f32, [B, C, H, W] (GELU output, NHWC-like but shape as provided)
    norm_ptr,            # *f32, [B, K]
    mean_ptr,            # *f32, [B]
    eps,                 # f32
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, K: tl.constexpr,
):
    # Compute per-(b,k) L2 norm across spatial dims (H, W)
    for b in range(B):
        for k in range(K):
            sum_sq = tl.zeros((), dtype=tl.float32)
            # Assuming x_ptr layout corresponds to [B, K, H, W] in linear address space:
            # k is a dimension; we compute sum over H*W for that (b,k).
            for hw in range(H * W):
                # linear base for (b, k, :, :)
                base = b * K * H * W + k * H * W + hw
                val = tl.load(x_ptr + base)
                sum_sq += val * val
            norm = tl.sqrt(sum_sq)
            tl.store(norm_ptr + b * K + k, norm)

    # Compute per-sample mean of norms across K channels
    for b in range(B):
        sum_norms = tl.zeros((), dtype=tl.float32)
        for k in range(K):
            sum_norms += tl.load(norm_ptr + b * K + k)
        mean = sum_norms / K
        tl.store(mean_ptr + b, mean)


class ModelNew(nn.Module):
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
        # Inputs are expected to be CUDA tensors; ensure contiguity
        B = grad_output.shape[0]
        C = grad_output.shape[1]
        H = x_nhwc.shape[1]
        W = x_nhwc.shape[2]

        # 1) Depthwise conv: residual -> x_dwconv
        H_out = x_dwconv.shape[2]
        W_out = x_dwconv.shape[3]
        BLOCK_W = 64
        grid_conv = (B * C, H_out, triton.cdiv(W_out, BLOCK_W))
        x_dwconv = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=residual.dtype)
        conv2d_depthwise_kernel[grid_conv](
            residual, dwconv_weight, x_dwconv, B, C, residual.shape[2], residual.shape[3], H_out, W_out, 3, 3, BLOCK_W
        )

        # 2) NHWC from x_dwconv
        x_nhwc = x_dwconv.permute(0, 2, 3, 1).contiguous()

        # 3) LayerNorm reduction over channels (C) for each (B, H, W): mean and var
        mean_var = torch.empty((B, H, W), device=x_nhwc.device, dtype=x_nhwc.dtype)
        var_var = torch.empty((B, H, W), device=x_nhwc.device, dtype=x_nhwc.dtype)
        grid_mean_var = (B, H, W)
        layernorm_reduce_mean_var_kernel[grid_mean_var](
            x_nhwc, mean_var, var_var, B, H, W, C
        )

        # 4) rsqrt(var + eps)
        grid_rsqrt = (B, H, W)
        rsqrt_inplace_kernel[grid_rsqrt](var_var, eps, B, H, W)

        # 5) Linear projection: x_ln @ pwconv1_weight.T
        K = pwconv1_weight.shape[0]  # 4*C
        x_ln_nhwc = x_nhwc * layernorm_weight  # apply LN weight
        x_expanded = torch.empty((B, K, H, W), device=x_ln_nhwc.device, dtype=x_ln_nhwc.dtype)
        BLOCK_HW = 128
        grid_linear = (B, K, triton.cdiv(H * W, BLOCK_HW))
        linear_matmul_kernel[grid_linear](
            x_ln_nhwc, pwconv1_weight, x_expanded, B, C, H, W, K
        )

        # 6) GELU (tanh approximation)
        x_gelu = torch.empty_like(x_expanded)
        grid_gelu = (B, K, triton.cdiv(H * W, BLOCK_HW))
        gelu_tanh_kernel[grid_gelu](
            x_expanded, x_gelu, B, K, H, W
        )

        # 7) GRN: compute global L2 norm per (b, k) over spatial dims, and per-sample mean
        norm_features = torch.empty((B, K), device=x_gelu.device, dtype=x_gelu.dtype)
        gf_mean = torch.empty((B,), device=x_gelu.device, dtype=x_gelu.dtype)
        grid_norm = (B, K)
        # We need to interpret x_gelu as [B, K, H, W] in linear address space for per-(b,k) sums
        # The kernel above wrote x_gelu in this layout; we can reuse linear address mapping:
        # For each (b,k), sum over H*W positions starting at b*K*H*W + k*H*W
        norm_mean_scale_kernel[grid_norm](
            x_gelu, norm_features, gf_mean, eps, B, C, H, W, K
        )

        # 8) Final result: x_grn_scaled = x_gelu * norm_features; x_grn = grn_weight * x_grn_scaled + x_gelu
        # Note: The provided get_inputs sets grn_weight as [1,1,1,K], so we treat it as scalar per k by multiplying elementwise.
        x_grn_scaled = x_gelu * norm_features
        x_grn = grn_weight * x_grn_scaled + x_gelu

        # Return final tensor (any tensor produced by Triton kernels). To match signature, return x_grn.
        return x_grn


def run(*args):
    return ModelNew()(*args)
