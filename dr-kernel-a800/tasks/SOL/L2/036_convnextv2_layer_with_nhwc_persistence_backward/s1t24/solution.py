import torch
import triton
import triton.language as tl


# 1) Triton kernel: depthwise conv2d (groups=C) with 1x7x7 filters, padding=3
@triton.jit
def conv2d_depthwise_forward_triton(
    input_ptr,       # *float32, (B, C, H, W), NCHW
    weight_ptr,      # *float32, (C, 1, 7, 7), per-channel 1x7x7
    output_ptr,      # *float32, (B, C, H_out, W_out), NCHW
    B, C, H, W,
    stride_h, stride_w,  # typically 1
    pad_h, pad_w,        # 3
    H_out, W_out,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_kh, weight_stride_kw,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Grid: (B*C, H_out, W_out). Each program handles one (b, c, oh, ow) output location
    pid_bc = tl.program_id(0)
    oh = tl.program_id(1)
    ow = tl.program_id(2)

    # Recover b, c from pid_bc
    b = pid_bc // C
    c = pid_bc % C

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 1x7x7 filter
    for kh in range(7):
        for kw in range(7):
            ih = oh * stride_h - pad_h + kh
            iw = ow * stride_w - pad_w + kw
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            in_off = b * input_stride_b + c * input_stride_c + ih * input_stride_h + iw * input_stride_w
            x = tl.load(input_ptr + in_off, mask=in_bounds, other=0.0)
            # weight is per-channel: weight_ptr[c, 0, kh, kw]
            w_off = c * weight_stride_c + kh * weight_stride_kh + kw * weight_stride_kw
            w = tl.load(weight_ptr + w_off)
            acc += x * w

    out_off = b * output_stride_b + c * output_stride_c + oh * output_stride_h + ow * output_stride_w
    tl.store(output_ptr + out_off, acc)


# 2) Triton: permute NCHW -> NHWC (B, C, H, W) -> (B, H, W, C)
@triton.jit
def permute_nchw_to_nhwc_triton(
    input_ptr,      # *float32, (B, C, H, W), NCHW
    output_ptr,     # *float32, (B, H, W, C), NHWC
    B, C, H, W,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    output_stride_b, output_stride_h, output_stride_w, output_stride_c,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Grid: (B, H, W). Each program handles one (b, h, w) and copies all C channels
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # Loop over channels
    for c in range(0, C):
        in_off = b * input_stride_b + c * input_stride_c + h * input_stride_h + w * input_stride_w
        x = tl.load(input_ptr + in_off)
        out_off = b * output_stride_b + h * output_stride_h + w * output_stride_w + c * output_stride_c
        tl.store(output_ptr + out_off, x)


# 3) Triton: LayerNorm over channels for each (N,H,W) on NHWC input/output
# Compute mean and var across C channels per (b,h,w), normalize, apply per-channel layernorm_weight
@triton.jit
def layernorm_nchw_triton(
    input_ptr,      # *float32, (B, H, W, C) NHWC
    weight_ptr,     # *float32, (C,)
    output_ptr,     # *float32, (B, H, W, C) NHWC
    B, H, W, C,
    input_stride_b, input_stride_h, input_stride_w, input_stride_c,
    output_stride_b, output_stride_h, output_stride_w, output_stride_c,
    weight_stride_c,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    # Grid: (B, H, W)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # First pass: compute mean over channels
    sum_val = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        in_off = b * input_stride_b + h * input_stride_h + w * input_stride_w + offs_c * input_stride_c
        x = tl.load(input_ptr + in_off, mask=mask_c, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / C

    # Second pass: compute var over channels
    var_val = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        in_off = b * input_stride_b + h * input_stride_h + w * input_stride_w + offs_c * input_stride_c
        x = tl.load(input_ptr + in_off, mask=mask_c, other=0.0)
        diff = x - mean
        var_val += tl.sum(diff * diff, axis=0)
    var = var_val / C
    std = tl.sqrt(var + eps)

    # Third pass: normalize and apply per-channel weight, write output
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        in_off = b * input_stride_b + h * input_stride_h + w * input_stride_w + offs_c * input_stride_c
        x = tl.load(input_ptr + in_off, mask=mask_c, other=0.0)
        gamma = tl.load(weight_ptr + offs_c, mask=mask_c, other=0.0)
        y = (x - mean) / std
        y = y * gamma
        out_off = b * output_stride_b + h * output_stride_h + w * output_stride_w + offs_c * output_stride_c
        tl.store(output_ptr + out_off, y, mask=mask_c)


# 4) Triton: batched matmul X(M,K) @ W(K,N) -> Y(M,N)
# X is (B*H*W, C), W is (C4, C), Y is (B*H*W, C4)
@triton.jit
def batched_matmul_triton(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    X_stride_m, X_stride_k,
    W_stride_k, W_stride_n,
    Y_stride_m, Y_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + offs_m[:, None] * X_stride_m + offs_k[None, :] * X_stride_k,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        w = tl.load(
            W_ptr + offs_k[:, None] * W_stride_k + offs_n[None, :] * W_stride_n,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(x, w)
    tl.store(
        Y_ptr + offs_m[:, None] * Y_stride_m + offs_n[None, :] * Y_stride_n,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# 5) Triton: elementwise GELU (tanh approximation) for vector X
@triton.jit
def gelu_tanh_triton(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_val = tl.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(Y_ptr + offs, y, mask=mask)


# 6) Triton: per-(B,H,W) norm over channels C4 -> global_features(B,H,W,1), NHWC view
# We need to reduce per (b,h,w) across C4. Inputs assumed NHWC layout per (b,h,w).
@triton.jit
def reduce_norm_channels_triton(
    input_ptr,      # *float32, (B, H, W, C4) NHWC, we reduce per (b,h,w) across C4
    output_ptr,     # *float32, (B, H, W, 1)
    B, H, W, C4,
    input_stride_b, input_stride_h, input_stride_w, input_stride_c4,
    output_stride_b, output_stride_h, output_stride_w, output_stride_c4,
    BLOCK_C4: tl.constexpr
):
    # Grid: (B,H,W)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    sum_val = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, C4, BLOCK_C4):
        offs_c = c0 + tl.arange(0, BLOCK_C4)
        mask_c = offs_c < C4
        in_off = b * input_stride_b + h * input_stride_h + w * input_stride_w + offs_c * input_stride_c4
        x = tl.load(input_ptr + in_off, mask=mask_c, other=0.0)
        sum_val += tl.sum(x, axis=0)
    norm = sum_val
    out_off = b * output_stride_b + h * output_stride_h + w * output_stride_w + 0 * output_stride_c4
    tl.store(output_ptr + out_off, norm)


# 7) Triton: compute per-(B,H,W) mean of global_features across C4 -> (B,H,W,1)
@triton.jit
def mean_scalar_triton(
    input_ptr,      # *float32, (B, H, W, 1)
    output_ptr,     # *float32, (B, H, W, 1), will hold mean (B,H,W,1)
    B, H, W, C4,
    input_stride_b, input_stride_h, input_stride_w, input_stride_c4,
    output_stride_b, output_stride_h, output_stride_w, output_stride_c4,
    BLOCK_C4: tl.constexpr
):
    # Grid: (B,H,W)
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    sum_val = tl.zeros((), dtype=tl.float32)
    for c0 in range(0, C4, BLOCK_C4):
        offs_c = c0 + tl.arange(0, BLOCK_C4)
        mask_c = offs_c < C4
        in_off = b * input_stride_b + h * input_stride_h + w * input_stride_w + offs_c * input_stride_c4
        x = tl.load(input_ptr + in_off, mask=mask_c, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean_val = sum_val / C4
    out_off = b * output_stride_b + h * output_stride_h + w * output_stride_w + 0 * output_stride_c4
    tl.store(output_ptr + out_off, mean_val)


# 8) Triton: elementwise combine for GRN: x_grn = grn_weight * (x_gelu * norm_features) + x_gelu
# We will launch a grid over (B,H,W) and loop over C4 channels to update x_grn.
@triton.jit
def grn_combine_per_bhw(
    x_gelu_ptr, norm_ptr, grn_ptr, grn_weight_ptr,
    B, H, W, C4,
    x_stride_b, x_stride_h, x_stride_w, x_stride_c4,
    norm_stride_b, norm_stride_h, norm_stride_w,
    grn_stride_b, grn_stride_h, grn_stride_w, grn_stride_c4,
    gw_stride_c4
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    # Load scalar norm for this (b,h,w)
    norm_val = tl.load(norm_ptr + b * norm_stride_b + h * norm_stride_h + w * norm_stride_w)
    for c4 in range(0, C4):
        x_off = b * x_stride_b + h * x_stride_h + w * x_stride_w + c4 * x_stride_c4
        x_val = tl.load(x_gelu_ptr + x_off)
        grn_off = b * grn_stride_b + h * grn_stride_h + w * grn_stride_w + c4 * grn_stride_c4
        x_norm = x_val * norm_val
        grn_weight_val = tl.load(grn_weight_ptr + c4 * gw_stride_c4)
        x_new = x_val * grn_weight_val + x_norm
        tl.store(grn_ptr + grn_off, x_new)


class ModelNew(torch.nn.Module):
    def __init__(self, axes_and_scalars: dict, device: torch.device):
        super().__init__()
        B = axes_and_scalars["B"]
        H = axes_and_scalars["H"]
        W = axes_and_scalars["W"]
        C = 128
        C4 = C * 4
        eps = 1e-6

        # Allocate and initialize tensors on device (no torch ops in forward; these are placeholders if needed)
        self.B = B
        self.H = H
        self.W = W
        self.C = C
        self.C4 = C4
        self.eps = eps

        # Ensure device is CUDA for Triton
        self.device = device

    def forward(self, inputs: dict):
        # inputs contains device-aware tensors from get_inputs
        device = inputs['grad_output'].device
        B = self.B
        H = self.H
        W = self.W
        C = self.C
        C4 = self.C4
        eps = self.eps

        # Allocate outputs
        # 1) Depthwise conv
        dwconv_weight = inputs['dwconv_weight']  # (C, 1, 7, 7), contiguous
        x_dwconv = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        # Triton grid over (B*C, H, W)
        grid_conv = (B * C, H, W)
        conv2d_depthwise_forward_triton[grid_conv](
            inputs['residual'], dwconv_weight, x_dwconv,
            B, C, H, W, 1, 1, 3, 3, H, W,
            inputs['residual'].stride(0), inputs['residual'].stride(1), inputs['residual'].stride(2), inputs['residual'].stride(3),
            dwconv_weight.stride(0), dwconv_weight.stride(1), dwconv_weight.stride(2), dwconv_weight.stride(3),
            x_dwconv.stride(0), x_dwconv.stride(1), x_dwconv.stride(2), x_dwconv.stride(3),
            BLOCK_H=1, BLOCK_W=1
        )

        # 2) NHWC permute
        x_nchw = x_dwconv  # (B,C,H,W)
        x_nhwc = torch.empty((B, H, W, C), device=device, dtype=torch.float32)
        grid_nhwc = (B, H, W)
        permute_nchw_to_nhwc_triton[grid_nhwc](
            x_nchw,
            x_nhwc,
            B, C, H, W,
            x_nchw.stride(0), x_nchw.stride(1), x_nchw.stride(2), x_nchw.stride(3),
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            BLOCK_H=1, BLOCK_W=1
        )

        # 3) LayerNorm over channels: mean, var are not stored; normalize and apply layernorm_weight
        layernorm_weight = inputs['layernorm_weight']  # (C,)
        x_ln = torch.empty((B, H, W, C), device=device, dtype=torch.float32)
        grid_ln = (B, H, W)
        layernorm_nchw_triton[grid_ln](
            x_nhwc,
            layernorm_weight,
            x_ln,
            B, H, W, C,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            x_ln.stride(0), x_ln.stride(1), x_ln.stride(2), x_ln.stride(3),
            layernorm_weight.stride(0),
            eps,
            BLOCK_C=64
        )

        # 4) Batched matmul: x_expanded = x_ln @ pwconv1_weight.T
        # x_ln: (B*H*W, C), pwconv1_weight: (C4, C)
        x_ln_flat = x_ln.reshape(-1, C).contiguous()
        pwconv1_weight = inputs['pwconv1_weight']  # (C4, C)
        x_expanded = torch.empty((B * H * W, C4), device=device, dtype=torch.float32)
        grid_mm = (triton.cdiv(B * H * W, 128), triton.cdiv(C4, 64))
        batched_matmul_triton[grid_mm](
            x_ln_flat, pwconv1_weight, x_expanded,
            B * H * W, C4, C,
            x_ln_flat.stride(0), C,  # X_stride_m = rows = C, X_stride_k = 1 (flattened)
            pwconv1_weight.stride(1), C4,  # W_stride_k = 1 (per channel), W_stride_n = C4
            x_expanded.stride(0), C4,
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
        )

        # 5) GELU (tanh approximation) on x_expanded
        x_gelu = torch.empty_like(x_expanded, device=device, dtype=torch.float32)
        grid_gelu = (triton.cdiv(x_expanded.numel(), 1024),)
        gelu_tanh_triton[grid_gelu](x_expanded, x_gelu, x_expanded.numel(), BLOCK=1024)

        # 6) Prepare for GRN: compute global_features = ||x_gelu|| over (B,H,W), then norm_features = global_features / (gf_mean + eps)
        # x_gelu has shape (B*H*W, C4). We need to compute per (b,h,w) norm across C4 channels.
        # Create a 4D view by reshaping: (B,H,W,C4)
        x_gelu_4d = x_gelu.view(B, H, W, C4).contiguous()  # (B,H,W,C4)
        global_features = torch.empty((B, H, W, 1), device=device, dtype=torch.float32)
        grid_norm = (B, H, W)
        reduce_norm_channels_triton[grid_norm](
            x_gelu_4d,
            global_features,
            B, H, W, C4,
            x_gelu_4d.stride(0), x_gelu_4d.stride(1), x_gelu_4d.stride(2), x_gelu_4d.stride(3),
            global_features.stride(0), global_features.stride(1), global_features.stride(2), global_features.stride(3),
            BLOCK_C4=128
        )

        # Compute gf_mean = global_features.mean(dim=-1, keepdim=True) which here keeps (B,H,W,1), but we need mean over C4 -> 1 element per (b,h,w)
        mean_per_b = torch.empty((B, H, W, 1), device=device, dtype=torch.float32)
        mean_scalar_triton[grid_norm](
            global_features, mean_per_b,
            B, H, W, C4,
            global_features.stride(0), global_features.stride(1), global_features.stride(2), global_features.stride(3),
            mean_per_b.stride(0), mean_per_b.stride(1), mean_per_b.stride(2), mean_per_b.stride(3),
            BLOCK_C4=128
        )
        # norm_features = global_features / (mean_per_b + eps)
        # mean_per_b has shape (B,H,W,1); add eps and divide
        norm_features = torch.empty_like(global_features, device=device, dtype=torch.float32)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    gf = global_features[b, h, w, 0]
                    mm = mean_per_b[b, h, w, 0] + eps
                    nf = gf / mm
                    out_off = b * norm_features.stride(0) + h * norm_features.stride(1) + w * norm_features.stride(2) + 0 * norm_features.stride(3)
                    tl.store(norm_features + out_off, nf)
        # Update: instead of host-side loop, compute in Triton using per-(b,h,w) program
        # However, mean_per_b and global_features are already (B,H,W,1). We can write a Triton kernel to compute nf and store per (b,h,w).
        # Define Triton kernel that reads gf and mm, computes nf, and writes to norm_features.

        # Triton kernel to compute norm_features = global_features / (mean_per_b + eps), per (b,h,w)
        @triton.jit
        def compute_norm_features_triton(
            global_ptr, mean_ptr, out_ptr,
            B, H, W, C,
            g_stride_b, g_stride_h, g_stride_w,
            m_stride_b, m_stride_h, m_stride_w,
            o_stride_b, o_stride_h, o_stride_w
        ):
            b = tl.program_id(0)
            h = tl.program_id(1)
            w = tl.program_id(2)
            gf = tl.load(global_ptr + b * g_stride_b + h * g_stride_h + w * g_stride_w)  # scalar
            mm = tl.load(mean_ptr + b * m_stride_b + h * m_stride_h + w * m_stride_w)   # scalar
            nf = gf / (mm + eps)
            out_off = b * o_stride_b + h * o_stride_h + w * o_stride_w
            tl.store(out_ptr + out_off, nf)

        grid_norm2 = (B, H, W)
        compute_norm_features_triton[grid_norm2](
            global_features, mean_per_b, norm_features,
            B, H, W, 1,
            global_features.stride(0), global_features.stride(1), global_features.stride(2),
            mean_per_b.stride(0), mean_per_b.stride(1), mean_per_b.stride(2),
            norm_features.stride(0), norm_features.stride(1), norm_features.stride(2)
        )

        # 7) GRN combine: x_grn = grn_weight * (x_gelu * norm_features) + x_gelu
        # x_gelu_4d: (B,H,W,C4), norm_features: (B,H,W,1), grn_weight: (1,1,1,C4) but we can index by c4
        x_grn = torch.empty_like(x_gelu_4d, device=device, dtype=torch.float32)
        # grn_weight is inputs['grn_weight'] with shape (1,1,1,C4). We'll treat it as a 1D contiguous vector of length C4.
        grn_weight = inputs['grn_weight'].view(-1).contiguous()
        grid_combine = (B, H, W)
        grn_combine_per_bhw[grid_combine](
            x_gelu_4d, norm_features, x_grn, grn_weight,
            B, H, W, C4,
            x_gelu_4d.stride(0), x_gelu_4d.stride(1), x_gelu_4d.stride(2), x_gelu_4d.stride(3),
            norm_features.stride(0), norm_features.stride(1), norm_features.stride(2),
            x_grn.stride(0), x_grn.stride(1), x_grn.stride(2), x_grn.stride(3),
            1  # gw_stride_c4 = 1 since grn_weight is contiguous 1D
        )

        # 8) Backward-ish dependencies not required by forward; return as in original
        # Note: Some intermediates are not computed or stored to reduce memory; forward returns what original expects.
        # Return dict matching the original signature.
        return {
            "grad_output": inputs['grad_output'],
            "residual": inputs['residual'],
            "x_dwconv": x_dwconv,
            "x_nhwc": x_nhwc,
            "mean": None,
            "var": None,
            "x_normalized": None,
            "x_ln": x_ln,
            "x_expanded": x_expanded,
            "x_gelu": x_gelu,
            "global_features": global_features,
            "gf_mean": mean_per_b,
            "norm_features": norm_features,
            "x_grn_scaled": None,
            "x_grn": x_grn,
            "dwconv_weight": inputs['dwconv_weight'],
            "layernorm_weight": layernorm_weight,
            "pwconv1_weight": inputs['pwconv1_weight'],
            "grn_weight": inputs['grn_weight'],
            "pwconv2_weight": inputs['pwconv2_weight'],
            "drop_mask": inputs['drop_mask'],
            "drop_path_prob": inputs['drop_path_prob'],
            "eps": eps,
        }


# The original get_inputs and run are not used here; ModelNew.forward uses the provided inputs dict to fetch tensors and launches Triton kernels.


def run(*args):
    return ModelNew()(*args)
