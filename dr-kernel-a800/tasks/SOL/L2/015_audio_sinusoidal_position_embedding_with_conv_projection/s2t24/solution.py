import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input: x[B, C_in, IH, IW], w[OC, C_in, 3, 3], bias[OC], output y[B, OC, OH, OW]
@triton.jit
def conv2d_stride2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, IH, IW,
    OC, OH, OW,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator for output element
    acc = tl.zeros([1], dtype=tl.float32)

    # Iterate over input channels and 3x3 taps
    for ic in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                ih = 2 * oh + kh - 1
                iw = 2 * ow + kw - 1
                in_bounds = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                # Load input scalar
                x_val = tl.load(
                    x_ptr + b * (C_in * IH * IW) + ic * (IH * IW) + ih * IW + iw,
                    mask=in_bounds,
                    other=0.0
                )
                # Load weight scalar
                w_val = tl.load(
                    w_ptr + oc * (C_in * 9) + ic * 9 + kh * 3 + kw
                )
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(b_ptr + oc)
    acc += bias_val

    # Store output
    tl.store(
        y_ptr + b * (OC * OH * OW) + oc * (OH * OW) + oh * OW + ow,
        acc[0]
    )


# Triton GELU (tanh approximation) over a flat tensor
@triton.jit
def gelu_tanh_kernel(
    x_ptr, y_ptr, numel,
    scale: tl.float32
):
    pid = tl.program_id(0)
    start = pid * 1024
    for i in range(start, start + 1024):
        mask = i < numel
        x = tl.load(x_ptr + i, mask=mask, other=0.0)
        # GELU tanh approximation
        # y = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        c = 0.044715
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        x3 = x * x * x
        t = sqrt_2_over_pi * (x + c * x3)
        y = 0.5 * x * (1.0 + tl.math.tanh(t))
        tl.store(y_ptr + i, y, mask=mask)


# Triton kernel for final linear projection and positional embedding addition
# Inputs: X_flat[B*T_final, N], W[M=1024, N], pos_emb[M], outputs Y_flat[B*T_final, M]
@triton.jit
def linear_pos_kernel(
    X_ptr, W_ptr, pos_ptr, Y_ptr,
    B, T_final, N, M, scale: tl.float32
):
    # Each program handles one (b, t) row
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Precompute base offsets
    base_in = b * T_final * N + t * N
    base_out = b * T_final * M + t * M

    # Accumulator for M outputs
    acc = tl.zeros([M], dtype=tl.float32)

    # Loop over N in tiles
    for n0 in range(0, N, 128):
        n_offsets = n0 + tl.arange(0, 128)
        mask_n = n_offsets < N
        x_vals = tl.load(X_ptr + base_in + n_offsets, mask=mask_n, other=0.0)  # [128]
        # Load W block [M, 128]
        w_ptrs = W_ptr + n_offsets[None, :] * M  # shape (128,) broadcast to (M,128)
        w_vals = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)  # [M, 128]
        # acc += sum over n of x * w per m
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Scale
    acc = acc * scale
    # Add pos_emb[t, :]
    pos_vec = tl.load(pos_ptr + t * M + tl.arange(0, M))
    acc = acc + pos_vec
    # Store
    tl.store(Y_ptr + base_out + tl.arange(0, M), acc)


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: (B, 1, 80, T), bfloat16
        conv2d1_weight: (384, 1, 3, 3), bfloat16
        conv2d1_bias: (384), bfloat16
        conv2d2_weight, conv2d3_weight: (384, 384, 3, 3), bfloat16
        conv2d2_bias, conv2d3_bias: (384), bfloat16
        conv_out_weight: (1024, 3840), bfloat16
        positional_embedding: (1500, 1024), bfloat16
        embed_scale: float (e.g., 32.0)
        Output: (B, time_after_conv, 1024)
        """
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Move all inputs to CUDA if available
        input_features = input_features.to(device, non_blocking=True)
        conv2d1_weight = conv2d1_weight.to(device, non_blocking=True)
        conv2d1_bias = conv2d1_bias.to(device, non_blocking=True)
        conv2d2_weight = conv2d2_weight.to(device, non_blocking=True)
        conv2d2_bias = conv2d2_bias.to(device, non_blocking=True)
        conv2d3_weight = conv2d3_weight.to(device, non_blocking=True)
        conv2d3_bias = conv2d3_bias.to(device, non_blocking=True)
        conv_out_weight = conv_out_weight.to(device, non_blocking=True)
        positional_embedding = positional_embedding.to(device, non_blocking=True)

        B, C_in, IH, IW = input_features.shape  # C_in=1
        OC1, OC2, OC3 = 384, 384, 384
        IH1, IW1 = IH, IW  # keep for conv1

        # Allocate outputs for convs
        y1 = torch.empty((B, OC1, (IH1 - 1) // 2 + 1, (IW1 + 1) // 2), device=device, dtype=torch.float32)
        y2 = torch.empty((B, OC2, (y1.shape[2] - 1) // 2 + 1, (y1.shape[3] + 1) // 2), device=device, dtype=torch.float32)
        y3 = torch.empty((B, OC3, (y2.shape[2] - 1) // 2 + 1, (y2.shape[3] + 1) // 2), device=device, dtype=torch.float32)

        # Launch conv1
        grid1 = (B, OC1, y1.shape[2], y1.shape[3])
        conv2d_stride2_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, C_in, IH1, IW1,
            OC1, y1.shape[2], y1.shape[3],
        )

        # GELU conv1
        y1_gelu = torch.empty_like(y1, dtype=torch.float32)
        numel1 = y1.numel()
        gelu_tanh_kernel[(numel1 + 1023) // 1024,](y1, y1_gelu, numel1, 1.0)

        # Launch conv2
        grid2 = (B, OC2, y2.shape[2], y2.shape[3])
        conv2d_stride2_kernel[grid2](
            y1_gelu, conv2d2_weight, conv2d2_bias, y2,
            B, OC1, y1_gelu.shape[2], y1_gelu.shape[3],
            OC2, y2.shape[2], y2.shape[3],
        )

        # GELU conv2
        y2_gelu = torch.empty_like(y2, dtype=torch.float32)
        numel2 = y2.numel()
        gelu_tanh_kernel[(numel2 + 1023) // 1024,](y2, y2_gelu, numel2, 1.0)

        # Launch conv3
        grid3 = (B, OC3, y3.shape[2], y3.shape[3])
        conv2d_stride2_kernel[grid3](
            y2_gelu, conv2d3_weight, conv2d3_bias, y3,
            B, OC2, y2_gelu.shape[2], y2_gelu.shape[3],
            OC3, y3.shape[2], y3.shape[3],
        )

        # GELU conv3 (optional: keep for robustness, but the next step permutes anyway)
        y3_gelu = torch.empty_like(y3, dtype=torch.float32)
        numel3 = y3.numel()
        gelu_tanh_kernel[(numel3 + 1023) // 1024,](y3, y3_gelu, numel3, 1.0)

        # Compute time_after_conv from y3 shape
        T_final = y3_gelu.shape[3]

        # Permute to (B, T_final, 384*10) and do linear projection in Triton
        # Reshape X: (B, T_final, N) where N = 384*10 = 3840
        X_reshaped = y3_gelu.permute(0, 3, 1, 2).contiguous().view(B, T_final, 384 * 10)

        # Allocate output Y_flat: (B * T_final, 1024)
        Y_flat = torch.empty((B * T_final, 1024), device=device, dtype=torch.float32)

        # Launch linear + pos emb kernel
        grid_lin = (B, T_final)
        linear_pos_kernel[grid_lin](
            X_reshaped, conv_out_weight, positional_embedding, Y_flat,
            B, T_final, 384 * 10, 1024, embed_scale
        )

        # Reshape to (B, T_final, 1024)
        Y = Y_flat.view(B, T_final, 1024)

        return Y


def run(*args):
    return ModelNew()(*args)
