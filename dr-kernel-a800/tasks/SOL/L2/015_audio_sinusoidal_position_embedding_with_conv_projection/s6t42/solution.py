import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_3x3_stride2_gelu_nchw_fp32(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    BLOCK_HO: tl.constexpr, BLOCK_WO: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)
    co = tl.program_id(1)
    tile_h = tl.program_id(2)
    tile_w = tl.program_id(3)

    # compute tile ranges
    ho_base = tile_h * BLOCK_HO
    wo_base = tile_w * BLOCK_WO

    # offsets within the tile
    ho_offsets = ho_base + tl.arange(0, BLOCK_HO)  # [BLOCK_HO]
    wo_offsets = wo_base + tl.arange(0, BLOCK_WO)  # [BLOCK_WO]

    # create 2D grid for the tile
    HO = ho_offsets[:, None]  # [BLOCK_HO, 1]
    WO = wo_offsets[None, :]  # [1, BLOCK_WO]

    # mask for valid output positions
    mask_out = (HO < H_out) & (WO < W_out)  # [BLOCK_HO, BLOCK_WO]

    # initialize output tile
    y_tile = tl.zeros((BLOCK_HO, BLOCK_WO), dtype=tl.float32)

    # loop over input channels
    for ic in range(0, C_in):
        # loop over 3x3 neighborhood
        for kh in range(0, 3):
            # ih = ho - 1 + kh
            ih = HO - 1 + kh  # [BLOCK_HO, 1]
            valid_h = (ih >= 0) & (ih < H)  # [BLOCK_HO, 1]

            for kw in range(0, 3):
                # iw = wo - 1 + kw
                iw = WO - 1 + kw  # [1, BLOCK_WO]
                valid_w = (iw >= 0) & (iw < W)  # [1, BLOCK_WO]

                # combine masks for input load
                mask_in = valid_h & valid_w & mask_out  # [BLOCK_HO, BLOCK_WO]
                # input linear indexing: ((b * C_in + ic) * H + ih) * W + iw
                x_off = ((b * C_in + ic) * H + ih) * W + iw  # [BLOCK_HO, BLOCK_WO]
                x_vals = tl.load(x_ptr + x_off, mask=mask_in, other=0.0)  # [BLOCK_HO, BLOCK_WO], fp32

                # weight scalar: weight[co, ic, kh, kw]
                # weight layout: [C_out, C_in, 3, 3]
                w_off = co * (C_in * 9) + ic * 9 + kh * 3 + kw  # scalar index
                w_val = tl.load(w_ptr + w_off)  # scalar
                # accumulate: broadcast w_val across tile
                y_tile += w_val * x_vals

    # add bias
    b_val = tl.load(bias_ptr + co)  # scalar
    y_tile = y_tile + b_val

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = y_tile * y_tile * y_tile
    tanh_arg = c * (y_tile + 0.044715 * x3)
    tanh_val = tl.math.tanh(tanh_arg)
    y_tile = 0.5 * y_tile * (1.0 + tanh_val)

    # store to output: out[b, co, HO, WO]
    # out linear indexing: ((b * C_out + co) * H_out + HO) * W_out + WO
    out_off = ((b * C_out + co) * H_out + HO) * W_out + WO  # [BLOCK_HO, BLOCK_WO]
    tl.store(out_ptr + out_off, y_tile, mask=mask_out)


@triton.jit
def linear_projection_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, T, K, N,
    # strides not needed for contiguous fp32 layout
):
    # Grid: (B, T, N)
    b = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)
    # compute y[b, t, d] = sum_k X[b, t, k] * W[d, k]
    acc = 0.0
    for k in range(0, K):
        # load X[b, t, k]
        x_off = (b * T + t) * K + k
        x_val = tl.load(X_ptr + x_off)
        # load W[d, k]
        w_off = d * K + k
        w_val = tl.load(W_ptr + w_off)
        acc += x_val * w_val
    # store y[b, t, d]
    y_off = (b * T + t) * N + d
    tl.store(Y_ptr + y_off, acc)


@triton.jit
def add_pos_embed_scaled_kernel(
    Y_ptr, pos_ptr, scale, B, T, N,
):
    # Grid: (B, T, N)
    b = tl.program_id(0)
    t = tl.program_id(1)
    n = tl.program_id(2)
    # load Y[b, t, n]
    y_off = (b * T + t) * N + n
    y_val = tl.load(Y_ptr + y_off)
    # load pos_emb[t, n] * scale
    pos_off = t * N + n
    pos_val = tl.load(pos_ptr + pos_off) * scale
    y_new = y_val + pos_val
    tl.store(Y_ptr + y_off, y_new)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # We expect the same input/output signatures as the original Model.forward:
        # input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        # Note: input_features, weights, biases are torch tensors, positional_embedding is [max_positions, d_model] tensor,
        # embed_scale is float.

        # Extract inputs
        (input_features, conv2d1_weight, conv2d1_bias,
         conv2d2_weight, conv2d2_bias,
         conv2d3_weight, conv2d3_bias,
         conv_out_weight, positional_embedding, embed_scale) = args

        # Ensure dtype is bfloat16 for input and weights, convert to fp32 for computation
        x = input_features.to(torch.bfloat16)
        x = x.contiguous()
        # conv weights and biases
        w1 = conv2d1_weight.to(torch.bfloat16).contiguous()
        b1 = conv2d1_bias.to(torch.bfloat16).contiguous()
        w2 = conv2d2_weight.to(torch.bfloat16).contiguous()
        b2 = conv2d2_bias.to(torch.bfloat16).contiguous()
        w3 = conv2d3_weight.to(torch.bfloat16).contiguous()
        b3 = conv2d3_bias.to(torch.bfloat16).contiguous()
        # conv_out_weight: [d_model, conv_out_dim]
        W = conv_out_weight.to(torch.bfloat16).contiguous()
        # positional_embedding: [max_positions, d_model]
        pos = positional_embedding.to(torch.bfloat16).contiguous()

        B, Cin, H, W_in = x.shape  # Cin=1
        # Conv1: in_channels=1, out_channels=384, kernel=3x3, stride=2, padding=1
        C_out1 = w1.shape[0]
        H1 = H // 2  # 80 // 2 = 40
        W1 = W_in // 2  # time_dim // 2
        # Allocate output conv1
        out1 = torch.empty((B, C_out1, H1, W1), dtype=torch.float32, device=x.device)

        # Launch Triton kernel for Conv1 + GELU
        BLOCK_HO = 8
        BLOCK_WO = 8
        grid1 = (B, C_out1, (H1 + BLOCK_HO - 1) // BLOCK_HO, (W1 + BLOCK_WO - 1) // BLOCK_WO)
        conv2d_3x3_stride2_gelu_nchw_fp32[grid1](
            x, w1, b1, out1,
            B, Cin, H, W_in, C_out1, H1, W1,
            BLOCK_HO=BLOCK_HO, BLOCK_WO=BLOCK_WO,
            num_warps=4, num_stages=2
        )

        # Conv2: in_channels=C_out1=384, out_channels=384
        C_in2 = C_out1
        H2 = H1 // 2
        W2 = W1 // 2
        out2 = torch.empty((B, C_in2, H2, W2), dtype=torch.float32, device=x.device)

        grid2 = (B, C_in2, (H2 + BLOCK_HO - 1) // BLOCK_HO, (W2 + BLOCK_WO - 1) // BLOCK_WO)
        conv2d_3x3_stride2_gelu_nchw_fp32[grid2](
            out1, w2, b2, out2,
            B, C_in2, H1, W1, C_in2, H2, W2,
            BLOCK_HO=BLOCK_HO, BLOCK_WO=BLOCK_WO,
            num_warps=4, num_stages=2
        )

        # Conv3: in_channels=C_in2=384, out_channels=384
        C_in3 = C_in2
        H3 = H2 // 2
        W3 = W2 // 2  # This equals time_after_conv per workload
        out3 = torch.empty((B, C_in3, H3, W3), dtype=torch.float32, device=x.device)

        grid3 = (B, C_in3, (H3 + BLOCK_HO - 1) // BLOCK_HO, (W3 + BLOCK_WO - 1) // BLOCK_WO)
        conv2d_3x3_stride2_gelu_nchw_fp32[grid3](
            out2, w3, b3, out3,
            B, C_in3, H2, W2, C_in3, H3, W3,
            BLOCK_HO=BLOCK_HO, BLOCK_WO=BLOCK_WO,
            num_warps=4, num_stages=2
        )

        # Reshape to [B, T, K] where T=W3 and K=C_in3*H3*W3 (this matches time_after_conv and final features)
        T = W3
        K = C_in3 * H3 * W3
        x_flat = out3.permute(0, 3, 1, 2).contiguous().view(B, T, K)  # fp32

        # Linear projection: Y[b, t, d] = sum_k x_flat[b, t, k] * W[d, k], where W shape [d_model, conv_out_dim]
        d_model = W.shape[0]
        conv_out_dim = W.shape[1]
        # Ensure conv_out_dim equals K (as in helper, conv_out_dim=3840). We compute K and set conv_out_dim accordingly.
        # If conv_out_dim != K, raise error to match PyTorch behavior.
        if conv_out_dim != K:
            raise RuntimeError(f"conv_out_dim ({conv_out_dim}) must equal K ({K})")

        Y = torch.empty((B, T, d_model), dtype=torch.float32, device=x.device)

        # Launch Triton linear projection kernel
        grid_linear = (B, T, d_model)
        linear_projection_kernel[grid_linear](
            x_flat, W, Y,
            B, T, K, d_model,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale (fp32)
        Y = Y * float(embed_scale)

        # Add scaled positional embedding: pos_emb[t, d] * embed_scale
        # pos shape: [max_positions, d_model], we slice to T rows
        # Note: positional_embedding provided by helper has max_source_positions >= T
        pos_sliced = pos[:T, :]  # [T, d_model]
        grid_pos = (B, T, d_model)
        add_pos_embed_scaled_kernel[grid_pos](
            Y, pos_sliced, float(embed_scale),
            B, T, d_model,
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original pipeline's dtype expectations
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
