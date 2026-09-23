import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv2d_3x3_stride2_gelu_nchw_kernel(
    X_ptr,       # *bf16, input [B, C_in, H, W]
    W_ptr,       # *bf16, weight [C_out, C_in, 3, 3]
    BIAS_ptr,    # *bf16, bias [C_out]
    Y_ptr,       # *bf16, output [B, C_out, H_out, W_out]
    B, C_in, H, W, C_out, H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wic, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,
):
    # One program per (b, co)
    b = tl.program_id(0)
    co = tl.program_id(1)

    # Initialize accumulator for this (b, co) over all H_out*W_out positions
    # We will iterate over output spatial positions and accumulate into Y.
    # y[b, co, :, :] will be stored after GELU.

    # Prepare constants
    c = 0.7978845608028654  # sqrt(2/pi) for GELU tanh approximation

    # Loop over output positions (row-major: oh then ow)
    for oh in range(0, H_out):
        for ow in range(0, W_out):
            # Accumulator for current (b, co, oh, ow)
            acc = tl.zeros((), dtype=tl.float32)

            # Loop over input channels and 3x3 neighborhood
            for ic in range(0, C_in):
                # For each (kh, kw), compute input index with padding=1
                # ih = oh + kh - 1, iw = ow + kw - 1
                for kh in range(0, 3):
                    ih = oh + kh - 1
                    valid_h = (ih >= 0) & (ih < H)
                    for kw in range(0, 3):
                        iw = ow + kw - 1
                        valid_w = (iw >= 0) & (iw < W)
                        valid = valid_h & valid_w

                        # Load input vector for this (b, ic, ih, iw)
                        x_off = b * stride_xb + ic * stride_xc + ih * stride_xh + iw * stride_xw
                        # We need a vector across output channels (co is scalar here, but we accumulate per output pixel).
                        # However, we load scalar x for current (ic, ih, iw) and multiply with weight vector across co.
                        x_val = tl.load(X_ptr + x_off, mask=valid, other=0.0)  # scalar bf16
                        x_val_f32 = x_val.to(tl.float32)

                        # Load weight vector for co across input channel ic and kernel (kh, kw): W[co, ic, kh, kw]
                        # W layout: [C_out, C_in, 3, 3]
                        w_off = co * stride_wc + ic * stride_wic + kh * stride_wkh + kw * stride_wkw
                        # But co is a scalar here; Triton needs vectorization across output channels: we will do it per co via BLOCK_CO.
                        # To implement: we need to iterate over BLOCK_CO and accumulate into Y per output pixel.
                        # Instead of looping in kernel per co, we can precompute a per-(ic,kh,kw) contribution to Y for all co by vectorizing.
                        # Since Triton does not allow dynamic loops over C_out easily in this way, we implement per-co accumulation.
                        # We'll iterate co in blocks of BLOCK_CO, and compute acc += w * x_val.
                        co_offsets = co  # only one co per program for simplicity; we will tile/co loop by launching grid over C_out.
                        # Since we want to accumulate into Y for all co, we should instead structure grid to (B, C_out), but
                        # to keep accumulation here, we'll set BLOCK_CO=1. This implies we handle one output channel per program.
                        # Therefore, we compute y[b, co, oh, ow] = sum_{ic,kh,kw} x[b, ic, ih, iw] * W[co, ic, kh, kw].
                        # We store y after finishing all kh,kw,ic and then apply GELU.

            # After finishing spatial accumulation, add bias and apply GELU, then store.
            bias_val = tl.load(BIAS_ptr + co, mask=True, other=0.0)
            acc = acc + bias_val.to(tl.float32)

            # Apply GELU tanh approximation
            x3 = acc * acc * acc
            tanh_arg = c * (acc + 0.044715 * x3)
            tanh_val = tl.math.tanh(tanh_arg)
            acc = 0.5 * acc * (1.0 + tanh_val)

            # Store to Y[b, co, oh, ow]
            y_off = b * stride_yb + co * stride_yc + oh * stride_yh + ow * stride_yw
            y_val = acc.to(tl.bfloat16)
            tl.store(Y_ptr + y_off, y_val, mask=True)

    # End of kernel (implicit). Note: we launched grid over (B, C_out) so this computes all output channels.
    # However, Triton expects a return; since we used masks and stores, we do nothing further.


@triton.jit
def linear_gemv_nobias_kernel(
    X_ptr,           # *bf16, shape [B, T, K]
    W_ptr,           # *bf16, shape [N, K]
    Y_ptr,           # *bf16, shape [B, T, N]
    B, T, K, N,
    stride_xb, stride_xt, stride_xk,
    stride_wd, stride_wk,
    stride_yb, stride_yt, stride_yd,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_tile = tl.program_id(2)

    d_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    valid_d = d_offsets < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        valid_k = k_offsets < K

        x_off = b * stride_xb + t * stride_xt + k_offsets * stride_xk
        x_vec = tl.load(X_ptr + x_off, mask=valid_k, other=0.0)  # bf16
        x_vec_f32 = x_vec.to(tl.float32)

        w_off = d_offsets[:, None] * stride_wd + k_offsets[None, :] * stride_wk
        w_mat = tl.load(W_ptr + w_off, mask=valid_d[:, None] & valid_k[None, :], other=0.0)  # bf16
        w_mat_f32 = w_mat.to(tl.float32)

        acc += tl.dot(w_mat_f32, x_vec_f32[None, :])  # [BLOCK_N]

    y_off = b * stride_yb + t * stride_yt + d_offsets * stride_yd
    y_vals = acc.to(tl.bfloat16)
    tl.store(Y_ptr + y_off, y_vals, mask=valid_d)


@triton.jit
def add_pos_emb_kernel(
    Y_ptr,        # *bf16, shape [B, T, N]
    Pos_ptr,      # *bf16, shape [T, N] (position embeddings for T rows)
    Out_ptr,      # *bf16, shape [B, T, N]
    B, T, N,
    stride_yb, stride_yt, stride_yd,
    stride_pt, stride_pd,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_tile = tl.program_id(2)

    d_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_d = d_offsets < N

    y_off = b * stride_yb + t * stride_yt + d_offsets * stride_yd
    y_vals = tl.load(Y_ptr + y_off, mask=valid_d, other=0.0)

    pos_off = t * stride_pt + d_offsets * stride_pd
    pos_vals = tl.load(Pos_ptr + pos_off, mask=valid_d, other=0.0)

    y_vals = y_vals + pos_vals
    tl.store(Out_ptr + y_off, y_vals, mask=valid_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features: torch.Tensor,
                conv2d1_weight: torch.Tensor, conv2d1_bias: torch.Tensor,
                conv2d2_weight: torch.Tensor, conv2d2_bias: torch.Tensor,
                conv2d3_weight: torch.Tensor, conv2d3_bias: torch.Tensor,
                conv_out_weight: torch.Tensor,
                positional_embedding: torch.Tensor,
                embed_scale: float):
        """
        input_features: [B, 1, 80, T]
        conv weights/bias: conv2d1..3, stride=2, padding=1, 3x3
        conv_out_weight: [N, K] where N=d_model=1024, K=C_out3*H_out3*W_out3
        positional_embedding: [max_source_positions, d_model] (bf16), use first T rows
        """
        # We'll implement convs + GELU in Triton to satisfy the requirement of doing computation in Triton.

        B, C_in, H, W = input_features.shape
        C_out1, C_in1, kH, kW = conv2d1_weight.shape
        assert kH == 3 and kW == 3 and C_in1 == 1, "Conv1 weight must be [384, 1, 3, 3]"
        assert conv2d1_weight.shape == (C_out1, C_in1, 3, 3), "Conv1 weight shape mismatch"
        # Compute output dimensions after each conv:
        def out_dim(in_d, k=3, s=2, p=1):
            return (in_d + 2*p - k) // s + 1

        H1 = out_dim(H, 3, 2, 1)
        W1 = out_dim(W, 3, 2, 1)

        H2 = out_dim(H1, 3, 2, 1)
        W2 = out_dim(W1, 3, 2, 1)

        H3 = out_dim(H2, 3, 2, 1)
        W3 = out_dim(W2, 3, 2, 1)

        # Allocate outputs for each conv stage
        # Conv1: [B, C_out1, H1, W1]
        y1 = torch.empty((B, C_out1, H1, W1), dtype=torch.bfloat16, device=input_features.device)
        # Conv2: [B, 384, H2, W2]
        y2 = torch.empty((B, 384, H2, W2), dtype=torch.bfloat16, device=input_features.device)
        # Conv3: [B, 384, H3, W3]
        y3 = torch.empty((B, 384, H3, W3), dtype=torch.bfloat16, device=input_features.device)

        # Launch Triton conv+GELU for each stage: grid over (B, C_out) to compute all output channels
        # Note: In the kernel, we process one output channel per program, i.e., BLOCK_CO=1.
        BLOCK_CO = 1

        # Stage 1: Conv1
        grid1 = (B, C_out1)
        conv2d_3x3_stride2_gelu_nchw_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, C_in, H, W, C_out1, H1, W1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2,
        )

        # Stage 2: Conv2
        grid2 = (B, 384)
        conv2d_3x3_stride2_gelu_nchw_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            B, 384, H1, W1, 384, H2, W2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2,
        )

        # Stage 3: Conv3
        grid3 = (B, 384)
        conv2d_3x3_stride2_gelu_nchw_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            B, 384, H2, W2, 384, H3, W3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_CO=BLOCK_CO,
            num_warps=4, num_stages=2,
        )

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        B, C, F, T = y3.size()


def run(*args):
    return ModelNew()(*args)
