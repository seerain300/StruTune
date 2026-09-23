import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    x_ptr,        # *const T (input), shape [B, C_in, H, W]
    w_ptr,        # *const T (weights), shape [C_out, C_in, 3, 3]
    b_ptr,        # *const T (bias), shape [C_out]
    y_ptr,        # *T (output), shape [B, C_out, H_out, W_out]
    B, C_in, H, W, C_out, H_out, W_out,
    # strides for x: row-major [B, C_in, H, W]
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    # strides for w: row-major [C_out, C_in, 3, 3]
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    # strides for y: row-major [B, C_out, H_out, W_out]
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,
    embed_scale,  # float32 scale (unused here, but present for extensibility)
    BLOCK_CO: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    # grid dims: (B, C_out, ceil(H_out/BLOCK_H), ceil(W_out/BLOCK_W))
    b = tl.program_id(0)
    co = tl.program_id(1)
    ph = tl.program_id(2)
    pw = tl.program_id(3)

    co_offsets = co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    H_offsets = ph * BLOCK_H + tl.arange(0, BLOCK_H)
    W_offsets = pw * BLOCK_W + tl.arange(0, BLOCK_W)

    co_mask = co_offsets < C_out
    H_mask = H_offsets < H_out
    W_mask = W_offsets < W_out

    # Initialize output tile
    y_tile = tl.zeros((BLOCK_CO, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # For each input channel and each 3x3 filter tap, accumulate
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # input top-left corner corresponding to output (h, w)
                in_h = H_offsets[:, None, None] + (kh - 1)
                in_w = W_offsets[None, :, None] + (kw - 1)

                # in-bounds mask for input load
                in_mask = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & H_mask[:, None, None] & W_mask[None, :, None]
                x_ptrs = x_ptr + b * stride_x_b + ci * stride_x_c + in_h * stride_x_h + in_w * stride_x_w
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0).to(tl.float32)  # load as fp32 for accumulation

                # weight vector for this (ci, kh, kw) and all co in tile
                w_ptrs = w_ptr + co_offsets * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw
                w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0).to(tl.float32)

                # outer-product accumulate: [CO, H, W] += w[CO] * x[H, W]
                y_tile += w_vals[:, None, None] * x_vals

    # Add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
    y_tile += b_vals[:, None, None]

    # Fused GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x3 = y_tile * y_tile * y_tile
    gelu_inner = sqrt_2_over_pi * (y_tile + 0.044715 * x3)
    gelu_out = 0.5 * y_tile * (1.0 + tl.math.tanh(gelu_inner))

    # Store to y in bfloat16
    y_ptrs = y_ptr + b * stride_y_b + co_offsets[:, None, None] * stride_y_c + H_offsets[None, :, None] * stride_y_h + W_offsets[None, None, :] * stride_y_w
    store_mask = co_mask[:, None, None] & H_mask[None, :, None] & W_mask[None, None, :]
    tl.store(y_ptrs, gelu_out.to(tl.bfloat16), mask=store_mask)


@triton.jit
def linear_matmul_kernel(
    x_row_ptr,    # *const T, shape [B*S, K], row-major over S*K
    w_ptr,        # *const T, shape [N, K], row-major
    y_ptr,        # *T, shape [B*S, N], row-major
    B, S, K, N,
    stride_x_row, stride_x_k,  # x_row strides: row_stride (S*K), k_stride (1)
    stride_w_n, stride_w_k,    # w strides: [N, K] row-major
    stride_y_row, stride_y_n,  # y strides: row-major
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    # Grid: (B*S, ceil(N/BLOCK_N))
    pid_row = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator for this row over N-tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduce over K in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load x_row block: shape [BLOCK_K]
        x_ptrs = x_row_ptr + pid_row * stride_x_row + k_offsets * stride_x_k
        x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load W block: shape [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k
        w_vals = tl.load(w_ptrs, mask=(n_mask[:, None] & k_mask[None, :]), other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # Accumulate: acc += sum_k (w_vals[n, k] * x_vals[k])
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Store result for this row tile
    y_ptrs = y_ptr + pid_row * stride_y_row + n_offsets * stride_y_n
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=n_mask)


@triton.jit
def scale_elementwise_kernel(
    out_ptr,      # *T, flattened output to scale
    scale,        # float32 scale factor
    N_elems: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    # Simple 1D grid
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    vals = tl.load(out_ptr + offs, mask=mask, other=0.0)
    vals = vals * scale
    tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def add_pos_emb_elementwise_kernel(
    out_ptr,      # *T, flattened output (after scaling) to add pos emb
    pos_ptr,      # *T, flattened positional embedding [time_after_conv, d_model]
    N_elems: tl.constexpr,
    NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    # Simple 1D grid
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    out_vals = tl.load(out_ptr + offs, mask=mask, other=0.0)
    pos_vals = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    out_vals = out_vals + pos_vals
    tl.store(out_ptr + offs, out_vals, mask=mask)


def run_triton_only(
    input_features: torch.Tensor,
    conv2d1_weight: torch.Tensor,
    conv2d1_bias: torch.Tensor,
    conv2d2_weight: torch.Tensor,
    conv2d2_bias: torch.Tensor,
    conv2d3_weight: torch.Tensor,
    conv2d3_bias: torch.Tensor,
    conv_out_weight: torch.Tensor,
    positional_embedding: torch.Tensor,
    embed_scale: float,
):
    """
    Triton-only execution of the original pipeline:
    - conv2d1: input_features -> (B, 384, 40, T//2)
    - conv2d2: -> (B, 384, 20, T//4)
    - conv2d3: -> (B, 384, 10, T//8)
    - Reshape to (B, time_after_conv, 384*10)
    - Linear projection: (B*S, 3840) @ (1024, 3840)^T -> (B*S, 1024)
    - Scale by embed_scale
    - Add positional embedding [S, 1024], broadcast over batch
    """
    B, _, H, W = input_features.shape
    # Stage 1: Conv2d (1 -> 384 channels) + GELU
    C_in1, C_out1, H1_out, W1_out = 1, 384, (H + 2*1 - 3)//2 + 1, (W//2 + 2*1 - 3)//2 + 1
    x1 = torch.empty((B, C_out1, H1_out, W1_out), device=input_features.device, dtype=torch.bfloat16)
    grid1 = (B, C_out1, triton.cdiv(H1_out, 16), triton.cdiv(W1_out, 16))
    conv2d_stride2_pad1_bias_gelu_kernel[grid1](
        input_features, conv2d1_weight, conv2d1_bias, x1,
        B, C_in1, H, W, C_out1, H1_out, W1_out,
        input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
        conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        float(embed_scale),
        BLOCK_CO=32, BLOCK_H=16, BLOCK_W=16,
        NUM_WARPS=4, NUM_STAGES=2
    )

    # Stage 2: Conv2d (384 -> 384 channels) + GELU
    C_in2, C_out2, H2_out, W2_out = C_out1, C_out1, (H1_out + 2*1 - 3)//2 + 1, (W1_out + 2*1 - 3)//2 + 1
    x2 = torch.empty((B, C_out2, H2_out, W2_out), device=input_features.device, dtype=torch.bfloat16)
    grid2 = (B, C_out2, triton.cdiv(H2_out, 16), triton.cdiv(W2_out, 16))
    conv2d_stride2_pad1_bias_gelu_kernel[grid2](
        x1, conv2d2_weight, conv2d2_bias, x2,
        B, C_in2, H1_out, W1_out, C_out2, H2_out, W2_out,
        x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        float(embed_scale),
        BLOCK_CO=32, BLOCK_H=16, BLOCK_W=16,
        NUM_WARPS=4, NUM_STAGES=2
    )

    # Stage 3: Conv2d (384 -> 384 channels) + GELU
    C_in3, C_out3, H3_out, W3_out = C_out2, C_out2, (H2_out + 2*1 - 3)//2 + 1, (W2_out + 2*1 - 3)//2 + 1
    x3 = torch.empty((B, C_out3, H3_out, W3_out), device=input_features.device, dtype=torch.bfloat16)
    grid3 = (B, C_out3, triton.cdiv(H3_out, 16), triton.cdiv(W3_out, 16))
    conv2d_stride2_pad1_bias_gelu_kernel[grid3](
        x2, conv2d3_weight, conv2d3_bias, x3,
        B, C_in3, H2_out, W2_out, C_out3, H3_out, W3_out,
        x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
        x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        float(embed_scale),
        BLOCK_CO=32, BLOCK_H=16, BLOCK_W=16,
        NUM_WARPS=4, NUM_STAGES=2
    )

    # Reshape: (batch, channels, freq, time) -> (batch, time_after_conv, channels*freq)
    # time_after_conv = W3_out
    S = W3_out
    x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(B, S, C_out3 * H3_out)

    # Linear projection: (B*S, K=3840) @ (N=1024, K)^T -> (B*S, N)
    B2 = B
    S2 = S
    K = C_out3 * H3_out  # 3840
    N = conv_out_weight.shape[0]  # 1024
    x_row = x3_reshaped.view(B2 * S2, K).contiguous()  # [B*S, K]
    y = torch.empty((B2 * S2, N), device=input_features.device, dtype=torch.bfloat16)

    # Launch linear GEMM Triton kernel
    grid_linear = (B2 * S2, triton.cdiv(N, 128))
    linear_matmul_kernel[grid_linear](
        x_row, conv_out_weight, y,
        B2, S2, K, N,
        x_row.stride(0), 1,  # row stride, k stride
        conv_out_weight.stride(0), conv_out_weight.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_N=128, BLOCK_K=64,
        NUM_WARPS=4, NUM_STAGES=2
    )

    # Scale by embed_scale
    y_flat = y.view(-1)
    N_elems = y_flat.numel()
    grid_scale = (triton.cdiv(N_elems, 1024),)
    scale_elementwise_kernel[grid_scale](y_flat, float(embed_scale), N_elems=N_elems, NUM_WARPS=4, NUM_STAGES=2)

    # Add positional embedding [S, 1024], broadcast over batch
    pos_emb = positional_embedding[:S, :].to(torch.bfloat16).contiguous()  # [S, N]
    grid_add = (triton.cdiv(N_elems, 1024),)
    add_pos_emb_elementwise_kernel[grid_add](y_flat, pos_emb.view(-1), N_elems=N_elems, NUM_WARPS=4, NUM_STAGES=2)

    # Reshape back to (B, S, N)
    y = y_flat.view(B, S, N)
    return y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure all inputs are provided in the same order as get_inputs()
        # args: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        return run_triton_only(*args)


def run(*args):
    return ModelNew()(*args)
