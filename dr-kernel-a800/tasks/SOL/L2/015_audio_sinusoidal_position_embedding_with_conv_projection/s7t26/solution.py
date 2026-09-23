import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    x, w, bias, y,
    B, C_in, H, W,
    C_out, H_out, W_out,
    stride_x_b, stride_x_ci, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_co, stride_y_h, stride_y_w,
    embed_scale,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    # Grid: (B, C_out, ceil(H_out/BLOCK_H), ceil(W_out/BLOCK_W))
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Tile offsets
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    c_out = pid_co  # one output channel per program for simplicity

    # Initialize accumulator for this output channel tile
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels in chunks
    for ci0 in range(0, C_in, BLOCK_C):
        ci = ci0 + tl.arange(0, BLOCK_C)  # vector of input channels

        # Build input spatial grid (oh, ow) for this tile
        oh_mat = h_offsets[:, None]  # shape [BLOCK_H, 1]
        ow_mat = w_offsets[None, :]  # shape [1, BLOCK_W]

        # Compute input coordinates for 3x3 kernel with padding=1
        # in_h = oh + kh - 1, in_w = ow + kw - 1
        # Valid positions satisfy 0 <= in_h < H and 0 <= in_w < W
        # We'll mask loads accordingly.
        # Loop over kernel window (3x3)
        for kh in range(0, 3):
            for kw in range(0, 3):
                in_h = oh_mat + kh - 1  # [BLOCK_H, 1]
                in_w = ow_mat + kw - 1  # [1, BLOCK_W]
                # Compute valid mask
                valid = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

                # Build pointers for x[b, ci, in_h, in_w]
                # Note: broadcasting ci and (in_h, in_w)
                x_ptrs = x + pid_b * stride_x_b + ci[:, None] * stride_x_ci + in_h * stride_x_h + in_w * stride_x_w
                # Load with mask; other=0
                x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
                x_tile = x_tile.to(tl.float32)

                # Load weight w[c_out, ci, kh, kw]
                w_ptrs = w + c_out * stride_w_co + (ci0 + tl.arange(0, BLOCK_C)) * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw
                w_vec = tl.load(w_ptrs, mask=(ci0 + tl.arange(0, BLOCK_C) < C_in), other=0.0)  # scalar w is 3x3 per c_out and ci chunk
                # Promote to [BLOCK_C, 1, 1] via broadcasting
                w_vec = w_vec[:, None, None]  # [BLOCK_C, 1, 1]

                # Outer product accumulate: acc += sum_ci x_tile[:, ci] * w_vec[ci]
                # We sum over ci axis (dimension 0 of x_tile) with shape [BLOCK_H, 1], broadcast across width
                # To do that, reshape x_tile to [BLOCK_H, BLOCK_C] and multiply with w_vec after expanding to [BLOCK_C, BLOCK_H, 1]
                # Instead, perform per-ci accumulation:
                for i in range(0, BLOCK_C):
                    # mask for ci[i] only
                    mask_ci = (ci0 + i) < C_in
                    x_vec_ci = x_tile[:, i]  # [BLOCK_H]
                    w_scalar = w_vec[i, 0, 0]  # scalar
                    acc += x_vec_ci[:, None] * w_scalar

    # After accumulation, add bias for this output channel
    b_val = bias[c_out]  # scalar
    acc = acc + b_val

    # Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 x^3)))
    # Compute in float32
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * x3)))

    # Store to y[b, c_out, h, w]
    y_ptrs = y + pid_b * stride_y_b + c_out * stride_y_co + oh_mat * stride_y_h + ow_mat * stride_y_w
    # Store with mask for valid (h,w)
    store_mask = (h_offsets[:, None] < H_out) & (w_offsets[None, :] < W_out)
    tl.store(y_ptrs, gelu, mask=store_mask)


@triton.jit
def linear_matmul_kernel(
    x, w, y,
    B, S, K, N,
    stride_x_row, stride_x_k,
    stride_w_n, stride_w_k,
    stride_y_row, stride_y_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B*S, ceil(N / BLOCK_N))
    pid_row = tl.program_id(0)  # row index in [0, B*S)
    pid_col = tl.program_id(1)  # output channel block id

    b = pid_row // S
    s = pid_row % S

    offs_n = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load x[b, s, k]
        x_ptrs = x + b * stride_x_row + s * stride_x_row + offs_k * stride_x_k
        x_vec = tl.load(x_ptrs, mask=(offs_k < K), other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load w[n, k] for offs_n -> [BLOCK_N, BLOCK_K]
        w_ptrs = w + (offs_n[:, None] * stride_w_n) + (offs_k[None, :] * stride_w_k)
        w_mat = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # Accumulate: acc[n] += sum_k w_mat[n, k] * x_vec[k]
        acc += tl.sum(w_mat * x_vec[None, :], axis=1)

    # Store y[b, s, n]
    y_ptrs = y + b * stride_y_row + s * stride_y_row + offs_n * stride_y_n
    tl.store(y_ptrs, acc, mask=(offs_n < N))


@triton.jit
def scale_elementwise_kernel(y_flat, scale, N_elems: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N_elems
    y = tl.load(y_flat + offsets, mask=mask, other=0.0)
    y = y * scale
    tl.store(y_flat + offsets, y, mask=mask)


@triton.jit
def add_pos_emb_kernel(y_flat, pos_flat, N_elems: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N_elems
    y = tl.load(y_flat + offsets, mask=mask, other=0.0)
    pos = tl.load(pos_flat + offsets, mask=mask, other=0.0)
    y = y + pos
    tl.store(y_flat + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args comes from get_inputs(...) in the evaluation harness
        # The order of args matches the original Model.forward signature
        # We need to implement all numerical ops via Triton kernels.
        # Identify tensors: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        # Let's unpack by name to ensure correct usage.

        # Helper to fetch an arg by name
        def arg(name):
            return args[args.index.__getitem__(name) if hasattr(args, '__getitem__') else None]

        # Since we cannot rely on dynamic attribute access, we instead rely on order:
        # args[0] -> input_features, args[1] -> conv2d1_weight, args[2] -> conv2d1_bias, etc.
        # Number of tensors returned by get_inputs: 10 (including positional_embedding and embed_scale)
        # So we can directly index by position.
        input_features = args[0]
        conv2d1_weight = args[1]  # [C_in=384, C_in=384, 3, 3]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # [d_model=1024, conv_out_dim=3840]
        positional_embedding = args[8]  # [max_source_positions, d_model]
        embed_scale = float(args[9])  # scalar float

        # Ensure dtype is bfloat16 for Triton kernels
        input_features = input_features.to(torch.bfloat16).contiguous()

        # Dimensions
        B = input_features.shape[0]
        C_in = 1  # original input has C_in=1, as per get_inputs
        H = 80
        W = input_features.shape[-1]  # time_dim

        # Conv1: (1, 80, W) -> (384, 40, W//2)
        C_out1 = conv2d1_weight.shape[0]
        H_out1 = (H + 2*1 - 3)//2 + 1
        W_out1 = (W + 2*1 - 3)//2 + 1
        x1 = torch.empty((B, C_out1, H_out1, W_out1), device=input_features.device, dtype=torch.bfloat16)
        grid1 = (B, C_out1, triton.cdiv(H_out1, 16), triton.cdiv(W_out1, 16))
        conv2d_stride2_pad1_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_in, H, W, C_out1, H_out1, W_out1,
            input_features.stride(0), 1, input_features.stride(2), input_features.stride(3),  # fake strides for C_in are not used since C_in=1
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            embed_scale,
            BLOCK_H=16, BLOCK_W=16, BLOCK_C=32,
            num_warps=4, num_stages=2
        )

        # Conv2: (384, 40, W//2) -> (384, 20, W//4)
        C_in2 = C_out1
        H2 = H_out1
        W2 = W_out1
        C_out2 = conv2d2_weight.shape[0]
        H_out2 = (H2 + 2*1 - 3)//2 + 1
        W_out2 = (W2 + 2*1 - 3)//2 + 1
        x2 = torch.empty((B, C_out2, H_out2, W_out2), device=input_features.device, dtype=torch.bfloat16)
        grid2 = (B, C_out2, triton.cdiv(H_out2, 16), triton.cdiv(W_out2, 16))
        conv2d_stride2_pad1_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            embed_scale,
            BLOCK_H=16, BLOCK_W=16, BLOCK_C=32,
            num_warps=4, num_stages=2
        )

        # Conv3: (384, 20, W//4) -> (384, 10, W//8)
        C_in3 = C_out2
        H3 = H_out2
        W3 = W_out2
        C_out3 = conv2d3_weight.shape[0]
        H_out3 = (H3 + 2*1 - 3)//2 + 1
        W_out3 = (W3 + 2*1 - 3)//2 + 1
        x3 = torch.empty((B, C_out3, H_out3, W_out3), device=input_features.device, dtype=torch.bfloat16)
        grid3 = (B, C_out3, triton.cdiv(H_out3, 16), triton.cdiv(W_out3, 16))
        conv2d_stride2_pad1_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, C_in3, H3, W3, C_out3, H_out3, W_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            embed_scale,
            BLOCK_H=16, BLOCK_W=16, BLOCK_C=32,
            num_warps=4, num_stages=2
        )

        # Reshape to [B, time_after_conv, 384*10]
        b, c, f, t = x3.size()
        x3 = x3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, time_after_conv, 3840]

        # Prepare for linear projection: x_row = [B*time_after_conv, 3840]
        B2 = b
        S = t
        K = 3840
        N = conv_out_weight.shape[0]  # 1024
        x_row = x3.view(B2 * S, K).contiguous().to(torch.bfloat16)

        # Output y: [B, S, N]
        y = torch.empty((B2, S, N), device=input_features.device, dtype=torch.bfloat16)

        # Launch Triton GEMM kernel
        stride_x_row = x_row.stride(0)  # K
        stride_x_k = x_row.stride(1)    # 1

        w = conv_out_weight.contiguous()  # [1024, 3840]
        stride_w_n = w.stride(0)  # 3840
        stride_w_k = w.stride(1)  # 1

        stride_y_row = y.stride(0)  # S
        stride_y_n = y.stride(2)    # 1

        BLOCK_N = 128
        BLOCK_K = 64
        grid_linear = (B2 * S, triton.cdiv(N, BLOCK_N))
        linear_matmul_kernel[grid_linear](
            x_row, w, y,
            B2, S, K, N,
            stride_x_row, stride_x_k,
            stride_w_n, stride_w_k,
            stride_y_row, stride_y_n,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Scale by embed_scale
        y_flat = y.view(-1)
        N_elems = y_flat.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](y_flat, float(embed_scale), N_elems=N_elems, num_warps=4, num_stages=2)

        # Add positional embedding [time_after_conv, 1024], broadcast over batch
        pos_emb = positional_embedding[:S, :].to(torch.bfloat16).contiguous()  # [S, N]
        grid_add = (triton.cdiv(N_elems, 1024),)
        add_pos_emb_kernel[grid_add](y_flat, pos_emb.view(-1), N_elems=N_elems, num_warps=4, num_stages=2)

        return y


def run(*args):
    return ModelNew()(*args)
