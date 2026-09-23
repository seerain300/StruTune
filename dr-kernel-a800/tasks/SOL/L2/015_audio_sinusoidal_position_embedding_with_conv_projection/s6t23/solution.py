import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_stride2_gelu(
    x_ptr,          # *ptr to input [B, C_in, H, W]
    w_ptr,          # *ptr to weight [C_out, C_in, 3, 3]
    b_ptr,          # *ptr to bias [C_out]
    out_ptr,        # *ptr to output [B, C_out, H_out, W_out]
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_OW: tl.constexpr,
):
    # Grid: (pid_b, pid_co, pid_h)
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Output height
    oh = pid_h  # each program handles one output height row
    # Vector of output width indices
    ow = tl.arange(0, BLOCK_OW)
    valid_ow = ow < W_out

    # Accumulator for output vector [BLOCK_OW]
    acc = tl.zeros([BLOCK_OW], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C_in):
        # Loop over 3x3 neighborhood
        for kh in range(0, 3):
            ih = oh + kh - 1  # output at oh, with kh neighborhood
            # Only process valid ih
            valid_h = (ih >= 0) & (ih < H)
            for kw in range(0, 3):
                iw = ow + kw - 1
                valid_w = (iw >= 0) & (iw < W)
                mask = valid_ow & valid_h & valid_w

                # Compute input offsets: ((b * C_in + ci) * H + ih) * W + iw
                x_off = ((pid_b * C_in + ci) * H + ih) * W + iw
                x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)  # cast to fp32 for accumulation

                # Load weight scalar w[co, ci, kh, kw]
                w_off = pid_co * (C_in * 9) + ci * 9 + kh * 3 + kw
                w_val = tl.load(w_ptr + w_off).to(tl.float32)

                # Outer product add: acc += w_val * x_val (broadcasted)
                acc += w_val * x_val

    # Add bias
    b_val = tl.load(b_ptr + pid_co).to(tl.float32)
    acc += b_val

    # Apply GELU (tanh approximation)
    # y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    tanh_arg = c * (acc + 0.044715 * x3)
    tanh_val = tl.math.tanh(tanh_arg)
    acc = 0.5 * acc * (1.0 + tanh_val)

    # Store output: out[b, co, oh, ow]
    out_off = (pid_b * C_out + pid_co) * (H_out * W_out) + oh * W_out + ow
    tl.store(out_ptr + out_off, acc, mask=valid_ow)


@triton.jit
def linear_nobias_kernel(
    x_ptr,      # *ptr to [B*T, K]
    w_ptr,      # *ptr to [d_model, K]
    y_ptr,      # *ptr to [B*T, d_model]
    B: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    d_model: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (outer=B*T, d=0..d_model-1, tile along K)
    pid_outer = tl.program_id(0)  # 0..B*T-1
    pid_d = tl.program_id(1)      # 0..d_model-1
    pid_k_tile = tl.program_id(2)

    # Map outer to (b, t)
    b = pid_outer // T
    t = pid_outer % T

    # Accumulator for y[b, t, pid_d]
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K in tiles
    k_start = pid_k_tile * BLOCK_K
    for k in range(0, BLOCK_K):
        kk = k_start + k
        k_valid = kk < K
        x_val = tl.load(x_ptr + (b * T + t) * K + kk, mask=k_valid, other=0.0).to(tl.float32)
        w_val = tl.load(w_ptr + pid_d * K + kk, mask=k_valid, other=0.0).to(tl.float32)
        acc += x_val * w_val

    # Store y[b, t, d]
    tl.store(y_ptr + pid_outer * d_model + pid_d, acc)


@triton.jit
def scale_kernel(
    y_ptr,      # *ptr to [B*T, d_model], float32
    scale,      # scalar float32
    B: tl.constexpr,
    T: tl.constexpr,
    d_model: tl.constexpr,
):
    # Grid: (outer=B*T, d=0..d_model-1)
    pid_outer = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_outer // T
    t = pid_outer % T

    val = tl.load(y_ptr + pid_outer * d_model + pid_d).to(tl.float32) * scale
    tl.store(y_ptr + pid_outer * d_model + pid_d, val)


@triton.jit
def add_pos_emb_kernel(
    y_ptr,        # *ptr to [B*T, d_model], float32
    pos_ptr,      # *ptr to [T, d_model], float32
    B: tl.constexpr,
    T: tl.constexpr,
    d_model: tl.constexpr,
):
    # Grid: (outer=B*T, d=0..d_model-1)
    pid_outer = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_outer // T
    t = pid_outer % T

    y_val = tl.load(y_ptr + pid_outer * d_model + pid_d).to(tl.float32)
    pos_val = tl.load(pos_ptr + t * d_model + pid_d).to(tl.float32)
    tl.store(y_ptr + pid_outer * d_model + pid_d, y_val + pos_val)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args order: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale

        # Input x: [B, 1, 80, time_dim], bfloat16 (layout NCHW)
        x = args[0]  # [B, 1, 80, time_dim]
        w1 = args[1]  # [C_out1=384, C_in=1, 3, 3]
        b1 = args[2]  # [384]
        w2 = args[3]  # [384, 384, 3, 3]
        b2 = args[4]  # [384]
        w3 = args[5]  # [384, 384, 3, 3]
        b3 = args[6]  # [384]
        conv_out_weight = args[7]  # [d_model=1024, K], K = C_out3*H_out3*W_out3
        positional_embedding = args[8]  # [max_source_positions, d_model], bfloat16
        embed_scale = args[9]  # float

        # Shapes
        B, C_in, H, W = x.shape
        C_out1 = w1.shape[0]
        C_out2 = w2.shape[0]
        C_out3 = w3.shape[0]
        H1 = (H + 2 * 1 - 3) // 2 + 1  # 79
        W1 = (W + 2 * 1 - 3) // 2 + 1  # floor(W/2)
        H2 = (H1 + 2 * 1 - 3) // 2 + 1  # 39
        W2 = (W1 + 2 * 1 - 3) // 2 + 1
        H3 = (H2 + 2 * 1 - 3) // 2 + 1
        W3 = (W2 + 2 * 1 - 3) // 2 + 1

        # Ensure inputs are contiguous; NCHW layout expected by conv kernel
        x = x.contiguous()
        w1 = w1.contiguous()
        b1 = b1.contiguous()
        w2 = w2.contiguous()
        b2 = b2.contiguous()
        w3 = w3.contiguous()
        b3 = b3.contiguous()
        # We'll pass conv_out_weight and positional_embedding as is; they're not modified in forward.

        # Allocate outputs for convs (fp32 accumulation, final cast handled after)
        y1 = torch.empty((B, C_out1, H1, W1), device=x.device, dtype=torch.float32)
        y2 = torch.empty((B, C_out2, H2, W2), device=x.device, dtype=torch.float32)
        y3 = torch.empty((B, C_out3, H3, W3), device=x.device, dtype=torch.float32)

        # Launch conv1 + GELU
        grid1 = (B, C_out1, H1)
        conv2d_nchw_stride2_gelu[grid1](
            x, w1, b1, y1,
            B, C_in, H, W, C_out1, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OW=32,
            num_warps=4, num_stages=2,
        )

        # Launch conv2 + GELU
        grid2 = (B, C_out2, H2)
        conv2d_nchw_stride2_gelu[grid2](
            y1, w2, b2, y2,
            B, C_out1, H1, W1, C_out2, H2, W2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OW=32,
            num_warps=4, num_stages=2,
        )

        # Launch conv3 + GELU
        grid3 = (B, C_out3, H3)
        conv2d_nchw_stride2_gelu[grid3](
            y2, w3, b3, y3,
            B, C_out2, H2, W2, C_out3, H3, W3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_OW=32,
            num_warps=4, num_stages=2,
        )

        # Reshape y3 to [B, W3, C_out3*H3] using view (metadata only, no data movement)
        B, C, F, T3 = y3.shape  # C=384, F=H3, T3=W3
        K = C * F  # 384 * H3 * W3

        # Create x2d_view as [B, T3, K] using permute and view; but Triton cannot permute, so allocate and copy.
        # Allocate target and copy with reshape view (data stays same, just change strides logically).
        # We can do a simple reshape: y3 is [B, C, F, T3]; we need [B, T3, C*F].
        # Allocate y3_reshaped [B, T3, K] and fill via pointer arithmetic by treating y3 as [B, C, F, T3].
        # Since y3 is contiguous in NCHW, we can map linear indices:
        # For b, t, k: b*stride_b + t*stride_t + k*stride_k
        # But better: allocate empty [B, T3, K] and copy row-wise from y3.
        y3_reshaped = torch.empty((B, T3, K), device=x.device, dtype=torch.float32)

        # Manually map from [B, C, F, T3] to [B, T3, C*F] using row-wise copy:
        # For each b and t, copy C*F elements from y3[b, :, :, t] into y3_reshaped[b, t, :].
        # Implement with torch ops (allowed minimal ops). This is only to produce input for linear kernel; convs are fully Triton.
        for b in range(B):
            for t in range(T3):
                # y3[b, :, :, t] is [C, F]
                # Flatten to [C*F]
                row = y3[b].view(C, F)[:, :, t]  # shape [C, F]
                # However row is [C, F], so we need to flatten C*F. Since F=H3 and C=384, we can use contiguous and view:
                row = y3[b].view(C, F)[:, :, t].contiguous().view(C * F)  # [C*F]
                y3_reshaped[b, t, :] = row

        # Prepare weights conv_out_weight: [d_model=1024, K], ensure fp32
        d_model = conv_out_weight.shape[0]
        w = conv_out_weight.contiguous().to(torch.float32)

        # Positional embedding slice: [T3, d_model], bfloat16 -> fp32 for compute
        pos = positional_embedding[:T3, :].contiguous().to(torch.float32)  # [T3, d_model]

        # Allocate y_lin [B*T3, d_model] in fp32
        y_lin = torch.empty((B * T3, d_model), device=x.device, dtype=torch.float32)

        # Launch Triton linear (GEMV without bias)
        BLOCK_K = 256
        grid_linear = (B * T3, d_model, (K + BLOCK_K - 1) // BLOCK_K)
        linear_nobias_kernel[grid_linear](
            y3_reshaped, w, y_lin,
            B, T3, K, d_model,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Scale
        y_scaled = torch.empty_like(y_lin)
        grid_scale = (B * T3, d_model)
        scale_kernel[grid_scale](
            y_lin, float(embed_scale),
            B, T3, d_model,
        )

        # Add positional embedding
        y_final = torch.empty((B * T3, d_model), device=x.device, dtype=torch.float32)
        grid_pos = (B * T3, d_model)
        add_pos_emb_kernel[grid_pos](
            y_scaled, pos,
            B, T3, d_model,
        )

        # Reshape to [B, T3, d_model]
        y_out = y_final.view(B, T3, d_model)

        # If original expected dtype is bfloat16 (as inputs), cast to bfloat16 for output
        # Note: The original helper uses bfloat16; we kept computation in fp32 for accuracy, but final output should match dtype expectation.
        # Since the helper returns bfloat16 inputs, we cast to bfloat16 to match.
        y_out = y_out.to(torch.bfloat16)

        return y_out


def run(*args):
    return ModelNew()(*args)
