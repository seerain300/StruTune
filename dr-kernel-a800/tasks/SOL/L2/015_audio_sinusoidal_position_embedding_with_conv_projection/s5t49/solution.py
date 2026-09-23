import math
import torch
import torch.nn as nn

# Triton is required
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: conv (stride=2, padding=1, bias, fused GELU), GEMV (linear projection), and scale+pos add.

@triton.jit
def conv2d_stride2_bias_gelu_kernel(
    x_ptr,           # input [N, Ci, Fi, Ti], fp32
    weight_ptr,      # weights [Co, Ci, 3, 3], fp32
    bias_ptr,        # bias [Co], fp32
    output_ptr,      # output [N, Co, F_out, T_out], fp32
    N, Ci, Co, Fi, Ti, F_out, T_out,
    x_sN, x_sCi, x_sF, x_sT,        # strides for x
    w_sCo, w_sCi, w_sKh, w_sKw,     # strides for weights
    o_sN, o_sCo, o_sF, o_sT         # strides for output
):
    # Grid: (N, Co, F_out, T_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_f = tl.program_id(2)
    pid_t = tl.program_id(3)

    # Accumulator for conv output before GELU
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over Ci and 3x3 kernel
    # ci in [0, Ci), kh in [0,3), kw in [0,3)
    for ci in range(0, Ci):
        for kh in range(0, 4):  # 3
            for kw in range(0, 4):  # 3
                # Compute input indices for this output position (padding=1, stride=2)
                fi_in = pid_f * 2 + 1 - 1 + kh  # kh is 0..2, but we bound later
                ti_in = pid_t * 2 + 1 - 1 + kw

                # Bounds check
                valid = (fi_in >= 0) & (fi_in < Fi) & (ti_in >= 0) & (ti_in < Ti)

                # Compute input pointers
                x_off = pid_n * x_sN + ci * x_sCi + fi_in * x_sF + ti_in * x_sT
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)

                # Weight for this co, ci, kh, kw
                w_off = pid_co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(weight_ptr + w_off)

                acc += x_val * w_val

    # Add bias
    b_val = tl.load(bias_ptr + pid_co)
    acc += b_val

    # Fused GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    acc_cub = acc * acc * acc
    gelu_in = acc + 0.044715 * acc_cub
    gelu_out = 0.5 * acc * (1.0 + tl.math.tanh(c * gelu_in))

    # Store output
    o_off = pid_n * o_sN + pid_co * o_sCo + pid_f * o_sF + pid_t * o_sT
    tl.store(output_ptr + o_off, gelu_out)


@triton.jit
def linear_gemv_kernel(
    X_ptr,          # input [N, T, M], fp32
    W_ptr,          # weight [M, K], fp32 (note: M is dimension of X's last dim, K is output channels)
    Y_ptr,          # output [N, T, K], fp32
    N, T, M, K,
    xsN, xsT, xsM,
    wsM, wsK,
    ysN, ysT, ysK,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, T, ceil_div(K, BLOCK_K))
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Row of X for this (n, t)
    row_base = pid_n * xsN + pid_t * xsT

    # Accumulator for this (n, t) across K
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < K

        # Load W chunk [BLOCK_K, BLOCK_K] with mask
        w_base = k_idx * wsM
        w_off = w_base[:, None] * wsM + (tl.arange(0, BLOCK_K)[None, :] * wsK)
        W_chunk = tl.load(W_ptr + w_off, mask=mask_k[:, None], other=0.0)

        # Compute dot: sum over M for each k in chunk
        dot_sum = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for m in range(0, M):
            x_val = tl.load(X_ptr + row_base + m * xsM)
            dot_sum += W_chunk[m, :] * x_val

        acc += dot_sum

    # Store Y[n, t, k]
    y_base = pid_n * ysN + pid_t * ysT
    y_off = y_base + tl.arange(0, BLOCK_K) * ysK
    tl.store(Y_ptr + y_off, acc, mask=tl.arange(0, BLOCK_K) < K)


@triton.jit
def scale_add_pos_kernel(
    Y_ptr,          # [N, T, K], fp32
    POS_ptr,        # [T, K], fp32
    scale,          # float32
    N, T, K,
    ysN, ysT, ysK,
    psT, psK,
):
    # Grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Load y and pos
    y_off = pid_n * ysN + pid_t * ysT + pid_k * ysK
    p_off = pid_t * psT + pid_k * psK
    y_val = tl.load(Y_ptr + y_off)
    pos_val = tl.load(POS_ptr + p_off)

    y_new = y_val + pos_val * scale
    tl.store(Y_ptr + y_off, y_new)


class ModelNew(nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: [N, 1, 80, time_dim], dtype=torch.bfloat16 (from get_inputs), but Triton kernels use float32 for computation
        conv* weights/bias: provided, dtype=torch.bfloat16, will cast to float32 for Triton kernels
        conv_out_weight: [1024, 3840] (xavier), dtype=torch.bfloat16, cast to float32
        positional_embedding: [max_time_after_conv, 1024], dtype=torch.bfloat16, cast to float32
        embed_scale: float, e.g., sqrt(1024) = 32.0
        Returns: [N, T_out3, 1024]
        """
        assert TRITON_AVAILABLE, "Triton is not available"

        device = input_features.device

        # Cast to float32 for Triton kernels
        input_f32 = input_features.to(torch.float32)         # [N, 1, 80, time_dim]
        w1 = conv2d1_weight.to(torch.float32)               # [384, 1, 3, 3]
        b1 = conv2d1_bias.to(torch.float32)                 # [384]
        w2 = conv2d2_weight.to(torch.float32)               # [384, 384, 3, 3]
        b2 = conv2d2_bias.to(torch.float32)                 # [384]
        w3 = conv2d3_weight.to(torch.float32)               # [384, 384, 3, 3]
        b3 = conv2d3_bias.to(torch.float32)                 # [384]
        cout_w = conv_out_weight.to(torch.float32)          # [1024, 3840] from get_inputs (xavier)
        pos_emb = positional_embedding.to(torch.float32)    # [max_time_after_conv, 1024]

        N = input_f32.shape[0]
        time_dim = input_f32.shape[3]

        # Dimensions for convs (stride=2, padding=1)
        # conv1: in (1), out (384), F_out = (80-3)//2 + 1 = 38, T_out = (time_dim-3)//2 + 1
        F1_out = (80 - 3) // 2 + 1  # 38
        T1_out = (time_dim - 3) // 2 + 1

        # Allocate conv1 output
        x1 = torch.empty((N, 384, F1_out, T1_out), device=device, dtype=torch.float32)

        # Launch conv1 kernel
        grid1 = (N, 384, F1_out, T1_out)
        conv2d_stride2_bias_gelu_kernel[grid1](
            input_f32, w1, b1, x1,
            N, 1, 384, 80, time_dim, F1_out, T1_out,
            input_f32.stride(0), input_f32.stride(1), input_f32.stride(2), input_f32.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # conv2: in (384), out (384), F_out2 = (F1_out - 3)//2 + 1 = 17, T_out2 = (T1_out - 3)//2 + 1
        F2_out = (F1_out - 3) // 2 + 1  # 17
        T2_out = (T1_out - 3) // 2 + 1

        x2 = torch.empty((N, 384, F2_out, T2_out), device=device, dtype=torch.float32)

        grid2 = (N, 384, F2_out, T2_out)
        conv2d_stride2_bias_gelu_kernel[grid2](
            x1, w2, b2, x2,
            N, 384, 384, F1_out, T1_out, F2_out, T2_out,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # conv3: in (384), out (384), F_out3 = (F2_out - 3)//2 + 1 = 8, T_out3 = (T2_out - 3)//2 + 1
        F3_out = (F2_out - 3) // 2 + 1  # 8
        T3_out = (T2_out - 3) // 2 + 1

        x3 = torch.empty((N, 384, F3_out, T3_out), device=device, dtype=torch.float32)

        grid3 = (N, 384, F3_out, T3_out)
        conv2d_stride2_bias_gelu_kernel[grid3](
            x2, w3, b3, x3,
            N, 384, 384, F2_out, T2_out, F3_out, T3_out,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            w3.stride(0), w3.stride(1), w3.stride(2), w3.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # Reshape: (N, C, F3_out, T3_out) -> (N, T3_out, C*F3_out)
        x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(N, T3_out, 384 * F3_out)

        # Linear projection: X [N, T_out3, M=384*F3_out], W [1024, 3840] provided, transpose to [M=3840, K=1024]
        M_linear = 3840  # per get_inputs
        K_linear = 1024
        W_t = cout_w.transpose(0, 1).contiguous()  # [3840, 1024]

        # Allocate output of GEMV
        y = torch.empty((N, T3_out, K_linear), device=device, dtype=torch.float32)

        # Launch GEMV kernel: grid = (N, T3_out, ceil_div(K, BLOCK_K))
        BLOCK_K = 128
        grid_linear = (N, T3_out, (K_linear + BLOCK_K - 1) // BLOCK_K)
        linear_gemv_kernel[grid_linear](
            x3_reshaped, W_t, y,
            N, T3_out, M_linear, K_linear,
            x3_reshaped.stride(0), x3_reshaped.stride(1), x3_reshaped.stride(2),
            W_t.stride(0), W_t.stride(1),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_K=BLOCK_K,
        )

        # Scale by embed_scale and add positional embedding: y += pos_emb[:T3_out, :] * embed_scale
        # Launch scale+pos kernel
        grid_scale = (N, T3_out, K_linear)
        scale_add_pos_kernel[grid_scale](
            y, pos_emb[:T3_out, :], float(embed_scale),
            N, T3_out, K_linear,
            y.stride(0), y.stride(1), y.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
        )

        return y


def run(*args):
    return ModelNew()(*args)
