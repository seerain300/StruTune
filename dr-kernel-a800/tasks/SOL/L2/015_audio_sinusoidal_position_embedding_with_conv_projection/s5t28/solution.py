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


# -------------------------
# Triton kernels (all launched in forward)
# -------------------------

# 1) Conv2d with Ci=1, 3x3, stride=2, padding=1, bias, GELU
@triton.jit
def conv_ci1_stride2_bias_gelu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, Co, F_in, T_in, T_out,
    x_strideN, x_strideC, x_strideF, x_strideT,
    w_strideCo, w_strideCi, w_strideKh, w_strideKw,
    y_strideN, y_strideCo, y_strideF, y_strideT,
    BLOCK_OUT: tl.constexpr,
):
    # Grid: (N, Co, F_in * T_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_ft = tl.program_id(2)

    f_in = pid_ft // T_out
    t_out = pid_ft % T_out

    acc = tl.zeros((), dtype=tl.float32)

    # Ci=1, loop over 3x3 window
    for kh in range(3):
        for kw in range(3):
            t_in = t_out * 2 + kh - 1
            if (t_in >= 0) and (t_in < T_in):
                x_ptr = X_ptr + pid_n * x_strideN + 0 * x_strideC + f_in * x_strideF + t_in * x_strideT
                x_val = tl.load(x_ptr).to(tl.float32)
                w_ptr = W_ptr + pid_co * w_strideCo + 0 * w_strideCi + kh * w_strideKh + kw * w_strideKw
                w_val = tl.load(w_ptr).to(tl.float32)
                acc += x_val * w_val

    # Add bias and GELU (tanh approximation)
    b_val = tl.load(B_ptr + pid_co).to(tl.float32)
    acc = acc + b_val

    c = 0.7978845608028654
    x3 = acc * acc * acc
    gelu_inner = c * (acc + 0.044715 * x3)
    gelu = 0.5 * acc * (1.0 + tl.tanh(gelu_inner))

    y_ptr = Y_ptr + pid_n * y_strideN + pid_co * y_strideCo + f_in * y_strideF + t_out * y_strideT
    tl.store(y_ptr, gelu)


# 2) Conv2d generic, Ci>1, 3x3, stride=2, padding=1, bias, GELU
@triton.jit
def conv_general_stride2_bias_gelu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, Ci, Co, F_in, T_in, T_out,
    x_strideN, x_strideC, x_strideF, x_strideT,
    w_strideCo, w_strideCi, w_strideKh, w_strideKw,
    y_strideN, y_strideCo, y_strideF, y_strideT,
    BLOCK_OUT: tl.constexpr,
):
    # Grid: (N, Co, F_in * T_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_ft = tl.program_id(2)

    f_in = pid_ft // T_out
    t_out = pid_ft % T_out

    acc = tl.zeros((), dtype=tl.float32)

    for ci in range(Ci):
        for kh in range(3):
            for kw in range(3):
                t_in = t_out * 2 + kh - 1
                if (t_in >= 0) and (t_in < T_in):
                    x_ptr = X_ptr + pid_n * x_strideN + ci * x_strideC + f_in * x_strideF + t_in * x_strideT
                    x_val = tl.load(x_ptr).to(tl.float32)
                    w_ptr = W_ptr + pid_co * w_strideCo + ci * w_strideCi + kh * w_strideKh + kw * w_strideKw
                    w_val = tl.load(w_ptr).to(tl.float32)
                    acc += x_val * w_val

    # Add bias and GELU (tanh approximation)
    b_val = tl.load(B_ptr + pid_co).to(tl.float32)
    acc = acc + b_val

    c = 0.7978845608028654
    x3 = acc * acc * acc
    gelu_inner = c * (acc + 0.044715 * x3)
    gelu = 0.5 * acc * (1.0 + tl.tanh(gelu_inner))

    y_ptr = Y_ptr + pid_n * y_strideN + pid_co * y_strideCo + f_in * y_strideF + t_out * y_strideT
    tl.store(y_ptr, gelu)


# 3) Batched GEMV: Y[n, t, k] = sum_j X[n, t, j] * W[j, k]
# X: [N, T, M], W: [M, K], Y: [N, T, K]
@triton.jit
def linear_bmm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, T, M, K,
    x_strideN, x_strideT, x_strideM,
    w_strideM, w_strideK,
    y_strideN, y_strideT, y_strideK,
    BLOCK_M: tl.constexpr,
):
    # Grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        x_ptrs = X_ptr + pid_n * x_strideN + pid_t * x_strideT + offs_m * x_strideM
        x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        w_ptrs = W_ptr + offs_m * w_strideM + pid_k * w_strideK
        w_vals = tl.load(w_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        acc += tl.sum(x_vals * w_vals, axis=0)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    tl.store(y_ptr, acc)


# 4) Elementwise scale: Y *= scale (embed_scale = sqrt(d_model))
@triton.jit
def scale_embed_kernel(
    Y_ptr, scale, N, T, K,
    y_strideN, y_strideT, y_strideK,
):
    # Grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    val = tl.load(y_ptr).to(tl.float32) * scale
    tl.store(y_ptr, val)


# 5) Elementwise add positional embedding: Y += pos_emb
@triton.jit
def add_pos_emb_kernel(
    Y_ptr, pos_ptr, N, T, K,
    y_strideN, y_strideT, y_strideK,
    pos_strideF, pos_strideE,
):
    # Grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    y_val = tl.load(y_ptr).to(tl.float32)
    pos_val = tl.load(pos_ptr + 0 * pos_strideF + pid_k * pos_strideE).to(tl.float32)
    tl.store(y_ptr, y_val + pos_val)


# -------------------------
# ModelNew: forward must launch all kernels
# -------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        # Ensure bfloat16 and contiguous
        device = input_features.device
        assert TRITON_AVAILABLE, "Triton is not available"
        input_features = input_features.to(torch.bfloat16).contiguous()
        conv2d1_weight = conv2d1_weight.to(torch.bfloat16).contiguous()
        conv2d1_bias = conv2d1_bias.to(torch.bfloat16).contiguous()
        conv2d2_weight = conv2d2_weight.to(torch.bfloat16).contiguous()
        conv2d2_bias = conv2d2_bias.to(torch.bfloat16).contiguous()
        conv2d3_weight = conv2d3_weight.to(torch.bfloat16).contiguous()
        conv2d3_bias = conv2d3_bias.to(torch.bfloat16).contiguous()
        # conv_out_weight is [K, M] in original (d_model=1024, conv_out_dim=3840); we will use it in linear kernel as transposed [M, K]
        conv_out_weight = conv_out_weight.to(torch.bfloat16).contiguous()
        positional_embedding = positional_embedding.to(torch.bfloat16).contiguous()

        N, C_in, F_in, T_in = input_features.shape
        Co1 = conv2d1_weight.shape[0]  # 384
        # Conv1: Ci=1, 3x3, stride=2, padding=1
        F_out1 = F_in  # same
        T_out1 = (T_in - 3) // 2 + 1
        x1 = torch.empty((N, Co1, F_out1, T_out1), dtype=torch.bfloat16, device=device)
        grid_conv1 = (N, Co1, F_out1 * T_out1)
        conv_ci1_stride2_bias_gelu_kernel[grid_conv1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            N, Co1, F_in, T_in, T_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            BLOCK_OUT=1,
        )

        # Conv2: Ci=Co1, 3x3, stride=2, padding=1
        Co2 = conv2d2_weight.shape[0]  # 384
        F_out2 = F_out1 // 2  # 80 -> 40
        T_out2 = (T_out1 - 3) // 2 + 1
        x2 = torch.empty((N, Co2, F_out2, T_out2), dtype=torch.bfloat16, device=device)
        grid_conv2 = (N, Co2, F_out2 * T_out2)
        conv_general_stride2_bias_gelu_kernel[grid_conv2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            N, Co1, Co2, F_out1, T_out1, T_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            BLOCK_OUT=1,
        )

        # Conv3: Ci=Co2, 3x3, stride=2, padding=1
        Co3 = conv2d3_weight.shape[0]  # 384
        F_out3 = F_out2 // 2  # 40 -> 20
        T_out3 = (T_out2 - 3) // 2 + 1
        x3 = torch.empty((N, Co3, F_out3, T_out3), dtype=torch.bfloat16, device=device)
        grid_conv3 = (N, Co3, F_out3 * T_out3)
        conv_general_stride2_bias_gelu_kernel[grid_conv3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            N, Co2, Co3, F_out2, T_out2, T_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            BLOCK_OUT=1,
        )

        # Permute: [N, T_out3, Co3*F_out3]
        X_for_linear = x3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, Co3 * F_out3)
        N2, T, M = X_for_linear.shape
        K = conv_out_weight.shape[0]  # 1024
        # We need W as [M, K], i.e., conv_out_weight transposed (original is [K, M]).
        Wt = conv_out_weight.transpose(0, 1).contiguous()  # [M=3840, K=1024]
        Y = torch.empty((N2, T, K), dtype=torch.bfloat16, device=device)

        # Launch linear BMM kernel
        grid_linear = (N2, T, K)
        linear_bmm_kernel[grid_linear](
            X_for_linear, Wt, Y,
            N2, T, M, K,
            X_for_linear.stride(0), X_for_linear.stride(1), X_for_linear.stride(2),
            Wt.stride(0), Wt.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=128,
        )

        # Scale by embed_scale = sqrt(1024) = 32.0
        scale = 1.0 / embed_scale  # use reciprocal to avoid a separate division kernel
        grid_scale = (N2, T, K)
        scale_embed_kernel[grid_scale](
            Y, scale, N2, T, K,
            Y.stride(0), Y.stride(1), Y.stride(2),
        )

        # Add positional embedding: shape [max, d_model] but we only need [:T, :]
        # Note: positional_embedding has shape [max_source_positions, 1024]
        grid_pos = (N2, T, K)
        add_pos_emb_kernel[grid_pos](
            Y, positional_embedding, N2, T, K,
            Y.stride(0), Y.stride(1), Y.stride(2),
            positional_embedding.stride(0), positional_embedding.stride(1),
        )

        return Y


def run(*args):
    return ModelNew()(*args)
