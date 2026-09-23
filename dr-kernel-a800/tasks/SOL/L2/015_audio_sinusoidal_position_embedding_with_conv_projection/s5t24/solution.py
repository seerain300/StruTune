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

# 1) Conv2d specialized for Ci=1: input [N, 1, F_in, T_in], weight [Co, 1, 3, 3], bias [Co]
# Output [N, Co, F_in, T_out] where T_out = (T_in - 3)//2 + 1
@triton.jit
def conv_ci1_stride2_bias_gelu_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    N, F_in, T_in, T_out, Co,
    x_strideN, x_strideC, x_strideF, x_strideT,
    w_strideCo, w_strideCi, w_strideKh, w_strideKw,
    out_strideN, out_strideCo, out_strideF, out_strideT,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_f = tl.program_id(2)  # output frequency index
    pid_to = tl.program_id(3) # output time index

    acc = 0.0

    # Reduction over 3x3 window and Ci=1
    for kh in range(3):
        for kw in range(3):
            t_in = pid_to * 2 + kh - 1
            if (t_in >= 0) and (t_in < T_in):
                x_ptr = X_ptr + pid_n * x_strideN + 0 * x_strideC + pid_f * x_strideF + t_in * x_strideT
                x_val = tl.load(x_ptr).to(tl.float32)
                w_ptr = W_ptr + pid_co * w_strideCo + 0 * w_strideCi + kh * w_strideKh + kw * w_strideKw
                w_val = tl.load(w_ptr).to(tl.float32)
                acc += x_val * w_val

    # Add bias
    b_val = tl.load(B_ptr + pid_co).to(tl.float32)
    acc = acc + b_val

    # GELU approximation
    c = 0.7978845608028654
    x3 = acc * acc * acc
    gelu_inner = c * (acc + 0.044715 * x3)
    gelu = 0.5 * acc * (1.0 + tl.tanh(gelu_inner))

    out_ptr = OUT_ptr + pid_n * out_strideN + pid_co * out_strideCo + pid_f * out_strideF + pid_to * out_strideT
    tl.store(out_ptr, gelu)


# 2) Conv2d general: input [N, Ci, F_in, T_in], weight [Co, Ci, 3, 3], bias [Co]
# Output [N, Co, F_out, T_out] where F_out = F_in // 2, T_out = (T_in - 3)//2 + 1
@triton.jit
def conv_general_stride2_bias_gelu_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    N, Ci, F_in, T_in, F_out, T_out, Co,
    x_strideN, x_strideC, x_strideF, x_strideT,
    w_strideCo, w_strideCi, w_strideKh, w_strideKw,
    out_strideN, out_strideCo, out_strideF, out_strideT,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_f = tl.program_id(2)  # output frequency index
    pid_to = tl.program_id(3) # output time index

    acc = 0.0

    # Reduction over input channels and 3x3 window
    for ci in range(Ci):
        for kh in range(3):
            for kw in range(3):
                t_in = pid_to * 2 + kh - 1
                if (t_in >= 0) and (t_in < T_in):
                    x_ptr = X_ptr + pid_n * x_strideN + ci * x_strideC + pid_f * x_strideF + t_in * x_strideT
                    x_val = tl.load(x_ptr).to(tl.float32)
                    w_ptr = W_ptr + pid_co * w_strideCo + ci * w_strideCi + kh * w_strideKh + kw * w_strideKw
                    w_val = tl.load(w_ptr).to(tl.float32)
                    acc += x_val * w_val

    # Add bias
    b_val = tl.load(B_ptr + pid_co).to(tl.float32)
    acc = acc + b_val

    # GELU approximation
    c = 0.7978845608028654
    x3 = acc * acc * acc
    gelu_inner = c * (acc + 0.044715 * x3)
    gelu = 0.5 * acc * (1.0 + tl.tanh(gelu_inner))

    out_ptr = OUT_ptr + pid_n * out_strideN + pid_co * out_strideCo + pid_f * out_strideF + pid_to * out_strideT
    tl.store(out_ptr, gelu)


# 3) Linear projection via batched GEMV:
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
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    acc = 0.0
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


# 4) Elementwise scale: Y *= scale (float32)
@triton.jit
def scale_embed_kernel(
    Y_ptr, Y_out_ptr,
    N, T, K,
    y_strideN, y_strideT, y_strideK,
    y_out_strideN, y_out_strideT, y_out_strideK,
    scale,  # float32
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    val = tl.load(y_ptr).to(tl.float32)
    val = val * scale
    y_out_ptr = Y_out_ptr + pid_n * y_out_strideN + pid_t * y_out_strideT + pid_k * y_out_strideK
    tl.store(y_out_ptr, val)


# 5) Add positional embedding: Y_out += pos_emb[:, :]
# pos_emb: [T_out, K], Y_out: [N, T_out, K]
@triton.jit
def add_pos_emb_kernel(
    Y_ptr, POS_ptr, Y_out_ptr,
    N, T_out, K,
    y_strideN, y_strideT, y_strideK,
    pos_strideT, pos_strideK,
    y_out_strideN, y_out_strideT, y_out_strideK,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    y_ptr = Y_ptr + pid_n * y_strideN + pid_t * y_strideT + pid_k * y_strideK
    val_y = tl.load(y_ptr).to(tl.float32)

    pos_ptr = POS_ptr + pid_t * pos_strideT + pid_k * pos_strideK
    val_pos = tl.load(pos_ptr).to(tl.float32)

    val = val_y + val_pos

    y_out_ptr = Y_out_ptr + pid_n * y_out_strideN + pid_t * y_out_strideT + pid_k * y_out_strideK
    tl.store(y_out_ptr, val)


# -------------------------
# ModelNew: forward uses Triton kernels
# -------------------------

class ModelNew(nn.Module):
    def forward(
        self,
        input_features,      # [N, 1, 80, T]
        conv2d1_weight,      # [384, 1, 3, 3]
        conv2d1_bias,        # [384]
        conv2d2_weight,      # [384, 384, 3, 3]
        conv2d2_bias,        # [384]
        conv2d3_weight,      # [384, 384, 3, 3]
        conv2d3_bias,        # [384]
        conv_out_weight,     # [1024, 3840] (d_model, conv_out_dim)
        positional_embedding,# [max_source_positions, 1024], dtype bfloat16
        embed_scale,         # float (sqrt(1024)=32.0)
    ):
        # Ensure contiguity
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()

        # Launch conv1: (1 -> 384)
        N, C1, F_in, T_in = input_features.shape
        Co1 = 384
        T_out1 = (T_in - 3) // 2 + 1
        x1 = torch.empty((N, Co1, F_in, T_out1), dtype=torch.float32, device=input_features.device)

        grid1 = (N, Co1, F_in, T_out1)
        conv_ci1_stride2_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            N, F_in, T_in, T_out1, Co1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # Launch conv2: (384 -> 384), F_out2 = F_in//2
        Co2 = 384
        F_out2 = F_in // 2
        T_out2 = (T_out1 - 3) // 2 + 1
        x2 = torch.empty((N, Co2, F_out2, T_out2), dtype=torch.float32, device=input_features.device)

        grid2 = (N, Co2, F_out2, T_out2)
        conv_general_stride2_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            N, Co1, F_in, T_out1, F_out2, T_out2, Co2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # Launch conv3: (384 -> 384), F_out3 = F_out2//2
        Co3 = 384
        F_out3 = F_out2 // 2
        T_out3 = (T_out2 - 3) // 2 + 1
        x3 = torch.empty((N, Co3, F_out3, T_out3), dtype=torch.float32, device=input_features.device)

        grid3 = (N, Co3, F_out3, T_out3)
        conv_general_stride2_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            N, Co2, F_out2, T_out2, F_out3, T_out3, Co3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # Reshape: [N, T_out3, channels*freq] where channels=384, freq=10
        # x3 has shape [N, 384, F_out3, T_out3]; F_out3=10 (given time_after_conv=131 in some configs)
        C_out = 384
        M = C_out * F_out3  # conv_out_dim = 3840
        x3_perm = x3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, M)

        # Prepare W for GEMV: conv_out_weight [1024, 3840] -> [M, K]
        # Note: original conv_out_weight is [d_model=1024, conv_out_dim=3840]; we need W[j, k] for our GEMV.
        # We transpose to [M, K] (since M=3840, K=1024).
        # This is a torch operation for data preparation, not heavy compute.
        W = conv_out_weight.transpose(0, 1).contiguous()  # [M, K]
        K = conv_out_weight.shape[0]  # 1024

        # Linear projection via Triton GEMV: grid (N, T_out3, K)
        Y = torch.empty((N, T_out3, K), dtype=torch.float32, device=input_features.device)

        grid4 = (N, T_out3, K)
        # We choose a BLOCK_M large enough to cover M=3840 in a few iterations. Using 1024 is fine.
        linear_bmm_kernel[grid4](
            x3_perm, W, Y,
            N, T_out3, M, K,
            x3_perm.stride(0), x3_perm.stride(1), x3_perm.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=1024,
        )

        # Scale by embed_scale = 32.0
        Y_scaled = torch.empty_like(Y, dtype=torch.float32, device=input_features.device)
        grid5 = (N, T_out3, K)
        scale_embed_kernel[grid5](
            Y, Y_scaled,
            N, T_out3, K,
            Y.stride(0), Y.stride(1), Y.stride(2),
            Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2),
            float(embed_scale),
        )

        # Prepare and add positional embedding: [T_out3, K], cast to float32 for addition
        # positional_embedding is [max_source_positions, 1024], bfloat16
        pos_emb = positional_embedding[:T_out3, :].to(torch.float32).contiguous()
        Y_out = torch.empty_like(Y_scaled, dtype=torch.float32, device=input_features.device)
        grid6 = (N, T_out3, K)
        add_pos_emb_kernel[grid6](
            Y_scaled, pos_emb, Y_out,
            N, T_out3, K,
            Y_scaled.stride(0), Y_scaled.stride(1), Y_scaled.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
            Y_out.stride(0), Y_out.stride(1), Y_out.stride(2),
        )

        # Return result in the original expected dtype (bfloat16). The original run likely returns bfloat16.
        # The original code multiplies by embed_scale and adds pos_emb. Here, we kept computation in float32
        # for numerical stability. Return as bfloat16.
        return Y_out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
