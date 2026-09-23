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


# ----------------------------
# Triton kernels
# ----------------------------

@triton.jit
def conv2d_stride2_bias_gelu_kernel(
    x_ptr,           # input [N, Ci, Fi, Ti]
    weight_ptr,      # weights [Co, Ci, 3, 3]
    bias_ptr,        # bias [Co]
    output_ptr,      # output [N, Co, F_out, T_out]
    N, Ci, Co, Fi, Ti, F_out, T_out,
    x_sN, x_sCi, x_sFi, x_sTi,
    w_sCo, w_sCi, w_sKh, w_sKw,
    out_sN, out_sCo, out_sF, out_sT,
):
    # grid: (pid_n, pid_co, pid_f, pid_t)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_f = tl.program_id(2)
    pid_t = tl.program_id(3)

    # Accumulator for GELU
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 window
    # Note: x_t idx corresponds to output position t; input t_idx = t - oh - 1 due to padding=1, stride=2
    # y[n, co, f, t] = bias[co] + sum_{ci=0..Ci-1} sum_{kh, kw in 3x3} x[n, ci, f+kh, t+kw] * weight[co, ci, kh, kw]
    for ci in range(Ci):
        for kh in range(3):
            for kw in range(3):
                in_f = pid_f + kh - 1
                in_t = pid_t + kw - 1  # oh, ow mapping relative to output
                # check bounds
                valid_f = (in_f >= 0) & (in_f < Fi)
                valid_t = (in_t >= 0) & (in_t < Ti)
                # compute input index for x
                x_idx = pid_n * x_sN + ci * x_sCi + in_f * x_sFi + in_t * x_sTi
                x_val = tl.load(x_ptr + x_idx, mask=valid_f & valid_t, other=0.0).to(tl.float32)
                # load weight scalar
                w_idx = pid_co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(weight_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + pid_co).to(tl.float32)
    acc += bias_val

    # GELU (tanh approximation)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * acc * acc * acc)))

    # Store
    out_idx = pid_n * out_sN + pid_co * out_sCo + pid_f * out_sF + pid_t * out_sT
    tl.store(output_ptr + out_idx, gelu)

@triton.jit
def linear_bmm_kernel(
    X_ptr,       # input [N, M, T_out3] where M=Co*F_out3
    W_ptr,       # weights [M, K] where K=1024
    Y_ptr,       # output [N, T_out3, K]
    N, T, M, K,
    X_sN, X_sM, X_sT,
    W_sM, W_sK,
    Y_sN, Y_sT, Y_sK,
    BLOCK_M: tl.constexpr,  # block for M
    BLOCK_K: tl.constexpr,  # block for K
):
    # grid: (N, T_out3, ceil_div(K, BLOCK_K))
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # tile indices
    m_start = pid_t * BLOCK_M
    k_start = pid_k * BLOCK_K

    # pointers for tiles
    X_tile_ptr = X_ptr + pid_n * X_sN + m_start * X_sM
    W_tile_ptr = W_ptr + m_start * W_sM + k_start * W_sK
    Y_tile_ptr = Y_ptr + pid_n * Y_sN + pid_t * Y_sT + k_start * Y_sK

    # accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # loop over M in chunks
    for mm in range(0, M, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        X_vec = tl.load(X_tile_ptr + mm + tl.arange(0, BLOCK_M) * X_sT, mask=m_mask, other=0.0)  # [BLOCK_M]
        # dot product with weight tile
        W_vec = tl.load(W_tile_ptr + (mm + tl.arange(0, BLOCK_M)) * W_sM + tl.arange(0, BLOCK_K) * W_sK,
                        mask=(m_offsets < M)[:, None], other=0.0)  # [BLOCK_M, BLOCK_K]
        acc += tl.dot(X_vec[:, None], W_vec)  # [BLOCK_M, BLOCK_K]

    # store acc to Y
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    tl.store(Y_tile_ptr, acc, mask=k_mask)

@triton.jit
def scale_kernel(
    X_ptr,           # input/output [N, T, K]
    scale,           # float32 scale
    N, T, K,
    X_sN, X_sT, X_sK,
):
    # grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    idx = pid_n * X_sN + pid_t * X_sT + pid_k * X_sK
    x_val = tl.load(X_ptr + idx).to(tl.float32)
    x_val *= scale
    tl.store(X_ptr + idx, x_val)

@triton.jit
def add_pos_emb_kernel(
    X_ptr,           # [N, T, K]
    pos_ptr,         # [T, K] positional embedding
    N, T, K,
    X_sN, X_sT, X_sK,
    pos_sT, pos_sK,
):
    # grid: (N, T, K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    x_idx = pid_n * X_sN + pid_t * X_sT + pid_k * X_sK
    x_val = tl.load(X_ptr + x_idx).to(tl.float32)

    pos_idx = pid_t * pos_sT + pid_k * pos_sK
    pos_val = tl.load(pos_ptr + pos_idx).to(tl.float32)

    x_val += pos_val
    tl.store(X_ptr + x_idx, x_val)


# ----------------------------
# ModelNew: Triton-only forward
# ----------------------------

class ModelNew(nn.Module):
    def forward(self, *args):
        # Expect: input_features [N, Ci=1, Fi=80, Ti], conv weights and biases, conv_out_weight [K, M], positional_embedding [max_source_positions, d_model], embed_scale float
        # Note: *args order follows get_inputs signature.
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        device = input_features.device
        dtype = input_features.dtype
        N = input_features.shape[0]
        Ci = 1  # given in inputs
        Fi = 80
        Ti = input_features.shape[3]

        # Ensure tensors are contiguous and dtype bfloat16 for computation
        input_features = input_features.to(torch.bfloat16).contiguous()
        conv2d1_weight = conv2d1_weight.to(torch.bfloat16).contiguous()
        conv2d1_bias = conv2d1_bias.to(torch.bfloat16).contiguous()
        conv2d2_weight = conv2d2_weight.to(torch.bfloat16).contiguous()
        conv2d2_bias = conv2d2_bias.to(torch.bfloat16).contiguous()
        conv2d3_weight = conv2d3_weight.to(torch.bfloat16).contiguous()
        conv2d3_bias = conv2d3_bias.to(torch.bfloat16).contiguous()
        conv_out_weight = conv_out_weight.to(torch.bfloat16).contiguous()
        positional_embedding = positional_embedding.to(torch.bfloat16).contiguous()

        # Compute T_out for conv1: T_out1 = (Ti - 3)//2 + 1
        T_out1 = (Ti - 3) // 2 + 1
        F_out1 = (Fi - 3) // 2 + 1  # output feature map size (spatial)

        # Allocate and launch conv1: [N, Co=384, F_out1, T_out1]
        x1 = torch.empty((N, 384, F_out1, T_out1), device=device, dtype=torch.bfloat16)
        grid1 = (N, 384, F_out1, T_out1)
        conv2d_stride2_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            N, Ci, 384, Fi, Ti, F_out1, T_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # Compute conv2 sizes
        Co = 384
        F_out2 = (F_out1 - 3) // 2 + 1
        T_out2 = (T_out1 - 3) // 2 + 1

        # Allocate and launch conv2: [N, 384, F_out2, T_out2]
        x2 = torch.empty((N, 384, F_out2, T_out2), device=device, dtype=torch.bfloat16)
        grid2 = (N, 384, F_out2, T_out2)
        conv2d_stride2_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            N, Co, 384, F_out1, T_out1, F_out2, T_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # Compute conv3 sizes
        F_out3 = (F_out2 - 3) // 2 + 1
        T_out3 = (T_out2 - 3) // 2 + 1

        # Allocate and launch conv3: [N, 384, F_out3, T_out3]
        x3 = torch.empty((N, 384, F_out3, T_out3), device=device, dtype=torch.bfloat16)
        grid3 = (N, 384, F_out3, T_out3)
        conv2d_stride2_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            N, Co, 384, F_out2, T_out2, F_out3, T_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # Reshape to [N, T_out3, Co*F_out3]
        x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, Co * F_out3)

        # Linear projection: X [N, T_out3, M=3840], W [M, K=1024] (given as [K, M] but we transpose for kernel)
        W_t = conv_out_weight.transpose(0, 1).contiguous()  # [3840, 1024]
        Y = torch.empty((N, T_out3, 1024), device=device, dtype=torch.bfloat16)

        # Launch linear_bmm_kernel with grid (N, T_out3, ceil_div(1024, BLOCK_K)), BLOCK_M=64, BLOCK_K=128
        BLOCK_M = 64
        BLOCK_K = 128
        grid_linear = (N, T_out3, (1024 + BLOCK_K - 1) // BLOCK_K)
        linear_bmm_kernel[grid_linear](
            x3_reshaped, W_t, Y,
            N, T_out3, Co * F_out3, 1024,
            x3_reshaped.stride(0), x3_reshaped.stride(1), x3_reshaped.stride(2),
            W_t.stride(0), W_t.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Scale by embed_scale = sqrt(1024) = 32.0
        Y_fp32 = Y.to(torch.float32)
        grid_scale = (N, T_out3, 1024)
        scale_kernel[grid_scale](
            Y_fp32, float(embed_scale),
            N, T_out3, 1024,
            Y_fp32.stride(0), Y_fp32.stride(1), Y_fp32.stride(2),
        )
        Y = Y_fp32.to(torch.bfloat16)

        # Add positional embedding: pos_emb shape [T_out3, 1024]
        pos_emb = positional_embedding[:T_out3, :].contiguous()  # [T_out3, 1024], dtype bfloat16
        grid_add = (N, T_out3, 1024)
        add_pos_emb_kernel[grid_add](
            Y, pos_emb,
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
        )

        return Y


def run(*args):
    return ModelNew()(*args)
