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
    x_sN, x_sCi, x_sF, x_sT,
    w_sCo, w_sCi, w_sK, w_sL,
    out_sN, out_sCo, out_sF, out_sT,
):
    # program ids for N, Co, F_out, T_out
    pid_N = tl.program_id(0)
    pid_Co = tl.program_id(1)
    pid_F = tl.program_id(2)
    pid_T = tl.program_id(3)

    # initialize accumulator
    acc = tl.zeros([], dtype=tl.float32)  # compute in fp32 for numerical stability

    # iterate over input channels and 3x3 window
    for ci in range(0, Ci):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # compute input index with padding: h = F_out + kh - 1, w = T_out + kw - 1
                h = pid_F + kh - 1
                w = pid_T + kw - 1
                in_bounds = (h >= 0) & (h < Fi) & (w >= 0) & (w < Ti)
                # compute input offset
                # x[n, ci, h, w]
                x_off = pid_N * x_sN + ci * x_sCi + h * x_sF + w * x_sT
                # load input, masked
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                x_val = x_val.to(tl.float32)
                # load weight [Co, Ci, 3, 3] at (Co=pid_Co, Ci, kh, kw)
                w_off = pid_Co * w_sCo + ci * w_sCi + kh * w_sK + kw * w_sL
                w_val = tl.load(weight_ptr + w_off)
                w_val = w_val.to(tl.float32)
                # accumulate
                acc += x_val * w_val

    # add bias
    b = tl.load(bias_ptr + pid_Co)
    acc = acc + b.to(tl.float32)

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * x3)))

    # store to output [N, Co, F_out, T_out]
    out_off = pid_N * out_sN + pid_Co * out_sCo + pid_F * out_sF + pid_T * out_sT
    # Cast to output dtype as needed (bfloat16 expected)
    tl.store(output_ptr + out_off, gelu.to(tl.float32))  # store as fp32; final casting handled by caller if needed


@triton.jit
def linear_bmm_kernel(
    X_ptr,       # [N, M, T] where T is T_out3 (rows), M is Co*F_out (features)
    W_ptr,       # [M, K] weights, M=3840, K=1024
    Y_ptr,       # [N, T, K] output
    N, T, M, K,
    X_sN, X_sM, X_sT,
    W_sM, W_sK,
    Y_sN, Y_sT, Y_sK,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_t * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # iterate over M dimension in chunks
    for m0 in range(0, M, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        # load X[n, m_idx, pid_t] as vector of length BLOCK_M
        X_off = pid_n * X_sN + m_idx * X_sM + pid_t * X_sT
        X_vec = tl.load(X_ptr + X_off, mask=m_idx < M, other=0.0)  # [BLOCK_M], fp32
        # load W[m_idx, k] as matrix [BLOCK_M, BLOCK_K]
        W_off = m_idx[:, None] * W_sM + offs_k[None, :] * W_sK
        W_mat = tl.load(W_ptr + W_off, mask=(m_idx[:, None] < M) & (offs_k[None, :] < K), other=0.0)  # [BLOCK_M, BLOCK_K]
        # acc += X_vec[:, None] * W_mat
        acc += X_vec[:, None] * W_mat

    # store results into Y[n, pid_t, offs_k]
    Y_off = pid_n * Y_sN + pid_t * Y_sT + offs_k[None, :] * Y_sK
    tl.store(Y_ptr + Y_off, acc, mask=(offs_k[None, :] < K))


@triton.jit
def scale_kernel(
    Y_ptr,         # [N, T, K]
    scale,         # scalar float
    N, T, K,
    Y_sN, Y_sT, Y_sK,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_k = pid_k * 64 + tl.arange(0, 64)
    Y_off = pid_n * Y_sN + pid_t * Y_sT + offs_k[None, :] * Y_sK
    Y_vals = tl.load(Y_ptr + Y_off, mask=(offs_k[None, :] < K), other=0.0)
    Y_vals = Y_vals * scale
    tl.store(Y_ptr + Y_off, Y_vals, mask=(offs_k[None, :] < K))


@triton.jit
def add_pos_emb_kernel(
    Y_ptr,         # [N, T, K]
    pos_emb_ptr,   # [T, K]
    N, T, K,
    Y_sN, Y_sT, Y_sK,
    pe_sT, pe_sK,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_k = pid_k * 64 + tl.arange(0, 64)
    Y_off = pid_n * Y_sN + pid_t * Y_sT + offs_k[None, :] * Y_sK
    Y_vals = tl.load(Y_ptr + Y_off, mask=(offs_k[None, :] < K), other=0.0)
    pe_off = pid_t * pe_sT + offs_k[None, :] * pe_sK
    pe_vals = tl.load(pos_emb_ptr + pe_off, mask=(offs_k[None, :] < K), other=0.0)
    Y_vals = Y_vals + pe_vals
    tl.store(Y_ptr + Y_off, Y_vals, mask=(offs_k[None, :] < K))


# ----------------------------
# Forward (ModelNew)
# ----------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expected args from get_inputs:
        # input_features [N, 1, 80, time_dim]
        # conv weights/biases for 3 layers
        # conv_out_weight [K=1024, M=3840]
        # positional_embedding [max_source_positions, d_model]
        # embed_scale (float)
        input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale = args

        device = input_features.device
        N, Ci, Fi, Ti = input_features.shape  # Ci=1
        # Ensure dtype and contiguity
        input_features = input_features.contiguous().to(torch.bfloat16)

        # Conv1: [N, Co=384, F_out, T_out1]
        F_out1 = (Fi - 3) // 2 + 1
        T_out1 = (Ti - 3) // 2 + 1
        x1 = torch.empty((N, 384, F_out1, T_out1), device=device, dtype=torch.bfloat16)
        grid1 = (N, 384, F_out1, T_out1)
        conv2d_stride2_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            N, Ci, 384, Fi, Ti, F_out1, T_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # Conv2: [N, 384, F_out2, T_out2]
        F_out2 = (F_out1 - 3) // 2 + 1
        T_out2 = (T_out1 - 3) // 2 + 1
        x2 = torch.empty((N, 384, F_out2, T_out2), device=device, dtype=torch.bfloat16)
        grid2 = (N, 384, F_out2, T_out2)
        conv2d_stride2_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            N, 384, 384, F_out1, T_out1, F_out2, T_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # Conv3: [N, 384, F_out3, T_out3]
        F_out3 = (F_out2 - 3) // 2 + 1
        T_out3 = (T_out2 - 3) // 2 + 1
        x3 = torch.empty((N, 384, F_out3, T_out3), device=device, dtype=torch.bfloat16)
        grid3 = (N, 384, F_out3, T_out3)
        conv2d_stride2_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            N, 384, 384, F_out2, T_out2, F_out3, T_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # Reshape: [N, T_out3, Co*F_out3]
        x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, 384 * F_out3)

        # Linear projection: X [N, M=3840, T_out3], W [K=1024, M=3840] -> Y [N, T_out3, K]
        W_t = conv_out_weight.transpose(0, 1).contiguous()  # [3840, 1024], bfloat16
        Y = torch.empty((N, T_out3, 1024), device=device, dtype=torch.bfloat16)

        BLOCK_M = 128
        BLOCK_K = 128
        grid_linear = (N, T_out3, (1024 + BLOCK_K - 1) // BLOCK_K)
        linear_bmm_kernel[grid_linear](
            x3_reshaped, W_t, Y,
            N, T_out3, 384 * F_out3, 1024,
            x3_reshaped.stride(0), x3_reshaped.stride(1), x3_reshaped.stride(2),
            W_t.stride(0), W_t.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Scale by embed_scale
        grid_scale = (N, T_out3, (1024 + 64 - 1) // 64)
        scale_kernel[grid_scale](
            Y, float(embed_scale),
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
        )

        # Add positional embedding: pos_emb [T_out3, 1024] (from get_inputs), bfloat16
        pos_emb = positional_embedding[:T_out3, :].contiguous()  # [T_out3, 1024], bfloat16

        grid_add = (N, T_out3, (1024 + 64 - 1) // 64)
        add_pos_emb_kernel[grid_add](
            Y, pos_emb,
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
        )

        return Y


def run(*args):
    return ModelNew()(*args)
