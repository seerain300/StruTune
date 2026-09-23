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
    x_ptr,           # input [N, Ci, Fi, Ti], dtype bfloat16
    weight_ptr,      # weights [Co, Ci, 3, 3], dtype bfloat16
    bias_ptr,        # bias [Co], dtype bfloat16
    output_ptr,      # output [N, Co, F_out, T_out], dtype bfloat16
    N, Ci, Co, Fi, Ti, F_out, T_out,
    x_sN, x_sCi, x_sF, x_sT,
    w_sCo, w_sCi, w_sKh, w_sKw,
    out_sN, out_sCo, out_sF, out_sT,
):
    n = tl.program_id(0)  # batch
    co = tl.program_id(1)  # output channel
    f_out = tl.program_id(2)  # output feature index
    t_out = tl.program_id(3)  # output time index

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # We will loop over input channels (Ci) and 3x3 kernel window
    # Compute input indices for stride=2, padding=1
    f_in_base = f_out * 2 + 1
    t_in_base = t_out * 2 + 1

    # Loop over kernel window and input channels
    for ci in range(0, Ci):
        # For each kernel element (kh, kw) in 3x3
        for kh in range(0, 3):
            for kw in range(0, 3):
                # Compute input index (handle padding: if out of bounds, skip)
                fi = f_in_base + kh - 1  # kh: 0,1,2 -> offset -1,0,1
                ti = t_in_base + kw - 1  # kw: 0,1,2 -> offset -1,0,1

                # Masks for bounds
                mask_f = (fi >= 0) & (fi < Fi)
                mask_t = (ti >= 0) & (ti < Ti)
                mask = mask_f & mask_t

                # Load input scalar x[n, ci, fi, ti] (bf16), default 0 when masked
                x_val = tl.load(
                    x_ptr + n * x_sN + ci * x_sCi + fi * x_sF + ti * x_sT,
                    mask=mask, other=0.0
                )
                x_val = x_val.to(tl.float32)

                # Load weight scalar w[co, ci, kh, kw] (bf16), default 0 when masked
                w_val = tl.load(
                    weight_ptr + co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw,
                    mask=True, other=0.0
                )
                w_val = w_val.to(tl.float32)

                acc += x_val * w_val

    # Add bias
    b = tl.load(bias_ptr + co, mask=True, other=0.0).to(tl.float32)
    acc = acc + b

    # Fused GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    acc_cubed = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * acc_cubed)))

    # Store result (bf16)
    tl.store(
        output_ptr + n * out_sN + co * out_sCo + f_out * out_sF + t_out * out_sT,
        gelu.to(tl.bfloat16)
    )


@triton.jit
def linear_bmm_kernel(
    x_ptr,      # input [N, M, T] (here T is time, M=3840), dtype bfloat16
    w_ptr,      # weights [M, K] (here K=1024), dtype bfloat16
    y_ptr,      # output [N, T, K], dtype bfloat16
    N, T, M, K,
    x_sN, x_sM, x_sT,
    w_sM, w_sK,
    y_sN, y_sT, y_sK,
    BLOCK_M: tl.constexpr,  # tile over input dimension M
    BLOCK_K: tl.constexpr,  # tile over output dimension K
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)

    k_start = k_block * BLOCK_K
    offs_k = k_start + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Loop over input dimension M in tiles
    for m_start in range(0, M, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        # Load x[n, offs_m, t] as a vector
        x_vec = tl.load(
            x_ptr + n * x_sN + offs_m * x_sM + t * x_sT,
            mask=mask_m, other=0.0
        ).to(tl.float32)  # [BLOCK_M]

        # Load W[offs_m, offs_k] as a matrix
        w_mat = tl.load(
            w_ptr + offs_m[:, None] * w_sM + offs_k[None, :] * w_sK,
            mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # acc += sum_m (x_vec[m] * w_mat[m, :]) for this K tile
        # Triton supports tl.dot between vectors and matrices
        acc += tl.dot(x_vec, w_mat)  # [BLOCK_K]

    # Store acc to y[n, t, offs_k]
    tl.store(
        y_ptr + n * y_sN + t * y_sT + offs_k * y_sK,
        acc.to(tl.bfloat16),
        mask=mask_k
    )


@triton.jit
def scale_kernel(
    y_ptr,        # input/output [N, T, K], dtype bfloat16
    scale,        # float32 scalar
    N, T, K,
    y_sN, y_sT, y_sK,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    k = tl.program_id(2)
    # Load scalar, scale, store
    val = tl.load(y_ptr + n * y_sN + t * y_sT + k * y_sK).to(tl.float32)
    val = val * scale
    tl.store(y_ptr + n * y_sN + t * y_sT + k * y_sK, val.to(tl.bfloat16))


@triton.jit
def add_pos_emb_kernel(
    y_ptr,        # input/output [N, T, K], dtype bfloat16
    pos_ptr,      # positional embedding [T, K], dtype bfloat16
    N, T, K,
    y_sN, y_sT, y_sK,
    pos_sT, pos_sK,
):
    n = tl.program_id(0)
    t = tl.program_id(1)
    k = tl.program_id(2)
    # Load y[n, t, k] and pos[t, k], add, store
    y_val = tl.load(y_ptr + n * y_sN + t * y_sT + k * y_sK).to(tl.float32)
    pos_val = tl.load(pos_ptr + t * pos_sT + k * pos_sK).to(tl.float32)
    y_val = y_val + pos_val
    tl.store(y_ptr + n * y_sN + t * y_sT + k * y_sK, y_val.to(tl.bfloat16))


# ----------------------------
# ModelNew: forward
# ----------------------------

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        input_features: torch.Tensor,          # [N, 1, 80, time_dim], bf16
        conv2d1_weight: torch.Tensor,         # [384, 1, 3, 3], bf16
        conv2d1_bias: torch.Tensor,           # [384], bf16
        conv2d2_weight: torch.Tensor,         # [384, 384, 3, 3], bf16
        conv2d2_bias: torch.Tensor,           # [384], bf16
        conv2d3_weight: torch.Tensor,         # [384, 384, 3, 3], bf16
        conv2d3_bias: torch.Tensor,           # [384], bf16
        conv_out_weight: torch.Tensor,        # [1024, 3840], bf16
        positional_embedding: torch.Tensor,   # [max_source_positions, 1024], bf16
        embed_scale: float,                   # float
    ):
        # Ensure device/dtype and contiguity
        device = input_features.device
        N, Ci, Fi, Ti = input_features.shape
        # Conv1: [N, 384, 80, (Ti - 3)//2 + 1]
        F_out1 = (Fi - 3) // 2 + 1
        T_out1 = (Ti - 3) // 2 + 1
        x1 = torch.empty((N, 384, Fi, T_out1), device=device, dtype=torch.bfloat16)
        grid1 = (N, 384, Fi, T_out1)
        conv2d_stride2_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            N, Ci, 384, Fi, Ti, F_out1, T_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # Conv2: [N, 384, 40, (T_out1 - 3)//2 + 1]
        F_out2 = (Fi - 3) // 2 + 1  # same as input feature maps
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

        # Conv3: [N, 384, 20, (T_out2 - 3)//2 + 1]
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

        # Reshape: (N, 384, 20, T_out3) -> (N, T_out3, 384*20)
        x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, 384 * 20)

        # Linear projection: X [N, T_out3, 3840], W [3840, 1024] (provided)
        W_t = conv_out_weight.transpose(0, 1).contiguous()  # [3840, 1024], bf16
        Y = torch.empty((N, T_out3, 1024), device=device, dtype=torch.bfloat16)

        # Launch linear_bmm_kernel
        BLOCK_M = 128
        BLOCK_K = 128
        grid_linear = (N, T_out3, (1024 + BLOCK_K - 1) // BLOCK_K)
        linear_bmm_kernel[grid_linear](
            x3_reshaped, W_t, Y,
            N, T_out3, 3840, 1024,
            x3_reshaped.stride(0), x3_reshaped.stride(1), x3_reshaped.stride(2),
            W_t.stride(0), W_t.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

        # Scale by embed_scale
        grid_scale = (N, T_out3, 1024)
        scale_kernel[grid_scale](
            Y, float(embed_scale),
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
        )

        # Add positional embedding: pos_emb [T_out3, 1024], bf16
        pos_emb = positional_embedding[:T_out3, :].contiguous()  # [T_out3, 1024], bf16
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
