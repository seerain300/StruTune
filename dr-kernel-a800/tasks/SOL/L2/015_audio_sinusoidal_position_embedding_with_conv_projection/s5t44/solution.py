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
    x_ptr,           # input [N, Ci, Fi, Ti], dtype: bf16/float
    weight_ptr,      # weights [Co, Ci, 3, 3], dtype: same as x
    bias_ptr,        # bias [Co], dtype: same as x
    output_ptr,      # output [N, Co, F_out, T_out], dtype: same as x
    N, Ci, Co, Fi, Ti, F_out, T_out,
    x_sN, x_sCi, x_sFi, x_sTi,
    w_sCo, w_sCi, w_sK_h, w_sK_w,
    out_sN, out_sCo, out_sFi, out_sTi,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_f = tl.program_id(2)
    pid_t = tl.program_id(3)

    # bounds check (usually grid matches, but keep safe)
    if pid_n >= N or pid_co >= Co or pid_f >= F_out or pid_t >= T_out:
        return

    # accumulate
    acc = tl.zeros((), dtype=tl.float32)

    # loop over Ci and 3x3 window
    for ci in range(0, Ci):
        for kh in range(0, 3):
            for kw in range(0, 3):
                fi = pid_f * 2 + 1 - kh  # due to padding=1 and stride=2
                ti = pid_t * 2 + 1 - kw  # due to padding=1 and stride=2
                valid = (fi >= 0) & (fi < Fi) & (ti >= 0) & (ti < Ti)
                if valid:
                    x_val = tl.load(
                        x_ptr + pid_n * x_sN + ci * x_sCi + fi * x_sFi + ti * x_sTi,
                        eviction_policy="evict_last",
                    )
                    # cast to f32 for accumulation
                    x_val = x_val.to(tl.float32)
                else:
                    x_val = 0.0

                # load weight for (co, ci, kh, kw)
                w_val = tl.load(
                    weight_ptr + pid_co * w_sCo + ci * w_sCi + kh * w_sK_h + kw * w_sK_w,
                    eviction_policy="evict_last",
                )
                w_val = w_val.to(tl.float32)

                acc += x_val * w_val

    # add bias
    b = tl.load(bias_ptr + pid_co).to(tl.float32)
    acc += b

    # GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * x3)))

    # store output as original dtype (assume bfloat16)
    gelu_out = gelu.to(tl.bfloat16)
    tl.store(
        output_ptr + pid_n * out_sN + pid_co * out_sCo + pid_f * out_sFi + pid_t * out_sTi,
        gelu_out,
        eviction_policy="evict_last",
    )


@triton.jit
def linear_bmm_kernel(
    X_ptr,          # input [N, M, T], dtype: bf16
    W_ptr,          # weights [M, K], dtype: bf16 (we will cast to f32 for compute)
    Y_ptr,          # output [N, T, K], dtype: bf16
    N, T, M, K,
    X_sN, X_sM, X_sT,
    W_sM, W_sK,
    Y_sN, Y_sT, Y_sK,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # tile over M
    m_offsets = pid_k * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = pid_t * BLOCK_K + tl.arange(0, BLOCK_K)

    # masks
    mask_m = m_offsets < M
    mask_k = k_offsets < K

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # loop over M in tiles
    for m0 in range(0, M, BLOCK_M):
        m_idx = m0 + tl.arange(0, BLOCK_M)
        mask_m_i = m_idx < M

        # X: [N, M, T]
        # load X[n, m, t]
        x_ptrs = X_ptr + pid_n * X_sN + m_idx[:, None] * X_sM + pid_t * X_sT
        x_mask = mask_m_i[:, None] & mask_k[None, :]
        X_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        X_tile = X_tile.to(tl.float32)

        # W: [M, K]
        w_ptrs = W_ptr + m_idx[:, None] * W_sM + k_offsets[None, :] * W_sK
        W_tile = tl.load(w_ptrs, mask=(mask_m_i[:, None] & mask_k[None, :]), other=0.0)
        W_tile = W_tile.to(tl.float32)

        # acc += X_tile @ W_tile.T
        acc += tl.dot(X_tile, tl.trans(W_tile))

    # store Y[n, t, k] = acc
    Y_ptrs = Y_ptr + pid_n * Y_sN + pid_t * Y_sT + k_offsets[None, :] * Y_sK
    tl.store(Y_ptrs, acc, mask=(mask_k[None, :]))

    # Note: we store f32 into Y_ptr; if Y_ptr is bf16, Triton will cast on store.


@triton.jit
def scale_kernel(
    Y_ptr,          # input/output [N, T, K], dtype: bf16
    scale,          # float32 scalar
    N, T, K,
    Y_sN, Y_sT, Y_sK,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    offsets = pid_k * 1 + tl.arange(0, 1)  # we tile K by pid_k
    mask_k = offsets < K

    Y_ptrs = Y_ptr + pid_n * Y_sN + pid_t * Y_sT + offsets * Y_sK
    Y_vals = tl.load(Y_ptrs, mask=mask_k, other=0.0).to(tl.float32)
    Y_vals = Y_vals * scale
    tl.store(Y_ptrs, Y_vals.to(tl.bfloat16), mask=mask_k)


@triton.jit
def add_pos_emb_kernel(
    Y_ptr,          # input [N, T, K], dtype: bf16
    pos_emb_ptr,    # input [T, K], dtype: bf16
    N, T, K,
    Y_sN, Y_sT, Y_sK,
    pe_sT, pe_sK,
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    offsets = pid_k * 1 + tl.arange(0, 1)  # we tile K by pid_k
    mask_k = offsets < K

    Y_ptrs = Y_ptr + pid_n * Y_sN + pid_t * Y_sT + offsets * Y_sK
    Y_vals = tl.load(Y_ptrs, mask=mask_k, other=0.0).to(tl.float32)

    pe_ptrs = pos_emb_ptr + pid_t * pe_sT + offsets * pe_sK
    pe_vals = tl.load(pe_ptrs, mask=mask_k, other=0.0).to(tl.float32)

    Y_vals = Y_vals + pe_vals
    tl.store(Y_ptrs, Y_vals.to(tl.bfloat16), mask=mask_k)


# ----------------------------
# ModelNew: Triton-only forward
# ----------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_features: torch.Tensor,
                conv2d1_weight: torch.Tensor, conv2d1_bias: torch.Tensor,
                conv2d2_weight: torch.Tensor, conv2d2_bias: torch.Tensor,
                conv2d3_weight: torch.Tensor, conv2d3_bias: torch.Tensor,
                conv_out_weight: torch.Tensor,
                positional_embedding: torch.Tensor,
                embed_scale: float):
        # Ensure device and dtype
        device = input_features.device
        dtype = input_features.dtype  # bfloat16 per get_inputs
        N, Ci, Fi, Ti = input_features.shape
        Co = 384

        # Ensure inputs are contiguous
        x = input_features.contiguous()

        # Allocate conv1 output: [N, Co, F_out1, T_out1]
        F_out1 = (Fi - 3) // 2 + 1
        T_out1 = (Ti - 3) // 2 + 1
        x1 = torch.empty((N, Co, F_out1, T_out1), device=device, dtype=dtype)

        # Launch conv1 kernel
        grid1 = (N, Co, F_out1, T_out1)
        conv2d_stride2_bias_gelu_kernel[grid1](
            x, conv2d1_weight, conv2d1_bias, x1,
            N, Ci, Co, Fi, Ti, F_out1, T_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
        )

        # Conv2: x1 -> [N, Co, F_out2, T_out2]
        F_out2 = (F_out1 - 3) // 2 + 1
        T_out2 = (T_out1 - 3) // 2 + 1
        x2 = torch.empty((N, Co, F_out2, T_out2), device=device, dtype=dtype)

        grid2 = (N, Co, F_out2, T_out2)
        conv2d_stride2_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            N, Co, Co, F_out1, T_out1, F_out2, T_out2,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
        )

        # Conv3: x2 -> [N, Co, F_out3, T_out3]
        F_out3 = (F_out2 - 3) // 2 + 1
        T_out3 = (T_out2 - 3) // 2 + 1
        x3 = torch.empty((N, Co, F_out3, T_out3), device=device, dtype=dtype)

        grid3 = (N, Co, F_out3, T_out3)
        conv2d_stride2_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            N, Co, Co, F_out2, T_out2, F_out3, T_out3,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
        )

        # Reshape to [N, T_out3, Co * F_out3]
        x3_reshaped = x3.permute(0, 3, 1, 2).contiguous().view(N, T_out3, Co * F_out3)

        # Linear projection: X [N, M=3840, T_out3], W [K=1024, M=3840] provided
        # We transpose W to [M, K] for kernel
        W_t = conv_out_weight.transpose(0, 1).contiguous()  # [3840, 1024], bf16

        Y = torch.empty((N, T_out3, 1024), device=device, dtype=torch.bfloat16)

        BLOCK_M = 128
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

        # Scale by embed_scale
        grid_scale = (N, T_out3, 1024)
        scale_kernel[grid_scale](
            Y, float(embed_scale),
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
        )

        # Add positional embedding [T_out3, 1024], dtype bf16
        pos_emb = positional_embedding[:T_out3, :].contiguous()  # [T_out3, 1024], bf16
        grid_add = (N, T_out3, 1024)
        add_pos_emb_kernel[grid_add](
            Y, pos_emb,
            N, T_out3, 1024,
            Y.stride(0), Y.stride(1), Y.stride(2),
            pos_emb.stride(0), pos_emb.stride(1),
        )

        return Y


# -------

# ... (middle omitted) ...


def run(*args):
    return ModelNew()(*args)
